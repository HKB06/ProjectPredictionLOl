"""Croise NOTRE watchlist Elo avec les cotes Polymarket (marché réellement pariable).

Pourquoi : nos meilleures ligues (LJL, TCL, LFL...) ne sont quasiment jamais cotées
chez les books grand public. Polymarket, lui, a une API publique gratuite et ses prix
en cents SONT des probabilités (≈ sans marge). On ne garde donc que les matchs qu'on
PEUT réellement parier, et on calcule l'edge = notre_proba − proba_implicite_marché.

API : Gamma (https://gamma-api.polymarket.com), sans clé.
  - /public-search?q=<équipe>     -> retrouve l'événement "LoL: A vs B (BOx) - Ligue"
  - /events?slug=<slug>           -> détail (liste des markets)
  - market.sportsMarketType == "moneyline"  -> le vainqueur de SÉRIE (Match Winner)

Sortie : WATCHLIST_PARIABLE.md (racine Projet_Perso) + résumé console trié par edge.

Usage :
    python -m src.update.polymarket
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import requests

from src.ingest.load_oracle import ROOT, load_config
from src.update.elo import RELIABLE_ACC
from src.update.paris import now_paris
from src.update.watchlist import _fmt_paris, _is_passed, build_rows, orient

BASE = "https://gamma-api.polymarket.com"
OUT_PATH = ROOT.parent / "WATCHLIST_PARIABLE.md"
EDGE_MIN = 0.04          # edge mini pour parler de "value" (sous ça = bruit/marge)
MAX_GAP_DAYS = 1         # écart toléré entre la date du marché et celle de notre match
TIMEOUT = 25


# ----------------------------------------------------------------------------- API
def _get(path: str, **params) -> object:
    r = requests.get(BASE + path, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _search_events(query: str) -> list[dict]:
    try:
        js = _get("/public-search", q=query, limit_per_type="20")
    except Exception:  # noqa: BLE001
        return []
    return js.get("events", []) if isinstance(js, dict) else []


def _event_by_slug(slug: str) -> dict | None:
    try:
        d = _get("/events", slug=slug)
    except Exception:  # noqa: BLE001
        return None
    return d[0] if isinstance(d, list) and d else None


# ------------------------------------------------------------------- appariement
def _parse_sides(title: str) -> tuple[str, str] | None:
    """'LoL: Gen.G vs KT Rolster (BO5) - LCK ...' -> ('Gen.G', 'KT Rolster')."""
    t = title.split(":", 1)[1] if ":" in title else title
    t = t.split("(")[0]
    if " - " in t:
        t = t.split(" - ")[0]
    low = t.lower()
    if " vs " not in low:
        return None
    i = low.index(" vs ")
    return t[:i].strip(), t[i + 4:].strip()


def _is_lol(title: str) -> bool:
    """Polymarket titre ses marchés « LoL: A vs B (BO5) - Ligue » ; le même affrontement
    existe aussi en Valorant ou en CS2 (G2 vs Fnatic)."""
    head = title.split(":", 1)[0].strip().lower() if ":" in title else ""
    return head in ("lol", "league of legends")


def _event_day(ev: dict) -> str:
    """Jour du MATCH en UTC. `startDate` n'est PAS utilisable : c'est la date de mise en
    ligne du marché, plusieurs jours avant le match."""
    for m in ev.get("markets") or []:
        raw = m.get("gameStartTime") or m.get("eventStartTime")
        if raw:
            return str(raw)[:10]
    return str(ev.get("startTime") or ev.get("eventDate") or "")[:10]


def _moneyline_market(event: dict) -> dict | None:
    mk = event.get("markets") or []
    for m in mk:
        if m.get("sportsMarketType") == "moneyline" and not m.get("closed", False):
            return m
    for m in mk:  # secours : groupe "Match Winner"
        if (m.get("groupItemTitle") or "").lower() == "match winner" and not m.get("closed", False):
            return m
    return None


def _implied(market: dict, a: str, b: str, title_straight: bool) -> tuple[float, float] | None:
    """Probas implicites Polymarket alignées sur (a, b).

    Les noms des issues désignent les équipes ; s'ils ne permettent pas de trancher, on suit
    l'ordre du titre (issue 0 = première équipe du titre), dont on sait s'il est dans notre
    ordre ou inversé (`title_straight`). Jamais l'ordre brut de Polymarket."""
    try:
        outs = json.loads(market["outcomes"])
        prices = [float(x) for x in json.loads(market["outcomePrices"])]
    except (KeyError, ValueError, TypeError):
        return None
    if len(outs) != 2 or len(prices) != 2:
        return None
    straight = orient(a, b, str(outs[0]), str(outs[1]))
    if straight is None:
        straight = title_straight
    return (prices[0], prices[1]) if straight else (prices[1], prices[0])


def find_market(a: str, b: str, when_iso: str) -> dict | None:
    """Retrouve le marché 'Match Winner' Polymarket pour a vs b proche de when_iso."""
    want_day = (when_iso or "")[:10]
    seen, cands = set(), []
    for q in (a, b):
        for ev in _search_events(q):
            sid = ev.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            cands.append(ev)

    # Même match = les DEUX équipes reconnues strictement, marché LoL, à la bonne date.
    # Deux marchés aussi bons l'un que l'autre -> on ne choisit pas (pas de prix plutôt
    # qu'un prix pris au hasard).
    best, best_key, tie, best_straight = None, None, False, True
    for ev in cands:
        title = ev.get("title", "") or ""
        sides = _parse_sides(title)
        if not sides or not _is_lol(title):
            continue
        straight = orient(a, b, sides[0], sides[1])
        if straight is None:
            continue
        sd = _event_day(ev)
        try:
            gap = abs((dt.date.fromisoformat(sd) - dt.date.fromisoformat(want_day)).days)
        except ValueError:
            continue                                  # date inconnue : on ne devine pas
        if gap > MAX_GAP_DAYS:
            continue
        key = (not ev.get("closed", False), -gap)   # ouvert d'abord, puis date la + proche
        if best_key is None or key > best_key:
            best, best_key, tie, best_straight = ev, key, False, straight
        elif key == best_key:
            tie = True
    if best is None or tie:
        return None

    full = _event_by_slug(best.get("slug")) or best
    m = _moneyline_market(full)
    if not m:
        return None
    imp = _implied(m, a, b, best_straight)
    if imp is None:
        return None
    return {
        "slug": full.get("slug"),
        "url": f"https://polymarket.com/event/{full.get('slug')}",
        "imp1": imp[0], "imp2": imp[1],
        "vol": float(m.get("volume") or 0),
        "closed": bool(m.get("closed", False)),
    }


# ------------------------------------------------------------------- enrichissement
def enrich(rows: list[dict]) -> list[dict]:
    out = []
    now = dt.datetime.now(dt.timezone.utc)
    for r in rows:
        r = dict(r)
        # Match commencé : le prix est un prix EN DIRECT (il a déjà encaissé les games
        # jouées), le comparer à notre proba d'avant-match fabrique une fausse value.
        if r.get("passed") or _is_passed(r.get("datetime", ""), now):
            out.append(r)
            continue
        pm = find_market(r["team1"], r["team2"], r.get("datetime", ""))
        if pm:
            r["pm"] = pm
            edge1 = r["p1"] - pm["imp1"]
            edge2 = r["p2"] - pm["imp2"]
            if edge1 >= edge2:
                r["value_team"], r["edge"], r["our_p"], r["mkt_p"] = r["team1"], edge1, r["p1"], pm["imp1"]
            else:
                r["value_team"], r["edge"], r["our_p"], r["mkt_p"] = r["team2"], edge2, r["p2"], pm["imp2"]
            r["actionable"] = (
                r["edge"] >= EDGE_MIN and r.get("reliable", False)
                and r.get("conf", False) and not r.get("xleague", False)
                and r.get("bettable", False)
            )
        out.append(r)
    return out


# ------------------------------------------------------------------------ sortie
def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _write_markdown(bettable: list[dict], not_listed: list[dict]) -> None:
    now = now_paris()
    lines = [
        "# Watchlist PARIABLE (Polymarket)",
        "",
        f"> Généré le **{now:%Y-%m-%d %H:%M}** (Paris) · {len(bettable)} matchs cotés sur Polymarket, "
        f"{len(not_listed)} de notre watchlist non cotés.",
        "",
        "**Lecture** : prix Polymarket (cents) = proba implicite ≈ **sans marge**. "
        "`edge = notre proba − proba marché`. ✅ = edge ≥ 4 % **ET** ligue fiable **ET** data ≥15 g "
        "**ET** pas cross-ligue. 🌪️ = ligue chaotique (edge = surtout du bruit, **ne pas suivre**).",
        "",
        "| Quand (Paris) | Ligue | Match | BO | Côté value | Notre proba | Marché | Edge | Vol $ | Signal |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(bettable, key=lambda x: -x.get("edge", -9)):
        pm = r["pm"]
        chaos = "🌪️" if not r.get("reliable", True) else ""
        xflag = "⚠️x" if r.get("xleague") else ""
        sig = "✅" if r.get("actionable") else (chaos or xflag or "—")
        edge = f"{r['edge'] * 100:+.1f} pts"
        lines.append(
            f"| {r['when']} | {r['league']} | [{r['team1']} vs {r['team2']}]({pm['url']}) | "
            f"BO{r['bestof']} | **{r['value_team']}** | {_fmt_pct(r['our_p'])} | "
            f"{_fmt_pct(r['mkt_p'])} | {edge} | {pm['vol']:,.0f} | {sig} |"
        )
    if not_listed:
        lines += ["", f"## Non cotés sur Polymarket ({len(not_listed)}) — injouables ici",
                  "*(nos bonnes ligues tier-2 type LJL/TCL/LFL : modèle fiable mais marché absent.)*", ""]
        for r in sorted(not_listed, key=lambda x: x["datetime"] or ""):
            lines.append(f"- {r['when']} · {r['league']} · {r['team1']} vs {r['team2']}")
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(days: int = 4, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    covered, _ = build_rows(days=days, cfg=cfg)
    enriched = enrich(covered)
    bettable = [r for r in enriched if "pm" in r]
    not_listed = [r for r in enriched if "pm" not in r]
    _write_markdown(bettable, not_listed)
    return {"bettable": bettable, "not_listed": not_listed, "path": OUT_PATH}


def main() -> None:
    res = generate(days=4)
    bettable, not_listed = res["bettable"], res["not_listed"]
    print(f"\n=== MATCHS PARIABLES SUR POLYMARKET ({len(bettable)}) ===\n")
    if not bettable:
        print("  (aucun match de notre watchlist coté sur Polymarket en ce moment)")
    for r in sorted(bettable, key=lambda x: -x.get("edge", -9)):
        chaos = " [chaos]" if not r.get("reliable", True) else ""
        xflag = " [x-ligue]" if r.get("xleague") else ""
        sig = "   >>> VALUE <<<" if r.get("actionable") else ""
        print(f"  {r['when']:14} {r['league']:5} {r['team1']} vs {r['team2']} (BO{r['bestof']})")
        print(f"       value: {r['value_team']:22} nous {_fmt_pct(r['our_p'])} | "
              f"marché {_fmt_pct(r['mkt_p'])} | edge {r['edge'] * 100:+.1f} pts"
              f"{chaos}{xflag}{sig}")
    print(f"\n  --- {len(not_listed)} matchs de notre watchlist NON cotés sur Polymarket "
          f"(injouables ici) ---")
    for r in sorted(not_listed, key=lambda x: x["datetime"] or ""):
        print(f"     {r['when']:14} {r['league']:5} {r['team1']} vs {r['team2']}")
    print(f"\nOK -> {res['path']}")


if __name__ == "__main__":
    main()

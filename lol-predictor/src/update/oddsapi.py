"""Croise NOTRE modèle Elo avec les cotes book réelles via odds-api.io.

Complément de Polymarket. odds-api.io couvre quelques ligues LoL (EMEA Masters, LCS,
Asia Masters, VCS...) chez 2 books « mous » (1xbet, GG.bet), avec **ML + Totals + cotes
LIVE + scores par map**. Couverture LoL limitée mais **gratuite**, et surtout exploitable
en **live** (auto-remplissage de la page « Série en cours »).

Discipline (identique au reste du projet) : une « value » n'est ACTIONNABLE que si
edge ≥ 4 pts **ET** ligue fiable **ET** data ≥15 g **ET** pas cross-ligue. En EM (chaos),
le book a souvent raison sur les favoris courts → on n'affiche pas de ✅ même à gros edge.

Clé API : variable d'environnement ODDS_API_KEY, sinon fichier `oddsapi.key` (gitignored).

Usage :
    python -m src.update.oddsapi            # scan pré-match + value vs modèle
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import requests

from src.ingest.load_oracle import ROOT, load_config
from src.update.elo import (RELIABLE_ACC, calibrate, compute_elo, resolve_league,
                            series_prob, win_prob)
from src.update.watchlist import MIN_GAMES_CONF, core_tokens, fuzzy_match, norm, norm_core

BASE = "https://api.odds-api.io/v3"
OUT_PATH = ROOT.parent / "WATCHLIST_ODDSAPI.md"
BOOKMAKERS = ("1xbet", "GG.bet")     # 2 books "mous" esports (plan gratuit = 2)
EDGE_MIN = 0.04
TIMEOUT = 40
PARIS_OFFSET = dt.timedelta(hours=2)  # CEST (affichage)


# --------------------------------------------------------------------------- clé
def load_key() -> str | None:
    """ODDS_API_KEY (env) en priorité, sinon fichier oddsapi.key (racine du projet)."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()
    for p in (ROOT / "oddsapi.key", ROOT.parent / "oddsapi.key"):
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    return None


# --------------------------------------------------------------------------- API
def _get(path: str, key: str, **params):
    params["apiKey"] = key
    r = requests.get(BASE + path, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _as_list(x, *keys):
    if isinstance(x, list):
        return x
    for k in keys:
        if isinstance(x, dict) and isinstance(x.get(k), list):
            return x[k]
    return []


def ensure_bookmakers(key: str, books=BOOKMAKERS) -> None:
    """Garantit que les `books` voulus sont sélectionnés (sinon /odds renvoie peu/rien)."""
    try:
        sel = _get("/bookmakers/selected", key)
        cur = set(sel.get("bookmakers") or []) if isinstance(sel, dict) else set()
        if set(books) - cur:
            requests.put(BASE + "/bookmakers/selected/select",
                         params={"bookmakers": ",".join(books), "apiKey": key},
                         timeout=TIMEOUT)
    except Exception:  # noqa: BLE001 - non bloquant (on passe les books en clair dans /odds)
        pass


def _league_name(ev: dict) -> str:
    lg = ev.get("league") or {}
    return lg.get("name", "") if isinstance(lg, dict) else str(lg)


def _is_lol(ev: dict) -> bool:
    return "league of legends" in _league_name(ev).lower()


def lol_events(key: str, days: int = 14, include_settled: bool = False) -> list[dict]:
    """Tous les events LoL des `days` prochains jours (paginé, cap 500/réponse)."""
    to = (dt.datetime.utcnow() + dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out, skip = [], 0
    while True:
        batch = _as_list(_get("/events", key, sport="esports", to=to, limit=500, skip=skip),
                         "events")
        if not batch:
            break
        out += batch
        if len(batch) < 500:
            break
        skip += 500
        if skip > 4000:
            break
    evs = [e for e in out if _is_lol(e)]
    if not include_settled:
        evs = [e for e in evs if e.get("status") not in ("settled", "cancelled")]
    return evs


def live_lol_events(key: str) -> list[dict]:
    return [e for e in _as_list(_get("/events/live", key), "events") if _is_lol(e)]


def _parse_markets(odds_obj: dict) -> dict:
    """Extrait ML (vainqueur série) + Totals d'une réponse /odds, par book + best."""
    bm = odds_obj.get("bookmakers") or {}
    ml: dict[str, tuple[float, float]] = {}
    totals: dict[str, list] = {}
    seen_markets: set[str] = set()
    for book, markets in bm.items():
        for m in markets or []:
            name = (m.get("name") or "").strip()
            seen_markets.add(name)
            odds = m.get("odds") or []
            if name.upper() == "ML" and odds:
                try:
                    ml[book] = (float(odds[0]["home"]), float(odds[0]["away"]))
                except (KeyError, ValueError, TypeError):
                    pass
            elif "total" in name.lower():
                rows = []
                for o in odds:
                    try:
                        rows.append((float(o["hdp"]), float(o["over"]), float(o["under"])))
                    except (KeyError, ValueError, TypeError):
                        continue
                if rows:
                    totals[book] = rows
    best_home = max((v[0] for v in ml.values()), default=None)
    best_away = max((v[1] for v in ml.values()), default=None)
    return {
        "ml": ml, "totals": totals, "markets": sorted(seen_markets),
        "best_home": best_home, "best_away": best_away,
        "urls": odds_obj.get("urls") or {}, "bookmakerIds": odds_obj.get("bookmakerIds") or {},
    }


def event_odds(key: str, event_id, books=BOOKMAKERS) -> dict | None:
    try:
        od = _get("/odds", key, eventId=event_id, bookmakers=",".join(books))
    except Exception:  # noqa: BLE001
        return None
    return _parse_markets(od)


def _guess_wins(totals: dict) -> int:
    """Estime le nombre de maps à gagner via le hdp des Totals (sinon BO3)."""
    hdps = [h for rows in totals.values() for (h, _o, _u) in rows]
    if hdps:
        mx = max(hdps)
        if mx >= 4:
            return 3   # BO5
        if mx >= 2:
            return 2   # BO3
        return 1       # BO1
    return 2           # défaut : BO3 (régulière régionale)


# ------------------------------------------------------------------- modèle Elo
class Scorer:
    """Charge l'état Elo une fois et score un match book -> proba calibrée + contexte."""

    def __init__(self, cfg: dict | None = None):
        st = compute_elo(cfg or load_config())
        self.elo, self.n, self.league = st["elo"], st["n"], st["league"]
        self.reliability, self.shrink, self.global_rel = (
            st["reliability"], st["shrink"], st["global_rel"])
        self._idx_full = {norm(t): t for t in self.elo}
        self._idx_core = {norm_core(t): t for t in self.elo}
        self._ttok = {t: core_tokens(t) for t in self.elo}

    def match(self, name: str) -> str | None:
        return (self._idx_full.get(norm(name)) or self._idx_core.get(norm_core(name))
                or fuzzy_match(core_tokens(name), self._ttok))

    def score(self, a: str, b: str, wins_needed: int, league_name: str = "") -> dict:
        code = resolve_league(league_name, self.reliability)
        if code is None:
            cand = [c for c in (self.league.get(a), self.league.get(b)) if c in self.reliability]
            code = min(cand, key=lambda c: self.reliability[c]) if cand else None
        rel = self.reliability.get(code, self.global_rel)
        shr = self.shrink.get(code, 1.0)
        pg_raw = win_prob(self.elo[a], self.elo[b])
        pg = calibrate(pg_raw, shr)
        p1 = series_prob(pg, wins_needed)
        return {
            "p1": p1, "p2": 1 - p1, "p_game": pg,
            "rel": rel, "reliable": rel >= RELIABLE_ACC,
            "league_code": code or "?",
            "conf": min(self.n[a], self.n[b]) >= MIN_GAMES_CONF,
            "xleague": self.league.get(a) != self.league.get(b),
            "elo1": self.elo[a], "elo2": self.elo[b],
            "n1": self.n[a], "n2": self.n[b],
        }


# ----------------------------------------------------------------------- scan
def _iso_to_paris(iso: str) -> str:
    try:
        d = dt.datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ") + PARIS_OFFSET
        return d.strftime("%a %d/%m %H:%M")
    except (TypeError, ValueError):
        return str(iso)


def scan(days: int = 14, key: str | None = None, scorer: Scorer | None = None) -> list[dict]:
    """Renvoie les matchs LoL cotés (odds-api.io) enrichis de NOTRE proba + edge/value."""
    key = key or load_key()
    if not key:
        raise RuntimeError("Clé odds-api.io absente (ODDS_API_KEY ou fichier oddsapi.key).")
    ensure_bookmakers(key)
    scorer = scorer or Scorer()
    rows: list[dict] = []
    for ev in lol_events(key, days=days):
        if ev.get("status") == "live":   # pré-match only : cote live vs proba pré-match = faux
            continue
        od = event_odds(key, ev["id"])
        if not od or (od["best_home"] is None and od["best_away"] is None):
            continue
        home, away = ev.get("home", ""), ev.get("away", "")
        a, b = scorer.match(home), scorer.match(away)
        wins = _guess_wins(od["totals"])
        row = {
            "id": ev["id"], "datetime": ev.get("date", ""),
            "when": _iso_to_paris(ev.get("date", "")),
            "league": _league_name(ev).replace("League of Legends - ", ""),
            "home": home, "away": away, "status": ev.get("status"),
            "bestof": wins * 2 - 1, "wins_needed": wins,
            "ml": od["ml"], "totals": od["totals"], "markets": od["markets"],
            "best_home": od["best_home"], "best_away": od["best_away"],
            "urls": od["urls"], "resolved": bool(a and b),
        }
        # désaccord entre books (sur le favori implicite home)
        homes = [v[0] for v in od["ml"].values()]
        if len(homes) >= 2:
            row["book_spread"] = abs(1 / min(homes) - 1 / max(homes))
        if a and b:
            sc = scorer.score(a, b, wins, _league_name(ev))
            row.update({"team1": a, "team2": b, **sc})
            eh = sc["p1"] - (1 / od["best_home"]) if od["best_home"] else -9
            ea = sc["p2"] - (1 / od["best_away"]) if od["best_away"] else -9
            if eh >= ea:
                row.update({"value_team": a, "value_side": "home", "edge": eh,
                            "our_p": sc["p1"], "best_odd": od["best_home"],
                            "mkt_p": (1 / od["best_home"]) if od["best_home"] else None})
            else:
                row.update({"value_team": b, "value_side": "away", "edge": ea,
                            "our_p": sc["p2"], "best_odd": od["best_away"],
                            "mkt_p": (1 / od["best_away"]) if od["best_away"] else None})
            row["trust"] = bool(sc["reliable"] and sc["conf"] and not sc["xleague"])
            row["has_value"] = row["edge"] is not None and row["edge"] > 0
            row["actionable"] = row["edge"] >= EDGE_MIN and row["trust"]
        else:
            row.update({"team1": home, "team2": away, "value_team": None,
                        "edge": None, "trust": False, "has_value": False,
                        "actionable": False})
        rows.append(row)
    rows.sort(key=lambda r: r.get("datetime") or "")
    return rows


# ------------------------------------------------------------------- live (page 3)
def _same_match(name_a: str, name_b: str, ha: str, hb: str) -> bool:
    sa, sb = core_tokens(name_a), core_tokens(name_b)
    ta, tb = core_tokens(ha), core_tokens(hb)
    return bool((sa & ta and sb & tb) or (sa & tb and sb & ta))


def live_series(team_a: str, team_b: str, key: str | None = None,
                books=BOOKMAKERS) -> dict | None:
    """Pour la page Série en cours : retrouve le match live/à venir et renvoie le score
    par map + les cotes ML live, ALIGNÉS sur (team_a, team_b)."""
    key = key or load_key()
    if not key:
        return None
    ensure_bookmakers(key)
    pool = []
    try:
        pool += live_lol_events(key)
    except Exception:  # noqa: BLE001
        pass
    try:
        pool += lol_events(key, days=2, include_settled=False)
    except Exception:  # noqa: BLE001
        pass

    ev = next((e for e in pool if _same_match(team_a, team_b, e.get("home", ""),
                                              e.get("away", ""))), None)
    if not ev:
        return None
    od = event_odds(key, ev["id"], books)
    sc = ev.get("scores") or {}
    home_maps, away_maps = int(sc.get("home") or 0), int(sc.get("away") or 0)
    a_is_home = bool(core_tokens(team_a) & core_tokens(ev.get("home", "")))
    ml = od["ml"] if od else {}
    odd_home = max((v[0] for v in ml.values()), default=None)
    odd_away = max((v[1] for v in ml.values()), default=None)
    return {
        "found": True, "status": ev.get("status"), "clock": ev.get("clock"),
        "when": _iso_to_paris(ev.get("date", "")),
        "league": _league_name(ev).replace("League of Legends - ", ""),
        "home": ev.get("home"), "away": ev.get("away"),
        "wa": home_maps if a_is_home else away_maps,
        "wb": away_maps if a_is_home else home_maps,
        "odd_a": (odd_home if a_is_home else odd_away),
        "odd_b": (odd_away if a_is_home else odd_home),
        "wins_needed": _guess_wins(od["totals"]) if od else 2,
        "url": (od["urls"].get(books[0]) or next(iter((od["urls"] or {}).values()), None))
        if od else None,
    }


# ------------------------------------------------------------------------ sortie
def _pct(x) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "—"


def _write_markdown(rows: list[dict]) -> None:
    now = dt.datetime.now() + PARIS_OFFSET
    actionable = [r for r in rows if r.get("actionable")]
    lines = [
        "# Watchlist BOOK (odds-api.io)",
        "",
        f"> Généré le **{now:%Y-%m-%d %H:%M}** (Paris) · {len(rows)} matchs LoL cotés "
        f"({', '.join(BOOKMAKERS)}) · {len(actionable)} value(s) actionnable(s).",
        "",
        "**Lecture** : `edge = notre proba série − 1/meilleure cote`. ✅ = edge ≥ 4 pts **ET** "
        "ligue fiable **ET** data ≥15 g **ET** pas cross-ligue. 🌪️ = ligue chaotique (EM…) : "
        "edge surtout du bruit → **ne pas suivre** (le book a souvent raison sur les favoris courts).",
        "",
        "| Quand (Paris) | Ligue | Match | BO | Côté value | Notre p | Cote | Marché | Edge | Signal |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    def _rank(x):
        return (x.get("actionable", False), x.get("trust", False),
                x.get("edge") if x.get("edge") is not None else -9)

    for r in sorted(rows, key=_rank, reverse=True):
        if not r.get("resolved"):
            continue
        chaos = "🌪️" if not r.get("reliable", True) else ""
        xflag = "⚠️x" if r.get("xleague") else ""
        sig = "✅" if r.get("actionable") else (chaos or xflag or "—")
        edge = f"{r['edge'] * 100:+.1f} pts" if r.get("edge") is not None else "—"
        odd = f"@{r['best_odd']:.2f}" if r.get("best_odd") else "—"
        side = f"**{r.get('value_team')}**" if r.get("has_value") else "—"
        lines.append(
            f"| {r['when']} | {r['league']} | {r['team1']} vs {r['team2']} | BO{r['bestof']} | "
            f"{side} | {_pct(r.get('our_p'))} | {odd} | "
            f"{_pct(r.get('mkt_p'))} | {edge} | {sig} |"
        )
    unres = [r for r in rows if not r.get("resolved")]
    if unres:
        lines += ["", f"## Cotés mais hors de notre data Elo ({len(unres)})",
                  "*(équipes absentes du CSV Oracle — on ne peut pas modéliser.)*", ""]
        for r in unres:
            lines.append(f"- {r['when']} · {r['league']} · {r['home']} vs {r['away']}")
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(days: int = 14) -> dict:
    rows = scan(days=days)
    _write_markdown(rows)
    return {"rows": rows, "path": OUT_PATH,
            "actionable": [r for r in rows if r.get("actionable")]}


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if not load_key():
        print("[oddsapi] Cle absente : pose ODDS_API_KEY ou un fichier oddsapi.key.")
        return
    res = generate(days=14)
    rows = res["rows"]
    print(f"\n=== MATCHS LoL COTES (odds-api.io, {', '.join(BOOKMAKERS)}) : {len(rows)} ===\n")
    if not rows:
        print("  (aucun match LoL cote en ce moment chez ces books)")
    for r in sorted(rows, key=lambda x: x.get("datetime") or ""):
        if not r.get("resolved"):
            print(f"  {r['when']:14} {r['league']:18} {r['home']} vs {r['away']}  "
                  f"[hors data Elo]")
            continue
        chaos = " [chaos]" if not r.get("reliable", True) else ""
        xfl = " [x-ligue]" if r.get("xleague") else ""
        tag = "   >>> VALUE <<<" if r.get("actionable") else ""
        odd = f"@{r['best_odd']:.2f}" if r.get("best_odd") else ""
        edge = f"{r['edge']*100:+.1f} pts" if r.get("edge") is not None else ""
        lead = f"value {r.get('value_team')} {odd}" if r.get("has_value") else "pas de value"
        print(f"  {r['when']:14} {r['league']:18} {r['team1']} vs {r['team2']} (BO{r['bestof']})")
        print(f"       {lead}  nous {_pct(r.get('our_p'))} | "
              f"marche {_pct(r.get('mkt_p'))} | edge {edge}{chaos}{xfl}{tag}")
    print(f"\nOK -> {res['path']}")


if __name__ == "__main__":
    main()

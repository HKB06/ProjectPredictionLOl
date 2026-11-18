"""Archive des prix Polymarket d'AVANT-MATCH pour tous les matchs LoL d'une saison.

Pourquoi : savoir si un modèle bat l'ancien ne dit pas s'il fait gagner de l'argent.
Le seul juge est le marché. Polymarket cote la quasi-totalité des matchs LoL
(~3 500 en 2026, contre ~700 dans `odds_history.csv`), son API est publique et sans
clé, et ses prix sont directement des probabilités (pas de marge de bookmaker).

Pour chaque match on archive :
  - le marché « Match Winner » (vainqueur de la SÉRIE) ;
  - le marché « Game 1 Winner » quand il existe (comparable à nos probas par game).
Prix relevés à l'heure de début annoncée, 30 min, 1 h et 24 h avant : si la game a
en réalité démarré plus tôt que prévu, `src.v2.market` retombe sur un relevé antérieur
au vrai coup d'envoi (jamais de prix « live » pris pour un prix d'avant-match).

API :
  - Gamma  /events?tag_id=…&closed=true&end_date_min/max   -> matchs terminés
  - CLOB   /prices-history?market=<token>&startTs&endTs&fidelity=10 -> prix (pas de 10 min)

Cache reprenable : `data/odds/polymarket_<année>.csv` (une ligne par match). Relancer
ne redemande que les matchs manquants.

Usage :
    python -m src.update.polymarket_history            # saison en cours
    python -m src.update.polymarket_history --year 2026 --stats
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.ingest.load_oracle import ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
TAG_SLUG = "league-of-legends"
HEADERS = {"User-Agent": "lol-predictor/0.2 (archive des prix d'avant-match)"}
TIMEOUT = 40
WORKERS = 4
LOOKBACK_H = 48            # fenêtre de prix demandée avant le coup d'envoi
FIDELITY_MIN = 10
OFFSETS = {"p_close": 0, "p_30m": 30, "p_1h": 60, "p_24h": 24 * 60}   # minutes avant le début
COLS = ["event_id", "slug", "title", "label", "team_a", "team_b", "bestof", "start",
        "volume", "winner", *OFFSETS, "n_pts",
        "g1_volume", "g1_winner", *(f"g1_{k}" for k in OFFSETS)]

_BO = re.compile(r"\(BO(\d)\)", re.I)


def cache_path(year: int):
    return ROOT / "data" / "odds" / f"polymarket_{year}.csv"


def _get(url: str, **params):
    """GET avec reprises (429 / erreurs réseau passagères)."""
    err: Exception | None = None
    for i in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            err = requests.HTTPError(f"{r.status_code} {url}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as exc:
            err = exc
        time.sleep(1.5 * (i + 1))
    raise err or RuntimeError(url)


# ------------------------------------------------------------------ découverte
def list_events(year: int) -> list[dict]:
    """Tous les événements LoL « A vs B » terminés de l'année (un par match)."""
    tag_id = _get(f"{GAMMA}/tags/slug/{TAG_SLUG}")["id"]
    out: dict[str, dict] = {}
    for month in range(1, 13):
        lo = f"{year}-{month:02d}-01T00:00:00Z"
        hi = f"{year + month // 12}-{month % 12 + 1:02d}-01T00:00:00Z"
        offset = 0
        while offset < 2000:                       # l'API refuse les offsets plus grands
            batch = _get(f"{GAMMA}/events", tag_id=tag_id, closed="true", limit=100,
                         offset=offset, end_date_min=lo, end_date_max=hi)
            if not batch:
                break
            for ev in batch:
                if " vs " in (ev.get("title") or "").lower():
                    out[str(ev["id"])] = ev
            offset += 100
    return list(out.values())


def _loads(x) -> list:
    try:
        return json.loads(x) if isinstance(x, str) else list(x or [])
    except ValueError:
        return []


def _start(market: dict) -> dt.datetime | None:
    raw = market.get("gameStartTime") or market.get("eventStartTime")
    if not raw:
        return None
    try:
        t = pd.Timestamp(raw)
    except ValueError:
        return None
    return (t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")).to_pydatetime()


def _winner(market: dict) -> str | None:
    """'a' / 'b' selon l'issue réglée, None si le marché n'a pas tranché (annulé, 50/50)."""
    prices = [float(p) for p in _loads(market.get("outcomePrices")) or []]
    if len(prices) != 2:
        return None
    if prices[0] > 0.99:
        return "a"
    if prices[1] > 0.99:
        return "b"
    return None


def _pick_markets(ev: dict) -> tuple[dict | None, dict | None]:
    series = game1 = None
    for m in ev.get("markets") or []:
        kind, group = m.get("sportsMarketType"), (m.get("groupItemTitle") or "").lower()
        if kind == "moneyline" or (series is None and group == "match winner"):
            series = m
        elif kind == "child_moneyline" and group == "game 1 winner":
            game1 = m
    return series, game1


# ---------------------------------------------------------------------- prix
def _prices(token: str, start: dt.datetime) -> tuple[dict, int]:
    """Prix du token aux instants `OFFSETS` avant `start` (dernier relevé connu à cet instant)."""
    end = int(start.timestamp())
    hist = _get(f"{CLOB}/prices-history", market=token, startTs=end - LOOKBACK_H * 3600,
                endTs=end, fidelity=FIDELITY_MIN).get("history") or []
    pts = sorted((int(h["t"]), float(h["p"])) for h in hist)
    out = {}
    for name, minutes in OFFSETS.items():
        cut = end - minutes * 60
        before = [p for t, p in pts if t <= cut]
        out[name] = before[-1] if before else float("nan")
    return out, len(pts)


def _row(ev: dict) -> dict | None:
    series, game1 = _pick_markets(ev)
    if series is None:
        return None
    teams, tokens, start = _loads(series.get("outcomes")), _loads(series.get("clobTokenIds")), _start(series)
    if len(teams) != 2 or len(tokens) != 2 or start is None:
        return None
    title = ev.get("title") or ""
    bo = _BO.search(title)
    row = {
        "event_id": str(ev["id"]), "slug": ev.get("slug"), "title": title,
        "label": title.split(" - ", 1)[1].strip() if " - " in title else "",
        "team_a": teams[0], "team_b": teams[1],
        "bestof": int(bo.group(1)) if bo else None,
        "start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "volume": float(series.get("volume") or 0), "winner": _winner(series),
    }
    prices, n = _prices(tokens[0], start)
    row.update(prices)
    row["n_pts"] = n
    if game1 is not None:
        g_teams, g_tokens = _loads(game1.get("outcomes")), _loads(game1.get("clobTokenIds"))
        if len(g_teams) == 2 and len(g_tokens) == 2:
            # même ordre que la série : on lit le token de `team_a`
            idx = 0 if g_teams[0] == teams[0] else 1
            gp, _ = _prices(g_tokens[idx], start)
            row.update({f"g1_{k}": v for k, v in gp.items()})
            row["g1_volume"] = float(game1.get("volume") or 0)
            w = _winner(game1)
            row["g1_winner"] = w if idx == 0 or w is None else ("b" if w == "a" else "a")
    return row


# ---------------------------------------------------------------------- cache
def load_cache(year: int) -> pd.DataFrame:
    path = cache_path(year)
    if not path.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(path, dtype={"event_id": str})
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLS]


def backfill(year: int, limit: int | None = None) -> dict:
    """Complète le cache : ne demande que les matchs absents. Reprenable."""
    cache = load_cache(year)
    known = set(cache["event_id"])
    todo = [e for e in list_events(year) if str(e["id"]) not in known]
    if limit:
        todo = todo[:limit]
    rows, failed = [], 0

    def save() -> None:
        path = cache_path(year)
        path.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows, columns=COLS)
        tmp = path.with_name(path.name + ".tmp")      # écriture atomique : le site lit ce fichier
        (pd.concat([cache, new], ignore_index=True) if len(cache) else new).to_csv(tmp, index=False)
        tmp.replace(path)

    def safe(ev: dict):
        try:
            return _row(ev)
        except Exception:  # noqa: BLE001 - un match en erreur sera redemandé au prochain passage
            return "failed"

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, row in enumerate(pool.map(safe, todo), 1):
            if row == "failed":
                failed += 1
            elif row:
                rows.append(row)
            if i % 200 == 0:
                save()
                print(f"  {i}/{len(todo)} matchs…", flush=True)
    save()
    return {"new": len(rows), "failed": failed, "skipped": len(todo) - len(rows) - failed,
            "total": len(cache) + len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive des prix Polymarket d'avant-match (LoL)")
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--limit", type=int, default=None, help="nombre max de nouveaux matchs")
    ap.add_argument("--stats", action="store_true", help="état du cache, sans requête")
    args = ap.parse_args()
    if not args.stats:
        res = backfill(args.year, args.limit)
        print(f"{res['new']} matchs ajoutés, {res['failed']} en erreur (redemandés au prochain "
              f"passage), {res['skipped']} sans marché exploitable.")
    c = load_cache(args.year)
    priced = c["p_close"].notna()
    print(f"Cache {args.year} : {len(c)} matchs, {int(priced.sum())} avec prix d'avant-match, "
          f"{int(c['g1_p_close'].notna().sum())} avec prix de la game 1.")
    if len(c):
        print(f"Période : {c['start'].min()[:10]} -> {c['start'].max()[:10]} · volume médian "
              f"{c.loc[priced, 'volume'].median():,.0f} $".replace(",", " "))


if __name__ == "__main__":
    main()

"""Le modèle « avec draft » bat-il le prix Polymarket une fois la draft connue ?

Polymarket cote un marché « Game 1 Winner » sur la plupart des matchs. Pour chaque série des
compétitions pariables, on relève le prix de ce marché autour du VRAI début de la game 1
(heure Oracle, donc après la draft) :

    p_m20 : 20 min avant le début en jeu   -> avant la draft
    p_m2  :  2 min avant                   -> draft terminée, game pas commencée
    p_0   : au début de la game
    p_p3  :  3 min après

et on compare, sur exactement les mêmes games :
    - le prix d'avant la draft et le prix d'après ;
    - notre modèle d'avant-match et notre modèle avec draft (probas hors échantillon).
Puis : notre modèle avec draft ajoute-t-il de l'information au prix d'après la draft, et
qu'aurait rapporté de parier nos désaccords à ce prix ?

Cache reprenable : data/odds/polymarket_g1_live_<année>.csv.

Usage :
    python -m src.research.draft_live            # relève ce qui manque puis analyse
    python -m src.research.draft_live --no-fetch
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ingest.load_oracle import ROOT, load_config
from src.update import polymarket_history as ph
from src.update.bet_universe import universe
from src.v2 import market as mk
from src.v2 import model as m
from src.v2 import precompute

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OFFSETS = {"p_m20": -20, "p_m10": -10, "p_m5": -5, "p_m2": -2, "p_0": 0, "p_p3": 3}
COLS = ["event_id", "t0", *OFFSETS, "n_pts"]
SLIPPAGE = 0.01
EDGES = (0.03, 0.05, 0.08, 0.12)


def cache_path(year: int):
    return ROOT / "data" / "odds" / f"polymarket_g1_live_{year}.csv"


def matched(eng: dict, year: int) -> pd.DataFrame:
    """Séries pariables rapprochées d'un marché : event_id, sens, début réel de la game 1."""
    s = eng["series"].copy()
    games = eng["games"]
    first = games.sort_values("date").drop_duplicates("series_id").set_index("series_id")
    cache = ph.load_cache(year)
    cache = cache.assign(start=pd.to_datetime(cache["start"]))
    resolve = mk._Resolver(pd.concat([s["team_a"], s["team_b"]]).unique())
    index: dict[frozenset, list] = {}
    for i, a, b, d in zip(s.index, s["team_a"], s["team_b"], s["date"]):
        index.setdefault(frozenset((a, b)), []).append((d, i))
    used, rows = set(), []
    for r in cache.itertuples():
        a, b = resolve(r.team_a), resolve(r.team_b)
        if not a or not b or a == b:
            continue
        cand = [(abs(d - r.start), i, d) for d, i in index.get(frozenset((a, b)), [])]
        if not cand:
            continue
        gap, i, _ = min(cand)
        if gap > mk.MATCH_WINDOW or i in used:
            continue
        used.add(i)
        sid = s.at[i, "series_id"]
        g = first.loc[sid]
        rows.append({"event_id": str(r.event_id), "series_id": sid, "league": s.at[i, "league"],
                     "flip": s.at[i, "team_a"] != a, "t0": g["date"], "gameid": g["gameid"],
                     "y": float(g["winner"] == g["blue"]), "rodage": bool(s.at[i, "rodage"]),
                     "low_data": bool(s.at[i, "low_data"]), "g1_volume": r.g1_volume,
                     "g1_pre": r.g1_p_close})
    return pd.DataFrame(rows)


def _fetch(event_id: str, t0: pd.Timestamp) -> dict | None:
    """Prix du marché Game 1 (côté du 1er nom Polymarket) autour de `t0`."""
    try:
        ev = ph._get(f"{ph.GAMMA}/events/{event_id}")
    except Exception:  # noqa: BLE001
        return None
    series, game1 = ph._pick_markets(ev)
    if series is None or game1 is None:
        return {"event_id": event_id, "t0": t0, "n_pts": 0}
    teams, g_teams = ph._loads(series.get("outcomes")), ph._loads(game1.get("outcomes"))
    tokens = ph._loads(game1.get("clobTokenIds"))
    if len(teams) != 2 or len(g_teams) != 2 or len(tokens) != 2:
        return {"event_id": event_id, "t0": t0, "n_pts": 0}
    idx = 0 if g_teams[0] == teams[0] else 1
    ts = int(pd.Timestamp(t0, tz="UTC").timestamp())
    try:
        hist = ph._get(f"{ph.CLOB}/prices-history", market=tokens[idx], startTs=ts - 3 * 3600,
                       endTs=ts + 15 * 60, fidelity=1).get("history") or []
    except Exception:  # noqa: BLE001
        return None
    pts = sorted((int(h["t"]), float(h["p"])) for h in hist)
    row = {"event_id": event_id, "t0": t0, "n_pts": len(pts)}
    for name, minutes in OFFSETS.items():
        before = [p for t, p in pts if t <= ts + minutes * 60]
        row[name] = before[-1] if before else np.nan
    return row


def fetch(todo: pd.DataFrame, year: int) -> pd.DataFrame:
    path = cache_path(year)
    have = pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame(columns=COLS)
    missing = todo[~todo["event_id"].isin(set(have["event_id"]))]
    print(f"prix à relever : {len(missing)} (déjà en cache : {len(have)})")
    rows = []
    with ThreadPoolExecutor(ph.WORKERS) as pool:
        for n, r in enumerate(pool.map(lambda x: _fetch(*x), zip(missing["event_id"], missing["t0"])), 1):
            if r is not None:
                rows.append(r)
            if n % 200 == 0:
                print(f"  {n}/{len(missing)}…", flush=True)
                pd.concat([have, pd.DataFrame(rows)], ignore_index=True)[COLS].to_csv(path, index=False)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out = out.reindex(columns=COLS)
    out.to_csv(path, index=False)
    return out


def _ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _ci(x, rng, n=2000):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def analyse(d: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    y = d["y"].to_numpy()
    print(f"\n=== {len(d)} games 1 (pariables, marché Game 1 avec prix avant et après la draft) ===")
    ref = _ll(y, d["post"].to_numpy())
    rows = []
    for label, col in (("Prix AVANT la draft (20 min avant)", "pre"), ("Prix APRÈS la draft (2 min avant)", "post"),
                       ("Prix au début de la game", "at0"),
                       ("Notre modèle d'avant-match", "p_full"), ("Notre modèle AVEC draft", "p_live")):
        p = d[col].to_numpy()
        diff = _ll(y, p) - ref
        lo, hi = _ci(diff, rng)
        rows.append({"source": label, "log-loss": round(float(_ll(y, p).mean()), 4),
                     "bon vainqueur %": round(float(((p >= 0.5) == (y == 1)).mean() * 100), 1),
                     "écart au prix d'après draft (x1000)": round(float(diff.mean() * 1000), 1),
                     "IC 95 %": f"[{lo * 1000:+.1f} ; {hi * 1000:+.1f}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    move = (d["post"] - d["pre"]).abs()
    print(f"\nLe prix bouge de {move.mean() * 100:.1f} points en moyenne pendant la draft "
          f"(médiane {move.median() * 100:.1f}, 9e décile {move.quantile(.9) * 100:.1f}).")
    ours = d["p_live"] - d["p_full"]
    same = np.sign(d["post"] - d["pre"]) == np.sign(ours)
    big = move >= 0.03
    print(f"Quand le prix bouge d'au moins 3 points ({int(big.sum())} games), il va dans le même "
          f"sens que notre lecture de la draft {same[big].mean():.0%} du temps.")

    # le modèle avec draft ajoute-t-il de l'info au prix d'après draft ? (1re moitié -> 2e moitié)
    h = len(d) // 2
    tr, te = d.iloc[:h], d.iloc[h:]
    X = lambda q, cols: np.column_stack([mk._logit(q[c].to_numpy()) for c in cols])
    base = LogisticRegression(C=1e6).fit(X(tr, ["post"]), tr["y"])
    both = LogisticRegression(C=1e6).fit(X(tr, ["post", "p_live"]), tr["y"])
    yb = te["y"].to_numpy()
    diff = _ll(yb, base.predict_proba(X(te, ["post"]))[:, 1]) - _ll(yb, both.predict_proba(X(te, ["post", "p_live"]))[:, 1])
    lo, hi = _ci(diff, rng)
    print(f"\nAjouter notre modèle avec draft au prix d'après draft (appris sur {len(tr)} games, jugé "
          f"sur {len(te)}) : poids du prix {both.coef_[0][0]:.2f}, poids du modèle {both.coef_[0][1]:.2f}, "
          f"gain {diff.mean() * 1000:+.1f} [{lo * 1000:+.1f} ; {hi * 1000:+.1f}]")

    print(f"\n=== PARIS SIMULÉS au prix d'APRÈS la draft (prix + {SLIPPAGE}) ===")
    rows = []
    for col, name in (("p_live", "modèle avec draft"), ("p_full", "modèle d'avant-match")):
        for edge in EDGES:
            gap = d[col] - d["post"]
            side_a = gap >= edge
            side_b = gap <= -edge
            price = np.where(side_a, d["post"], 1 - d["post"]) + SLIPPAGE
            won = np.where(side_a, d["y"] == 1, d["y"] == 0)
            sel = (side_a | side_b).to_numpy()
            if not sel.any():
                continue
            profit = np.where(won[sel], 1 / np.clip(price[sel], 0.02, 0.99) - 1, -1.0)
            lo, hi = _ci(profit, rng)
            hh = len(profit) // 2
            rows.append({"modèle": name, "écart mini": f"{edge * 100:.0f} pts", "paris": int(sel.sum()),
                         "gagnés %": round(float(won[sel].mean() * 100), 1),
                         "prix moyen": round(float(price[sel].mean()), 2),
                         "rendement %": round(float(profit.mean() * 100), 1),
                         "IC 95 %": f"[{lo * 100:+.1f} ; {hi * 100:+.1f}]",
                         "moitiés": f"{profit[:hh].mean() * 100:+.1f} / {profit[hh:].mean() * 100:+.1f}"})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Modèle avec draft contre le prix Polymarket en direct")
    ap.add_argument("--no-fetch", action="store_true", help="analyse le cache sans rien relever")
    args = ap.parse_args()
    pd.set_option("display.width", 230)
    cfg = load_config()
    eng = precompute.peek()
    if eng is None:
        sys.exit("Pas de moteur calculé : lancer d'abord python -m src.v2.precompute")
    year = int(pd.to_datetime(eng["series"]["date"]).dt.year.max())
    leagues, events = universe(cfg)
    s = matched(eng, year)
    s = s[s["league"].isin(leagues | events) & s["g1_pre"].notna() & ~s["rodage"] & ~s["low_data"]]
    print(f"{len(s)} séries pariables avec un marché Game 1")
    path = cache_path(year)
    if args.no_fetch:
        live = pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame(columns=COLS)
    else:
        live = fetch(s, year)
    d = s.merge(live.drop(columns="t0"), on="event_id")
    res = eng["res"].drop_duplicates("gameid").set_index("gameid")
    d["p_full"] = d["gameid"].map(res[m.FULL])
    d["p_live"] = d["gameid"].map(res[m.LIVE])
    for src, dst in (("p_m20", "pre"), ("p_m2", "post"), ("p_0", "at0")):
        v = pd.to_numeric(d[src], errors="coerce").to_numpy(dtype=float)
        d[dst] = np.where(d["flip"].astype(bool).to_numpy(), 1 - v, v)
    d = d.dropna(subset=["pre", "post", "at0", "p_full", "p_live"])
    d = d[(d["g1_volume"] >= mk.MIN_VOLUME) & (d["n_pts"] >= 10)].sort_values("t0").reset_index(drop=True)
    if len(d) < 100:
        sys.exit(f"Trop peu de games avec prix ({len(d)})")
    analyse(d)


if __name__ == "__main__":
    main()

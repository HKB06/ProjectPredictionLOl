"""Prévisibilité comparée des ligues (LCK vs mineures) — répond à :
"les ligues mineures sont-elles plus 50-50 / plus dures à prédire ?"

Méthode : Elo walk-forward (sans fuite) PAR ligue, K=24, warmup 5 games/équipe.
Métriques : accuracy du favori Elo, Brier, AUC, log-loss. Plus prévisible = acc/AUC ↑, Brier ↓.
Aucune cote nécessaire (mesure interne de prévisibilité, pas d'edge).

Usage : python -m src.models.league_predictability
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ingest.load_oracle import ROOT, load_config

K = 24.0
WARMUP = 5
MIN_SCORED = 80   # ligues avec assez de matchs notés pour être fiables


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(ra - rb) / 400.0))


def eval_league(games: pd.DataFrame) -> dict | None:
    elo: dict[str, float] = defaultdict(lambda: 1500.0)
    played: dict[str, int] = defaultdict(int)
    ps, ys = [], []
    for _, g in games.iterrows():
        a, b = g["blue_team"], g["red_team"]
        p = expected(elo[a], elo[b])
        if played[a] >= WARMUP and played[b] >= WARMUP:
            ps.append(p)
            ys.append(int(g["y_blue_win"]))
        # update
        outcome = float(g["y_blue_win"])
        elo[a] += K * (outcome - p)
        elo[b] += K * ((1 - outcome) - (1 - p))
        played[a] += 1
        played[b] += 1

    if len(ys) < MIN_SCORED:
        return None
    ps, ys = np.array(ps), np.array(ys)
    pred = (ps >= 0.5).astype(int)
    acc = (pred == ys).mean()
    brier = np.mean((ps - ys) ** 2)
    ll = -np.mean(ys * np.log(np.clip(ps, 1e-6, 1)) + (1 - ys) * np.log(np.clip(1 - ps, 1e-6, 1)))
    try:
        auc = roc_auc_score(ys, ps)
    except ValueError:
        auc = float("nan")
    base = max(ys.mean(), 1 - ys.mean())   # baseline "toujours le côté majoritaire"
    return {"n": len(ys), "base%": base * 100, "acc%": acc * 100,
            "AUC": auc, "Brier": brier, "logloss": ll}


def main() -> None:
    cfg = load_config()
    csv_path = ROOT / cfg["data"]["oracle_csv"]
    df = pd.read_csv(csv_path, low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    teams = df[df["position"].str.lower() == "team"].copy()
    rows = []
    for gid, g in teams.groupby("gameid"):
        blue = g[g["side"].str.lower() == "blue"]
        red = g[g["side"].str.lower() == "red"]
        if len(blue) != 1 or len(red) != 1:
            continue
        blue, red = blue.iloc[0], red.iloc[0]
        rows.append({"league": blue["league"], "date": blue["date"],
                     "blue_team": blue["teamname"], "red_team": red["teamname"],
                     "y_blue_win": int(blue["result"])})
    matches = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")

    print("=" * 84)
    print(f"  PRÉVISIBILITÉ PAR LIGUE (Elo walk-forward, 2026)  | total games={len(matches)}")
    print("=" * 84)
    print(f"  {'ligue':<10} {'games':>6} {'base%':>7} {'acc%':>7} {'AUC':>6} {'Brier':>7} {'logloss':>8}")
    print("  " + "-" * 80)

    results = []
    for lg, g in matches.groupby("league"):
        res = eval_league(g.sort_values("date"))
        if res:
            results.append((lg, res))
    # tri par AUC décroissant (plus prévisible en haut)
    for lg, r in sorted(results, key=lambda x: -x[1]["AUC"]):
        tag = "  <- LCK" if lg == "LCK" else ""
        print(f"  {lg:<10} {r['n']:>6} {r['base%']:>6.1f}% {r['acc%']:>6.1f}% "
              f"{r['AUC']:>6.3f} {r['Brier']:>7.3f} {r['logloss']:>8.3f}{tag}")

    print("=" * 84)
    print("  Plus previsible = AUC/acc en haut, Brier bas. (mesure la PREVISIBILITE, pas l'edge)")
    print("  Edge = écart modèle vs cote -> nécessite les cotes de la ligue.")
    print("=" * 84)


if __name__ == "__main__":
    main()

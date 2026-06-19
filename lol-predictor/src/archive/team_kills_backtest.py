"""Backtest du modèle attaque/défense des KILLS PAR ÉQUIPE (sans fuite).

But : quantifier l'ERREUR réelle de la prédiction kills par équipe, pour savoir si un
écart de ~1 kill avec le book (vu sur KT) est un vrai edge ou du bruit.

pred_kills(A vs B) = (kills_pour_A_passés + kills_contre_B_passés) / 2   (expanding, no leak)

Usage : python -m src.models.team_kills_backtest
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.ingest.load_oracle import ROOT, load_config


def main() -> None:
    cfg = load_config()
    tg = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "team_games.parquet")
    tg = tg.sort_values(["date", "gameid"]).reset_index(drop=True)

    fors: dict[str, list] = defaultdict(list)      # kills infligés (attaque)
    againsts: dict[str, list] = defaultdict(list)   # kills encaissés (défense)
    league: list[float] = []

    preds, actuals, base = [], [], []
    MIN = 5

    for gid, g in tg.groupby("gameid", sort=False):
        if len(g) != 2:
            continue
        r1, r2 = g.iloc[0], g.iloc[1]
        pairs = [(r1["teamname"], r2["teamname"], r1["kills"], r1["deaths"]),
                 (r2["teamname"], r1["teamname"], r2["kills"], r2["deaths"])]
        for team, opp, k_for, k_against in pairs:
            if len(fors[team]) >= MIN and len(againsts[opp]) >= MIN and league:
                pred = (np.mean(fors[team]) + np.mean(againsts[opp])) / 2
                preds.append(pred)
                actuals.append(float(k_for))
                base.append(float(np.mean(league)))
        # MAJ après prédiction (anti-fuite)
        for team, _opp, k_for, k_against in pairs:
            fors[team].append(float(k_for))
            againsts[team].append(float(k_against))
            league.append(float(k_for))

    preds, actuals, base = map(np.array, (preds, actuals, base))
    err = np.abs(preds - actuals)
    err_base = np.abs(base - actuals)

    print("=" * 64)
    print(f"  BACKTEST KILLS PAR ÉQUIPE  (n={len(preds)} prédictions, sans fuite)")
    print("=" * 64)
    print(f"  MAE modèle attaque/défense : {err.mean():.2f} kills")
    print(f"  MAE baseline (moyenne ligue): {err_base.mean():.2f} kills")
    gain = (err_base.mean() - err.mean()) / err_base.mean() * 100
    print(f"  Gain vs baseline           : {gain:+.1f}%")
    print(f"  Écart-type des kills réels : {actuals.std():.2f}")
    print(f"  % prédictions à ±1 kill    : {(err <= 1).mean() * 100:.0f}%")
    print(f"  % prédictions à ±3 kills   : {(err <= 3).mean() * 100:.0f}%")
    print("=" * 64)
    print("  -> Si MAE ~ 4-5 kills, un écart de 1 kill avec le book = BRUIT, pas un edge.")
    print("=" * 64)


if __name__ == "__main__":
    main()

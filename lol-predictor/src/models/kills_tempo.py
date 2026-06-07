"""Modèle TOTAL KILLS dédié au TEMPO (suite du finding style_edge).

Finding : le "kill volume" d'une équipe (kills + deaths par game = total kills de ses
games) est STABLE (split-half 0.60) et INDÉPENDANT de la force (corr winrate 0.12).
Donc le total kills d'un match devrait se prédire surtout via le tempo des 2 équipes.

On teste si un modèle SIMPLE basé tempo bat (a) la moyenne et (b) le LGBM générique.
Puis test "betting-relevant" : contre une ligne PARESSEUSE (= moyenne glissante, ce
qu'un book peu attentif posterait), notre prédiction tempo donne-t-elle le bon côté
(over/under) assez souvent pour battre la marge (~53-54% requis) ?

Anti-fuite : tempo = kills_avg + deaths_avg AS-OF (déjà sans fuite dans features.parquet) ;
ligne paresseuse = moyenne des totals des games STRICTEMENT antérieures.

Usage : python -m src.models.kills_tempo
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.ingest.load_oracle import ROOT, load_config

warnings.filterwarnings("ignore")


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet").sort_values(["date", "gameid"]).reset_index(drop=True)
    feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()

    # tempo "total kills" de chaque équipe (as-of) = kills encaissés + infligés
    feats["blue_tempo"] = feats["blue_kills_avg"] + feats["blue_deaths_avg"]
    feats["red_tempo"] = feats["red_kills_avg"] + feats["red_deaths_avg"]
    feats = feats.dropna(subset=["y_total_kills", "blue_tempo", "red_tempo"])

    n_test = max(1, int(cfg["split"]["test_fraction"] * len(feats)))
    train, test = feats.iloc[:-n_test], feats.iloc[-n_test:]
    ytr, yte = train["y_total_kills"], test["y_total_kills"]

    print("=" * 70)
    print(f"  MODÈLE TOTAL KILLS — TEMPO  |  train={len(train)}  test={len(test)}")
    print("=" * 70)

    # (a) baseline = moyenne d'entraînement
    mae_base = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))

    # (b) tempo "naïf" = moyenne des 2 tempos
    pred_avg = (test["blue_tempo"] + test["red_tempo"]) / 2
    mae_avg = mean_absolute_error(yte, pred_avg)

    # (c) régression linéaire sur les 2 tempos
    lin = LinearRegression().fit(train[["blue_tempo", "red_tempo"]], ytr)
    pred_lin = lin.predict(test[["blue_tempo", "red_tempo"]])
    mae_lin = mean_absolute_error(yte, pred_lin)

    # (d) LGBM générique (toutes features) pour comparaison
    fcols = [c for c in feats.columns if not c.startswith("y_")
             and c not in ("gameid", "date", "blue_tempo", "red_tempo")]
    lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=42, verbose=-1)
    lgbm.fit(train[fcols], ytr)
    mae_lgbm = mean_absolute_error(yte, lgbm.predict(test[fcols]))

    print(f"  MAE baseline (moyenne)        : {mae_base:.2f}")
    print(f"  MAE tempo (moyenne 2 tempos)  : {mae_avg:.2f}   ({(mae_base-mae_avg)/mae_base*100:+.1f}% vs moyenne)")
    print(f"  MAE tempo (rég. linéaire)     : {mae_lin:.2f}   ({(mae_base-mae_lin)/mae_base*100:+.1f}% vs moyenne)")
    print(f"  MAE LGBM générique            : {mae_lgbm:.2f}   ({(mae_base-mae_lgbm)/mae_base*100:+.1f}% vs moyenne)")

    # ---- test betting : contre une ligne PARESSEUSE (moyenne glissante) ----
    print("\n  --- O/U vs ligne 'paresseuse' (moyenne glissante des totals passés) ---")
    all_tot = feats["y_total_kills"].values
    test_start = len(feats) - n_test
    best = None
    for delta in (0.0, 1.0, 2.0, 3.0):
        wins = bets = 0
        for i in range(test_start, len(feats)):
            line = float(np.mean(all_tot[:i]))   # ligne = moyenne des games avant i (no leak)
            pred = float(lin.predict(feats.iloc[[i]][["blue_tempo", "red_tempo"]])[0])
            if abs(pred - line) <= delta:
                continue
            actual = all_tot[i]
            if actual == line:
                continue
            bets += 1
            if np.sign(pred - line) == np.sign(actual - line):
                wins += 1
        wr = wins / bets * 100 if bets else 0
        flag = "  <- bat la marge (~54%)" if wr >= 54 and bets >= 15 else ""
        print(f"    seuil |pred-ligne|>{delta:.0f} : {bets:3d} paris, {wr:5.1f}% bon côté{flag}")

    print("=" * 70)
    print("  NB : ligne paresseuse = hypothèse OPTIMISTE (book qui poste ~la moyenne).")
    print("  Si un vrai book ajuste la ligne au matchup, l'edge fond. -> besoin lignes réelles.")
    print("=" * 70)


if __name__ == "__main__":
    main()

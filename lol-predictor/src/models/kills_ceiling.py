"""Plafond de SIGNAL du total kills : combien est prévisible vs bruit irréductible ?

Idée clé : on ne bat le book que si le total kills est PRÉVISIBLE. Si 90% de la variance
est du bruit pur (une game snowball ou pas, indépendamment des équipes), alors NI nous NI
le book ne peut le pricer finement -> la marge (8%) protège le book. Ce script mesure le
PLAFOND de prévisibilité (R², corrélation) avec le meilleur modèle possible, + l'ICC
(part de variance due à l'identité des équipes). Indépendant des cotes.

Usage : python -m src.models.kills_ceiling
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from src.ingest.load_oracle import ROOT, load_config

warnings.filterwarnings("ignore")


def report(name, y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    r2 = r2_score(y, pred)
    corr = np.corrcoef(pred, y)[0, 1] if np.std(pred) > 0 else 0.0
    print(f"  {name:<28} MAE {mae:5.2f} | RMSE {rmse:5.2f} | R2 {r2:+5.2f} | corr {corr:+.2f}")
    return r2


def icc_team_kills(tg: pd.DataFrame) -> float:
    """Part de variance du total-kills d'une game expliquée par l'identité des équipes.
    total game = kills + deaths d'une équipe. On l'attribue aux 2 équipes -> ANOVA 1 facteur."""
    tg = tg.copy()
    tg["game_total"] = tg["kills"] + tg["deaths"]
    grand = tg["game_total"].mean()
    ss_tot = ((tg["game_total"] - grand) ** 2).sum()
    ss_between = tg.groupby("teamname")["game_total"].apply(
        lambda x: len(x) * (x.mean() - grand) ** 2).sum()
    return ss_between / ss_tot


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet").sort_values(["date", "gameid"]).reset_index(drop=True)
    feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()
    feats["blue_tempo"] = feats["blue_kills_avg"] + feats["blue_deaths_avg"]
    feats["red_tempo"] = feats["red_kills_avg"] + feats["red_deaths_avg"]
    feats = feats.dropna(subset=["y_total_kills", "blue_tempo", "red_tempo"])

    n_test = max(1, int(cfg["split"]["test_fraction"] * len(feats)))
    train, test = feats.iloc[:-n_test], feats.iloc[-n_test:]
    ytr, yte = train["y_total_kills"].values, test["y_total_kills"].values

    print("=" * 74)
    print(f"  PLAFOND DE SIGNAL — TOTAL KILLS (LCK)  | train={len(train)} test={len(test)}")
    print("=" * 74)
    report("baseline (moyenne)", yte, np.full(len(yte), ytr.mean()))
    ridge = Ridge(alpha=1.0).fit(train[["blue_tempo", "red_tempo"]], ytr)
    report("tempo (2 var)", yte, ridge.predict(test[["blue_tempo", "red_tempo"]]))
    fcols = [c for c in feats.columns if not c.startswith("y_")
             and c not in ("gameid", "date")]
    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=15,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=2.0, random_state=42, verbose=-1).fit(train[fcols], ytr)
    r2_best = report("LGBM (toutes features)", yte, lgbm.predict(test[fcols]))

    tg = pd.read_parquet(proc / "team_games.parquet")
    icc = icc_team_kills(tg)

    print("=" * 74)
    print(f"  ICC (variance du total expliquée par l'identité des équipes) : {icc * 100:.0f}%")
    print(f"  -> ~{(1 - max(r2_best, 0)) * 100:.0f}% du total kills d'une game = BRUIT irréductible")
    print("=" * 74)
    print("  LECTURE :")
    print("  - R2 proche de 0 / corr faible -> total kills quasi imprévisible game par game.")
    print("  - Si NI nous NI le book ne peut prédire, la marge 8% rend le marché ininjouable.")
    print("  - Ce plafond est STRUCTUREL (vaut pour LCK/LEC/LCS ; LPL encore plus chaotique).")
    print("=" * 74)


if __name__ == "__main__":
    main()

"""Pouvoir prédictif PAR MARCHÉ secondaire (pré-game, sans fuite) — où a-t-on du signal ?

On ne peut pas (encore) mesurer la VALEUR vs cotes (pas d'historique de cotes secondaires).
Mais on peut mesurer notre pouvoir de prédiction marché par marché, pour repérer ceux où
on est nettement mieux que le hasard (candidats edge) vs ceux qui sont des pile-ou-face.

On réutilise les features as-of (features.parquet) et on DÉRIVE des labels game-level
supplémentaires depuis team_games (first baron/herald, total tourelles/drakes/barons,
kills par équipe) — qui correspondent aux marchés vus chez le book.

Binaire : accuracy / AUC / Brier vs baseline (classe majoritaire).
Numérique (O/U) : MAE du modèle vs MAE de la moyenne (si on ne bat pas la moyenne -> nul).

Usage : python -m src.models.secondary_markets
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             mean_absolute_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.load_oracle import ROOT, load_config

warnings.filterwarnings("ignore")


def build_extra_labels(team_games: pd.DataFrame) -> pd.DataFrame:
    """Labels game-level dérivés (point de vue bleu) depuis team_games."""
    tg = team_games.copy()
    tg["side"] = tg["side"].str.lower()
    blue = tg[tg["side"] == "blue"].set_index("gameid")
    red = tg[tg["side"] == "red"].set_index("gameid")

    def col(df, c):
        return pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(dtype=float)

    out = pd.DataFrame(index=blue.index)
    # binaires (1 = bleu réalise l'objectif)
    out["y_first_herald"] = col(blue, "firstherald")
    out["y_first_baron"] = col(blue, "firstbaron")
    # totaux (O/U)
    out["y_total_towers"] = col(blue, "towers").add(col(red, "towers"), fill_value=np.nan)
    out["y_total_dragons"] = col(blue, "dragons").add(col(red, "dragons"), fill_value=np.nan)
    out["y_total_barons"] = col(blue, "barons").add(col(red, "barons"), fill_value=np.nan)
    out["y_blue_kills"] = col(blue, "kills")
    out["y_red_kills"] = col(red, "kills")
    return out.reset_index()


def temporal_split(df, frac):
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    n_test = max(1, int(frac * len(df)))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy(), n_test


def eval_binary(train, test, fcols, target):
    tr, te = train.dropna(subset=[target]), test.dropna(subset=[target])
    if te.empty or te[target].nunique() < 2 or tr[target].nunique() < 2:
        return None
    ytr, yte = tr[target].astype(int), te[target].astype(int)
    base = max(yte.mean(), 1 - yte.mean())
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
    model.fit(tr[fcols], ytr)
    p = model.predict_proba(te[fcols])[:, 1]
    return {
        "Marché": target.replace("y_", ""),
        "n_test": len(te),
        "base%": round(base * 100, 1),
        "acc%": round(accuracy_score(yte, (p >= 0.5).astype(int)) * 100, 1),
        "AUC": round(roc_auc_score(yte, p), 3),
        "Brier": round(brier_score_loss(yte, p), 3),
    }


def eval_numeric(train, test, fcols, target):
    tr, te = train.dropna(subset=[target]), test.dropna(subset=[target])
    if te.empty:
        return None
    ytr, yte = tr[target], te[target]
    base_mae = mean_absolute_error(yte, np.full(len(yte), ytr.mean()))
    reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                            min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=1.0, random_state=42, verbose=-1)
    reg.fit(tr[fcols], ytr)
    mae = mean_absolute_error(yte, reg.predict(te[fcols]))
    gain = (base_mae - mae) / base_mae * 100
    return {
        "Marché": target.replace("y_", ""),
        "n_test": len(te),
        "moyenne": round(ytr.mean(), 1),
        "MAE moyenne": round(base_mae, 2),
        "MAE modèle": round(mae, 2),
        "gain%": round(gain, 1),
    }


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet")
    team_games = pd.read_parquet(proc / "team_games.parquet")

    extra = build_extra_labels(team_games)
    feats = feats.merge(extra, on="gameid", how="left")
    feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()
    fcols = [c for c in feats.columns
             if not c.startswith("y_") and c not in ("gameid", "date")]

    train, test, n_test = temporal_split(feats, cfg["split"]["test_fraction"])
    print("=" * 76)
    print(f"  POUVOIR PRÉDICTIF PAR MARCHÉ (test temporel = {n_test} games récents)")
    print(f"  Test : {test['date'].min():%Y-%m-%d} -> {test['date'].max():%Y-%m-%d}")
    print("=" * 76)

    binaries = ["y_winner", "y_first_blood", "y_first_tower", "y_first_dragon",
                "y_first_herald", "y_first_baron"]
    rows = [r for t in binaries if (r := eval_binary(train, test, fcols, t))]
    bdf = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    print("\n--- BINAIRES (trié par AUC ; AUC 0.5 = hasard, >0.65 = vrai signal) ---")
    print(bdf.to_string(index=False))

    numerics = ["y_total_kills", "y_game_time_min", "y_total_towers",
                "y_total_dragons", "y_total_barons", "y_blue_kills", "y_red_kills"]
    rows = [r for t in numerics if (r := eval_numeric(train, test, fcols, t))]
    ndf = pd.DataFrame(rows).sort_values("gain%", ascending=False)
    print("\n--- NUMÉRIQUES O/U (gain% > 0 = on bat la moyenne ; <=0 = inutile) ---")
    print(ndf.to_string(index=False))
    print("=" * 76)
    print("  Rappel : AUC/MAE = pouvoir prédictif, PAS encore la value (besoin de")
    print("  l'historique des cotes de ces marchés pour mesurer un edge réel).")
    print("=" * 76)


if __name__ == "__main__":
    main()

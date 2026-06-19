"""P4 : entraînement + backtest temporel par marché.

Split CHRONOLOGIQUE strict (train = passé, test = matchs les plus récents).
Modèles : régression logistique et LightGBM, avec calibration (Platt/sigmoïde).
On compare toujours aux baselines (toujours bleu / signe d'Elo / moyenne).

Usage :
    python -m src.models.train_backtest
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             mean_absolute_error, mean_squared_error,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.load_oracle import ROOT, load_config

warnings.filterwarnings("ignore")

BINARY_MARKETS = {
    "y_winner": "Vainqueur (bleu)",
    "y_first_blood": "First blood (bleu)",
    "y_first_tower": "First tower (bleu)",
    "y_first_dragon": "First dragon (bleu)",
}
REG_MARKETS = {
    "y_total_kills": "Total kills",
    "y_game_time_min": "Durée (min)",
}


def _lgbm_clf() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )


def _lgbm_reg() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )


def temporal_split(df: pd.DataFrame, frac: float):
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    n_test = max(1, int(frac * len(df)))
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy(), n_test


def eval_binary(train, test, fcols, target, results):
    tr = train.dropna(subset=[target])
    te = test.dropna(subset=[target])
    ytr, yte = tr[target].astype(int), te[target].astype(int)
    if yte.nunique() < 2:
        return
    Xtr, Xte = tr[fcols], te[fcols]

    base = max(yte.mean(), 1 - yte.mean())  # toujours la classe majoritaire
    row = {"Marché": BINARY_MARKETS[target], "Baseline%": round(base * 100, 1)}

    if target == "y_winner":
        row["Elo-sign%"] = round(accuracy_score(yte, (te["d_elo"] > 0).astype(int)) * 100, 1)

    # LogReg brut C=0.1 (la calibration sigmoid cv=3 dégrade sur petit dataset, cf. tune_winner).
    # LGBM gardé calibré pour comparaison.
    models = (
        ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1)), False),
        ("LGBM", _lgbm_clf(), True),
    )
    for name, est, calibrate in models:
        model = CalibratedClassifierCV(est, method="sigmoid", cv=3) if calibrate else est
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        row[f"{name}_acc%"] = round(accuracy_score(yte, (p >= 0.5).astype(int)) * 100, 1)
        row[f"{name}_AUC"] = round(roc_auc_score(yte, p), 3)
        row[f"{name}_Brier"] = round(brier_score_loss(yte, p), 3)
    results.append(row)


def eval_regression(train, test, fcols, target, results):
    tr = train.dropna(subset=[target])
    te = test.dropna(subset=[target])
    ytr, yte = tr[target], te[target]
    Xtr, Xte = tr[fcols], te[fcols]

    base_pred = np.full(len(yte), ytr.mean())
    base_mae = mean_absolute_error(yte, base_pred)

    reg = _lgbm_reg()
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    results.append({
        "Marché": REG_MARKETS[target],
        "Baseline MAE (moyenne)": round(base_mae, 2),
        "LGBM MAE": round(mean_absolute_error(yte, pred), 2),
        "LGBM RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 2),
    })


def feature_importance(train, fcols, target="y_winner", top=12):
    tr = train.dropna(subset=[target])
    model = _lgbm_clf()
    model.fit(tr[fcols], tr[target].astype(int))
    imp = pd.Series(model.feature_importances_, index=fcols).sort_values(ascending=False)
    return imp.head(top)


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet")

    # On garde les matchs avec un minimum d'historique des deux côtés
    feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()

    fcols = [c for c in feats.columns
             if not c.startswith("y_") and c not in ("gameid", "date")]

    train, test, n_test = temporal_split(feats, cfg["split"]["test_fraction"])
    print("=" * 64)
    print(f"  BACKTEST TEMPOREL  |  train={len(train)}  test={n_test}")
    print(f"  Test = {test['date'].min():%Y-%m-%d} -> {test['date'].max():%Y-%m-%d}")
    print("=" * 64)

    bin_results: list[dict] = []
    for target in BINARY_MARKETS:
        eval_binary(train, test, fcols, target, bin_results)
    print("\n--- MARCHÉS BINAIRES (accuracy / AUC / Brier) ---")
    print(pd.DataFrame(bin_results).to_string(index=False))

    reg_results: list[dict] = []
    for target in REG_MARKETS:
        eval_regression(train, test, fcols, target, reg_results)
    print("\n--- MARCHÉS NUMÉRIQUES (erreur moyenne) ---")
    print(pd.DataFrame(reg_results).to_string(index=False))

    print("\n--- TOP FEATURES (importance LightGBM, marché vainqueur) ---")
    print(feature_importance(train, fcols).to_string())

    pd.DataFrame(bin_results).to_csv(ROOT / "reports" / "backtest_binary.csv", index=False)
    pd.DataFrame(reg_results).to_csv(ROOT / "reports" / "backtest_regression.csv", index=False)
    print(f"\nRapports -> {ROOT / 'reports'}")


if __name__ == "__main__":
    main()

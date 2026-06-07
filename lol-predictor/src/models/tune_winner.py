"""Réglage du marché VAINQUEUR : régularisation (C) + calibration.

On a constaté que le pipeline actuel (C=0.5 + calibration sigmoïde cv=3 sur ~250
games) est SOUS-CONFIANT (il comprime les probas vers 50 %). On balaie donc C et la
méthode de calibration, évalués en ROLLING-ORIGIN (chaque match prédit par un modèle
entraîné uniquement sur son passé), et on regarde :
  - AUC   : qualité du classement (favori vs outsider)
  - acc%  : accuracy
  - Brier : qualité des PROBA (plus bas = mieux) -> métrique clé pour la calibration
  - ECE   : erreur de calibration (écart |proba prédite - fréquence réelle|)

Usage :
    python -m src.models.tune_winner
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.load_oracle import ROOT, load_config

warnings.filterwarnings("ignore")

ROLLING_START = 100

# (label, C, méthode de calibration ou None)
CONFIGS = [
    ("C=0.5 + sigmoid (ACTUEL)", 0.5, "sigmoid"),
    ("C=0.5 brut", 0.5, None),
    ("C=0.25 brut", 0.25, None),
    ("C=0.1 brut", 0.1, None),
    ("C=0.05 brut", 0.05, None),
    ("C=0.1 + sigmoid", 0.1, "sigmoid"),
    ("C=0.25 + sigmoid", 0.25, "sigmoid"),
]


def _make(C: float, calib: str | None):
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))
    if calib:
        return CalibratedClassifierCV(est, method=calib, cv=3)
    return est


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        total += abs(p[m].mean() - y[m].mean()) * m.sum()
    return total / len(y)


def rolling(feats: pd.DataFrame, fcols: list[str], C: float, calib: str | None,
            start: int = ROLLING_START) -> dict:
    feats = feats.sort_values(["date", "gameid"]).reset_index(drop=True)
    preds, ys = [], []
    for i in range(start, len(feats)):
        tr, te = feats.iloc[:i], feats.iloc[i:i + 1]
        if tr["y_winner"].nunique() < 2:
            continue
        model = _make(C, calib).fit(tr[fcols], tr["y_winner"].astype(int))
        preds.append(model.predict_proba(te[fcols])[:, 1][0])
        ys.append(int(te["y_winner"].iloc[0]))
    ys, preds = np.array(ys), np.array(preds)
    return {
        "AUC": roc_auc_score(ys, preds),
        "acc": accuracy_score(ys, (preds >= 0.5).astype(int)),
        "Brier": brier_score_loss(ys, preds),
        "ECE": ece(ys, preds),
        "p_min": preds.min(), "p_max": preds.max(),
    }


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    feats = pd.read_parquet(proc / "features.parquet")
    feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()
    fcols = [c for c in feats.columns if not c.startswith("y_") and c not in ("gameid", "date")]

    print(f"Rolling-origin sur le marché vainqueur (n test = {len(feats) - ROLLING_START})...\n")
    rows = []
    for label, C, calib in CONFIGS:
        r = rolling(feats, fcols, C, calib)
        rows.append({
            "Config": label,
            "AUC": round(r["AUC"], 3),
            "acc%": round(r["acc"] * 100, 1),
            "Brier": round(r["Brier"], 4),
            "ECE": round(r["ECE"], 4),
            "étendue proba": f"{r['p_min']*100:.0f}-{r['p_max']*100:.0f}%",
        })
        print(f"  [ok] {label}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  RÉGLAGE VAINQUEUR  (Brier = métrique clé ; étendue large = plus confiant)")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    best = df.loc[df["Brier"].idxmin()]
    print(f"  Meilleur Brier : {best['Config']}  (Brier {best['Brier']}, AUC {best['AUC']})")
    df.to_csv(ROOT / "reports" / "tune_winner.csv", index=False)


if __name__ == "__main__":
    main()

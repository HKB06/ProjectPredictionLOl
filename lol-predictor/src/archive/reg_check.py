"""Diagnostic : le modèle est-il SOUS-CONFIANT (compresse les favoris) ?

Les features favorisent clairement DK (Elo 1539 vs 1429 -> ~65%), mais le modèle
sort ~52% side-neutre. Hypothèse : la régularisation (C=0.5) + les features
redondantes écrasent le signal de force vers 50%.

On réentraîne la régression logistique avec différentes forces de régularisation
(C) et on regarde la proba DK vs BRION side-neutre. Si elle remonte vers ~65%
quand C augmente, l'hypothèse est confirmée.

Usage :
    python -m src.models.reg_check
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import build_features
from src.ingest.load_oracle import ROOT, load_config
from src.models.predict import MatchPredictor

warnings.filterwarnings("ignore")

DK, BRO = "Dplus Kia", "HANJIN BRION"


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    matches = pd.read_parquet(proc / "matches.parquet")
    team_games = pd.read_parquet(proc / "team_games.parquet")

    mp = MatchPredictor().fit(cfg)
    feats = build_features(matches, team_games, mp.champ_idx)
    feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)]
    fcols = mp.fcols
    y = feats["y_winner"].astype(int)

    row_dk_blue = mp._feature_row(DK, BRO, {}, {})   # DK côté bleu
    row_bro_blue = mp._feature_row(BRO, DK, {}, {})   # BRION côté bleu

    print("=" * 64)
    print("  SOUS-CONFIANCE ? proba DK vs BRION selon la régularisation")
    print("  (Elo seul implique ~65.3% pour DK)")
    print("=" * 64)
    print(f"  {'C':>5} | {'DK bleu':>8} | {'DK rouge':>9} | {'DK neutre':>10}")
    print("-" * 64)
    for C in [0.1, 0.25, 0.5, 1, 2, 4, 8, 50]:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=C)).fit(feats[fcols], y)
        p_dk_blue = model.predict_proba(row_dk_blue[fcols])[:, 1][0]
        p_bro_blue = model.predict_proba(row_bro_blue[fcols])[:, 1][0]
        p_dk_red = 1 - p_bro_blue
        neutral = 0.5 * (p_dk_blue + p_dk_red)
        flag = "  <- C actuel" if C == 0.5 else ""
        print(f"  {C:>5} | {p_dk_blue*100:7.1f}% | {p_dk_red*100:8.1f}% | {neutral*100:9.1f}%{flag}")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""Démonstration de FUITE : re-prédire une game déjà dans la data n'est PAS un test.

Quand on rentre dans le front une game qui est dans le dataset, la "mémoire" des
équipes (Elo/forme/H2H) contient déjà le résultat de cette game -> la proba est
gonflée. Ce script compare, pour un match historique :
  - AVEUGLE (as-of) : features calculées avant la game + modèle entraîné uniquement
    sur le passé de la game (vrai hold-out, sans fuite) ;
  - FUITÉ (front)   : état final (toutes games incluses) -> ce que montre le front.

Usage :
    python -m src.models.blind_check
"""
from __future__ import annotations

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest.load_oracle import ROOT, load_config
from src.models.predict import MatchPredictor

BLUE_NAME = "Hanwha"
RED_NAME = "BRION"
DATE = "2026-05-31"


def _logreg_cal():
    return CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        method="sigmoid", cv=3)


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    m = pd.read_parquet(proc / "matches.parquet")
    f = pd.read_parquet(proc / "features.parquet")
    m["date"] = pd.to_datetime(m["date"])
    f["date"] = pd.to_datetime(f["date"])

    target_day = pd.to_datetime(DATE).date()
    mask = ((m["date"].dt.date == target_day)
            & m["blue_team"].str.contains(BLUE_NAME, case=False)
            & m["red_team"].str.contains(RED_NAME, case=False))
    games = m[mask]
    if games.empty:
        print("Aucune game trouvée pour ce filtre.")
        return

    fcols = [c for c in f.columns if not c.startswith("y_") and c not in ("gameid", "date")]
    ff = f[(f["blue_n_games"] >= 3) & (f["red_n_games"] >= 3)].copy()

    # Front (fuité) : on entraîne le MatchPredictor (état final) et on prédit avec la vraie draft
    print("Entraînement du MatchPredictor (état final = ce que voit le front)...")
    mp = MatchPredictor().fit(cfg)
    ROLES = ["top", "jng", "mid", "bot", "sup"]

    print("\n" + "=" * 72)
    print(f"  {games.iloc[0]['blue_team']} (bleu) vs {games.iloc[0]['red_team']} (rouge) — {DATE}")
    print("=" * 72)
    for _, g in games.iterrows():
        gid = g["gameid"]
        row = f[f["gameid"] == gid]
        gdate = row["date"].iloc[0]

        # AVEUGLE : modèle entraîné uniquement sur le passé strict de la game
        train = ff[ff["date"] < gdate]
        est = _logreg_cal().fit(train[fcols], train["y_winner"].astype(int))
        p_blind = est.predict_proba(row[fcols])[:, 1][0]

        # FUITÉ : front avec la vraie draft de la game
        blue_champs = {r: g.get(f"blue_{r}") for r in ROLES}
        red_champs = {r: g.get(f"red_{r}") for r in ROLES}
        p_leak = mp.predict_match(g["blue_team"], g["red_team"],
                                  blue_champs, red_champs)["winner"]["blue"]

        real = int(row["y_winner"].iloc[0])
        print(f"\n  game {gid}")
        print(f"    Résultat réel               : {'BLEU gagne' if real else 'ROUGE gagne'}")
        print(f"    P(bleu) AVEUGLE (as-of)     : {p_blind*100:5.1f}%   <- vraie capacité (sans fuite)")
        print(f"    P(bleu) FRONT (état final)  : {p_leak*100:5.1f}%   <- gonflé (game déjà connue)")
        print(f"    Écart dû à la fuite         : {(p_leak - p_blind)*100:+5.1f} pts")
    print("=" * 72)


if __name__ == "__main__":
    main()

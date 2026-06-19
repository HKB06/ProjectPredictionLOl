"""Expérience : la source des priors champion (par ligue) fausse-t-elle le modèle ?

Question posée par l'utilisateur : dans le pool de ~5500 games, il y a des ligues
mineures (LFL, AL...) plus faibles que LCK/LEC. Calculer le winrate d'un champion
sur ces ligues risque de fausser le signal.

On compare donc 4 sources de priors champion, toujours évaluées sur du LCK que le
modèle n'a JAMAIS vu (held-out), via 2 protocoles complémentaires :
  - SPLIT TEMPOREL : train = LCK passé, test = ~62 derniers matchs LCK.
  - ROLLING-ORIGIN : pour chaque match (à partir du 100e), on entraîne sur tout son
    passé et on prédit ce match -> ~200 matchs LCK testés, chacun hors échantillon.

Anti-fuite : le winrate d'un champion n'utilise que les games antérieures à la date
du match (le match lui-même est exclu). C'est valable quelle que soit la ligue source.

Usage :
    python -m src.models.exp_champ_leagues
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

from src.features.build_features import build_features
from src.features.champion_priors import (ChampionWinrateIndex,
                                          load_full_champion_results)
from src.ingest.load_oracle import ROOT, load_config

warnings.filterwarnings("ignore")

LEAGUE_SETS: dict[str, object] = {
    "Aucune draft": "NODRAFT",
    "Toutes ligues": None,
    "Majeures (LPL/LCK/LEC/LCS)": ["LPL", "LCK", "LEC", "LCS"],
    "LCK seule": ["LCK"],
}

ROLLING_START = 100  # nb de matchs d'amorçage avant de commencer à tester


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def _fcols(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if not c.startswith("y_") and c not in ("gameid", "date")]


def single_split(feats: pd.DataFrame, frac: float = 0.2) -> dict:
    feats = feats.sort_values(["date", "gameid"]).reset_index(drop=True)
    n_test = max(1, int(frac * len(feats)))
    train, test = feats.iloc[:-n_test], feats.iloc[-n_test:]
    fcols = _fcols(feats)
    ytr, yte = train["y_winner"].astype(int), test["y_winner"].astype(int)
    cal = CalibratedClassifierCV(_logreg(), method="sigmoid", cv=3)
    cal.fit(train[fcols], ytr)
    p = cal.predict_proba(test[fcols])[:, 1]
    return {
        "n_test": len(yte),
        "acc": accuracy_score(yte, (p >= 0.5).astype(int)),
        "auc": roc_auc_score(yte, p),
        "brier": brier_score_loss(yte, p),
    }


def rolling_origin(feats: pd.DataFrame, start: int = ROLLING_START) -> dict:
    feats = feats.sort_values(["date", "gameid"]).reset_index(drop=True)
    fcols = _fcols(feats)
    preds, ys = [], []
    for i in range(start, len(feats)):
        tr = feats.iloc[:i]
        te = feats.iloc[i:i + 1]
        if tr["y_winner"].nunique() < 2:
            continue
        est = _logreg()
        est.fit(tr[fcols], tr["y_winner"].astype(int))
        preds.append(est.predict_proba(te[fcols])[:, 1][0])
        ys.append(int(te["y_winner"].iloc[0]))
    ys, preds = np.array(ys), np.array(preds)
    return {
        "n_test": len(ys),
        "acc": accuracy_score(ys, (preds >= 0.5).astype(int)),
        "auc": roc_auc_score(ys, preds),
        "brier": brier_score_loss(ys, preds),
    }


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    matches = pd.read_parquet(proc / "matches.parquet")
    team_games = pd.read_parquet(proc / "team_games.parquet")

    print("Construction des index champion par source (peut prendre ~1 min)...")
    rows = []
    for label, leagues in LEAGUE_SETS.items():
        if leagues == "NODRAFT":
            feats = build_features(matches, team_games, champ_idx=None)
            pool = 0
        else:
            players = load_full_champion_results(cfg, leagues)
            pool = len(players) // 10
            feats = build_features(matches, team_games,
                                   champ_idx=ChampionWinrateIndex(players))
        feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()

        s = single_split(feats)
        r = rolling_origin(feats)
        rows.append({
            "Source priors": label,
            "Pool games": pool,
            "Split AUC": round(s["auc"], 3),
            "Split acc%": round(s["acc"] * 100, 1),
            "Split Brier": round(s["brier"], 3),
            "Roll AUC": round(r["auc"], 3),
            "Roll acc%": round(r["acc"] * 100, 1),
            "Roll Brier": round(r["brier"], 3),
        })
        print(f"  [ok] {label}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("  IMPACT DE LA SOURCE DES PRIORS CHAMPION  (marché VAINQUEUR, LCK held-out)")
    print("=" * 78)
    print(f"  Split temporel : test = {single_split.__defaults__[0]:.0%} des derniers matchs")
    print(f"  Rolling-origin : test = chaque match à partir du {ROLLING_START}e (chacun hors échantillon)")
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    out = ROOT / "reports" / "exp_champ_leagues.csv"
    df.to_csv(out, index=False)
    print(f"Rapport -> {out}")


if __name__ == "__main__":
    main()

"""Inférence "1 match à venir" : entraîne tous les marchés et prédit un match isolé.

Brique commune au front. `MatchPredictor.fit()` :
- rejoue l'historique pour obtenir l'état FINAL des équipes (Elo, forme, H2H) ;
- construit l'index winrate champion (priors draft, toutes ligues) ;
- entraîne 1 modèle calibré par marché binaire + 1 régression par marché numérique.

`predict_match(blue, red, blue_champs, red_champs)` recalcule le même vecteur de
features que l'entraînement (forme actuelle + draft) et renvoie les probas/valeurs.

Anti-fuite : les winrates champion sont pris "à aujourd'hui" (après tout l'historique
connu), ce qui est correct pour un match futur. Pour l'entraînement, ils restent
calculés as-of (dans build_features), donc le modèle n'a jamais vu le futur.

Usage (test CLI) :
    python -m src.models.predict
"""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from src.features.build_features import (DELTA_KEYS, K_H2H, ROLES, _shrink,
                                         _team_features, build_features)
from src.features.champion_priors import (ChampionWinrateIndex,
                                          load_full_champion_results)
from src.ingest.load_oracle import ROOT, load_config

BINARY_MARKETS = {
    "y_winner": "Vainqueur",
    "y_first_blood": "First blood",
    "y_first_tower": "First tower",
    "y_first_dragon": "First dragon",
}
REG_MARKETS = {
    "y_total_kills": "Total kills",
    "y_game_time_min": "Durée (min)",
}


def bo_series_prob(p_game: float, wins_needed: int = 3) -> float:
    """Proba de gagner une série au meilleur des (2*wins_needed-1).

    Hypothèse : proba par game `p_game` constante et games indépendantes.
    BO5 -> wins_needed=3 : p^3 (1 + 3q + 6q^2), q = 1-p.
    """
    q = 1.0 - p_game
    return float(sum(
        comb(wins_needed - 1 + losses, losses) * p_game ** wins_needed * q ** losses
        for losses in range(wins_needed)
    ))


# C=0.1 + PAS de wrapper de calibration : le rolling-origin (tune_winner) montre que
# CalibratedClassifierCV(sigmoid, cv=3) sur ~250 games dégrade tout (AUC 0.59 vs 0.74).
# La régression logistique régularisée est déjà bien calibrée (Brier 0.199).
WINNER_C = 0.1


def _logreg(C: float = WINNER_C):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))


def _lgbm_reg():
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, verbose=-1,
    )


class MatchPredictor:
    """Entraîne tous les marchés et prédit un match (2 équipes + draft)."""

    def __init__(self) -> None:
        self.fcols: list[str] = []
        self.bin_models: dict[str, object] = {}
        self.reg_models: dict[str, object] = {}
        self.state: dict = {}
        self.champ_idx: ChampionWinrateIndex | None = None
        self.asof_date: pd.Timestamp | None = None
        self.teams: list[str] = []
        self.champions: list[str] = []

    def fit(self, cfg: dict | None = None) -> "MatchPredictor":
        cfg = cfg or load_config()
        proc = ROOT / cfg["data"]["processed_dir"]
        matches = pd.read_parquet(proc / "matches.parquet")
        team_games = pd.read_parquet(proc / "team_games.parquet")

        prior_leagues = cfg.get("features", {}).get("champ_prior_leagues")
        self.champ_idx = ChampionWinrateIndex(load_full_champion_results(cfg, prior_leagues))
        self.asof_date = pd.to_datetime(matches["date"]).max() + pd.Timedelta(days=1)

        feats, state = build_features(matches, team_games, self.champ_idx, return_state=True)
        self.state = state
        feats = feats[(feats["blue_n_games"] >= 3) & (feats["red_n_games"] >= 3)].copy()
        self.fcols = [c for c in feats.columns
                      if not c.startswith("y_") and c not in ("gameid", "date")]

        for target in BINARY_MARKETS:
            sub = feats.dropna(subset=[target])
            y = sub[target].astype(int)
            if y.nunique() < 2:
                continue
            model = _logreg()
            model.fit(sub[self.fcols], y)
            self.bin_models[target] = model

        for target in REG_MARKETS:
            sub = feats.dropna(subset=[target])
            reg = _lgbm_reg()
            reg.fit(sub[self.fcols], sub[target])
            self.reg_models[target] = reg

        self.teams = sorted(state["history"].keys())
        self.champions = sorted(self.champ_idx.dates.keys())
        return self

    def _comp_wr(self, champs: dict[str, str]) -> float:
        wrs = [self.champ_idx.asof(c, self.asof_date)
               for c in champs.values()
               if c and not (isinstance(c, float) and pd.isna(c))]
        return float(np.mean(wrs)) if wrs else 0.5

    def _feature_row(self, blue: str, red: str,
                     blue_champs: dict[str, str], red_champs: dict[str, str],
                     is_playoffs: int = 0) -> pd.DataFrame:
        hist, elo, h2h = self.state["history"], self.state["elo"], self.state["h2h"]
        fb = _team_features(hist[blue], elo[blue], "Blue")
        fr = _team_features(hist[red], elo[red], "Red")

        meetings = h2h.get(frozenset((blue, red)), [])
        nh = len(meetings)
        blue_wins = sum(1 for w in meetings if w == blue)

        row: dict = {"is_playoffs": int(is_playoffs)}
        for key, val in fb.items():
            row[f"blue_{key}"] = val
        for key, val in fr.items():
            row[f"red_{key}"] = val
        for key in DELTA_KEYS:
            row[f"d_{key}"] = fb[key] - fr[key]
        row["h2h_blue_wr"] = _shrink(blue_wins, nh, k=K_H2H)
        row["h2h_n"] = nh

        bcw, rcw = self._comp_wr(blue_champs), self._comp_wr(red_champs)
        row["blue_champ_wr"], row["red_champ_wr"], row["d_champ_wr"] = bcw, rcw, bcw - rcw

        return pd.DataFrame([row]).reindex(columns=self.fcols)

    def predict_match(self, blue: str, red: str,
                      blue_champs: dict[str, str], red_champs: dict[str, str],
                      is_playoffs: int = 0) -> dict:
        X = self._feature_row(blue, red, blue_champs, red_champs, is_playoffs)
        out: dict = {"markets": {}}

        p_blue = float(self.bin_models["y_winner"].predict_proba(X)[:, 1][0])
        out["winner"] = {"blue": p_blue, "red": 1 - p_blue}

        for target, label in BINARY_MARKETS.items():
            if target in self.bin_models:
                out["markets"][label] = float(self.bin_models[target].predict_proba(X)[:, 1][0])
        for target, label in REG_MARKETS.items():
            if target in self.reg_models:
                out["markets"][label] = float(self.reg_models[target].predict(X)[0])
        return out

    def predict_series(self, blue: str, red: str,
                       blue_champs: dict[str, str], red_champs: dict[str, str],
                       wins_needed: int = 3, is_playoffs: int = 0) -> dict:
        """Proba de SÉRIE (BO3/BO5) en neutralisant l'avantage de side.

        On calcule la proba par game de `blue` quand il est côté bleu PUIS côté
        rouge, on en fait la moyenne (alternance des sides en série), puis on
        convertit en proba de série.
        """
        p_on_blue = self.predict_match(blue, red, blue_champs, red_champs, is_playoffs)["winner"]["blue"]
        p_on_red = 1.0 - self.predict_match(red, blue, red_champs, blue_champs, is_playoffs)["winner"]["blue"]
        # Neutralisation EXACTE du side : moyenne en log-odds (le side est un décalage
        # de log-odds dans le modèle logistique -> la moyenne des 2 sides l'annule).
        eps = 1e-6
        lb = np.log(np.clip(p_on_blue, eps, 1 - eps) / np.clip(1 - p_on_blue, eps, 1 - eps))
        lr = np.log(np.clip(p_on_red, eps, 1 - eps) / np.clip(1 - p_on_red, eps, 1 - eps))
        p_neutral = float(1.0 / (1.0 + np.exp(-(lb + lr) / 2.0)))
        series_blue = bo_series_prob(p_neutral, wins_needed)
        return {
            "p_on_blue": p_on_blue,
            "p_on_red": p_on_red,
            "p_neutral": p_neutral,
            "series_blue": series_blue,
            "series_red": 1.0 - series_blue,
        }


def main() -> None:
    print("Entraînement du MatchPredictor...")
    mp = MatchPredictor().fit()
    print(f"  Équipes connues   : {len(mp.teams)}")
    print(f"  Champions connus  : {len(mp.champions)}")
    print(f"  Date de référence : {mp.asof_date:%Y-%m-%d}")

    if len(mp.teams) >= 2:
        b, r = mp.teams[0], mp.teams[1]
        blue_champs = {role: mp.champions[i] for i, role in enumerate(ROLES)}
        red_champs = {role: mp.champions[i + 5] for i, role in enumerate(ROLES)}
        res = mp.predict_match(b, r, blue_champs, red_champs)
        print("\n  Exemple de prédiction :")
        print(f"    BLEU {b}  vs  ROUGE {r}")
        print(f"    P(victoire bleu) = {res['winner']['blue']*100:.1f}%")
        for label, val in res["markets"].items():
            if label in ("Total kills", "Durée (min)"):
                print(f"    {label:<14} : {val:.1f}")
            else:
                print(f"    {label:<14} : {val*100:.1f}% (bleu)")


if __name__ == "__main__":
    main()

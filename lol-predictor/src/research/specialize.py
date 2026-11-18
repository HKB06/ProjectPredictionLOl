"""ESSAI — un modèle qui connaît les particularités de chaque ligue, compétition et BO.

Question : le modèle gagne-t-il en précision si chaque ligue, chaque type de
compétition (saison régulière / playoffs / événement international) et chaque format
(BO1 / BO3 / BO5) a SES PROPRES réglages ?

Pourquoi pas un modèle totalement séparé par case : une case ligue × BO × compétition
n'a souvent que quelques dizaines de séries par an (LCK BO5 2026 : 34). Un modèle
appris là-dessus apprendrait le hasard. On fait donc un modèle COMMUN plus des
CORRECTIONS propres à chaque case, freinées (`ALPHA`) : une ligue avec beaucoup de
games s'écarte du modèle commun autant que ses données le justifient, une petite ligue
reste près du commun. (Une colonne multipliée par alpha < 1 subit une pénalité L2
1/alpha² plus forte : c'est ce qui freine la correction.)

Deux étages, tous deux en walk-forward (réentraînés tous les 14 jours sur le passé seul) :
  1. GAME : proba que le côté bleu gagne la game.
     - « actuel »       : le v2 en service (régression + calibration par ligue).
     - « spécialisé »   : + corrections par ligue, par compétition et par BO.
     - « boosting »     : arbres de décision boostés, ligue/compétition/BO en catégories.
  2. SÉRIE : proba que l'équipe A gagne la série, à partir de la proba de sa game 1.
     - « conversion »   : formule actuelle (games indépendantes).
     - « appris »       : couche apprise sur les séries passées, propre à chaque BO,
                          corrigée par ligue × BO et compétition × BO (freinée).

Protocole : réglages choisis sur 2025, verdict sur 2026 (jamais regardé pour régler).
Rien ici n'est branché au site.

Usage :
    python -m src.research.specialize
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.ingest.load_oracle import ROOT, load_config, season_files
from src.models.picks_roi import family_of
from src.update.elo import series_prob
from src.v2.model import ALL_FEATURES, C_REG, CLIP_Z, TRAIN_MIN_GAMES, V2Model, _logit
from src.v2.season import _series_ids

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

START = pd.Timestamp("2024-01-01")       # premières prédictions (2023 = rodage)
STEP = pd.Timedelta(days=14)
EVAL_MIN = 15                            # games de data mini par équipe pour juger
MIN_GROUP = 40                           # games mini pour qu'une case ait sa correction
KEY = ["logit_team", "logit_player", "d_form_elo", "d_lineup_games", "d_patch_adapt",
       "d_intl_games", "d_n_subs"]       # signaux dont le poids peut varier par case
ALPHAS = (0.15, 0.3, 0.6)                # force des corrections (petit = très freiné)
FAMS = {"Saison régulière": "régulière", "Playoffs de ligue": "playoffs",
        "Événement international / coupe": "événement"}


# ------------------------------------------------------------------ données
def game_meta(cfg: dict) -> pd.DataFrame:
    """gameid -> playoffs, n° de game dans la série (toutes les saisons rejouées)."""
    parts = []
    for _, path in season_files(cfg):
        d = pd.read_csv(path, low_memory=False,
                        usecols=lambda c: c.strip() in {"gameid", "position", "playoffs", "game"})
        d.columns = [c.strip() for c in d.columns]
        d = d[d["position"].str.lower() == "team"].drop_duplicates("gameid")
        parts.append(pd.DataFrame({
            "gameid": d["gameid"],
            "playoffs": pd.to_numeric(d["playoffs"], errors="coerce").fillna(0).astype(int),
            "game_no": pd.to_numeric(d["game"], errors="coerce").fillna(1).astype(int)}))
    return pd.concat(parts, ignore_index=True).drop_duplicates("gameid")


def load() -> pd.DataFrame:
    """Jeu de données du moteur + série, BO et compétition de chaque game."""
    eng = pickle.load(open(ROOT / "data" / "cache" / "engine.pkl", "rb"))
    df = eng["df"].merge(game_meta(load_config()), on="gameid", how="left")
    df["game_no"] = df["game_no"].fillna(1).astype(int)
    df["playoffs"] = df["playoffs"].fillna(0).astype(int)
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    df["series_id"] = _series_ids(df)
    g = df.groupby("series_id")
    # BO : déduit du nombre de victoires du vainqueur (2 -> BO3, 3 -> BO5).
    wins_a = g.apply(lambda s: int(((s["blue"] == s["blue"].iat[0]) == (s["yb"] == 1)).sum()),
                     include_groups=False)
    n = g.size()
    wn = np.maximum(wins_a, n - wins_a)
    bo = (2 * wn - 1).where(wins_a != n - wins_a, n)
    df["bo"] = df["series_id"].map(bo).clip(upper=5)
    df["bo"] = df["bo"].replace({2: 3, 4: 5}).astype(int)
    df["family"] = [FAMS[family_of(lg, po)] for lg, po in zip(df["league"], df["playoffs"])]
    df["fam_bo"] = df["family"] + " BO" + df["bo"].astype(str)
    return df


# ------------------------------------------------------------------ modèles game
class SpecModel:
    """Régression commune + corrections freinées par ligue, compétition et BO."""

    def __init__(self, alpha: float, groups=("league", "family", "bo")) -> None:
        self.alpha, self.groups = alpha, groups

    def _base(self, df):
        Xi = self.imp.transform(df[ALL_FEATURES].astype(float))
        return np.clip((Xi - self.mu) / self.sd, -CLIP_Z, CLIP_Z)

    def _design(self, df):
        Z = self._base(df)
        k = Z[:, self.key_idx]
        cols = [Z]
        for grp, levels in self.levels.items():
            vals = df[grp].astype(str).to_numpy()
            for lv in levels:
                ind = (vals == lv).astype(float)[:, None]
                cols.append(self.alpha * ind)                 # biais côté bleu propre
                cols.append(self.alpha * ind * k)             # poids propres des signaux clés
        return np.hstack(cols)

    def fit(self, df):
        X = df[ALL_FEATURES].astype(float)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imp.transform(X)
        self.mu, self.sd = Xi.mean(axis=0), Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.key_idx = [ALL_FEATURES.index(c) for c in KEY]
        self.levels = {}
        for grp in self.groups:
            vc = df[grp].astype(str).value_counts()
            self.levels[grp] = list(vc[vc >= MIN_GROUP].index)
        self.lr = LogisticRegression(C=C_REG, max_iter=3000).fit(self._design(df), df["yb"])
        return self

    def predict(self, df):
        return self.lr.predict_proba(self._design(df))[:, 1]


class BoostModel:
    """Arbres boostés : croisent eux-mêmes les signaux et les particularités des cases."""

    CATS = ["league", "family", "bo"]

    def fit(self, df):
        self.cats = {c: {v: i for i, v in enumerate(df[c].astype(str).value_counts().index)}
                     for c in self.CATS}
        self.m = HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=400, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=1.0, categorical_features=list(range(len(self.CATS))),
            early_stopping=False, random_state=0).fit(self._X(df), df["yb"])
        return self

    def _X(self, df):
        c = np.column_stack([df[k].astype(str).map(self.cats[k]).fillna(-1).to_numpy()
                             for k in self.CATS]).astype(float)
        c[c < 0] = np.nan
        return np.hstack([c, df[ALL_FEATURES].astype(float).to_numpy()])

    def predict(self, df):
        return self.m.predict_proba(self._X(df))[:, 1]


def game_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    makers = {"actuel": lambda: V2Model(ALL_FEATURES),
              **{f"spécialisé a={a}": (lambda a=a: SpecModel(a)) for a in ALPHAS},
              "boosting": BoostModel}
    usable = df["nmin"] >= TRAIN_MIN_GAMES
    out, t, t0 = [], START, time.time()
    while t <= df["date"].max():
        nxt = t + STEP
        train = df[(df["date"] < t) & usable]
        block = df[(df["date"] >= t) & (df["date"] < nxt)].copy()
        if len(block):
            for name, make in makers.items():
                block[name] = make().fit(train).predict(block)
            out.append(block)
        t = nxt
    print(f"walk-forward game : {len(out)} blocs en {time.time() - t0:.0f} s")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ étage série
def series_table(res: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """1 ligne par série complète : proba game 1 de chaque modèle, vainqueur, BO."""
    rows = []
    for sid, g in res.groupby("series_id", sort=False):
        f = g.iloc[0]
        if f["game_no"] != 1:
            continue                                  # série coupée : pas de game 1
        a = f["blue"]
        wa = int(((g["blue"] == a) == (g["yb"] == 1)).sum())
        wb = len(g) - wa
        if wa == wb:
            continue
        r = {"series_id": sid, "date": f["date"], "league": f["league"],
             "family": f["family"], "bo": int(f["bo"]), "nmin": f["nmin"], "ya": int(wa > wb)}
        for m in models:
            r[m] = f[m]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


class SeriesLayer:
    """P(A gagne la série) = f(logit proba game 1), propre à chaque BO, + corrections
    freinées par ligue × BO et compétition × BO."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def _X(self, s, col):
        x = _logit(s[col].to_numpy())
        bo = s["bo"].to_numpy()
        cols = []
        for b in (1, 3, 5):
            ind = (bo == b).astype(float)
            cols += [ind, ind * x]
        for key, levels in self.levels.items():
            vals = s[key].to_numpy()
            for lv in levels:
                ind = (vals == lv).astype(float)
                cols += [self.alpha * ind, self.alpha * ind * x]
        return np.column_stack(cols)

    def fit(self, s, col):
        s = s.assign(lg_bo=s["league"] + " BO" + s["bo"].astype(str),
                     fam_bo=s["family"] + " BO" + s["bo"].astype(str))
        self.levels = {}
        for key in ("lg_bo", "fam_bo"):
            vc = s[key].value_counts()
            self.levels[key] = list(vc[vc >= 25].index)
        self.lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(
            self._X(s, col), s["ya"])
        return self

    def predict(self, s, col):
        s = s.assign(lg_bo=s["league"] + " BO" + s["bo"].astype(str),
                     fam_bo=s["family"] + " BO" + s["bo"].astype(str))
        return self.lr.predict_proba(self._X(s, col))[:, 1]


def series_walk_forward(s: pd.DataFrame, col: str, alpha: float, since) -> np.ndarray:
    out = np.full(len(s), np.nan)
    t = pd.Timestamp(since)
    while t <= s["date"].max():
        nxt = t + STEP
        tr = s[s["date"] < t]
        idx = np.where((s["date"] >= t) & (s["date"] < nxt))[0]
        if len(idx) and len(tr) > 200:
            out[idx] = SeriesLayer(alpha).fit(tr, col).predict(s.iloc[idx], col)
        t = nxt
    return out


# ------------------------------------------------------------------ mesures
def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def board(d: pd.DataFrame, y: str, cols: list[str], ref: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    yv = d[y].to_numpy()
    base = ll(yv, d[ref])
    idx = [rng.integers(0, len(d), len(d)) for _ in range(500)]
    rows = []
    for c in cols:
        p = d[c].to_numpy()
        gain = (base - ll(yv, p)) * 1000
        lo, hi = np.percentile([gain[i].mean() for i in idx], [5, 95])
        rows.append({"modèle": c, "n": len(d), "réussite %": round(((p >= .5) == (yv == 1)).mean() * 100, 1),
                     "log-loss": round(ll(yv, p).mean(), 4), "gain x1000": round(gain.mean(), 1),
                     "IC 90 %": f"[{lo:+.1f} ; {hi:+.1f}]"})
    return pd.DataFrame(rows)


def by_group(d, y, cols, key, min_n=60):
    rows = []
    for k, g in d.groupby(key):
        if len(g) < min_n:
            continue
        r = {key: k, "n": len(g)}
        for c in cols:
            r[c] = round(((g[c] >= .5) == (g[y] == 1)).mean() * 100, 1)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df = load()
    print(f"{len(df)} games · BO : {df.drop_duplicates('series_id')['bo'].value_counts().to_dict()}")
    res = game_walk_forward(df)
    res.to_pickle(ROOT / "data" / "cache" / "research_specialize.pkl")
    gm = [c for c in res.columns if c in ("actuel", "boosting") or c.startswith("spécialisé")]
    ev = res[res["nmin"] >= EVAL_MIN]
    for yr in (2024, 2025, 2026):
        d = ev[ev["season"] == yr]
        print(f"\n=== GAMES {yr} (gain = baisse de log-loss x1000 face à « actuel ») ===")
        print(board(d, "yb", gm, "actuel").to_string(index=False))

    # étage série, sur les probas game 1 de l'actuel et du meilleur candidat
    s = series_table(res, gm)
    cols = []
    for m in gm:
        s[f"{m} · conversion"] = [series_prob(p, (b + 1) // 2) for p, b in zip(s[m], s["bo"])]
        cols.append(f"{m} · conversion")
    for m in ("actuel", "boosting", f"spécialisé a={ALPHAS[1]}"):
        for a in (0.3, 1.0):
            c = f"{m} · série apprise a={a}"
            s[c] = series_walk_forward(s, m, a, "2025-01-01")
            cols.append(c)
    s.to_pickle(ROOT / "data" / "cache" / "research_specialize_series.pkl")
    se = s[(s["nmin"] >= EVAL_MIN) & s[cols].notna().all(axis=1)]
    for yr in (2025, 2026):
        d = se[se["date"].dt.year == yr]
        print(f"\n=== SÉRIES {yr} (gain face à « actuel · conversion ») ===")
        print(board(d, "ya", cols, "actuel · conversion").to_string(index=False))
    d26 = se[se["date"].dt.year == 2026]
    show = ["actuel · conversion", "boosting · conversion", f"spécialisé a={ALPHAS[1]} · conversion",
            "actuel · série apprise a=0.3", "boosting · série apprise a=0.3"]
    print("\n=== 2026 : réussite séries par BO ===")
    print(by_group(d26, "ya", show, "bo", 30).to_string(index=False))
    print("\n=== 2026 : réussite séries par compétition ===")
    print(by_group(d26, "ya", show, "family", 30).to_string(index=False))
    print("\n=== 2026 : réussite séries par ligue (n >= 60) ===")
    print(by_group(d26, "ya", show, "league", 60).to_string(index=False))


if __name__ == "__main__":
    main()

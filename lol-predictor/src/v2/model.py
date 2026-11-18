"""Modèle v2 et banc de comparaison avec l'ancien modèle (Elo d'équipe K32+MOV).

Les deux modèles sont jugés sur EXACTEMENT les mêmes games, en walk-forward :
le v2 est réentraîné tous les `REFIT_DAYS` jours sur les seules games antérieures,
puis prédit le bloc suivant. Aucune game de test n'a servi à l'entraînement.

Variantes comparées :
  - Ancien · Elo équipe brut           : la proba actuelle des picks (side-neutre).
  - Ancien · calibré par ligue         : celle affichée par l'app (shrink appris sur
                                         TOUT l'historique -> léger avantage en sa faveur).
  - Ancien · recalibré walk-forward    : l'ancien Elo réappris en walk-forward avec
                                         l'avantage du côté bleu ET la calibration par
                                         ligue. BASELINE ÉQUITABLE : le v2 reçoit exactement
                                         les mêmes traitements, son gain au-delà de cette
                                         ligne vient donc de ses vrais signaux.
  - Elo joueur seul                    : la force des 5 joueurs, sans rien d'autre.
  - Nouveau · v2 sans Elo équipe       : uniquement les blocs « ligne directrice ».
  - Nouveau · v2 sans soloQ            : tous les blocs retenus + Elo équipe.
  - Nouveau · v2 complet               : le précédent + le correctif soloQ, appliqué aux
                                         seules games où les deux équipes sont couvertes.
  - Nouveau · v2 + draft (direct)      : le v2 sans soloQ + les picks et bans de la game.
                                         N'est PAS un prono d'avant-match : la draft n'est
                                         connue qu'au lancement de la game (pari en direct).

Revue du 18/11/2026 : blocs « Objectifs » et « Face-à-face » ajoutés ; cinq blocs mesurés
inutiles ou nuisibles (`RETIRED`) sortis du calcul, mais toujours calculés pour pouvoir
les remesurer.

Usage :
    python -m src.v2.model
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.features.context_factors import FACTORS, build_context, team_snapshot
from src.ingest.load_oracle import load_config
from src.update.elo import (calibrate, compute_elo, series_ids, series_prob,
                            shrink_walk_forward, win_prob)
from src.v2.draft import DRAFT_COLS, DRAFT_FACTORS, build_draft, draft_row
from src.v2.external import EXT_FACTORS, build_external, external_snapshot
from src.v2.extra import EXTRA_FACTORS, build_extra, extra_pair, soloq_team_now
from src.v2.lineups import build_index, expected
from src.v2.player_engine import SIGNALS, build_player_table, player_snapshot

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REFIT_DAYS = 14
WARMUP_DAYS = 90
# Sans saisons passées : rodage court pour couvrir toute la saison. 34 + 4 × 14 = 90 : les blocs de
# réentraînement à partir de J+90 tombent aux mêmes dates qu'avec WARMUP_DAYS.
SEASON_WARMUP_DAYS = 34
TRAIN_MIN_GAMES = 5
EVAL_MIN_GAMES = 15
CONF = 0.70
C_REG = 0.3
CLIP_Z = 3.0
SLOPE_REG = 120.0   # même régularisation que SHRINK_REG de l'ancien modèle

CTX = {f"d_{k}": v for k, v in FACTORS.items()}
GROUPS: dict[str, list[str]] = {
    "Elo équipe": ["logit_team"],
    "Elo joueur": ["logit_player"],
    "Méta & pool": ["d_meta_comfort", "d_pool_meta", "d_pool_min", "d_pool_mean"],
    "Draft (staff)": ["d_draft_comfort", "d_draft_meta", "d_draft_edge", "d_target_bans"],
    "Calendrier & international": ["d_days_rest", "d_games_7d", "d_intl_games"],
    # Leaguepedia (src.v2.external) : vides tant que le cache n'a pas été téléchargé.
    "Coach": [f"d_{k}" for k, v in EXT_FACTORS.items() if v[0] == "Coach"],
    "Lieu & parcours": [f"d_{k}" for k, v in EXT_FACTORS.items() if v[0] == "Lieu & parcours"],
    # src.v2.extra (18/11/2026)
    "Objectifs": [f"d_{k}" for k, v in EXTRA_FACTORS.items() if v[0] == "Objectifs"],
    "Face-à-face": [f"d_{k}" for k, v in EXTRA_FACTORS.items() if v[0] == "Face-à-face"],
}
# Sortis du calcul le 18/11/2026 : retirés un par un, le modèle ne perd rien ; retirés
# ensemble, il gagne (log-loss −0,6 pour 1000 toutes ligues, −2,8 sur les ligues pariables,
# positif sur les deux moitiés de l'historique). Toujours calculés dans le jeu de données.
RETIRED: dict[str, list[str]] = {
    "Tempo": ["d_early", "d_late", "d_comeback"],
    "Roster & remplaçants": ["d_lineup_games", "d_duo_bot", "d_duo_mid_jng", "d_n_subs"],
    "Forme": ["d_form_elo", "d_form_lane"],
    "Patch": ["d_patch_adapt"],
    "Grands matchs": ["d_playoff_games", "d_playoff_adapt"],
}
ALL_FEATURES = [c for cols in GROUPS.values() for c in cols]
V2_PURE = [c for g, cols in GROUPS.items() if g != "Elo équipe" for c in cols]
SOLOQ = ["d_sq_vol7", "d_sq_role14"]
SOLOQ_MIN_TRAIN = 300
# Écart soloQ plafonné à 1,5 écart-type : mesuré sans perte (gain 6,9 pour 1000 sur les 499
# games jugées, contre 6,5 à 3 écarts-types) et le correctif ne déplace jamais un prono de
# plus de 13 points.
SOLOQ_CLIP = 1.5
# « Présence dans la méta » des champions pris : mesurée sans effet, non retenue.
DRAFT_LIVE = [c for c in DRAFT_COLS if c != "d_dr_presence"]
PURE, FULL, LIVE = "Nouveau · v2 sans soloQ", "Nouveau · v2 complet", "Nouveau · v2 + draft (direct)"
LABELS = {**{k: v[1] for k, v in SIGNALS.items()}, **CTX,
          **{f"d_{k}": v[1] for k, v in EXT_FACTORS.items()},
          **{f"d_{k}": v[1] for k, v in EXTRA_FACTORS.items()},
          **{f"d_{k}": v for k, v in DRAFT_FACTORS.items()},
          "logit_team": "Elo équipe (ancien)", "logit_player": "Elo joueur"}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class V2Model:
    """Régression logistique régularisée sur features standardisées et bornées,
    suivie d'une calibration par ligue (pente sur les log-odds, régularisée vers 1)
    apprise sur les seules games d'entraînement — l'équivalent walk-forward du
    `shrink` de `src.update.elo`."""

    def __init__(self, features: list[str], league_cal: bool = True) -> None:
        self.features = features
        self.league_cal = league_cal
        self.slopes: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "V2Model":
        X = df[self.features].astype(float)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imp.transform(X)
        self.mu, self.sd = Xi.mean(axis=0), Xi.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.lr = LogisticRegression(C=C_REG, max_iter=2000).fit(self._z(Xi), df["yb"])
        if self.league_cal:
            z = _logit(self.lr.predict_proba(self._z(Xi))[:, 1])
            for lg, idx in df.groupby("league").indices.items():
                if len(idx) < 50:
                    continue
                m = LogisticRegression(C=1e6, fit_intercept=False).fit(
                    z[idx].reshape(-1, 1), df["yb"].to_numpy()[idx])
                n = len(idx)
                self.slopes[lg] = (n * float(m.coef_[0][0]) + SLOPE_REG) / (n + SLOPE_REG)
        return self

    def _z(self, Xi):
        return np.clip((Xi - self.mu) / self.sd, -CLIP_Z, CLIP_Z)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        Xi = self.imp.transform(df[self.features].astype(float))
        p = self.lr.predict_proba(self._z(Xi))[:, 1]
        if not self.slopes:
            return p
        s = np.array([self.slopes.get(lg, 1.0) for lg in df["league"]])
        return 1 / (1 + np.exp(-s * _logit(p)))

    def weights(self) -> pd.Series:
        return pd.Series(self.lr.coef_[0], index=self.features)


class SoloqLayer:
    """Correctif soloQ, en 2e étage : `logit(p) + parties sur 7 jours + parties au poste`.

    Appris sur les seules games où les deux équipes ont au moins 4 joueurs suivis à
    l'historique complet, et sur des probas déjà hors-échantillon. Mis dans le modèle
    principal, le signal se perdrait : 98 % des games n'ont pas de soloQ. Sans assez de
    games couvertes (`SOLOQ_MIN_TRAIN`), ou pour une game non couverte, la proba est rendue
    telle quelle."""

    def __init__(self) -> None:
        self.ok = False

    def fit(self, d: pd.DataFrame, base: str = PURE) -> "SoloqLayer":
        d = d.dropna(subset=SOLOQ)
        self.n = len(d)
        if self.n < SOLOQ_MIN_TRAIN:
            return self
        self.mu = d[SOLOQ].mean().to_numpy()
        self.sd = d[SOLOQ].std().replace(0, 1).to_numpy()
        self.lr = LogisticRegression(C=C_REG, max_iter=1000).fit(
            self._x(d[base].to_numpy(), d[SOLOQ].to_numpy()), d["yb"])
        self.ok = True
        return self

    def _x(self, p, sq) -> np.ndarray:
        z = np.clip((np.asarray(sq, dtype=float) - self.mu) / self.sd, -SOLOQ_CLIP, SOLOQ_CLIP)
        return np.column_stack([_logit(p), z])

    def apply(self, p, sq) -> np.ndarray:
        """Probas corrigées là où les deux signaux soloQ sont connus, inchangées ailleurs."""
        p = np.array(p, dtype=float)
        sq = np.asarray(sq, dtype=float).reshape(len(p), len(SOLOQ))
        if not self.ok:
            return p
        has = ~np.isnan(sq).any(axis=1)
        if has.any():
            p[has] = self.lr.predict_proba(self._x(p[has], sq[has]))[:, 1]
        return p


def _soloq_walk(res: pd.DataFrame) -> np.ndarray:
    """`PURE` corrigé par le soloQ, le correctif n'étant appris que sur les games d'AVANT."""
    out = res[PURE].to_numpy().copy()
    if not all(c in res for c in SOLOQ):
        return out
    cov = res.dropna(subset=SOLOQ).sort_values("date")
    if len(cov) <= SOLOQ_MIN_TRAIN:
        return out
    step = pd.Timedelta(days=REFIT_DAYS)
    t, end = cov["date"].iloc[SOLOQ_MIN_TRAIN].normalize(), cov["date"].max()
    pos = pd.Series(np.arange(len(res)), index=res.index)
    while t <= end:
        tr, te = cov[cov["date"] < t], cov[(cov["date"] >= t) & (cov["date"] < t + step)]
        if len(te):
            layer = SoloqLayer().fit(tr)
            out[pos[te.index].to_numpy()] = layer.apply(te[PURE].to_numpy(), te[SOLOQ].to_numpy())
        t += step
    return out


def build_dataset(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Table complète (1 ligne/game) + états finaux pour la prédiction pré-match."""
    cfg = cfg or load_config()
    ctx, ctx_state = build_context(cfg)
    pl, pl_state = build_player_table(cfg)
    df = ctx.merge(pl, on="gameid", how="left")
    df["logit_team"] = _logit(df["p_elo"])
    df["logit_player"] = _logit(df["p_player"])
    elo = compute_elo(cfg, history=True)
    # calibration par ligue apprise game après game, jamais sur l'avenir (l'état final
    # `elo["shrink"]` ne sert qu'aux matchs à venir)
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)
    df["p_elo_cal"] = shrink_walk_forward(df["p_elo"], df["yb"], df["league"], df["nmin"])
    ext, ext_state = build_external(df)
    df = df.merge(ext, on="gameid", how="left")
    extra, extra_state = build_extra(df, cfg)
    df = df.merge(extra, on="gameid", how="left")
    draft, draft_state = build_draft(cfg)
    df = df.merge(draft, on="gameid", how="left")
    try:
        lineups = build_index(df, cfg)
    except Exception as e:                    # sans cache Leaguepedia : dernier cinq aligné
        print(f"[lineups] effectif du jour indisponible ({type(e).__name__}: {e})")
        lineups = None
    return df.sort_values("date").reset_index(drop=True), {
        "ctx": ctx_state, "player": pl_state, "elo": elo, "ext": ext_state,
        "extra": extra_state, "draft": draft_state, "lineups": lineups}


def season_origin(df: pd.DataFrame) -> pd.Timestamp:
    """Premier jour de la saison en cours (la dernière du jeu de données)."""
    return df.loc[df["season"] == df["season"].max(), "date"].min().normalize()


def has_history(df: pd.DataFrame) -> bool:
    """Des saisons passées ont-elles été rejouées avant la saison en cours ?"""
    return df["season"].nunique() > 1


def comparison_window(res: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Games servant à comparer les modèles sur la saison en cours.

    Avec les saisons passées : toute la saison (les modèles arrivent déjà rodés).
    Sans elles : après WARMUP_DAYS de rodage.
    """
    start = season_origin(df)
    if not has_history(df):
        start += pd.Timedelta(days=WARMUP_DAYS)
    return res[res["date"] >= start]


def walk_forward(df: pd.DataFrame, warmup_days: int = WARMUP_DAYS,
                 since: pd.Timestamp | None = None, until: pd.Timestamp | None = None,
                 ablation: bool = True) -> pd.DataFrame:
    """Prédictions hors-échantillon de chaque variante, bloc de REFIT_DAYS par bloc.

    Chaque bloc est prédit par un modèle entraîné sur les seules games ANTÉRIEURES
    (saisons passées comprises). Première game prédite (`since`) :
      - avec les saisons passées : le premier jour de la saison en cours ;
      - sans elles : `warmup_days` après. `SEASON_WARMUP_DAYS` étend alors les
        prédictions au début de saison (page Saison).
    Pour juger aussi les saisons passées : `since=first_evaluable(df)` ; `until` arrête
    les prédictions avant une date (ex. le début de la saison en cours, déjà calculée).
    `ablation=False` saute les variantes « v2 sans un bloc » (4 fois plus rapide).

    Les blocs tombent toujours aux mêmes dates (début de saison + WARMUP_DAYS, à
    REFIT_DAYS près), quel que soit `since` : deux appels donnent les mêmes
    prédictions sur les games qu'ils ont en commun.
    """
    variants = {
        "Ancien · recalibré walk-forward": ["logit_team"],
        "Nouveau · v2 sans Elo équipe": V2_PURE,
        PURE: ALL_FEATURES,
        LIVE: ALL_FEATURES + DRAFT_LIVE,
    }
    for g in GROUPS if ablation else ():
        if g != "Elo équipe":
            variants[f"v2 sans « {g} »"] = [c for c in ALL_FEATURES if c not in GROUPS[g]]

    origin = season_origin(df)
    if since is None:
        since = origin if has_history(df) else origin + pd.Timedelta(days=warmup_days)
    step = pd.Timedelta(days=REFIT_DAYS)
    anchor = origin + pd.Timedelta(days=WARMUP_DAYS)
    t = anchor + ((since - anchor) // step) * step            # dernier début de bloc <= since
    usable = df["nmin"] >= TRAIN_MIN_GAMES
    end = df["date"].max() if until is None else until - pd.Timedelta(seconds=1)
    out = []
    while t <= end:
        nxt = t + step
        lo = max(t, since)
        train = df[(df["date"] < lo) & usable]
        block = df[(df["date"] >= lo) & (df["date"] < nxt) & (df["date"] <= end)].copy()
        if len(block):
            for name, feats in variants.items():
                block[name] = V2Model(feats).fit(train).predict(block)
            out.append(block)
        t = nxt
    res = pd.concat(out, ignore_index=True)
    res["Ancien · Elo équipe brut"] = res["p_elo"]
    res["Ancien · calibré par ligue"] = res["p_elo_cal"]
    res["Elo joueur seul"] = res["p_player"]
    res[FULL] = _soloq_walk(res)
    return res


def first_evaluable(df: pd.DataFrame) -> pd.Timestamp:
    """Première date jugeable de tout l'historique : WARMUP_DAYS après la première game."""
    return df["date"].min().normalize() + pd.Timedelta(days=WARMUP_DAYS)


MAIN = ["Ancien · Elo équipe brut", "Ancien · calibré par ligue", "Ancien · recalibré walk-forward",
        "Elo joueur seul", "Nouveau · v2 sans Elo équipe", PURE, FULL, LIVE]


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scoreboard(res: pd.DataFrame, models: list[str] | None = None,
               baseline: str = "Ancien · recalibré walk-forward") -> pd.DataFrame:
    """Métriques par variante sur les games de test (≥ EVAL_MIN_GAMES par équipe)."""
    d = res[res["nmin"] >= EVAL_MIN_GAMES]
    y = d["yb"].to_numpy()
    ref = _ll(y, d[baseline].to_numpy())
    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    rows = []
    for m in models or MAIN:
        p = d[m].to_numpy()
        pf = np.maximum(p, 1 - p)
        correct = (p >= 0.5) == (y == 1)
        hc = pf >= CONF
        diff = ref - _ll(y, p)
        lo, hi = np.percentile([diff[i].mean() for i in idx], [5, 95])
        hit = correct[hc].mean() if hc.any() else np.nan
        rows.append({
            "modèle": m,
            "log-loss": round(float(_ll(y, p).mean()), 4),
            "Brier": round(float(np.mean((p - y) ** 2)), 4),
            "réussite %": round(float(correct.mean() * 100), 1),
            "picks ≥70 %": int(hc.sum()),
            "réussite picks %": round(float(hit * 100), 1) if hit == hit else np.nan,
            "cote mini picks": round(float(1 / hit), 2) if hit == hit and hit > 0 else np.nan,
            "gain vs baseline (x1000)": round(float(diff.mean() * 1000), 2),
            "IC 90 %": f"[{lo*1000:+.1f} ; {hi*1000:+.1f}]",
        })
    out = pd.DataFrame(rows)
    out.attrs["n"] = len(d)
    out.attrs["span"] = (d["date"].min(), d["date"].max())
    return out


def series_board(res: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Réussite au niveau SÉRIE : proba de la game 1 convertie en proba de série."""
    d = res.sort_values("date").copy()
    d["series"] = series_ids(d["league"], d["blue"], d["red"], d["date"])
    d = d[d["nmin"] >= EVAL_MIN_GAMES]
    rows = []
    for m in models or MAIN:
        won, picked = [], []
        for _, g in d.groupby("series"):
            g = g.sort_values("date")
            first = g.iloc[0]
            fav = first["blue"] if first[m] >= 0.5 else first["red"]
            wins = [b if y == 1 else r for b, r, y in zip(g["blue"], g["red"], g["yb"])]
            wf, wo = wins.count(fav), len(wins) - wins.count(fav)
            if wf == wo:
                continue
            ps = series_prob(max(first[m], 1 - first[m]), max(wf, wo))
            won.append(wf > wo)
            picked.append(ps >= CONF)
        won, picked = np.array(won), np.array(picked)
        rows.append({"modèle": m, "séries": len(won),
                     "bon vainqueur %": round(won.mean() * 100, 1),
                     "séries ≥70 %": int(picked.sum()),
                     "réussite séries ≥70 %": round(won[picked].mean() * 100, 1)
                     if picked.any() else np.nan})
    return pd.DataFrame(rows)


def ablation(res: pd.DataFrame) -> pd.DataFrame:
    """Ce que chaque bloc apporte : perte de log-loss quand on le retire du v2 complet."""
    d = res[res["nmin"] >= EVAL_MIN_GAMES]
    y = d["yb"].to_numpy()
    pure = _ll(y, d[PURE].to_numpy())          # les variantes « sans un bloc » n'ont pas le soloQ
    rows = []
    for g in GROUPS:
        col = f"v2 sans « {g} »"
        if col not in d:
            continue
        diff = _ll(y, d[col].to_numpy()) - pure
        rows.append({"bloc": g, "apport (x1000)": round(float(diff.mean() * 1000), 2),
                     "verdict": "utile" if diff.mean() > 0 else "bruit / nuisible"})
    for g, col in (("SoloQ (correctif)", FULL), ("Draft de la game (direct seulement)", LIVE)):
        if col in d:
            diff = pure - _ll(y, d[col].to_numpy())
            rows.append({"bloc": g, "apport (x1000)": round(float(diff.mean() * 1000), 2),
                         "verdict": "utile" if diff.mean() > 0 else "bruit / nuisible"})
    return pd.DataFrame(rows).sort_values("apport (x1000)", ascending=False)


# ------------------------------------------------------------------ pré-match
class Predictor:
    """Entraîne le v2 sur tout l'historique et prédit un BO à venir (ancien vs nouveau)."""

    def __init__(self, df: pd.DataFrame, states: dict, res: pd.DataFrame | None = None) -> None:
        """`res` : prédictions hors-échantillon du walk-forward, pour apprendre le correctif
        soloQ. Sans elles, le v2 complet est rendu sans correctif."""
        self.states = states
        train = df[df["nmin"] >= TRAIN_MIN_GAMES]
        self.models = {"Nouveau · v2 complet": V2Model(ALL_FEATURES).fit(train),
                       "Nouveau · v2 sans Elo équipe": V2Model(V2_PURE).fit(train),
                       "Ancien · recalibré walk-forward": V2Model(["logit_team"]).fit(train)}
        self.live = V2Model(ALL_FEATURES + DRAFT_LIVE).fit(train)
        self.soloq = SoloqLayer()
        if res is not None and PURE in res and all(c in res for c in SOLOQ):
            self.soloq.fit(res)
        # Équipes de la saison en cours seulement, les plus actives d'abord.
        cur = df[df["season"] == df["season"].max()]
        played = pd.concat([cur["blue"], cur["red"]]).value_counts()
        self.teams = list(played.index)

    def lineup(self, team: str) -> dict:
        """Le cinq attendu de `team` (`src.v2.lineups.expected`)."""
        last = self.states["ctx"].last_lineup.get(team, {})
        return expected(self.states.get("lineups"), team, last)

    def _row(self, a: str, b: str, ca: dict, cb: dict, pa: dict, pb: dict,
             ea: dict, eb: dict, league: str | None,
             ra: dict | None = None, rb: dict | None = None) -> pd.DataFrame:
        ctx = self.states["ctx"]
        row = {"league": league,
               "logit_team": _logit(win_prob(ctx.elo[a], ctx.elo[b])),
               "logit_player": _logit(win_prob(pa["rating"], pb["rating"]))}
        for k in FACTORS:
            row[f"d_{k}"] = ca[k] - cb[k]
        for key in SIGNALS:
            row[key] = pa[key[2:]] - pb[key[2:]]
        for k in EXT_FACTORS:
            row[f"d_{k}"] = ea[k] - eb[k]
        ra, rb = ra or {}, rb or {}
        row.update(extra_pair(self.states.get("extra"), a, b,
                              la=ra.get("lineup"), lb=rb.get("lineup"),
                              pa=ra.get("pages"), pb=rb.get("pages")))
        return pd.DataFrame([row])

    def matchup(self, a: str, b: str, bestof: int = 3, league: str | None = None) -> dict:
        """`league` : code Oracle de la compétition du match, quand il est connu
        (calendrier). Sinon on retombe sur la ligue la moins fiable des deux équipes."""
        ctx, pl, elo = self.states["ctx"], self.states["player"], self.states["elo"]
        ra, rb = self.lineup(a), self.lineup(b)
        ca, cb = (team_snapshot(ctx, a, lineup=ra["lineup"]),
                  team_snapshot(ctx, b, lineup=rb["lineup"]))
        pa, pb = (player_snapshot(pl, a, lineup=ra["lineup"], names=ra["names"]),
                  player_snapshot(pl, b, lineup=rb["lineup"], names=rb["names"]))
        ext = self.states.get("ext")
        ea, eb = external_snapshot(ext, a), external_snapshot(ext, b)
        p_raw = win_prob(ctx.elo[a], ctx.elo[b])
        la, lb = elo["league"].get(a), elo["league"].get(b)
        cand = [x for x in (la, lb) if x in elo["reliability"]]
        lg = league if league in elo["reliability"] else (
            min(cand, key=lambda c: elo["reliability"][c]) if cand else None)
        ab = self._row(a, b, ca, cb, pa, pb, ea, eb, lg, ra, rb)
        ba = self._row(b, a, cb, ca, pb, pa, eb, ea, lg, rb, ra)
        wn = (int(bestof) + 1) // 2
        games = {"Ancien · Elo équipe brut": p_raw,
                 "Ancien · calibré par ligue": calibrate(p_raw, elo["shrink"].get(lg, 1.0)),
                 "Elo joueur seul": win_prob(pa["rating"], pb["rating"])}
        soloq = getattr(self, "soloq", None)
        for name, m in self.models.items():
            # side-neutre : moyenne en log-odds de A côté bleu et A côté rouge
            p1, p2 = m.predict(ab), m.predict(ba)
            if name == "Nouveau · v2 complet" and soloq is not None and soloq.ok:
                games[PURE] = float(1 / (1 + np.exp(-(_logit(p1)[0] - _logit(p2)[0]) / 2)))
                p1 = soloq.apply(p1, ab[SOLOQ].to_numpy())
                p2 = soloq.apply(p2, ba[SOLOQ].to_numpy())
            l1, l2 = _logit(p1)[0], _logit(p2)[0]
            games[name] = float(1 / (1 + np.exp(-(l1 - l2) / 2)))
        xs = self.states.get("extra")
        h2h = {k[2:]: float(ab[k].iloc[0]) for k in GROUPS["Face-à-face"]} if xs is not None else None
        return {
            "games": games,
            "series": {k: series_prob(v, wn) for k, v in games.items()},
            "ctx": (ca, cb), "player": (pa, pb), "ext": (ea, eb),
            "soloq": (soloq_team_now(xs, a, ra["pages"]), soloq_team_now(xs, b, rb["pages"]))
            if xs is not None else None,
            "lineups": (ra, rb),
            "extra": (xs.obj_features(a), xs.obj_features(b)) if xs is not None else None,
            "h2h": h2h,
            "contrib": self._contrib(ab, ba),
        }

    def draft_game(self, blue: str, red: str, picks_blue: dict, picks_red: dict,
                   bans_blue: list | None = None, bans_red: list | None = None,
                   league: str | None = None) -> dict:
        """Proba que le côté BLEU gagne CETTE game, avant puis après la draft.

        `picks_*` : {poste: champion} (top, jng, mid, bot, sup). Les côtés étant connus, la
        proba garde l'avantage du côté bleu (pas de moyenne des deux côtés). À utiliser
        entre la fin de la draft et le début de la game."""
        ctx, pl, elo = self.states["ctx"], self.states["player"], self.states["elo"]
        ra, rb = self.lineup(blue), self.lineup(red)
        ca, cb = (team_snapshot(ctx, blue, lineup=ra["lineup"]),
                  team_snapshot(ctx, red, lineup=rb["lineup"]))
        pa, pb = (player_snapshot(pl, blue, lineup=ra["lineup"], names=ra["names"]),
                  player_snapshot(pl, red, lineup=rb["lineup"], names=rb["names"]))
        ext = self.states.get("ext")
        ea, eb = external_snapshot(ext, blue), external_snapshot(ext, red)
        la, lb = elo["league"].get(blue), elo["league"].get(red)
        cand = [x for x in (la, lb) if x in elo["reliability"]]
        lg = league if league in elo["reliability"] else (
            min(cand, key=lambda c: elo["reliability"][c]) if cand else None)
        row = self._row(blue, red, ca, cb, pa, pb, ea, eb, lg, ra, rb)
        before = float(self.models["Nouveau · v2 complet"].predict(row)[0])
        feats = draft_row(self.states["draft"], blue, red, picks_blue, picks_red,
                          bans_blue, bans_red, lineup_blue=ra["lineup"], lineup_red=rb["lineup"])
        for k, v in feats.items():
            row[k] = v
        after = float(self.live.predict(row)[0])
        m = self.live
        z = m._z(m.imp.transform(row[m.features].astype(float)))[0]
        contrib = pd.Series(z * m.lr.coef_[0], index=m.features)
        contrib = contrib[[c for c in DRAFT_LIVE]].rename(index=LABELS)
        return {"before": before, "after": after, "draft": feats,
                "contrib": contrib.sort_values(key=np.abs, ascending=False)}

    def _contrib(self, ab: pd.DataFrame, ba: pd.DataFrame) -> pd.Series:
        """Poids de chaque signal dans la décision du v2 complet (en log-odds pour A)."""
        m = self.models["Nouveau · v2 complet"]
        za = m._z(m.imp.transform(ab[m.features].astype(float)))[0]
        zb = m._z(m.imp.transform(ba[m.features].astype(float)))[0]
        c = pd.Series((za - zb) / 2 * m.lr.coef_[0], index=m.features)
        return c.rename(index=LABELS).sort_values(key=np.abs, ascending=False)


def season_board(res: pd.DataFrame, model: str = "Nouveau · v2 complet",
                 baseline: str = "Ancien · recalibré walk-forward") -> pd.DataFrame:
    """Le gain du nouveau modèle, saison par saison puis toutes saisons confondues.

    Chaque saison est un test indépendant : le modèle n'y a jamais vu que le passé.
    """
    d = res[res["nmin"] >= EVAL_MIN_GAMES]
    rng = np.random.default_rng(42)
    rows = []
    for label, part in [*((str(y), g) for y, g in d.groupby("season")), ("toutes", d)]:
        y = part["yb"].to_numpy()
        old, new = _ll(y, part[baseline].to_numpy()), _ll(y, part[model].to_numpy())
        diff = old - new
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        lo, hi = np.percentile(boots, [5, 95])
        p_old, p_new = part[baseline].to_numpy(), part[model].to_numpy()
        rows.append({
            "saison": label, "games": len(part),
            "log-loss ancien": round(float(old.mean()), 4),
            "log-loss nouveau": round(float(new.mean()), 4),
            "réussite ancien %": round(float(((p_old >= 0.5) == (y == 1)).mean() * 100), 1),
            "réussite nouveau %": round(float(((p_new >= 0.5) == (y == 1)).mean() * 100), 1),
            "gain (x1000)": round(float(diff.mean() * 1000), 2),
            "IC 90 %": f"[{lo*1000:+.1f} ; {hi*1000:+.1f}]",
            "verdict": "nouveau meilleur" if lo > 0 else "ancien meilleur" if hi < 0
                       else "rien de prouvé",
        })
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Comparaison ancien modèle / modèle v2")
    ap.add_argument("--no-history", action="store_true",
                    help="ignore les saisons passées (data.history_years)")
    ap.add_argument("--all-seasons", action="store_true",
                    help="juge aussi les saisons passées (plus long)")
    args = ap.parse_args()
    cfg = load_config()
    if args.no_history:
        cfg["data"]["history_years"] = []

    print("Construction du jeu de données (contexte + joueurs)…")
    df, _ = build_dataset(cfg)
    seasons = sorted(int(y) for y in df["season"].unique())
    print(f"  {len(df)} games · saisons {', '.join(map(str, seasons))} · "
          f"walk-forward : refit tous les {REFIT_DAYS} j")
    if args.all_seasons and has_history(df):
        full = walk_forward(df, since=first_evaluable(df))
        print("\n=== NOUVEAU CONTRE ANCIEN, SAISON PAR SAISON ===")
        print(season_board(full).to_string(index=False))
        res = comparison_window(full, df)
    else:
        res = comparison_window(walk_forward(df, SEASON_WARMUP_DAYS), df)
    sb = scoreboard(res)
    span = sb.attrs["span"]
    print(f"\n=== COMPARAISON PAR GAME ({sb.attrs['n']} games test, "
          f"{span[0]:%Y-%m-%d} -> {span[1]:%Y-%m-%d}, équipes ≥{EVAL_MIN_GAMES} games) ===")
    print(sb.to_string(index=False))
    print("\n=== COMPARAISON PAR SÉRIE (proba game 1 -> proba série) ===")
    print(series_board(res).to_string(index=False))
    print("\n=== APPORT DE CHAQUE BLOC (retiré du v2 complet) ===")
    print(ablation(res).to_string(index=False))


if __name__ == "__main__":
    main()

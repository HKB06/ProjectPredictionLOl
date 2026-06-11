"""Combien la DRAFT ajoute-t-elle à l'Elo ? (mesuré, split temporel sans fuite)

On compare l'accuracy / AUC sur un test (20 % les plus récents) de :
  M0 — Elo seul, **pré-game** (side inconnu) : on prédit le mieux noté.
  M1 — Elo + side, **au début du match** (on sait qui est bleu) : ajoute l'avantage bleu.
  M2 — Elo + side + **DRAFT** : ajoute le penchant des champions (ratings logistiques).

Le gain de la draft = M2 − M1 (ce que tu apportes en me donnant les 10 champions).
Le gain du side  = M1 − M0.

Anti-fuite : Elo en walk-forward ; ratings champions appris sur le TRAIN seulement
(scores du train obtenus par validation croisée pour ne pas surévaluer la draft).

Usage : python -m src.models.eval_draft
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.ingest.load_oracle import ROOT, load_config
from src.update.elo import BASE, K, _mov_mult, win_prob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BURN_IN = 5
TEST_FRAC = 0.2


def _build_games(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    pos = df["position"].str.lower()
    teams, players = df[pos == "team"].copy(), df[pos != "team"].copy()

    tb = teams[teams["side"].str.lower() == "blue"][["gameid", "teamname", "result", "date", "league", "kills"]]
    tr = teams[teams["side"].str.lower() == "red"][["gameid", "teamname", "kills"]]
    g = (tb.merge(tr, on="gameid", suffixes=("_b", "_r"))
           .dropna(subset=["teamname_b", "teamname_r", "result"]))

    players["side_l"] = players["side"].str.lower()
    champ = players.groupby(["gameid", "side_l"])["champion"].apply(list).unstack("side_l")
    g = g.merge(champ, left_on="gameid", right_index=True).dropna(subset=["blue", "red"])
    g = g[g["blue"].apply(lambda x: isinstance(x, list) and len(x) >= 5)
          & g["red"].apply(lambda x: isinstance(x, list) and len(x) >= 5)]
    g = g.rename(columns={"teamname_b": "tblue", "teamname_r": "tred", "result": "y",
                          "kills_b": "kb", "kills_r": "kr"})
    return g.sort_values("date").reset_index(drop=True)


def _elo_diff_walkforward(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    elo: dict[str, float] = defaultdict(lambda: BASE)
    n: dict[str, int] = defaultdict(int)
    diffs, nmin = [], []
    for a, b, y, kb, kr in zip(g["tblue"], g["tred"], g["y"], g["kb"], g["kr"]):
        diffs.append(elo[a] - elo[b])
        nmin.append(min(n[a], n[b]))
        ea = win_prob(elo[a], elo[b])
        mult = _mov_mult(kb, kr)
        elo[a] += K * mult * (y - ea)
        elo[b] += K * mult * ((1 - y) - (1 - ea))
        n[a] += 1
        n[b] += 1
    return np.array(diffs), np.array(nmin)


def _champ_matrix(blue_lists, red_lists, idx) -> np.ndarray:
    X = np.zeros((len(blue_lists), len(idx)))
    for i, (b, r) in enumerate(zip(blue_lists, red_lists)):
        for c in b:
            if c in idx:
                X[i, idx[c]] += 1
        for c in r:
            if c in idx:
                X[i, idx[c]] -= 1
    return X


def _report(name, y, p):
    pred = (p >= 0.5).astype(int)
    return f"  {name:32} accuracy {accuracy_score(y, pred)*100:5.2f}%   AUC {roc_auc_score(y, p):.3f}"


def main() -> None:
    cfg = load_config()
    g = _build_games(cfg)
    ediff, nmin = _elo_diff_walkforward(g)
    g["elo_diff"] = ediff
    g = g[nmin >= BURN_IN].reset_index(drop=True)
    y = g["y"].astype(int).to_numpy()

    n_test = int(len(g) * TEST_FRAC)
    tr = np.arange(len(g)) < (len(g) - n_test)   # 80 % anciens
    te = ~tr
    cut = g["date"].iloc[len(g) - n_test]
    print(f"Games (hors cold-start) : {len(g)}  | train {tr.sum()} / test {te.sum()}  "
          f"(coupe au {cut:%Y-%m-%d})\n")

    # Ratings champions : appris sur le TRAIN seulement
    allc = sorted({c for lst in g["blue"][tr] for c in lst} | {c for lst in g["red"][tr] for c in lst})
    idx = {c: i for i, c in enumerate(allc)}
    Xc = _champ_matrix(list(g["blue"]), list(g["red"]), idx)
    champ_clf = LogisticRegression(C=0.3, max_iter=3000)
    # score draft du test = hors échantillon (modèle entraîné sur train)
    champ_clf.fit(Xc[tr], y[tr])
    draft_logit = np.zeros(len(g))
    draft_logit[te] = champ_clf.decision_function(Xc[te])
    # score draft du train = validation croisée (non biaisé) pour fitter M2 proprement
    draft_logit[tr] = cross_val_predict(LogisticRegression(C=0.3, max_iter=3000),
                                        Xc[tr], y[tr], cv=5, method="decision_function")

    ed = g["elo_diff"].to_numpy().reshape(-1, 1)
    dl = draft_logit.reshape(-1, 1)

    # M0 : Elo seul, side-neutre (pré-game) -> proba sigmoïde de l'écart Elo, SANS intercept
    p0 = win_prob(g["elo_diff"].to_numpy(), 0.0)  # 1/(1+10^(-elo_diff/400))

    # M1 : Elo + side (intercept libre = avantage bleu)
    m1 = LogisticRegression(max_iter=2000).fit(ed[tr], y[tr])
    p1 = m1.predict_proba(ed[te])[:, 1]

    # M2 : Elo + side + draft
    m2 = LogisticRegression(max_iter=2000).fit(np.hstack([ed, dl])[tr], y[tr])
    p2 = m2.predict_proba(np.hstack([ed, dl])[te])[:, 1]

    # Draft seule (référence)
    md = LogisticRegression(max_iter=2000).fit(dl[tr], y[tr])
    pd_ = md.predict_proba(dl[te])[:, 1]

    yte = y[te]
    print("=== TEST (20 % les plus récents) ===")
    print(_report("M0  Elo seul (pré-game)", yte, p0[te]))
    print(_report("    Draft seule", yte, pd_))
    print(_report("M1  Elo + side (début match)", yte, p1))
    print(_report("M2  Elo + side + DRAFT", yte, p2))
    a1 = accuracy_score(yte, (p1 >= .5)) * 100
    a2 = accuracy_score(yte, (p2 >= .5)) * 100
    a0 = accuracy_score(yte, (p0[te] >= .5)) * 100
    print(f"\n  Gain SIDE  (M1 − M0) : {a1 - a0:+.2f} pts d'accuracy")
    print(f"  Gain DRAFT (M2 − M1) : {a2 - a1:+.2f} pts d'accuracy  "
          f"(AUC {roc_auc_score(yte, p1):.3f} -> {roc_auc_score(yte, p2):.3f})")


if __name__ == "__main__":
    main()

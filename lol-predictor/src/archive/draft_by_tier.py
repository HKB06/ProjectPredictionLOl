"""La draft prédit-elle MIEUX entre équipes de même niveau (top ligues) ?

Hypothèse (user) : en LCK/LPL/LEC le skill est ~égal -> la draft devient décisive ->
draft-only AUC plus haut qu'en régionales (où le meilleur gagne quoi qu'il arrive).

Méthode : ratings de champions appris (logistic, +1 bleu/-1 rouge) sur le train (80%
ancien de CHAQUE ligue), test = 20% récents de chaque ligue. On compare le draft-only AUC
par tier. TOP3 = LCK, LPL, LEC.

Usage : python -m src.models.draft_by_tier
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.ingest.load_oracle import ROOT, load_config

TOP3 = {"LCK", "LPL", "LEC"}


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    pos = df["position"].str.lower()
    teams, players = df[pos == "team"], df[pos != "team"].copy()

    # game-level (vectorisé) : 1 ligne / game côté bleu
    tb = teams[teams["side"].str.lower() == "blue"][["gameid", "date", "league", "result"]]
    tb = tb.rename(columns={"result": "y"}).drop_duplicates("gameid")

    players["side_l"] = players["side"].str.lower()
    champ = players.groupby(["gameid", "side_l"])["champion"].apply(list).unstack("side_l")

    m = tb.merge(champ, left_on="gameid", right_index=True).dropna(subset=["blue", "red"])
    m = m[m["blue"].apply(lambda x: isinstance(x, list) and len(x) >= 5)
          & m["red"].apply(lambda x: isinstance(x, list) and len(x) >= 5)]
    m = m.sort_values("date").reset_index(drop=True)

    # split 80/20 par ligue (test = 20% récents de chaque ligue)
    test_idx = []
    for _, sub in m.groupby("league"):
        sub = sub.sort_values("date")
        k = max(1, int(0.2 * len(sub)))
        test_idx += list(sub.index[-k:])
    test = m.loc[test_idx]
    train = m.drop(index=test_idx)

    allc = sorted({c for lst in train["blue"] for c in lst} | {c for lst in train["red"] for c in lst})
    idx = {c: i for i, c in enumerate(allc)}

    def design(sub):
        X = np.zeros((len(sub), len(allc)))
        for i, (b, r) in enumerate(zip(sub["blue"], sub["red"])):
            for c in b:
                if c in idx:
                    X[i, idx[c]] += 1
            for c in r:
                if c in idx:
                    X[i, idx[c]] -= 1
        return X

    clf = LogisticRegression(C=0.3, max_iter=3000).fit(design(train), train["y"].values)
    test = test.copy()
    test["p"] = clf.predict_proba(design(test))[:, 1]

    def metrics(sub):
        y = sub["y"].values
        if len(sub) < 20 or len(set(y)) < 2:
            return None
        base = max(y.mean(), 1 - y.mean())
        return len(sub), base * 100, roc_auc_score(y, sub["p"]), accuracy_score(y, (sub["p"] >= 0.5).astype(int)) * 100

    print("=" * 70)
    print("  DRAFT SEULE -> VAINQUEUR, PAR NIVEAU DE LIGUE")
    print(f"  (ratings appris sur {len(train)} games train ; test 20% recent/ligue)")
    print("=" * 70)
    print(f"  {'ligue':<14} {'n_test':>6} {'base%':>7} {'AUC':>6} {'acc%':>7}")
    print("  " + "-" * 64)
    for lg in ["LCK", "LPL", "LEC", "LCS"]:
        r = metrics(test[test["league"] == lg])
        if r:
            print(f"  {lg:<14} {r[0]:>6} {r[1]:>6.1f}% {r[2]:>6.3f} {r[3]:>6.1f}%")

    r_top = metrics(test[test["league"].isin(TOP3)])
    r_oth = metrics(test[~test["league"].isin(TOP3)])
    print("  " + "-" * 64)
    if r_top:
        print(f"  {'TOP3 (LCK+LPL+LEC)':<14} {r_top[0]:>6} {r_top[1]:>6.1f}% {r_top[2]:>6.3f} {r_top[3]:>6.1f}%")
    if r_oth:
        print(f"  {'AUTRES ligues':<14} {r_oth[0]:>6} {r_oth[1]:>6.1f}% {r_oth[2]:>6.3f} {r_oth[3]:>6.1f}%")
    print("=" * 70)
    if r_top and r_oth:
        diff = r_top[2] - r_oth[2]
        if diff > 0.03:
            print(f"  -> Draft predit MIEUX en top3 (+{diff:.3f} AUC) : hypothese SOUTENUE.")
        elif diff < -0.03:
            print(f"  -> Draft predit MOINS en top3 ({diff:.3f} AUC) : hypothese REJETEE.")
        else:
            print(f"  -> Difference faible ({diff:+.3f} AUC) : pas de difference nette.")
    print("  (AUC 0.50 = hasard ; rappel : draft seule reste un signal limite)")
    print("=" * 70)


if __name__ == "__main__":
    main()

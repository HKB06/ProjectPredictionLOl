"""Modèle DRAFT v2 (façon DraftGap, version pro) : à quel point la DRAFT SEULE prédit
le vainqueur ?

v1 (actuel) = moyenne des winrates des 5 champions -> 1 feature, trop bête.
v2 (ici)   = ratings de champions APPRIS par régression logistique sur ~5500 games pro :
   X[c] = +1 si le bleu a le champion c, -1 si le rouge l'a. Coef = contribution au win.
   (capture la force réelle de chaque champ dans le méta, pas juste sa winrate brute)

Anti-fuite : entraînement sur tous les games AVANT la fenêtre de test LCK ; test = 20%
LCK les plus récents. On compare v2 vs v1 vs baseline côté bleu.

Usage : python -m src.models.draft_model
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from src.ingest.load_oracle import ROOT, load_config

# Draft live KT vs DK (game 1) pour sanity-check
KT = ["Rumble", "Xin Zhao", "Viktor", "Jhin", "Nautilus"]
DK = ["Ornn", "Wukong", "Orianna", "Ashe", "Seraphine"]


def main() -> None:
    cfg = load_config()
    csv = ROOT / cfg["data"]["oracle_csv"]
    df = pd.read_csv(csv, low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    pos = df["position"].str.lower()
    teams = df[pos == "team"]
    players = df[pos != "team"].copy()

    grows = []
    for gid, g in teams.groupby("gameid"):
        b = g[g["side"].str.lower() == "blue"]
        r = g[g["side"].str.lower() == "red"]
        if len(b) != 1 or len(r) != 1:
            continue
        grows.append({"gameid": gid, "date": b.iloc[0]["date"],
                      "league": b.iloc[0]["league"], "y": int(b.iloc[0]["result"])})
    gdf = pd.DataFrame(grows).set_index("gameid")

    players["side_l"] = players["side"].str.lower()
    by = players.groupby(["gameid", "side_l"])["champion"].apply(list)

    data = []
    for gid in gdf.index:
        try:
            bc = [c for c in by[(gid, "blue")] if isinstance(c, str)]
            rc = [c for c in by[(gid, "red")] if isinstance(c, str)]
        except KeyError:
            continue
        if len(bc) >= 5 and len(rc) >= 5:
            data.append((gid, bc[:5], rc[:5]))

    meta = gdf.loc[[d[0] for d in data]].reset_index()
    meta["blue"] = [d[1] for d in data]
    meta["red"] = [d[2] for d in data]
    meta = meta.sort_values("date").reset_index(drop=True)

    allc = sorted({c for d in data for c in d[1] + d[2]})
    idx = {c: i for i, c in enumerate(allc)}

    lck = meta[meta["league"] == "LCK"]
    n_test = max(1, int(0.2 * len(lck)))
    test_min = lck["date"].iloc[-n_test]
    train = meta[meta["date"] < test_min]
    test = meta[(meta["date"] >= test_min) & (meta["league"] == "LCK")]

    def design(sub):
        X = np.zeros((len(sub), len(allc)))
        for i, (_, row) in enumerate(sub.iterrows()):
            for c in row["blue"]:
                if c in idx:
                    X[i, idx[c]] += 1
            for c in row["red"]:
                if c in idx:
                    X[i, idx[c]] -= 1
        return X

    Xtr, ytr = design(train), train["y"].values
    Xte, yte = design(test), test["y"].values

    clf = LogisticRegression(C=0.3, max_iter=3000).fit(Xtr, ytr)
    p2 = clf.predict_proba(Xte)[:, 1]

    # v1 : moyenne winrate champions (base train)
    wins, games = defaultdict(int), defaultdict(int)
    for _, row in train.iterrows():
        for c in row["blue"]:
            games[c] += 1; wins[c] += row["y"]
        for c in row["red"]:
            games[c] += 1; wins[c] += (1 - row["y"])

    def cwr(c):
        return wins[c] / games[c] if games[c] >= 10 else 0.5

    def feat(sub):
        return np.array([[np.mean([cwr(c) for c in row["blue"]]) - np.mean([cwr(c) for c in row["red"]])]
                         for _, row in sub.iterrows()])

    clf1 = LogisticRegression(max_iter=2000).fit(feat(train), ytr)
    p1 = clf1.predict_proba(feat(test))[:, 1]

    def line(name, p):
        print(f"  {name:<26} AUC {roc_auc_score(yte, p):.3f} | "
              f"acc {accuracy_score(yte, (p >= 0.5).astype(int)) * 100:4.1f}% | "
              f"Brier {brier_score_loss(yte, p):.3f}")

    base = max(yte.mean(), 1 - yte.mean())
    print("=" * 68)
    print(f"  DRAFT -> VAINQUEUR  | train={len(train)} (toutes ligues) test={len(test)} LCK")
    print("=" * 68)
    print(f"  baseline cote bleu         acc {base * 100:4.1f}% (toujours le cote majoritaire)")
    line("v1 moyenne winrate", p1)
    line("v2 ratings champ appris", p2)
    print("=" * 68)

    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    print("  TOP champions (draft rating le + fort) :")
    for i in order[::-1][:8]:
        print(f"    {allc[i]:<16} {coefs[i]:+.2f}")
    print("  FLOP champions (rating le + faible) :")
    for i in order[:8]:
        print(f"    {allc[i]:<16} {coefs[i]:+.2f}")

    # sanity KT vs DK game 1
    x = np.zeros((1, len(allc)))
    miss = []
    for c in KT:
        x[0, idx[c]] += 1 if c in idx else miss.append(c)
    for c in DK:
        x[0, idx[c]] -= 1 if c in idx else miss.append(c)
    pkt = clf.predict_proba(x)[0, 1]
    print("=" * 68)
    print(f"  SANITY KT(bleu) vs DK(rouge) — DRAFT SEULE : KT {pkt * 100:.1f}%")
    if miss:
        print(f"    (champions hors vocab, ignores : {miss})")
    print("  (rappel : DK a ecrase -> on regarde si la draft seule le voyait)")
    print("=" * 68)


if __name__ == "__main__":
    main()

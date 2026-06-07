"""Prédiction DRAFT-ONLY pour un matchup arbitraire (toutes ligues, LEC inclus).

Utilise les ratings de champions appris (logistic +1 bleu / -1 rouge) sur tout le CSV.
Sert à donner le "penchant draft" quand notre modèle d'équipe ne couvre pas la ligue.

Modifier TEAM_A / TEAM_B ci-dessous. Affiche P(A gagne) en A=bleu et A=rouge.

Usage : python -m src.models.draft_predict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ingest.load_oracle import ROOT, load_config

# --- matchup à prédire --- (A = côté BLEU)
A_NAME = "G2"
B_NAME = "KC"
TEAM_A = ["K'Sante", "Nocturne", "Aurora", "Xayah", "Neeko"]         # G2 bleu (5)
TEAM_B = ["Zaahen", "Vi", "Viktor", "Zeri", "Rakan"]                  # KC rouge (Canna -> Zaahen top)


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    pos = df["position"].str.lower()
    teams, players = df[pos == "team"], df[pos != "team"].copy()

    tb = teams[teams["side"].str.lower() == "blue"][["gameid", "result"]].rename(columns={"result": "y"}).drop_duplicates("gameid")
    players["side_l"] = players["side"].str.lower()
    champ = players.groupby(["gameid", "side_l"])["champion"].apply(list).unstack("side_l")
    m = tb.merge(champ, left_on="gameid", right_index=True).dropna(subset=["blue", "red"])
    m = m[m["blue"].apply(lambda x: isinstance(x, list) and len(x) >= 5)
          & m["red"].apply(lambda x: isinstance(x, list) and len(x) >= 5)]

    allc = sorted({c for lst in m["blue"] for c in lst} | {c for lst in m["red"] for c in lst})
    idx = {c: i for i, c in enumerate(allc)}

    X = np.zeros((len(m), len(allc)))
    for i, (b, r) in enumerate(zip(m["blue"], m["red"])):
        for c in b:
            if c in idx:
                X[i, idx[c]] += 1
        for c in r:
            if c in idx:
                X[i, idx[c]] -= 1
    clf = LogisticRegression(C=0.3, max_iter=3000).fit(X, m["y"].values)

    missing = [c for c in TEAM_A + TEAM_B if c not in idx]

    def vec(blue_team, red_team):
        x = np.zeros((1, len(allc)))
        for c in blue_team:
            if c in idx:
                x[0, idx[c]] += 1
        for c in red_team:
            if c in idx:
                x[0, idx[c]] -= 1
        return x

    p_a_blue = clf.predict_proba(vec(TEAM_A, TEAM_B))[0, 1]
    p_a_red = 1 - clf.predict_proba(vec(TEAM_B, TEAM_A))[0, 1]

    print("=" * 60)
    print(f"  DRAFT-ONLY : {A_NAME} vs {B_NAME}  (ratings sur {len(m)} games, toutes ligues)")
    print("=" * 60)
    print(f"  {A_NAME} = {TEAM_A}")
    print(f"  {B_NAME} = {TEAM_B}")
    if missing:
        print(f"  /!\\ champions hors data (ignores) : {missing}")
    print("-" * 60)
    print(f"  P({A_NAME} gagne)  si {A_NAME} BLEU : {p_a_blue * 100:.1f}%")
    print(f"  P({A_NAME} gagne)  si {A_NAME} ROUGE: {p_a_red * 100:.1f}%")
    print(f"  -> penchant draft (side-neutre) ~ {((p_a_blue + p_a_red) / 2) * 100:.1f}% {A_NAME}")
    print("=" * 60)
    print("  Rappel : LEC draft-AUC ~0.65 (n47, bruite) ; draft-only = signal PARTIEL,")
    print("  sans la force d'equipe (KC en feu, etc.). A comparer a la cote du book.")
    print("=" * 60)


if __name__ == "__main__":
    main()

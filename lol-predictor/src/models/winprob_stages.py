"""Win-prob par ETAPES : combien la DRAFT vs l'EXECUTION (gold/kills @10,@15) expliquent la victoire.

Repond a deux questions :
  1) "Faut-il combiner draft + etat de jeu @10/15 ?"  -> gain d'AUC a chaque etape.
  2) "La draft ajoute-t-elle qqch APRES avoir vu le gold@15 ?"  -> hypothese scaling (Viktor/late).

Methode (anti-fuite) :
  - split temporel 80/20 (train = passe, test = futur).
  - modele de DRAFT (ratings champions logistic) appris sur le TRAIN -> proba draft par equipe.
  - puis logistic emboitees : draft-only / +@10 / +@15, et @15-only (sans draft).
  - on lit les AUC/Brier + les coefs standardises (poids relatif draft vs gold@15).

Usage : python -m src.models.winprob_stages
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.ingest.load_oracle import ROOT, load_config

STATE10 = ["golddiffat10", "xpdiffat10", "killdiffat10"]
STATE15 = ["golddiffat15", "xpdiffat15", "killdiffat15"]


def champ_ratings(g: pd.DataFrame):
    """Logistic champions (+1 bleu / -1 rouge) -> P(bleu gagne). Renvoie clf + index champ."""
    allc = sorted({c for lst in g["blue"] for c in lst} | {c for lst in g["red"] for c in lst})
    idx = {c: i for i, c in enumerate(allc)}
    X = np.zeros((len(g), len(allc)))
    for i, (b, r) in enumerate(zip(g["blue"], g["red"])):
        for c in b:
            X[i, idx[c]] += 1
        for c in r:
            X[i, idx[c]] -= 1
    clf = LogisticRegression(C=0.3, max_iter=3000).fit(X, g["yblue"].values)
    return clf, idx


def p_blue(g: pd.DataFrame, clf, idx) -> np.ndarray:
    X = np.zeros((len(g), len(idx)))
    for i, (b, r) in enumerate(zip(g["blue"], g["red"])):
        for c in b:
            if c in idx:
                X[i, idx[c]] += 1
        for c in r:
            if c in idx:
                X[i, idx[c]] -= 1
    return clf.predict_proba(X)[:, 1]


def evaluate(name, feats, tr, te):
    sc = StandardScaler().fit(tr[feats])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[feats]), tr["y"].values)
    p = clf.predict_proba(sc.transform(te[feats]))[:, 1]
    auc = roc_auc_score(te["y"].values, p)
    brier = brier_score_loss(te["y"].values, p)
    print(f"  {name:34s} AUC={auc:.3f}  Brier={brier:.3f}")
    return dict(zip(feats, clf.coef_[0]))


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["datacompleteness"].astype(str).str.lower() == "complete"]
    pos = df["position"].str.lower()
    teams, players = df[pos == "team"].copy(), df[pos != "team"].copy()

    # --- listes de champions par game/side ---
    players["side_l"] = players["side"].str.lower()
    champ = players.groupby(["gameid", "side_l"])["champion"].apply(list).unstack("side_l")
    champ = champ.dropna(subset=["blue", "red"])
    champ = champ[champ["blue"].apply(lambda x: isinstance(x, list) and len(x) >= 5)
                  & champ["red"].apply(lambda x: isinstance(x, list) and len(x) >= 5)]

    tb = (teams[teams["side"].str.lower() == "blue"][["gameid", "result", "date"]]
          .rename(columns={"result": "yblue"}).drop_duplicates("gameid"))
    games = tb.merge(champ, left_on="gameid", right_index=True).sort_values("date").reset_index(drop=True)

    # --- split temporel + modele draft sur le train ---
    cut = games["date"].quantile(0.8)
    gtr = games[games["date"] <= cut]
    clf_d, idx = champ_ratings(gtr)
    games["p_blue"] = p_blue(games, clf_d, idx)

    # --- dataset par EQUIPE (etat @10/@15) ---
    t = teams.copy()
    for c in ["golddiffat10", "xpdiffat10", "killsat10", "deathsat10",
              "golddiffat15", "xpdiffat15", "killsat15", "deathsat15", "result"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["killdiffat10"] = t["killsat10"] - t["deathsat10"]
    t["killdiffat15"] = t["killsat15"] - t["deathsat15"]
    t["side_blue"] = (t["side"].str.lower() == "blue").astype(int)
    t = t.merge(games[["gameid", "p_blue"]], on="gameid", how="inner")
    t["draft_prob"] = np.where(t["side_blue"] == 1, t["p_blue"], 1 - t["p_blue"])
    t["y"] = t["result"].astype(int)

    feats_all = ["draft_prob", "side_blue"] + STATE10 + STATE15
    t = t.dropna(subset=feats_all + ["y"]).reset_index(drop=True)

    tr = t[t["date"] <= cut]
    te = t[t["date"] > cut]
    print("=" * 64)
    print(f"  WIN-PROB PAR ETAPES  (train={len(tr)} / test={len(te)} lignes-equipe)")
    print(f"  draft model appris sur {len(gtr)} games ; toutes ligues")
    print("=" * 64)

    print("\n  [A] Que vaut chaque bloc SEUL ?")
    evaluate("draft-only (champions)", ["draft_prob", "side_blue"], tr, te)
    evaluate("gold/kills @10 seul", STATE10 + ["side_blue"], tr, te)
    evaluate("gold/kills @15 seul", STATE15 + ["side_blue"], tr, te)

    print("\n  [B] On EMPILE (ta these : draft + etat @10/@15) :")
    evaluate("draft", ["draft_prob", "side_blue"], tr, te)
    evaluate("draft + @10", ["draft_prob", "side_blue"] + STATE10, tr, te)
    coefs = evaluate("draft + @10 + @15", feats_all, tr, te)

    print("\n  [C] La DRAFT ajoute-t-elle qqch APRES le gold@15 ? (effet scaling)")
    evaluate("@15 seul (sans draft)", STATE15 + ["side_blue"], tr, te)
    evaluate("@15 + draft", STATE15 + ["side_blue", "draft_prob"], tr, te)

    print("\n  Poids standardises (modele complet, |coef| = importance) :")
    for k, v in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:16s} {v:+.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()

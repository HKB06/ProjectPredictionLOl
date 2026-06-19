"""Picks HAUTE CONFIANCE — la règle qui donne ≥80 % de vrais vainqueurs.

Constat (backtest walk-forward strict, sans fuite, via `eval_models.production_records`) :

- Prédire **TOUS** les matchs plafonne à ~65 % : impossible d'avoir 80 % sur les
  matchs serrés (un coinflip reste un coinflip).
- Le levier n'est donc **pas le modèle** mais la **SÉLECTIVITÉ**. Deux filtres :
    1. **Ligues fiables** seulement : accuracy historique ≥ `RELIABLE_ACC`.
       On exclut les ligues chaotiques (EM ~55 %, LPL/LCS/CBLOL ~58 %) où le modèle
       est sur-confiant — c'est exactement là que Galions 3-0 Solary explose un "70 %".
    2. **Confiance ≥ `CONF_GAME`** par game (proba du favori, calibrée).

  Résultat mesuré : **~83 % de vainqueurs trouvés** sur ~23 % des matchs des ligues
  fiables (vs 65 % en pariant tout). Le prix : on joue moins souvent.

Pour une SÉRIE (BO3/BO5), l'avantage par game s'amplifie : un favori game ≥70 %
gagne la série encore plus souvent — *à condition* d'être dans une ligue fiable.

Usage :
    python -m src.models.high_confidence
    python -m src.models.high_confidence --recent-days 60
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config
from src.models.eval_models import BURN_IN, production_records
from src.update.elo import RELIABLE_ACC, compute_elo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONF_GAME = 0.70          # seuil de confiance PAR GAME pour un pick "haute confiance"
OUT_PATH = ROOT.parent / "HIGH_CONFIDENCE.md"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def df_to_md(df: pd.DataFrame) -> str:
    """Table Markdown sans dépendance externe (évite `tabulate`)."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def reliable_set(cfg: dict | None = None) -> tuple[set[str], dict]:
    """Ensemble des ligues fiables (accuracy hist. ≥ RELIABLE_ACC) + dict reliability."""
    st = compute_elo(cfg or load_config())
    rel = {lg for lg, acc in st["reliability"].items() if acc >= RELIABLE_ACC}
    return rel, st["reliability"]


def threshold_table(rec: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque seuil de confiance : nb de matchs gardés, couverture, accuracy réelle."""
    rows = []
    for thr in THRESHOLDS:
        sub = rec[rec["proba_fav"] >= thr]
        if len(sub):
            rows.append({"seuil": f">={thr*100:.0f}%", "matchs": len(sub),
                         "couv%": round(len(sub) / len(rec) * 100, 1),
                         "acc_reel%": round(sub["correct"].mean() * 100, 1)})
    return pd.DataFrame(rows)


def first_threshold_at(rec: pd.DataFrame, target: float = 0.80) -> dict | None:
    """Plus petit seuil de confiance atteignant `target` d'accuracy réelle (max de matchs)."""
    for thr in THRESHOLDS:
        sub = rec[rec["proba_fav"] >= thr]
        if len(sub) >= 30 and sub["correct"].mean() >= target:
            return {"seuil": thr, "matchs": len(sub), "acc": sub["correct"].mean()}
    return None


def per_league(rec: pd.DataFrame, conf: float = CONF_GAME, min_n: int = 10) -> pd.DataFrame:
    rows = []
    for lg, sub in rec.groupby("league"):
        hi = sub[sub["proba_fav"] >= conf]
        if len(hi) >= min_n:
            rows.append({"ligue": lg, "matchs_HC": len(hi),
                         "acc_HC%": round(hi["correct"].mean() * 100, 1)})
    return pd.DataFrame(rows).sort_values("acc_HC%", ascending=False)


def load_records(cfg: dict | None = None, recent_days: int | None = None) -> pd.DataFrame:
    rec = production_records(cfg or load_config())
    rec = rec[rec["nmin"] >= BURN_IN].copy()
    if recent_days:
        cutoff = rec["date"].max() - pd.Timedelta(days=recent_days - 1)
        rec = rec[rec["date"] >= cutoff]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Règle picks haute confiance (>=80% reel)")
    ap.add_argument("--recent-days", type=int, default=None, help="restreindre aux N derniers jours")
    args = ap.parse_args()

    cfg = load_config()
    rec = load_records(cfg, args.recent_days)
    rel, reliability = reliable_set(cfg)
    recR = rec[rec["league"].isin(rel)]

    span = f"{rec['date'].min():%Y-%m-%d} -> {rec['date'].max():%Y-%m-%d}"
    chaotic = sorted(lg for lg, a in reliability.items() if a < RELIABLE_ACC)

    L: list[str] = []
    L.append("# Picks HAUTE CONFIANCE — viser ≥80 % de vrais vainqueurs")
    L.append("")
    L.append(f"> Backtest walk-forward **sans fuite** · {len(rec)} games notées · {span}")
    L.append("")
    L.append("## La vérité en 1 phrase")
    L.append("Prédire **tout** = ~65 % (un coinflip reste un coinflip). Pour **≥80 % réel** il faut "
             "être **sélectif** : (1) **ligues fiables** uniquement, (2) **confiance élevée**.")
    L.append("")
    L.append("## Filtre 1 — TOUTES ligues (seuil de confiance seul)")
    L.append(df_to_md(threshold_table(rec)))
    L.append("")
    L.append(f"## Filtre 2 — LIGUES FIABLES seulement (on exclut {chaotic})")
    L.append(df_to_md(threshold_table(recR)))
    L.append("")
    hit = first_threshold_at(recR, 0.80)
    if hit:
        L.append(f"➡️ **Recette ≥80 %** : ligues fiables + confiance **≥{hit['seuil']*100:.0f}%/game** "
                 f"→ **{hit['acc']*100:.1f}%** réel sur **{hit['matchs']} matchs** "
                 f"({hit['matchs']/len(recR)*100:.0f}% des matchs fiables).")
    L.append("")
    L.append(f"## Par ligue fiable (confiance ≥{CONF_GAME*100:.0f}%/game)")
    L.append(df_to_md(per_league(recR)))
    L.append("")
    L.append("**Règle d'or** : un pick 🎯 = ligue fiable **+** favori ≥70 %/game **+** data ≥15 g "
             "**+** pas de cross-ligue. En BO3/BO5, la série amplifie encore l'avantage. "
             "⚠️ Haute *accuracy* ≠ profit auto : à cote courte le gain est faible — la value vient "
             "des ligues **mineures fiables** (LJL, LAS, PRM, TCL...) où le book est plus mou.")

    text = "\n".join(L) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")

    # Console (ascii-safe)
    repl = {"➡️": "->", "🎯": "*", "≥": ">=", "≠": "!=", "⚠️": "(!)", "→": "->"}
    for ln in L:
        for k, v in repl.items():
            ln = ln.replace(k, v)
        print(ln.encode("ascii", "replace").decode("ascii"))
    print(f"\n-> ecrit dans {OUT_PATH}")


if __name__ == "__main__":
    main()

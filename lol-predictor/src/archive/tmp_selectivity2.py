"""Selectivite : quel winrate atteint-on en filtrant par confiance + fiabilite + pariabilite ?
Walk-forward strict (production_records = K32+MOV, zero fuite)."""
import pandas as pd

from src.ingest.load_oracle import load_config
from src.models.eval_models import production_records
from src.update.elo import RELIABLE_ACC, compute_elo

cfg = load_config()
rec = production_records(cfg)
s = compute_elo(cfg)
rel, grel = s["reliability"], s["global_rel"]

rec = rec[rec["nmin"] >= 15].copy()          # >=15 games pour les 2 equipes
rec["rel"] = rec["league"].map(lambda l: rel.get(l, grel))
weeks = max(1, (rec["date"].max() - rec["date"].min()).days) / 7

# Ligues realistement pariables (majeures + mid-tier EU/regionales couvertes par les books)
BETTABLE = {"LCK", "LPL", "LEC", "LCS", "LCP", "VCS", "LJL", "LFL",
            "PRM", "TCL", "LAS", "EM", "NLC", "EBL", "HLL"}


def table(df, label):
    print(f"\n=== {label}  (n={len(df)}) ===")
    print(f"  {'seuil':<9}{'games':>7}{'winrate':>9}{'picks/sem':>11}")
    for thr in (0.60, 0.65, 0.70, 0.75, 0.80):
        sub = df[df["proba_fav"] >= thr]
        if len(sub) == 0:
            continue
        print(f"  p>={thr:.2f}{len(sub):>8}{sub['correct'].mean()*100:>8.1f}%{len(sub)/weeks:>10.1f}")


reliable = rec[rec["rel"] >= RELIABLE_ACC]
table(reliable, "Ligues FIABLES (toutes, y compris petites)")
table(reliable[reliable["league"].isin(BETTABLE)], "Ligues FIABLES *ET* pariables")
table(rec[~(rec["rel"] >= RELIABLE_ACC)], "Ligues CHAOS (EM, LPL, LCS...) -- pour comparaison")

print("\n=== Par ligue FIABLE : winrate du favori selon le seuil ===")
rows = []
for lg, sub in reliable.groupby("league"):
    s65 = sub[sub["proba_fav"] >= 0.65]
    s70 = sub[sub["proba_fav"] >= 0.70]
    rows.append({
        "ligue": lg, "n": len(sub), "acc%": round(sub["correct"].mean() * 100, 1),
        "n>=65": len(s65), "wr>=65": round(s65["correct"].mean() * 100, 1) if len(s65) else None,
        "n>=70": len(s70), "wr>=70": round(s70["correct"].mean() * 100, 1) if len(s70) else None,
        "pariable": "oui" if lg in BETTABLE else "-",
    })
print(pd.DataFrame(rows).sort_values("n", ascending=False).to_string(index=False))

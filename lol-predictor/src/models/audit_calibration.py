"""Audit de calibration & discrimination du modèle retenu (Elo K32 + MOV).

Rejoue tout l'historique en WALK-FORWARD strict (zéro fuite, via
`production_records`) puis régénère de façon **reproductible** l'ensemble des
métriques d'évaluation + les figures pour le mémoire :

Scores globaux
- Accuracy (bon vainqueur), base-rate du favori.
- Brier + Brier Skill Score (vs baseline « prédire toujours la base-rate »).
- LogLoss + LogLoss de la baseline (entropie de la base-rate).
- AUC dans les DEUX cadrages :
    * AUC(bleu | P(bleu))      = discrimination brute « qui gagne ? »
    * AUC(favori | proba_fav)  = signal AU-DELÀ du fait de connaître le favori.
- ECE (Expected Calibration Error).
- Décomposition de Murphy : Brier = Reliability − Resolution + Uncertainty.

Segments
- Par ligue (n ≥ min), par région (Asie vs Ouest/Autres).
- Par tranche de confiance, avec intervalles de Wilson 95 %.
- Biais de side (taux de victoire du bleu).

Figures (PNG)
- Diagramme de fiabilité (calibration prédit vs réel, barres de Wilson).
- Courbe ROC (cadrage « qui gagne ? »).

Usage :
    python -m src.models.audit_calibration
    python -m src.models.audit_calibration --league LCK
    python -m src.models.audit_calibration --outdir reports/audit
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend fichier (pas d'affichage) pour générer les PNG
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.ingest.load_oracle import ROOT, load_config  # noqa: E402
from src.models.eval_models import BURN_IN, production_records  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = 1e-12
# Ligues asiatiques (angle mort pointé par l'audit : moins de contexte roster/data).
ASIA = {"LCK", "LCKC", "LPL", "LPLOL", "LCP", "VCS", "PCS", "LJL",
        "KeSPA Cup", "Asia Master", "CCWS"}


def region_of(league: str) -> str:
    return "Asie" if league in ASIA else "Ouest/Autres"


def _md_table(df: pd.DataFrame) -> str:
    """Rend un DataFrame en table Markdown (sans dépendance 'tabulate')."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson (proportion) — robuste en petit n."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binary_entropy(q: float) -> float:
    q = min(max(q, EPS), 1 - EPS)
    return -(q * math.log(q) + (1 - q) * math.log(1 - q))


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 1000,
                     seed: int = 42) -> tuple[float, float]:
    """IC95 de l'AUC par bootstrap (rééchantillonnage avec remise). Blinde le
    chiffre en petit échantillon (ex. focus une ligue) contre l'accusation de
    cherry-pick / résultat non significatif."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        ys = y[s]
        if len(np.unique(ys)) < 2:
            continue
        aucs.append(roc_auc_score(ys, score[s]))
    if not aucs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return (float(lo), float(hi))


def bootstrap_ece_ci(proba_fav: np.ndarray, correct: np.ndarray, n_boot: int = 1000,
                     seed: int = 42, width: float = 0.05) -> tuple[float, float]:
    """IC95 de l'ECE par bootstrap : quantifie l'incertitude d'échantillonnage sur
    l'erreur de calibration (à citer à côté de la valeur ponctuelle dans le texte)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(proba_fav))
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals.append(ece(proba_fav[s], correct[s], width=width))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def murphy(p: np.ndarray, y: np.ndarray, n_bins: int = 10,
           lo: float = 0.5, hi: float = 1.0) -> dict:
    """Décomposition de Murphy : Brier = Reliability − Resolution + Uncertainty.

    Calculée sur (p = proba du favori, y = le favori a gagné) → l'uncertainty
    vaut base·(1−base) et la resolution mesure le signal AU-DELÀ du favori.
    """
    y = y.astype(float)
    obar = float(y.mean())
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    n = len(p)
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        pk, ok = float(p[m].mean()), float(y[m].mean())
        rel += nb * (pk - ok) ** 2
        res += nb * (ok - obar) ** 2
    rel, res = rel / n, res / n
    unc = obar * (1 - obar)
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier_check": rel - res + unc}


def ece(proba_fav: np.ndarray, correct: np.ndarray, width: float = 0.05) -> float:
    """Expected Calibration Error (pondéré par l'effectif de chaque tranche)."""
    n = len(proba_fav)
    if n == 0:
        return float("nan")
    out = 0.0
    lo = 0.5
    while lo < 1.0 - 1e-9:
        hi = lo + width
        m = (proba_fav >= lo) & (proba_fav < hi + (1e-9 if hi >= 1.0 else 0))
        nb = int(m.sum())
        if nb:
            out += nb / n * abs(float(correct[m].mean()) - float(proba_fav[m].mean()))
        lo = hi
    return out


def global_scores(rec: pd.DataFrame) -> dict:
    p = rec["p"].to_numpy()
    y = rec["yb"].to_numpy().astype(float)
    pf = rec["proba_fav"].to_numpy()
    corr = rec["correct"].to_numpy().astype(float)
    base = float(corr.mean())
    br = brier(p, y)
    ll = logloss(p, y)
    d = murphy(pf, corr, n_bins=10, lo=0.5, hi=1.0)
    return {
        "n": len(rec),
        "acc": float(corr.mean()),
        "base_fav": base,
        "brier": br,
        "brier_baseline": base * (1 - base),
        "bss": 1 - br / (base * (1 - base)),
        "logloss": ll,
        "logloss_baseline": binary_entropy(base),
        "auc_blue": float(roc_auc_score(y, p)),
        "auc_fav": float(roc_auc_score(corr, pf)),
        "auc_blue_ci": bootstrap_auc_ci(y, p),
        "auc_fav_ci": bootstrap_auc_ci(corr, pf),
        "ece": ece(pf, corr, width=0.05),
        "ece_ci": bootstrap_ece_ci(pf, corr),
        "blue_winrate": float(y.mean()),
        **d,
    }


def confidence_table(rec: pd.DataFrame) -> pd.DataFrame:
    pf = rec["proba_fav"].to_numpy()
    corr = rec["correct"].to_numpy().astype(int)
    rows = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        hi = lo + 0.1
        m = (pf >= lo) & (pf < hi + (1e-9 if hi >= 1.0 else 0))
        nb = int(m.sum())
        if nb == 0:
            continue
        k = int(corr[m].sum())
        wlo, whi = wilson(k, nb)
        rows.append({
            "tranche": f"{lo*100:.0f}-{hi*100:.0f}%",
            "n": nb,
            "proba_annoncee": round(float(pf[m].mean()) * 100, 1),
            "gagne_reel": round(k / nb * 100, 1),
            "IC95_Wilson": f"[{wlo*100:.0f}%, {whi*100:.0f}%]",
            "ecart": round((float(pf[m].mean()) - k / nb) * 100, 1),
        })
    return pd.DataFrame(rows)


def segment_table(rec: pd.DataFrame, by: str, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for key, sub in rec.groupby(by):
        if len(sub) < min_n:
            continue
        p = sub["p"].to_numpy()
        y = sub["yb"].to_numpy().astype(float)
        pf = sub["proba_fav"].to_numpy()
        corr = sub["correct"].to_numpy().astype(float)
        try:
            auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")
        rows.append({
            by: key,
            "n": len(sub),
            "acc": round(float(corr.mean()) * 100, 1),
            "proba_moy": round(float(pf.mean()) * 100, 1),
            "brier": round(brier(p, y), 3),
            "ece": round(ece(pf, corr), 3),
            "auc_bleu": round(auc, 3),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Figures                                                                     #
# --------------------------------------------------------------------------- #
def fig_reliability(rec: pd.DataFrame, path: Path, title_suffix: str = "",
                    min_pts: int = 15) -> None:
    pf = rec["proba_fav"].to_numpy()
    corr = rec["correct"].to_numpy().astype(float)
    edges = np.arange(0.5, 1.0001, 0.05)
    xs, ys, los, his, ns = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pf >= lo) & (pf < hi + (1e-9 if hi >= 1.0 else 0))
        nb = int(m.sum())
        if nb < min_pts:          # tranche trop peu peuplée -> hors nuage (reste dans l'histo)
            continue
        k = int(corr[m].sum())
        wlo, whi = wilson(k, nb)
        xs.append(float(pf[m].mean()))
        ys.append(k / nb)
        los.append(k / nb - wlo)
        his.append(whi - k / nb)
        ns.append(nb)

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(7, 7.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="gray", lw=1, label="Calibration parfaite")
    ax.errorbar(xs, ys, yerr=[los, his], fmt="o", color="#1f77b4", capsize=3,
                ms=6, lw=1, label="Observé (IC95 Wilson)")
    for x, y, nb in zip(xs, ys, ns):
        ax.annotate(f"n={nb}", (x, y), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color="#444")
    ax.set_ylabel("Taux de victoire réel du favori")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"Diagramme de fiabilité — Elo K32 + MOV{title_suffix}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    axh.hist(pf, bins=edges, color="#9ecae1", edgecolor="#3182bd")
    axh.set_ylabel("Games")
    axh.set_xlabel("Probabilité annoncée pour le favori")
    axh.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_roc(rec: pd.DataFrame, path: Path, title_suffix: str = "") -> None:
    y = rec["yb"].to_numpy().astype(int)
    p = rec["p"].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot(fpr, tpr, color="#d62728", lw=2, label=f"Elo K32+MOV (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#d62728")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Hasard (AUC = 0,5)")
    ax.set_xlabel("Taux de faux positifs (1 − spécificité)")
    ax.set_ylabel("Taux de vrais positifs (sensibilité)")
    ax.set_title(f"Courbe ROC — « qui gagne ? »{title_suffix}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration                                                               #
# --------------------------------------------------------------------------- #
def run(league: str | None = None, outdir: str = "reports/audit",
        min_n: int = 30) -> None:
    rec = production_records(load_config())
    rec = rec[rec["nmin"] >= BURN_IN].copy()          # hors cold-start
    rec["region"] = rec["league"].map(region_of)
    scope = "tout le dataset"
    if league:
        rec = rec[rec["league"] == league].copy()
        scope = f"ligue {league}"
    if rec.empty:
        print(f"[audit] aucune game pour {scope}.")
        return

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f" · {league}" if league else ""

    s = global_scores(rec)
    conf = confidence_table(rec)
    by_lg = segment_table(rec, "league", min_n=min_n)
    by_reg = segment_table(rec, "region", min_n=min_n)

    # --- Rapport texte (console + Markdown) ---
    lines = []
    add = lines.append
    add(f"# Audit calibration & discrimination — {scope}")
    add("")
    add(f"Modèle : **Elo K32 + MOV** · walk-forward strict · burn-in ≥ {BURN_IN} games/équipe.")
    add(f"Games évaluées : **{s['n']}** · période Oracle's Elixir.")
    add("")
    add("## Scores globaux")
    add("")
    add("| Métrique | Valeur | Lecture |")
    add("|---|---|---|")
    add(f"| Accuracy (bon vainqueur) | {s['acc']*100:.1f}% | base-rate favori = {s['base_fav']*100:.1f}% |")
    add(f"| Brier | {s['brier']:.4f} | 0 = parfait, {s['brier_baseline']:.3f} = baseline base-rate |")
    add(f"| Brier Skill Score | {s['bss']*100:+.1f}% | gain vs baseline « toujours la base-rate » |")
    add(f"| LogLoss | {s['logloss']:.4f} | baseline base-rate = {s['logloss_baseline']:.4f} ; hasard = 0.693 |")
    add(f"| **AUC « qui gagne ? »** | **{s['auc_blue']:.3f}** "
        f"[{s['auc_blue_ci'][0]:.3f}, {s['auc_blue_ci'][1]:.3f}] | "
        "P(bleu) vs bleu gagne · IC95 bootstrap |")
    add(f"| AUC « au-delà du favori » | {s['auc_fav']:.3f} "
        f"[{s['auc_fav_ci'][0]:.3f}, {s['auc_fav_ci'][1]:.3f}] | "
        "proba_fav vs favori gagne — signal fin |")
    add(f"| ECE | {s['ece']:.4f} [{s['ece_ci'][0]:.4f}, {s['ece_ci'][1]:.4f}] | "
        "erreur de calibration · IC95 bootstrap (plus bas = mieux) |")
    add("")
    add("## Décomposition de Murphy (cadrage favori)")
    add("")
    add("Brier = Reliability − Resolution + Uncertainty")
    add("")
    add("| Composante | Valeur | Lecture |")
    add("|---|---|---|")
    add(f"| Reliability | {s['reliability']:.4f} | bas = probabilités honnêtes |")
    add(f"| Resolution | {s['resolution']:.4f} | haut = pouvoir de séparation |")
    add(f"| Uncertainty | {s['uncertainty']:.4f} | difficulté intrinsèque (fixe) |")
    add(f"| (contrôle Brier) | {s['brier_check']:.4f} | doit ≈ Brier {s['brier']:.4f} |")
    add("")
    add(f"Side bias : le bleu gagne **{s['blue_winrate']*100:.1f}%** des games.")
    add("")
    add("## Par tranche de confiance (IC95 Wilson)")
    add("")
    add(_md_table(conf))
    add("")
    add("## Par région")
    add("")
    add(_md_table(by_reg))
    add("")
    add(f"## Par ligue (n ≥ {min_n})")
    add("")
    add(_md_table(by_lg))
    add("")

    report = "\n".join(lines)
    print(report)

    stem = f"audit_{league}" if league else "audit_global"
    (out / f"{stem}.md").write_text(report + "\n", encoding="utf-8")
    conf.to_csv(out / f"{stem}_confiance.csv", index=False)
    by_lg.to_csv(out / f"{stem}_ligues.csv", index=False)

    fig_reliability(rec, out / f"{stem}_fiabilite.png", suffix)
    fig_roc(rec, out / f"{stem}_roc.png", suffix)
    print(f"\n[audit] Rapport + figures écrits dans : {out.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit calibration/discrimination (walk-forward).")
    ap.add_argument("--league", default=None, help="restreindre à une ligue (ex. LCK)")
    ap.add_argument("--outdir", default="reports/audit", help="dossier de sortie (figures + md)")
    ap.add_argument("--min-n", type=int, default=30, help="effectif min par segment")
    args = ap.parse_args()
    run(league=args.league, outdir=args.outdir, min_n=args.min_n)


if __name__ == "__main__":
    main()

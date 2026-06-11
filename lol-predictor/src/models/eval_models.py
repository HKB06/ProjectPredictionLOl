"""Banc d'essai scientifique : compare des modèles de rating en WALK-FORWARD strict.

On rejoue toutes les games dans l'ordre chronologique. Pour CHAQUE game, on prédit
AVANT (avec l'état du modèle d'avant la game), puis on met à jour. Zéro fuite.

Métriques (plus c'est bas, mieux c'est sauf Accuracy) :
- Accuracy : a-t-on désigné le bon vainqueur (favori côté-neutre) ?
- Brier    : (proba − résultat)²  → qualité probabiliste.
- LogLoss  : pénalise les certitudes fausses (clé pour la mise/EV).
- Calibration (ECE) : un "70%" gagne-t-il ~70% du temps ?

Les métriques sont calculées après un burn-in (≥5 games par équipe) pour ne pas
polluer avec le cold-start. On rapporte aussi une fenêtre récente.

Usage :
    python -m src.models.eval_models                 # comparatif complet
    python -m src.models.eval_models --recent-days 45
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.ingest.load_oracle import load_config
from src.update.elo import load_games  # source unique du chargement des games

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = 1500.0
BURN_IN = 5          # games min par équipe avant de compter dans l'éval
EPS = 1e-12


# --------------------------------------------------------------------------- #
#  Raters : interface commune  prob(blue, red) -> P(blue gagne) ; update(...)  #
# --------------------------------------------------------------------------- #
class EloRater:
    """Elo glissant. Options : MOV (marge via kills), side (avantage bleu),
    K par 'famille' de ligue, forme (bonus série en cours)."""

    def __init__(self, k=24.0, scale=400.0, mov=False, side=0.0,
                 league_k: dict | None = None, form=0.0, name="elo"):
        self.k0 = k
        self.scale = scale
        self.mov = mov
        self.side = side          # avantage bleu en points d'Elo (0 = side-neutre)
        self.league_k = league_k or {}
        self.form = form          # poids du momentum (winrate récent)
        self.name = name
        self.r: dict[str, float] = defaultdict(lambda: BASE)
        self.n: dict[str, int] = defaultdict(int)
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

    def _form_bonus(self, team: str) -> float:
        if not self.form or not self.recent[team]:
            return 0.0
        wr = sum(self.recent[team]) / len(self.recent[team])
        return self.form * (wr - 0.5) * 2.0  # [-form, +form] points d'Elo

    def prob(self, blue: str, red: str) -> float:
        ra = self.r[blue] + self.side + self._form_bonus(blue)
        rb = self.r[red] + self._form_bonus(red)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / self.scale))

    def update(self, blue: str, red: str, yb: int, kblue=None, kred=None, lg=None) -> None:
        ea = self.prob(blue, red)
        k = self.league_k.get(lg, self.k0)
        mult = 1.0
        if self.mov and kblue is not None and kred is not None and not (
                math.isnan(kblue) or math.isnan(kred)):
            margin = abs(float(kblue) - float(kred))
            mult = math.log1p(margin) / math.log1p(8.0)  # 8 kills d'écart ≈ multiplicateur 1
        self.r[blue] += k * mult * (yb - ea)
        self.r[red] += k * mult * ((1 - yb) - (1 - ea))
        self.n[blue] += 1
        self.n[red] += 1
        if self.form:
            self.recent[blue].append(yb)
            self.recent[red].append(1 - yb)


# --------------------------------------------------------------------------- #
#  Replay walk-forward + métriques                                            #
# --------------------------------------------------------------------------- #
def replay(rater: EloRater, g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for blue, red, yb, lg, d, kb, kr in zip(
            g["blue"], g["red"], g["yb"], g["league"], g["date"], g["kblue"], g["kred"]):
        p = rater.prob(blue, red)                       # P(bleu gagne), pré-game
        nmin = min(rater.n[blue], rater.n[red])
        rows.append({"date": d, "league": lg, "blue": blue, "red": red,
                     "p": p, "yb": yb, "nmin": nmin})
        rater.update(blue, red, yb, kb, kr, lg)
    return pd.DataFrame(rows)


def production_records(cfg: dict | None = None) -> pd.DataFrame:
    """Rejoue le modèle RETENU (K32 + MOV) et renvoie un détail par game prêt à afficher.

    Colonnes : date, league, blue, red, p (P bleu), yb, nmin, favori, proba_fav,
    vainqueur, correct. Walk-forward strict (la prédiction n'utilise jamais le résultat).
    """
    g = load_games(cfg or load_config())
    rec = replay(EloRater(k=32, mov=True), g)
    rec["favori"] = rec.apply(lambda r: r["blue"] if r["p"] >= 0.5 else r["red"], axis=1)
    rec["proba_fav"] = rec["p"].where(rec["p"] >= 0.5, 1 - rec["p"])
    rec["vainqueur"] = rec.apply(lambda r: r["blue"] if r["yb"] == 1 else r["red"], axis=1)
    rec["correct"] = rec["favori"] == rec["vainqueur"]
    return rec


def metrics(rec: pd.DataFrame) -> dict:
    if rec.empty:
        return {"n": 0}
    p = rec["p"].clip(EPS, 1 - EPS).to_numpy()
    y = rec["yb"].to_numpy()
    acc = float(np.mean((p >= 0.5) == (y == 1)))
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # ECE sur la proba du favori
    pf = np.maximum(p, 1 - p)                 # confiance affichée
    won = (p >= 0.5) == (y == 1)
    ece = 0.0
    for lo in np.arange(0.5, 1.0, 0.1):
        m = (pf >= lo) & (pf < lo + 0.1)
        if m.sum() > 0:
            ece += m.sum() / len(pf) * abs(won[m].mean() - pf[m].mean())
    return {"n": len(rec), "acc": acc, "brier": brier, "logloss": logloss, "ece": ece}


def eval_rater(rater: EloRater, g: pd.DataFrame, recent_days: int) -> dict:
    rec = replay(rater, g)
    rec = rec[rec["nmin"] >= BURN_IN]                  # hors cold-start
    cutoff = g["date"].max().normalize() - pd.Timedelta(days=recent_days - 1)
    out = {"all": metrics(rec), "recent": metrics(rec[rec["date"] >= cutoff])}
    out["by_league"] = {lg: metrics(sub) for lg, sub in rec.groupby("league")}
    out["records"] = rec
    return out


# --------------------------------------------------------------------------- #
#  Comparatif                                                                  #
# --------------------------------------------------------------------------- #
def variants() -> list[EloRater]:
    return [
        EloRater(k=24, name="baseline K24"),
        EloRater(k=16, name="K16"),
        EloRater(k=32, name="K32"),
        EloRater(k=40, name="K40"),
        EloRater(k=24, scale=300, name="scale300"),
        EloRater(k=24, scale=500, name="scale500"),
        EloRater(k=24, mov=True, name="K24 +MOV"),
        EloRater(k=32, mov=True, name="K32 +MOV"),
        EloRater(k=40, mov=True, name="K40 +MOV"),
        EloRater(k=24, form=40, name="K24 +forme"),
        EloRater(k=32, mov=True, form=40, name="K32 +MOV +forme"),
        EloRater(k=24, side=20, name="K24 +side (info)"),  # non utilisable en pré-game
    ]


def compare(recent_days: int = 45) -> pd.DataFrame:
    cfg = load_config()
    g = load_games(cfg)
    print(f"Games chargées : {len(g)}  ({g['date'].min():%Y-%m-%d} -> {g['date'].max():%Y-%m-%d})")
    print(f"Burn-in : ≥{BURN_IN} games/équipe · fenêtre récente : {recent_days} j\n")

    rows = []
    for v in variants():
        r = eval_rater(v, g, recent_days)
        a, rc = r["all"], r["recent"]
        rows.append({
            "modèle": v.name,
            "acc_all": a["acc"], "brier_all": a["brier"], "logloss_all": a["logloss"], "ece_all": a["ece"],
            "acc_rec": rc.get("acc", float("nan")), "brier_rec": rc.get("brier", float("nan")),
            "logloss_rec": rc.get("logloss", float("nan")), "n_rec": rc.get("n", 0),
        })
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== COMPARATIF (all = saison hors cold-start ; rec = fenêtre récente) ===")
    print(df.round(4).to_string(index=False))

    best = df.sort_values("logloss_all").iloc[0]
    print(f"\nMeilleur log-loss saison : {best['modèle']}  "
          f"(acc {best['acc_all']:.3f}, brier {best['brier_all']:.3f})")
    return df


def calib_table(rec: pd.DataFrame) -> pd.DataFrame:
    """Fiabilité : par tranche de confiance affichée, taux de victoire réel du favori."""
    p = rec["p"].to_numpy()
    pf = np.maximum(p, 1 - p)
    won = (p >= 0.5) == (rec["yb"].to_numpy() == 1)
    rows = []
    for lo in np.arange(0.5, 1.0, 0.1):
        m = (pf >= lo) & (pf < lo + 0.1 + (1e-9 if lo > 0.85 else 0))
        if m.sum() == 0:
            continue
        rows.append({"tranche": f"{lo*100:.0f}-{(lo+0.1)*100:.0f}%", "n": int(m.sum()),
                     "proba_dite": round(pf[m].mean() * 100, 1),
                     "gagné_réel": round(won[m].mean() * 100, 1),
                     "écart": round((pf[m].mean() - won[m].mean()) * 100, 1)})
    return pd.DataFrame(rows)


def detail(recent_days: int = 45) -> None:
    """Analyse fine du modèle retenu (K32 + MOV) : par ligue + calibration."""
    cfg = load_config()
    g = load_games(cfg)
    rater = EloRater(k=32, mov=True, name="K32 +MOV")
    rec = replay(rater, g)
    rec = rec[rec["nmin"] >= BURN_IN]

    print("=== MODÈLE RETENU : K32 + MOV ===\n")
    print("--- Par ligue (n≥30, triées par volume) ---")
    rows = []
    for lg, sub in rec.groupby("league"):
        if len(sub) < 30:
            continue
        mt = metrics(sub)
        rows.append({"ligue": lg, "n": mt["n"], "acc": round(mt["acc"], 3),
                     "brier": round(mt["brier"], 3), "ece": round(mt["ece"], 3)})
    dfl = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(dfl.to_string(index=False))

    print("\n--- Calibration GLOBALE ---")
    print(calib_table(rec).to_string(index=False))

    for lg in ("EM", "LCK", "LEC"):
        sub = rec[rec["league"] == lg]
        if len(sub) >= 30:
            print(f"\n--- Calibration {lg} (n={len(sub)}) ---")
            print(calib_table(sub).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent-days", type=int, default=45)
    ap.add_argument("--detail", action="store_true", help="analyse par ligue + calibration du modèle retenu")
    args = ap.parse_args()
    if args.detail:
        detail(args.recent_days)
    else:
        compare(args.recent_days)


if __name__ == "__main__":
    main()

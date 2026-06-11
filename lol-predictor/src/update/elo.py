"""Modèle de rating Elo toutes-ligues — version retenue après banc d'essai.

Choix validés par `src/models/eval_models.py` (walk-forward, 5700+ games) :
  - **K = 32** (réagit plus vite à la forme qu'un K=24, meilleur Brier/log-loss).
  - **Marge de victoire (MOV)** : un stomp (gros écart de kills) bouge plus l'Elo
    qu'une victoire serrée → +1 pt d'accuracy (64.0 → 65.0 %), surtout en récent (→65.7 %).
  - **Calibration par ligue** : certaines ligues sont quasi imprévisibles (EM ~0.56,
    LPL/LCS/CBLOL ~0.58) et le modèle y est SUR-confiant (un "70 %" en EM gagne ~55 %).
    On apprend un facteur de rétrécissement (shrink) par ligue qui aplatit la proba vers
    50 % là où le modèle se trompe, et garde la confiance là où il est fiable (LCK/LEC/LJL).

`compute_elo` renvoie aussi `reliability` (accuracy historique par ligue) et `shrink`,
utilisés par la watchlist pour BRIDER le flag ⭐ dans les ligues chaotiques.

Usage :
    python -m src.update.elo
"""
from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from math import comb

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config

K = 32.0
BASE = 1500.0
SCALE = 400.0
MOV_REF = 8.0       # écart de kills donnant un multiplicateur MOV ≈ 1
BURN_IN = 5         # games min/équipe avant de compter dans la calibration
RELIABLE_ACC = 0.62  # seuil d'accuracy ligue pour autoriser un flag ⭐
SHRINK_REG = 120.0   # régularisation du shrink vers 1.0 (anti-petit échantillon)

# Alias de noms de compétition -> code Oracle's Elixir (pour retrouver la calibration).
_LEAGUE_ALIASES = {
    "emeamasters": "EM", "emea": "EM", "em": "EM",
    "cblol": "CBLOL", "lck": "LCK", "lckchallengers": "LCKC", "lpl": "LPL",
    "lec": "LEC", "lcs": "LCS", "ljl": "LJL", "primeleague": "PRM", "prm": "PRM",
    "tcl": "TCL", "arabianleague": "AL", "northernleague": "NLC", "nlc": "NLC",
    "hellenic": "HLL", "hitpoint": "HM", "ultraliga": "PRM",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return "".join(c for c in s if c.isalnum())


def _mov_mult(kb, kr) -> float:
    """Multiplicateur de marge de victoire à partir de l'écart de kills."""
    try:
        if kb is None or kr is None or math.isnan(float(kb)) or math.isnan(float(kr)):
            return 1.0
        return math.log1p(abs(float(kb) - float(kr))) / math.log1p(MOV_REF)
    except (TypeError, ValueError):
        return 1.0


def load_games(cfg: dict | None = None) -> pd.DataFrame:
    """Une ligne par game : blue, red, yb (1=bleu gagne), kblue, kred, date, league."""
    cfg = cfg or load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    t = df[df["position"].str.lower() == "team"]
    cb = ["gameid", "teamname", "result", "date", "league", "kills"]
    tb = t[t["side"].str.lower() == "blue"][cb]
    tr = t[t["side"].str.lower() == "red"][["gameid", "teamname", "kills"]]
    g = (tb.merge(tr, on="gameid", suffixes=("_b", "_r"))
           .dropna(subset=["teamname_b", "teamname_r", "result"])
           .sort_values("date").reset_index(drop=True))
    g = g.rename(columns={"teamname_b": "blue", "teamname_r": "red", "result": "yb",
                          "kills_b": "kblue", "kills_r": "kred"})
    g["yb"] = g["yb"].astype(int)
    return g


def compute_elo(cfg: dict | None = None, k: float = K, base: float = BASE) -> dict:
    """Rejoue toutes les games (K32 + MOV) et renvoie l'état + la calibration par ligue.

    Retour :
      elo, n, league : comme avant.
      reliability[lg] : accuracy historique du modèle dans la ligue (hors cold-start).
      shrink[lg]      : facteur d'aplatissement de la proba (calibration, 0..1.2).
      global_rel      : accuracy globale (fallback).
    """
    cfg = cfg or load_config()
    g = load_games(cfg)

    elo: dict[str, float] = defaultdict(lambda: base)
    n: dict[str, int] = defaultdict(int)
    league: dict[str, str] = {}
    pf_sum: dict[str, float] = defaultdict(float)   # somme des probas favori
    won_sum: dict[str, float] = defaultdict(float)  # nb de fois où le favori a gagné
    cnt: dict[str, int] = defaultdict(int)

    for blue, red, yb, lg, kb, kr in zip(g["blue"], g["red"], g["yb"], g["league"],
                                         g["kblue"], g["kred"]):
        ea = win_prob(elo[blue], elo[red])
        if min(n[blue], n[red]) >= BURN_IN:           # calibration hors cold-start
            pf = max(ea, 1 - ea)
            pf_sum[lg] += pf
            won_sum[lg] += float((ea >= 0.5) == (yb == 1))
            cnt[lg] += 1
        mult = _mov_mult(kb, kr)
        elo[blue] += k * mult * (yb - ea)
        elo[red] += k * mult * ((1 - yb) - (1 - ea))
        n[blue] += 1
        n[red] += 1
        league[blue] = lg
        league[red] = lg

    total = sum(cnt.values())
    global_rel = sum(won_sum.values()) / total if total else 0.5
    reliability, shrink = {}, {}
    for lg, c in cnt.items():
        obs = won_sum[lg] / c
        predf = pf_sum[lg] / c
        reliability[lg] = obs
        slope = (obs - 0.5) / (predf - 0.5) if predf > 0.5 else 1.0
        slope = (c * slope + SHRINK_REG * 1.0) / (c + SHRINK_REG)  # régularisé vers 1
        shrink[lg] = max(0.0, min(1.2, slope))

    return {"elo": dict(elo), "n": dict(n), "league": league,
            "reliability": reliability, "shrink": shrink, "global_rel": global_rel}


def win_prob(elo_a: float, elo_b: float, scale: float = SCALE) -> float:
    """P(A gagne une game) selon l'écart d'Elo (side-neutre, AVANT calibration)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / scale))


def calibrate(p_game: float, shrink: float) -> float:
    """Aplatit la proba vers 50 % selon le facteur de calibration de la ligue."""
    return 0.5 + (p_game - 0.5) * shrink


def resolve_league(name: str, reliability: dict) -> str | None:
    """Mappe un nom de compétition (overview lolesports) vers un code OE connu."""
    nm = _norm(name)
    if not nm:
        return None
    for code in reliability:
        if _norm(code) == nm:
            return code
    for key, code in _LEAGUE_ALIASES.items():
        if key in nm and code in reliability:
            return code
    return None


def series_prob(p_game: float, wins_needed: int) -> float:
    """P(gagner la série) au meilleur des (2*wins_needed-1), games i.i.d."""
    q = 1.0 - p_game
    return float(sum(
        comb(wins_needed - 1 + losses, losses) * p_game ** wins_needed * q ** losses
        for losses in range(wins_needed)
    ))


def main() -> None:
    state = compute_elo()
    top = sorted(state["elo"].items(), key=lambda kv: kv[1], reverse=True)[:20]
    print(f"Elo (K32 + MOV) sur {len(state['elo'])} équipes. Top 20 :")
    for name, rating in top:
        print(f"  {rating:6.0f}  {name}  ({state['n'][name]} g, {state['league'].get(name, '?')})")
    print("\nFiabilité par ligue (accuracy hist. · shrink calibration) :")
    rel = state["reliability"]
    for lg in sorted(rel, key=rel.get, reverse=True):
        tag = "fiable" if rel[lg] >= RELIABLE_ACC else "CHAOTIQUE"
        print(f"  {lg:6} acc={rel[lg]*100:4.1f}%  shrink={state['shrink'][lg]:.2f}  [{tag}]")


if __name__ == "__main__":
    main()

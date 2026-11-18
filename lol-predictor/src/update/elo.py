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

from src.ingest.load_oracle import ROOT, load_config, read_seasons, season_files  # noqa: F401

K = 32.0
BASE = 1500.0
SCALE = 400.0
MOV_REF = 8.0       # écart de kills donnant un multiplicateur MOV ≈ 1
BURN_IN = 5         # games min/équipe avant de compter dans la calibration
RELIABLE_ACC = 0.62  # seuil d'accuracy ligue pour autoriser un flag ⭐
SHRINK_REG = 120.0   # régularisation du shrink vers 1.0 (anti-petit échantillon)

# Compétitions qui ne sont pas une ligue « maison » (une équipe y vient de sa ligue).
EVENT_LEAGUES = {"WLDs", "WLRQ", "MSI", "EWC", "FST", "AM", "AC", "CCWS", "WSCI",
                 "DCup", "DCGI", "KeSPA Cup", "IEM", "MSC"}

# Alias de noms de compétition -> code Oracle's Elixir (pour retrouver la calibration).
_LEAGUE_ALIASES = {
    "emeamasters": "EM", "emea": "EM", "em": "EM",
    "cblol": "CBLOL", "lck": "LCK", "lckchallengers": "LCKC", "lckcl": "LCKC", "lpl": "LPL",
    "lec": "LEC", "lcs": "LCS", "ljl": "LJL", "primeleague": "PRM", "prm": "PRM",
    "tcl": "TCL", "arabianleague": "AL", "northernleague": "NLC", "nlc": "NLC",
    "hellenic": "HLL", "hitpoint": "HM", "ultraliga": "PRM",
    "lfl": "LFL", "laliguefrancaise": "LFL", "ligaespanola": "LES", "superliga": "LES",
    "les": "LES", "circuitodesafiante": "CD", "lcp": "LCP", "nacl": "NACL",
    "northamericanchallengers": "NACL",
    "worlds": "WLDs", "worldchampionship": "WLDs", "msi": "MSI",
    "midseasoninvitational": "MSI", "firststand": "FST", "esportsworldcup": "EWC",
    "kespacup": "KeSPA Cup", "demaciacup": "DCup",
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


# Colonnes réellement utilisées (sur ~170 du CSV) -> lecture frugale en RAM (cloud).
_GAME_COLS = {"gameid", "teamname", "result", "date", "league", "kills", "position", "side",
              "playoffs"}


_games_cache: dict = {}


def load_games(cfg: dict | None = None, history: bool = False,
               scope: str | None = None) -> pd.DataFrame:
    """Une ligne par game : blue, red, yb (1=bleu gagne), kblue, kred, date, league, season.

    `history=True` (modèle v2) : les saisons passées de `season_files` sont rejouées
    avant la saison en cours. Par défaut : la saison en cours seule, comme l'ancien site.
    Le résultat est gardé en mémoire tant que les fichiers ne changent pas : les trois
    moteurs (Elo, contexte, joueurs) rejouent les mêmes games.

    `scope` : « bettable » = seulement les ligues et événements pariables (`config.yaml`,
    `betting`), « all » = tout le CSV. Par défaut : `betting.elo_scope` pour l'Elo de la
    saison (picks, rejeu des paris), « all » pour le modèle v2 (`history=True`), qui n'a pas
    encore été testé sur le périmètre restreint.
    """
    cfg = cfg or load_config()
    if scope is None:
        scope = "all" if history else (cfg.get("betting") or {}).get("elo_scope", "all")
    key = tuple((str(p), p.stat().st_mtime) for _, p in season_files(cfg, history))
    if key not in _games_cache:
        if len(_games_cache) >= 2:
            _games_cache.clear()
        _games_cache[key] = _read_games(cfg, history)
    g = _games_cache[key]
    if scope == "bettable":
        b = cfg.get("betting") or {}
        keep = set(b.get("leagues") or []) | set(b.get("events") or [])
        if keep:
            g = g[g["league"].isin(keep)].reset_index(drop=True)
    return g.copy()


def _read_games(cfg: dict, history: bool) -> pd.DataFrame:
    df = read_seasons(cfg, _GAME_COLS, history=history)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    t = df[df["position"].str.lower() == "team"]
    cb = ["gameid", "teamname", "result", "date", "league", "kills", "season"]
    if "playoffs" in t.columns:            # phase du match (0 = régulière, 1 = playoffs)
        cb.append("playoffs")
    tb = t[t["side"].str.lower() == "blue"][cb]
    tr = t[t["side"].str.lower() == "red"][["gameid", "teamname", "kills"]]
    g = (tb.merge(tr, on="gameid", suffixes=("_b", "_r"))
           .dropna(subset=["teamname_b", "teamname_r", "result"])
           # ordre total et reproductible : quelques games portent exactement la même date
           .sort_values(["date", "gameid"]).reset_index(drop=True))
    g = g.rename(columns={"teamname_b": "blue", "teamname_r": "red", "result": "yb",
                          "kills_b": "kblue", "kills_r": "kred"})
    g["yb"] = g["yb"].astype(int)
    if "playoffs" in g.columns:
        g["playoffs"] = pd.to_numeric(g["playoffs"], errors="coerce").fillna(0).astype(int)
    return g


def compute_elo(cfg: dict | None = None, k: float = K, base: float = BASE,
                history: bool = False) -> dict:
    """Rejoue toutes les games (K32 + MOV) et renvoie l'état + la calibration par ligue.

    Retour :
      elo, n, league : comme avant (league = compétition de la dernière game jouée).
      home[team]      : dernière ligue « maison » de l'équipe (hors événements), pour
                        savoir d'où elle vient quand elle joue un Mondial ou un MSI.
      reliability[lg] : accuracy historique du modèle dans la ligue (hors cold-start).
      shrink[lg]      : facteur d'aplatissement de la proba (calibration, 0..1.2).
      global_rel      : accuracy globale (fallback).
    """
    cfg = cfg or load_config()
    g = load_games(cfg, history)

    elo: dict[str, float] = defaultdict(lambda: base)
    n: dict[str, int] = defaultdict(int)
    league: dict[str, str] = {}
    home: dict[str, str] = {}
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
        if lg not in EVENT_LEAGUES:
            home[blue] = lg
            home[red] = lg

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

    return {"elo": dict(elo), "n": dict(n), "league": league, "home": home,
            "reliability": reliability, "shrink": shrink, "global_rel": global_rel}


def win_prob(elo_a: float, elo_b: float, scale: float = SCALE) -> float:
    """P(A gagne une game) selon l'écart d'Elo (side-neutre, AVANT calibration)."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / scale))


def series_ids(league, blue, red, date, gap_hours: float = 10.0) -> list[int]:
    """Numéro de série pour des games en ordre chronologique : mêmes équipes, même ligue,
    games qui se suivent à moins de `gap_hours`. (Regrouper par jour coupait en deux les
    séries à cheval sur minuit UTC.)"""
    import pandas as pd
    gap = pd.Timedelta(hours=gap_hours)
    last: dict = {}
    ids, n = [], 0
    for lg, b, r, d in zip(league, blue, red, date):
        key = (lg, frozenset((b, r)))
        prev = last.get(key)
        if prev is None or d - prev[1] > gap:
            n += 1
            cur = n
        else:
            cur = prev[0]
        last[key] = (cur, d)
        ids.append(cur)
    return ids


def race(p: float, na: int, nb: int) -> float:
    """P(A prend `na` manches avant que B en prenne `nb`), p = P(A gagne une manche)."""
    if na <= 0:
        return 1.0
    if nb <= 0:
        return 0.0
    return p * race(p, na - 1, nb) + (1 - p) * race(p, na, nb - 1)


def series_prob_first(p_first: float, p_next: float, wins_needed: int) -> float:
    """P(gagner la série) quand la game 1 a sa propre proba (côté connu) et les suivantes
    une autre (côté non connu d'avance)."""
    if wins_needed <= 1:
        return float(p_first)
    return float(p_first * race(p_next, wins_needed - 1, wins_needed)
                 + (1 - p_first) * race(p_next, wins_needed, wins_needed - 1))


def shrink_walk_forward(p_blue, yb, league, nmin) -> list[float]:
    """Proba calibrée par ligue, le facteur de calibration de chaque game n'étant appris que
    sur les games d'AVANT (mêmes formules que `compute_elo`). Les games doivent être dans
    l'ordre chronologique. Calibrer avec le facteur de fin de saison donnait à l'ancien
    modèle un avantage qu'il n'a pas en vrai : le facteur contenait le résultat des games
    qu'il servait ensuite à juger."""
    pf_sum: dict = defaultdict(float)
    won_sum: dict = defaultdict(float)
    cnt: dict = defaultdict(int)
    out = []
    for p, y, lg, n in zip(p_blue, yb, league, nmin):
        c = cnt[lg]
        slope = 1.0
        if c:
            obs, predf = won_sum[lg] / c, pf_sum[lg] / c
            slope = (obs - 0.5) / (predf - 0.5) if predf > 0.5 else 1.0
            slope = (c * slope + SHRINK_REG) / (c + SHRINK_REG)
            slope = max(0.0, min(1.2, slope))
        out.append(calibrate(float(p), slope))
        if n >= BURN_IN:
            pf_sum[lg] += max(p, 1 - p)
            won_sum[lg] += float((p >= 0.5) == (y == 1))
            cnt[lg] += 1
    return out


def calibrate(p_game: float, shrink: float) -> float:
    """Aplatit (ou accentue) la proba autour de 50 % selon le facteur de calibration
    de la ligue. `shrink` pouvant aller jusqu'à 1.2 (ligues sous-confiantes), on borne
    le résultat à [0.01, 0.99] : une probabilité ne peut pas sortir de [0, 1]."""
    p = 0.5 + (p_game - 0.5) * shrink
    return min(max(p, 0.01), 0.99)


def competition_code(name: str) -> str | None:
    """Code OE d'une compétition d'après son nom, qu'on ait une calibration pour elle ou non.

    L'alias le plus long gagne (« LCK Challengers League » -> LCKC, pas LCK) ; un alias court
    doit commencer le nom (« em » ne doit pas reconnaître « Demacia Cup »)."""
    nm = _norm(name)
    if not nm:
        return None
    for key, code in sorted(_LEAGUE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if nm == key or (len(key) > 4 and key in nm) or (2 < len(key) <= 4 and nm.startswith(key)):
            return code
    return None


def resolve_league(name: str, reliability: dict) -> str | None:
    """Mappe un nom de compétition (overview lolesports) vers un code OE connu."""
    nm = _norm(name)
    if not nm:
        return None
    for code in reliability:
        if _norm(code) == nm:
            return code
    code = competition_code(name)
    return code if code in reliability else None


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

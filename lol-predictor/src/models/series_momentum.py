"""Analyse des patterns de MOMENTUM en série (BO3 / BO5) sur la data Oracle's Elixir.

Idée (demande user) : « quand une équipe est 2-0, a-t-elle l'avantage psychologique ?
y a-t-il beaucoup de 3-0 ? souvent en 2-0 l'équipe perdante revient ? » -> on mesure
TOUT sur la data, par état de score et par ligue, pour voir si un schéma exploitable
ressort (base d'un edge de trading live).

Reconstruction des séries : Oracle's Elixir a une colonne `game` = n° de la map dans la
série. On groupe les games par (jour, paire d'équipes), on ordonne par `game`, et on
déduit le format depuis le nb de maps du vainqueur :
    - vainqueur à 2 maps  -> BO3
    - vainqueur à 3 maps  -> BO5
    - vainqueur à 1 map   -> BO1 (ignoré : pas de progression de score)

Pour chaque état intermédiaire (1-0, 2-0, 2-1...) on mesure, du point de vue du LEADER :
    - P(gagne la SÉRIE)      -> close-out vs comeback
    - P(gagne la MAP suivante) -> momentum carte par carte
Global + par ligue + distribution des scores finaux.

ATTENTION biais : une équipe qui mène 2-0 est souvent la plus forte au départ -> un fort
taux de close-out reflète en partie la force, pas seulement le "momentum". Pour PARIER,
ce qui compte c'est le taux empirique brut vs la cote live -> ces chiffres sont
directement la bonne référence. Pour ISOLER le momentum pur, voir le test "même équipe
gagne 2 maps de suite" (à comparer à 50%).

Usage :
    python -m src.models.series_momentum
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "2026_LoL_esports_match_data_from_OraclesElixir.csv"

# Seuils d'affichage (sample minimal pour qu'un chiffre soit montré)
MIN_N_STATE = 30      # global : nb mini de séries dans un état pour l'afficher
MIN_N_LEAGUE = 15     # par ligue : nb mini de séries pour afficher la ligue


# --------------------------------------------------------------------------- #
#  Reconstruction des séries
# --------------------------------------------------------------------------- #
def load_games(csv: Path = CSV) -> pd.DataFrame:
    """Une ligne par game : blue, red, winner, league, jour, n° de map, playoffs."""
    df = pd.read_csv(csv, low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    t = df[df["position"].astype(str).str.lower() == "team"].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date", "teamname", "result"])
    t["day"] = t["date"].dt.date
    t["game"] = pd.to_numeric(t["game"], errors="coerce")

    side = t["side"].astype(str).str.lower()
    blue = t[side == "blue"][
        ["gameid", "day", "league", "game", "playoffs", "teamname", "result", "date"]
    ].rename(columns={"teamname": "blue", "result": "blue_res"})
    red = t[side == "red"][["gameid", "teamname"]].rename(columns={"teamname": "red"})

    g = blue.merge(red, on="gameid", how="inner")
    g = g.dropna(subset=["blue", "red"])
    g["winner"] = np.where(g["blue_res"].astype(float) == 1, g["blue"], g["red"])
    # paire ordonnée (clé de série stable quel que soit le side)
    g["t1"] = np.where(g["blue"] < g["red"], g["blue"], g["red"])
    g["t2"] = np.where(g["blue"] < g["red"], g["red"], g["blue"])
    return g


def reconstruct_series(g: pd.DataFrame) -> list[dict]:
    """Liste de séries : {league, bo, winners:[...], t1, t2, final:(hi,lo)}."""
    series: list[dict] = []
    for (_, t1, t2), grp in g.groupby(["day", "t1", "t2"], sort=False):
        grp = grp.sort_values(["game", "date"], kind="stable")
        winners = list(grp["winner"])
        if any(w not in (t1, t2) for w in winners):
            continue
        w1 = sum(w == t1 for w in winners)
        w2 = sum(w == t2 for w in winners)
        maxw = max(w1, w2)
        if maxw == 2:
            bo = "BO3"
        elif maxw == 3:
            bo = "BO5"
        else:
            continue  # BO1 ou série fusionnée/invalide
        if w1 + w2 != len(winners):
            continue
        series.append({
            "league": str(grp["league"].iloc[0]),
            "bo": bo,
            "winners": winners,
            "t1": t1,
            "t2": t2,
            "final": tuple(sorted((w1, w2), reverse=True)),
        })
    return series


# --------------------------------------------------------------------------- #
#  Agrégation par état de score
# --------------------------------------------------------------------------- #
def analyse(series: list[dict]):
    # (bo, state) -> [ (leader_gagne_serie, leader_gagne_map_suivante), ... ]
    by_state: dict = defaultdict(list)
    # (league, bo, state) -> idem
    by_league: dict = defaultdict(list)
    final_dist: dict = defaultdict(Counter)
    bo_counts: Counter = Counter()
    # momentum pur : la même équipe gagne 2 maps de suite ?
    streak_same = streak_tot = 0

    for s in series:
        bo, winners, t1, t2 = s["bo"], s["winners"], s["t1"], s["t2"]
        league = s["league"]
        bo_counts[bo] += 1
        final_dist[bo][s["final"]] += 1
        series_winner = t1 if winners.count(t1) > winners.count(t2) else t2

        ca = cb = 0
        for i, w in enumerate(winners):
            if i + 1 < len(winners):
                streak_tot += 1
                streak_same += int(winners[i + 1] == w)
            if w == t1:
                ca += 1
            else:
                cb += 1
            if i >= len(winners) - 1:
                continue  # état final : pas de "suite"
            if ca == cb:
                continue  # égalité (1-1, 2-2) : pas de leader
            leader = t1 if ca > cb else t2
            state = tuple(sorted((ca, cb), reverse=True))
            rec = (int(leader == series_winner), int(leader == winners[i + 1]))
            by_state[(bo, state)].append(rec)
            by_league[(league, bo, state)].append(rec)

    return {
        "by_state": by_state,
        "by_league": by_league,
        "final_dist": final_dist,
        "bo_counts": bo_counts,
        "streak": (streak_same, streak_tot),
    }


# --------------------------------------------------------------------------- #
#  Affichage
# --------------------------------------------------------------------------- #
def _rate(records):
    n = len(records)
    if n == 0:
        return float("nan"), float("nan"), 0
    serie = 100 * sum(r[0] for r in records) / n
    nextm = 100 * sum(r[1] for r in records) / n
    return serie, nextm, n


def report(res: dict) -> None:
    bo_counts = res["bo_counts"]
    by_state = res["by_state"]
    final_dist = res["final_dist"]

    print("=" * 70)
    print("  MOMENTUM DE SÉRIE — Oracle's Elixir 2026 (toutes ligues)")
    print("=" * 70)
    print(f"  Séries reconstruites : BO3={bo_counts['BO3']}   BO5={bo_counts['BO5']}")

    ss, st = res["streak"]
    if st:
        print(f"  Momentum brut : la même équipe gagne 2 maps de suite "
              f"{100*ss/st:.1f}% du temps (n={st} transitions ; 50% = pas de momentum)")

    # ---- Distribution des scores finaux ----
    for bo, total in (("BO3", bo_counts["BO3"]), ("BO5", bo_counts["BO5"])):
        if not total:
            continue
        print("\n" + "-" * 70)
        print(f"  {bo} — distribution des scores finaux (n={total})")
        for score, c in sorted(final_dist[bo].items(), key=lambda x: -x[1]):
            print(f"     {score[0]}-{score[1]} : {100*c/total:5.1f}%  ({c})")

    # ---- Conditionnel par état ----
    order = {
        "BO3": [(1, 0)],
        "BO5": [(1, 0), (2, 0), (2, 1)],
    }
    print("\n" + "=" * 70)
    print("  CONDITIONNEL — du point de vue du LEADER")
    print("    P(série) = il gagne la série   |   P(map+1) = il gagne la map suivante")
    print("=" * 70)
    for bo in ("BO3", "BO5"):
        print(f"\n  [{bo}]")
        for state in order[bo]:
            serie, nextm, n = _rate(by_state.get((bo, state), []))
            if n < MIN_N_STATE:
                print(f"    mène {state[0]}-{state[1]} : n={n} (insuffisant)")
                continue
            comeback = 100 - serie
            print(f"    mène {state[0]}-{state[1]} (n={n:4d}) : "
                  f"P(série)={serie:5.1f}%  | comeback adverse={comeback:4.1f}%  "
                  f"| P(map+1)={nextm:5.1f}%")

    # reverse sweep BO5 = perdre après 2-0
    rs = by_state.get(("BO5", (2, 0)), [])
    if len(rs) >= MIN_N_STATE:
        serie, _, n = _rate(rs)
        print(f"\n  >>> Reverse sweep BO5 (mener 2-0 puis perdre) : {100-serie:.1f}%  (n={n})")


def report_by_league(res: dict) -> None:
    by_league = res["by_league"]
    print("\n" + "=" * 70)
    print("  PAR LIGUE — close-out du leader (P gagne la série)")
    print(f"  (ligues avec n >= {MIN_N_LEAGUE} séries dans l'état)")
    print("=" * 70)

    for bo, state, label in (("BO3", (1, 0), "BO3 mène 1-0"),
                             ("BO5", (2, 0), "BO5 mène 2-0")):
        rows = []
        for (lg, b, stt), recs in by_league.items():
            if b == bo and stt == state and len(recs) >= MIN_N_LEAGUE:
                serie, nextm, n = _rate(recs)
                rows.append((lg, serie, nextm, n))
        if not rows:
            continue
        rows.sort(key=lambda x: -x[1])
        print(f"\n  --- {label} ---")
        print(f"    {'ligue':<10} {'P(série)':>9} {'P(map+1)':>9} {'n':>5}")
        for lg, serie, nextm, n in rows:
            print(f"    {lg:<10} {serie:8.1f}% {nextm:8.1f}% {n:5d}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    g = load_games()
    series = reconstruct_series(g)
    res = analyse(series)
    report(res)
    report_by_league(res)
    print("\n" + "=" * 70)
    print("  Rappel : un fort taux de close-out vient en partie de la FORCE de l'équipe,")
    print("  pas seulement du momentum. Pour parier, comparer ces % à la cote LIVE.")
    print("=" * 70)


if __name__ == "__main__":
    main()

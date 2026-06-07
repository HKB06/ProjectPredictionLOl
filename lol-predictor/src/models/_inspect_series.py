"""Inspection ponctuelle : noms d'équipes, plage de dates, reconstruction des séries.

Sert à préparer le backtest de valeur (join avec les cotes BO3) :
- liste les noms d'équipes exacts (pour la table de correspondance avec les cotes) ;
- montre comment les games (1 ligne) s'agrègent en séries (même jour, mêmes 2 équipes) ;
- affiche les séries autour des dates des cotes fournies (14-15 jan 2026) pour vérifier le mapping.

Usage : python -m src.models._inspect_series
"""
from __future__ import annotations

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config


def reconstruct_series(matches: pd.DataFrame) -> pd.DataFrame:
    """Agrège les games en séries : clé = (jour, paire d'équipes)."""
    m = matches.copy()
    m["day"] = pd.to_datetime(m["date"]).dt.date
    m["pair"] = m.apply(lambda r: tuple(sorted((r["blue_team"], r["red_team"]))), axis=1)

    rows = []
    for (day, pair), g in m.groupby(["day", "pair"]):
        g = g.sort_values("date")
        t1, t2 = pair
        wins = {t1: 0, t2: 0}
        for r in g.itertuples():
            winner = r.blue_team if r.y_winner == 1 else r.red_team
            wins[winner] += 1
        n_games = len(g)
        series_winner = t1 if wins[t1] > wins[t2] else t2
        rows.append({
            "day": day, "team1": t1, "team2": t2,
            "n_games": n_games, "score1": wins[t1], "score2": wins[t2],
            "series_winner": series_winner,
        })
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    matches = pd.read_parquet(proc / "matches.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    print("=" * 70)
    print("  DONNÉES MATCHES")
    print("=" * 70)
    print(f"  Games          : {len(matches)}")
    print(f"  Période        : {matches['date'].min():%Y-%m-%d} -> {matches['date'].max():%Y-%m-%d}")

    teams = sorted(set(matches["blue_team"]) | set(matches["red_team"]))
    print(f"\n  Équipes ({len(teams)}) :")
    for t in teams:
        print(f"    - {t!r}")

    series = reconstruct_series(matches)
    print(f"\n  Séries reconstruites : {len(series)}")
    print(f"  Répartition n_games par série :")
    print(series["n_games"].value_counts().sort_index().to_string())

    print("\n  Premières séries (jusqu'au 20 jan 2026) :")
    early = series[series["day"] <= pd.to_datetime("2026-01-20").date()]
    for r in early.itertuples():
        print(f"    {r.day}  {r.team1} {r.score1}-{r.score2} {r.team2}  -> {r.series_winner}")

    print("=" * 70)


if __name__ == "__main__":
    main()

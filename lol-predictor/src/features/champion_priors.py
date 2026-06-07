"""Priors de draft : winrate de base par champion, calculé SANS FUITE.

Idée (composante n°1 de DraftGap) : un champion fort sur le patch a un winrate
> 0.5. On veut donner au modèle, pour chaque match, le winrate "as-of" des 10
champions draftés — c.-à-d. estimé uniquement sur les games ANTÉRIEURES au match.

Astuce data : la LCK seule (~329 games) est trop maigre pour des winrates champion
fiables. On calcule donc ces priors sur le POOL PRO COMPLET du CSV Oracle
(toutes ligues, ~5500 games, 171 champions). Utiliser d'autres ligues *antérieures*
à la date du match n'est PAS une fuite : ces games ont réellement eu lieu avant.

Anti-fuite : pour un match daté D, on ne compte que les lignes de champion dont la
date est STRICTEMENT antérieure à D (bisect_left), ce qui exclut le match lui-même
et tout game au même horodatage.

Usage (sanity-check) :
    python -m src.features.champion_priors
"""
from __future__ import annotations

import bisect

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config

K_CHAMP = 30.0  # lissage du winrate champion vers 0.5 (pool large -> k modéré)


def load_full_champion_results(cfg: dict, leagues: list[str] | None = None) -> pd.DataFrame:
    """Lignes joueur : date, champion, result. Base des priors champion.

    `leagues=None` -> tout le pool pro (toutes ligues). Sinon on restreint aux
    ligues données (ex. ["LPL","LCK","LEC","LCS"] pour ne garder que les majeures,
    afin d'éviter qu'une ligue plus faible ne fausse le winrate d'un champion).
    """
    csv_path = ROOT / cfg["data"]["oracle_csv"]
    df = pd.read_csv(
        csv_path,
        low_memory=False,
        usecols=["league", "date", "position", "champion", "result"],
    )
    if leagues:
        df = df[df["league"].isin(leagues)]
    players = df[df["position"].str.lower() != "team"].copy()
    players["date"] = pd.to_datetime(players["date"], errors="coerce")
    players = players.dropna(subset=["date", "champion", "result"])
    return players.sort_values("date").reset_index(drop=True)


class ChampionWinrateIndex:
    """Winrate champion "as-of" via préfixes cumulés + recherche dichotomique.

    Pour chaque champion on stocke la liste triée de ses dates de game et la somme
    cumulée de ses victoires. Le winrate avant une date D s'obtient en O(log n).
    """

    def __init__(self, players: pd.DataFrame, k: float = K_CHAMP) -> None:
        self.k = k
        self.dates: dict[str, list] = {}
        self.cumwins: dict[str, list[float]] = {}
        for champ, grp in players.groupby("champion"):
            grp = grp.sort_values("date")
            self.dates[champ] = grp["date"].tolist()
            cum = [0.0]
            for w in grp["result"].astype(float):
                cum.append(cum[-1] + w)
            self.cumwins[champ] = cum

    def asof(self, champ, date) -> float:
        """Winrate lissé du champion, sur ses games STRICTEMENT avant `date`."""
        if champ is None or (isinstance(champ, float) and pd.isna(champ)):
            return 0.5
        ds = self.dates.get(champ)
        if not ds:
            return 0.5
        idx = bisect.bisect_left(ds, date)  # nb de games avant D (exclut D)
        games = idx
        wins = self.cumwins[champ][idx]
        return (wins + 0.5 * self.k) / (games + self.k)

    def asof_games(self, champ, date) -> int:
        """Nombre de games du champion avant `date` (mesure de fiabilité)."""
        ds = self.dates.get(champ)
        if not ds:
            return 0
        return bisect.bisect_left(ds, date)


def main() -> None:
    cfg = load_config()
    players = load_full_champion_results(cfg)
    idx = ChampionWinrateIndex(players)

    n_games = len(players) // 10
    print("=" * 60)
    print("  PRIORS CHAMPION (pool pro complet, toutes ligues)")
    print("=" * 60)
    print(f"  Lignes joueur     : {len(players)}  (~{n_games} games)")
    print(f"  Champions connus  : {len(idx.dates)}")
    print(f"  Période           : {players['date'].min():%Y-%m-%d} -> {players['date'].max():%Y-%m-%d}")

    # Exemple : top winrate "à aujourd'hui" (fin de période), min 50 games
    last_date = players["date"].max() + pd.Timedelta(seconds=1)
    rows = []
    for champ in idx.dates:
        g = idx.asof_games(champ, last_date)
        if g >= 50:
            rows.append((champ, idx.asof(champ, last_date), g))
    rows.sort(key=lambda x: x[1], reverse=True)
    print("\n  Top 10 winrate (>=50 games, fin de période) :")
    for champ, wr, g in rows[:10]:
        print(f"    {champ:<16} {wr*100:5.1f}%  ({g} games)")
    print("=" * 60)


if __name__ == "__main__":
    main()

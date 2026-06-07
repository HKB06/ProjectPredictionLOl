"""P2 : construit la table d'analyse (1 ligne par match) à partir du format Oracle.

Le format Oracle a 12 lignes par game (10 joueurs + 2 équipes). On en tire :
- une table `matches` (1 ligne/game) : contexte + draft + LABELS par marché ;
- une table `team_games` (1 ligne par équipe×game) : stats brutes, base des features historiques.

Convention des labels (du point de vue BLEU) :
- y_winner       : 1 si le bleu gagne
- y_first_blood  : 1 si le bleu prend le premier sang
- y_first_tower  : 1 si le bleu prend la première tour
- y_first_dragon : 1 si le bleu prend le premier dragon
- y_total_kills  : kills bleu + kills rouge
- y_game_time_min: durée en minutes

Usage :
    python -m src.ingest.build_match_table
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config, load_oracle

ROLES = ["top", "jng", "mid", "bot", "sup"]
TEAM_STAT_COLS = [
    "kills", "deaths", "assists", "result", "firstblood", "firsttower",
    "firstdragon", "firstherald", "firstbaron", "dragons", "barons", "towers",
    "golddiffat15", "xpdiffat15", "csdiffat15", "team kpm", "ckpm",
]


def _to_int(value, default=None):
    if pd.isna(value):
        return default
    return int(value)


def build_team_games(df: pd.DataFrame) -> pd.DataFrame:
    """1 ligne par (équipe, game) : stats brutes + contexte (base des features)."""
    teams = df[df["position"].str.lower() == "team"].copy()
    keep = ["gameid", "date", "patch", "split", "playoffs", "side", "teamname", "gamelength"]
    keep += [c for c in TEAM_STAT_COLS if c in teams.columns]
    return teams[keep].reset_index(drop=True)


def build_matches(df: pd.DataFrame) -> pd.DataFrame:
    teams = df[df["position"].str.lower() == "team"].copy()
    players = df[df["position"].str.lower() != "team"].copy()
    players_by_game = {gid: g for gid, g in players.groupby("gameid")}

    records = []
    for gid, g in teams.groupby("gameid"):
        blue = g[g["side"].str.lower() == "blue"]
        red = g[g["side"].str.lower() == "red"]
        if len(blue) != 1 or len(red) != 1:
            continue
        blue, red = blue.iloc[0], red.iloc[0]

        rec: dict = {
            "gameid": gid,
            "date": blue["date"],
            "patch": blue.get("patch"),
            "split": blue.get("split"),
            "playoffs": blue.get("playoffs"),
            "blue_team": blue["teamname"],
            "red_team": red["teamname"],
            "gamelength": blue["gamelength"],
            # LABELS (point de vue bleu)
            "y_winner": _to_int(blue["result"]),
            "y_total_kills": _to_int(blue["kills"], 0) + _to_int(red["kills"], 0),
            "y_first_blood": _to_int(blue.get("firstblood")),
            "y_first_tower": _to_int(blue.get("firsttower")),
            "y_first_dragon": _to_int(blue.get("firstdragon")),
            "y_game_time_min": round(blue["gamelength"] / 60, 2),
            # stats brutes utiles
            "blue_kills": _to_int(blue["kills"], 0),
            "red_kills": _to_int(red["kills"], 0),
            "blue_golddiffat15": blue.get("golddiffat15"),
        }

        # bans
        for i in range(1, 6):
            rec[f"blue_ban{i}"] = blue.get(f"ban{i}")
            rec[f"red_ban{i}"] = red.get(f"ban{i}")

        # picks par rôle
        pg = players_by_game.get(gid)
        if pg is not None:
            for side_key, side_val in (("blue", blue["side"]), ("red", red["side"])):
                side_players = pg[pg["side"] == side_val]
                by_role = {str(r).lower(): c for r, c in zip(side_players["position"], side_players["champion"])}
                for role in ROLES:
                    rec[f"{side_key}_{role}"] = by_role.get(role)

        records.append(rec)

    matches = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return matches


def print_market_baselines(matches: pd.DataFrame) -> None:
    n = len(matches)
    print("=" * 60)
    print(f"  BASELINES PAR MARCHÉ  (n={n} matchs)")
    print("=" * 60)
    print(f"  Vainqueur (bleu)      : {matches['y_winner'].mean() * 100:5.1f}%  <- baseline à battre")
    print(f"  First blood (bleu)    : {matches['y_first_blood'].mean() * 100:5.1f}%")
    print(f"  First tower (bleu)    : {matches['y_first_tower'].mean() * 100:5.1f}%")
    print(f"  First dragon (bleu)   : {matches['y_first_dragon'].mean() * 100:5.1f}%")
    print(f"  Total kills moyen     : {matches['y_total_kills'].mean():5.1f}  (médiane {matches['y_total_kills'].median():.0f})")
    print(f"  Durée moyenne (min)   : {matches['y_game_time_min'].mean():5.1f}")
    print("=" * 60)


def main() -> None:
    cfg = load_config()
    df = load_oracle(cfg)

    matches = build_matches(df)
    team_games = build_team_games(df)

    out_dir = ROOT / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out_dir / "matches.parquet", index=False)
    team_games.to_parquet(out_dir / "team_games.parquet", index=False)

    print(f"matches     -> {out_dir / 'matches.parquet'}  ({matches.shape})")
    print(f"team_games  -> {out_dir / 'team_games.parquet'}  ({team_games.shape})")
    print()
    print_market_baselines(matches)


if __name__ == "__main__":
    main()

"""Inspecte le(s) dernier(s) match(s) LCK pour comparer avec Gol.gg.

Affiche, pour les N derniers games : équipes, side, résultat, durée, kills,
first objectives, total kills, et le détail par joueur (champion, KDA, GD@15).

Usage :
    python -m src.ingest.inspect_game            # dernier game
    python -m src.ingest.inspect_game 3          # 3 derniers games
"""
from __future__ import annotations

import sys

import pandas as pd

from src.ingest.load_oracle import load_config, load_oracle

ROLE_ORDER = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}


def _g(row, col, default="-"):
    val = row.get(col, default)
    if pd.isna(val):
        return default
    return val


def inspect(df: pd.DataFrame, n: int = 1) -> None:
    df = df.sort_values("date")
    last_ids = df.drop_duplicates("gameid", keep="last")["gameid"].tolist()[-n:]

    for gid in last_ids:
        g = df[df["gameid"] == gid]
        teams = g[g["position"].str.lower() == "team"]
        players = g[g["position"].str.lower() != "team"]
        date = g["date"].iloc[0]
        patch = _g(g.iloc[0], "patch")
        length = g["gamelength"].iloc[0]
        mm, ss = divmod(int(length), 60)

        print("=" * 70)
        print(f"  GAME {gid}  |  {date:%Y-%m-%d}  |  patch {patch}  |  durée {mm}:{ss:02d}")
        print("=" * 70)

        total_kills = int(teams["kills"].sum())
        for _, t in teams.iterrows():
            side = str(_g(t, "side"))
            win = "WIN " if t.get("result") == 1 else "LOSS"
            print(f"\n  [{side.upper():>4}] {_g(t, 'teamname')}  -> {win}")
            print(
                f"        kills={int(_g(t, 'kills', 0))}  "
                f"FB={_g(t, 'firstblood', 0)}  FT={_g(t, 'firsttower', 0)}  "
                f"FD={_g(t, 'firstdragon', 0)}  dragons={_g(t, 'dragons', 0)}  "
                f"towers={_g(t, 'towers', 0)}  barons={_g(t, 'barons', 0)}"
            )
            side_players = players[players["side"] == t["side"]].copy()
            side_players["ord"] = side_players["position"].str.lower().map(ROLE_ORDER)
            for _, p in side_players.sort_values("ord").iterrows():
                pos = str(_g(p, "position")).upper()
                print(
                    f"          {pos:<4} {str(_g(p, 'playername')):<12} "
                    f"{str(_g(p, 'champion')):<13} "
                    f"{int(_g(p,'kills',0))}/{int(_g(p,'deaths',0))}/{int(_g(p,'assists',0)):<3} "
                    f"GD@15={_g(p, 'golddiffat15')}  XPD@15={_g(p, 'xpdiffat15')}"
                )

        bans = teams.iloc[0]
        print(f"\n  Total kills (combiné) : {total_kills}")
        print("=" * 70 + "\n")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cfg = load_config()
    df = load_oracle(cfg)
    inspect(df, n)


if __name__ == "__main__":
    main()

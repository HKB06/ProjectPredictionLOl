"""Cherche un EDGE de STYLE : un trait d'équipe stable et indépendant de la force.

Idée (hypothèse utilisateur) : une équipe peut être faible mais très agressive ->
beaucoup de first blood / kills, peu importe qui gagne. Si ce trait est STABLE dans le
temps ET INDÉPENDANT de la force (winrate), alors les marchés O/U (kills, durée, first
blood) sont prévisibles là où le book ancre ses lignes sur le favori -> edge potentiel.

On teste pour chaque trait :
1. STABILITÉ (split-half) : corrélation entre la 1re moitié et la 2e moitié de saison
   des moyennes par équipe. Élevée = vrai trait persistant ; ~0 = bruit (non prévisible).
2. INDÉPENDANCE : corrélation du trait avec le winrate. Proche de 0 = style pur
   (orthogonal à la force) = là où le book peut se tromper.

Un trait STABLE + INDÉPENDANT = candidat edge. Stable + corrélé force = déjà price-é.

Usage : python -m src.models.style_edge
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.load_oracle import ROOT, load_config

TRAITS = {
    "winrate": "result",
    "fb_rate": "firstblood",
    "fh_rate": "firstherald",
    "fd_rate": "firstdragon",
    "tot_kills": "_total_kills",   # kills + deaths (kills totaux du game)
    "ckpm": "ckpm",                # kills combinés / min = tempo/agressivité
    "duration": "_duration_min",
}


def team_table(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.copy()
    tg["date"] = pd.to_datetime(tg["date"])
    for c in ["result", "firstblood", "firstherald", "firstdragon", "kills", "deaths",
              "ckpm", "gamelength"]:
        if c in tg:
            tg[c] = pd.to_numeric(tg[c], errors="coerce")
    tg["_total_kills"] = tg["kills"] + tg["deaths"]
    tg["_duration_min"] = tg["gamelength"] / 60.0
    return tg


def split_half_stability(tg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait, col in TRAITS.items():
        h1_means, h2_means = [], []
        for team, g in tg.groupby("teamname"):
            g = g.sort_values("date")
            if len(g) < 10:
                continue
            mid = len(g) // 2
            h1_means.append(g.iloc[:mid][col].mean())
            h2_means.append(g.iloc[mid:][col].mean())
        h1, h2 = np.array(h1_means), np.array(h2_means)
        stab = float(np.corrcoef(h1, h2)[0, 1]) if len(h1) > 2 else np.nan
        rows.append({"trait": trait, "stabilité (split-half)": round(stab, 2)})
    return pd.DataFrame(rows)


def independence_from_strength(team_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait in TRAITS:
        if trait == "winrate":
            continue
        c = float(team_means["winrate"].corr(team_means[trait]))
        rows.append({"trait": trait, "corr avec winrate": round(c, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    tg = team_table(pd.read_parquet(proc / "team_games.parquet"))

    # moyennes par équipe
    agg = {t: (col, "mean") for t, col in TRAITS.items()}
    team_means = tg.groupby("teamname").agg(**agg)
    team_means["n"] = tg.groupby("teamname").size()
    team_means = team_means.sort_values("winrate", ascending=False)

    print("=" * 84)
    print("  TRAITS PAR ÉQUIPE (LCK 2026)")
    print("=" * 84)
    show = team_means.copy()
    show["winrate"] = (show["winrate"] * 100).round(0)
    show["fb_rate"] = (show["fb_rate"] * 100).round(0)
    show["fh_rate"] = (show["fh_rate"] * 100).round(0)
    show["fd_rate"] = (show["fd_rate"] * 100).round(0)
    show["tot_kills"] = show["tot_kills"].round(1)
    show["ckpm"] = show["ckpm"].round(2)
    show["duration"] = show["duration"].round(1)
    print(show.to_string())

    print("\n" + "=" * 84)
    print("  1) STABILITÉ d'un trait (split-half) : >0.5 = persistant/prévisible, ~0 = bruit")
    print("=" * 84)
    print(split_half_stability(tg).to_string(index=False))

    print("\n" + "=" * 84)
    print("  2) INDÉPENDANCE vs force : |corr| faible = style pur (là où le book peut louper)")
    print("=" * 84)
    print(independence_from_strength(team_means).to_string(index=False))

    print("\n" + "=" * 84)
    print("  LECTURE : trait STABLE (1) + INDÉPENDANT de la force (2) = candidat edge O/U.")
    print("  (reste à confirmer avec l'historique des cotes O/U de ce marché.)")
    print("=" * 84)


if __name__ == "__main__":
    main()

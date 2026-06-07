"""Test empirique : "miser uniquement sur les gros favoris évidents", ça rapporte ?

Stratégie naïve très répandue : parier le favori (cote la plus basse) quand l'écart
est "évident" (cote <= seuil). On la teste sur TOUTES les séries LCK 2026 (résultats
réels via les scores du fichier de cotes). Aucune modélisation : juste cotes + résultats.

But : montrer pourquoi "favori imbattable" != argent (les cotes basses paient peu, les
rares upsets effacent les gains, et le book prend sa marge).

Usage : python -m src.models.fav_strategy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.load_oracle import ROOT


def main() -> None:
    odds = pd.read_csv(ROOT / "data" / "odds" / "lck_2026_odds.csv")
    odds = odds.dropna(subset=["score1", "score2", "odd1", "odd2"]).copy()

    # favori = cote la plus basse ; a-t-il gagné la série ?
    odds["fav_is_t1"] = odds["odd1"] < odds["odd2"]
    odds["fav_odd"] = np.where(odds["fav_is_t1"], odds["odd1"], odds["odd2"])
    odds["dog_odd"] = np.where(odds["fav_is_t1"], odds["odd2"], odds["odd1"])
    t1_won = odds["score1"] > odds["score2"]
    odds["fav_won"] = np.where(odds["fav_is_t1"], t1_won, ~t1_won)

    print("=" * 78)
    print(f"  STRATÉGIE 'MISER LE FAVORI' — {len(odds)} séries LCK 2026 (cotes + résultats réels)")
    print("=" * 78)
    print(f"  Taux de victoire des favoris (toutes séries) : {odds['fav_won'].mean()*100:.1f}%")
    print(f"  -> les outsiders gagnent quand même {(1-odds['fav_won'].mean())*100:.1f}% du temps\n")

    print(f"  {'seuil cote':<12}{'#paris':>7}{'%gagnés':>9}{'seuil rentab.':>15}"
          f"{'profit (u)':>12}{'ROI':>9}")
    print("  " + "-" * 64)
    for T in [1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00, 99]:
        sub = odds[odds["fav_odd"] <= T]
        if sub.empty:
            continue
        n = len(sub)
        wr = sub["fav_won"].mean()
        profit = float(((sub["fav_odd"] - 1) * sub["fav_won"] - (~sub["fav_won"])).sum())
        roi = profit / n
        breakeven = 1.0 / sub["fav_odd"].mean()  # taux de victoire requis (cote moyenne)
        label = f"<= {T:.2f}" if T < 99 else "tous"
        print(f"  {label:<12}{n:>7}{wr*100:>8.1f}%{breakeven*100:>14.1f}%"
              f"{profit:>12.2f}{roi*100:>8.1f}%")

    # Et si on misait les OUTSIDERS au lieu des favoris ?
    dog_profit = float(((odds["dog_odd"] - 1) * (~odds["fav_won"]) - odds["fav_won"]).sum())
    print("  " + "-" * 64)
    print(f"  (Comparaison) miser TOUS les outsiders : profit {dog_profit:+.2f}u "
          f"-> ROI {dog_profit/len(odds)*100:+.1f}%")

    # Focus Dplus Kia (l'exemple "cadeau") : son bilan QUAND elle est favorite
    dk = odds[((odds["team1"] == "Dplus Kia") & odds["fav_is_t1"]) |
              ((odds["team2"] == "Dplus Kia") & ~odds["fav_is_t1"])]
    if len(dk):
        dk_won = dk["fav_won"]
        dk_profit = float(((dk["fav_odd"] - 1) * dk_won - (~dk_won)).sum())
        print("  " + "-" * 64)
        print(f"  FOCUS 'Dplus Kia favorite' : {len(dk)} séries, gagnées {dk_won.sum()} "
              f"({dk_won.mean()*100:.0f}%), profit {dk_profit:+.2f}u "
              f"-> ROI {dk_profit/len(dk)*100:+.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()

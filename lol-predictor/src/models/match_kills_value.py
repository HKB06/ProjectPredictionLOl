"""Analyse de VALEUR sur le marché TOTAL KILLS PAR ÉQUIPE (lignes réelles du book).

Match : KT Rolster vs Dplus Kia (LCK, 7 juin, map 1).

Modèle attaque/défense (standard pour les totals, type Poisson foot) :
    kills_A_attendus = (kills_pour_A_moy + kills_contre_B_moy) / 2
On modélise les kills d'une équipe par une loi Binomiale Négative (kills sur-dispersés
vs Poisson) calibrée sur la dispersion réelle de la LCK, puis on calcule P(over ligne),
la proba dé-viggée du book, et l'EV de chaque pari.

But : la ligne du book est-elle PARESSEUSE (proche de la moyenne -> edge) ou SHARP
(collée à notre tempo -> pas d'edge) ? C'est LE test du lead identifié hier soir.

Usage : python -m src.models.match_kills_value
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.ingest.load_oracle import ROOT, load_config

# ----- lignes réelles relevées (map 1) : ligne -> (cote_over, cote_under) -----
DK_LINES = {9.5: (1.36, 2.82), 10.5: (1.43, 2.52), 11.5: (1.52, 2.30),
            12.5: (1.62, 2.10), 13.5: (1.73, 1.94), 14.5: (1.87, 1.79), 15.5: (2.08, 1.63)}
KT_LINES = {11.5: (1.49, 2.38), 12.5: (1.59, 2.16), 13.5: (1.71, 1.96),
            14.5: (1.87, 1.79), 15.5: (2.08, 1.64), 16.5: (2.34, 1.50)}


def find_team(names, *needles):
    for n in names:
        low = str(n).lower()
        if all(x in low for x in needles):
            return n
    return None


def team_stats(tg: pd.DataFrame, name: str, last_n: int = 12):
    sub = tg[tg["teamname"] == name].sort_values("date")
    return {
        "n": len(sub),
        "for_season": sub["kills"].mean(),
        "against_season": sub["deaths"].mean(),
        "for_recent": sub["kills"].tail(last_n).mean(),
        "against_recent": sub["deaths"].tail(last_n).mean(),
    }


def devig_over(odds_over: float, odds_under: float) -> float:
    io, iu = 1 / odds_over, 1 / odds_under
    return io / (io + iu)


def implied_line(lines: dict) -> float:
    """Ligne 'médiane' du book = ligne où P(over) dé-viggée = 0.5 (interpolée)."""
    xs = sorted(lines)
    ps = [devig_over(*lines[x]) for x in xs]
    for i in range(len(xs) - 1):
        if (ps[i] - 0.5) * (ps[i + 1] - 0.5) <= 0:
            t = (0.5 - ps[i]) / (ps[i + 1] - ps[i])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return xs[0] if ps[0] < 0.5 else xs[-1]


def main() -> None:
    cfg = load_config()
    tg = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "team_games.parquet")
    names = tg["teamname"].unique()

    dk = find_team(names, "dplus") or find_team(names, "kia")
    kt = find_team(names, "kt")
    print(f"Équipes trouvées : DK='{dk}'  KT='{kt}'  | data jusqu'au {str(tg['date'].max())[:10]}")

    sdk, skt = team_stats(tg, dk), team_stats(tg, kt)

    # dispersion LCK (binomiale négative) calibrée sur kills par équipe/game
    k = tg["kills"].dropna().astype(float)
    m, v = k.mean(), k.var()
    r = m * m / (v - m) if v > m else 50.0       # paramètre de dispersion NB
    print(f"Kills/équipe LCK : moy={m:.1f}  var={v:.1f}  (sur-dispersion r={r:.1f})\n")

    def p_over(line: float, lam: float) -> float:
        n = r
        p = n / (n + lam)
        return 1 - stats.nbinom.cdf(math.floor(line), n, p)

    # prédictions attaque/défense (recent + saison, on prend la moyenne des deux)
    def lam(att, dfn):
        a = (att["for_recent"] + dfn["against_recent"]) / 2
        b = (att["for_season"] + dfn["against_season"]) / 2
        return (a + b) / 2, a, b

    lam_dk, dk_rec, dk_sea = lam(sdk, skt)
    lam_kt, kt_rec, kt_sea = lam(skt, sdk)

    print("=" * 78)
    print("  NOTRE ESTIMATION (attaque/défense)        |  BOOK (ligne médiane dé-viggée)")
    print("=" * 78)
    print(f"  DK kills  : {lam_dk:5.1f}  (récent {dk_rec:.1f} / saison {dk_sea:.1f})  |  {implied_line(DK_LINES):5.1f}")
    print(f"  KT kills  : {lam_kt:5.1f}  (récent {kt_rec:.1f} / saison {kt_sea:.1f})  |  {implied_line(KT_LINES):5.1f}")
    print(f"  COMBINÉ   : {lam_dk + lam_kt:5.1f}                            |  "
          f"{implied_line(DK_LINES) + implied_line(KT_LINES):5.1f}")
    print(f"\n  Profils : DK pour/contre {sdk['for_season']:.1f}/{sdk['against_season']:.1f}"
          f"  |  KT pour/contre {skt['for_season']:.1f}/{skt['against_season']:.1f}")

    for label, lines, lam_team in (("DK", DK_LINES, lam_dk), ("KT", KT_LINES, lam_kt)):
        print("\n" + "=" * 78)
        print(f"  {label} TOTAL KILLS  (notre lam={lam_team:.1f})")
        print(f"  {'ligne':>6} | {'cote O/U':>11} | {'P(over) nous':>12} | {'P(over) book':>12} | {'meilleur EV':>22}")
        print("  " + "-" * 74)
        for line in sorted(lines):
            o_over, o_under = lines[line]
            po = p_over(line, lam_team)
            pb = devig_over(o_over, o_under)
            ev_over = po * o_over - 1
            ev_under = (1 - po) * o_under - 1
            if ev_over >= ev_under:
                best = f"OVER  {ev_over * 100:+5.1f}%"
            else:
                best = f"UNDER {ev_under * 100:+5.1f}%"
            flag = "  <<<" if max(ev_over, ev_under) > 0.05 else ""
            print(f"  {line:>6} | {o_over:>4}/{o_under:<5} | {po * 100:>11.1f}% | {pb * 100:>11.1f}% | {best:>14}{flag}")

    print("\n" + "=" * 78)
    print("  LECTURE :")
    print("  - Si nos lam ~ lignes du book -> book SHARP, pas d'edge (lead mort).")
    print("  - Si écart net + EV>5% cohérents -> ligne PARESSEUSE, edge candidat.")
    print("  - 1 match ne PROUVE rien : à vérifier sur le résultat réel (forward test).")
    print("=" * 78)


if __name__ == "__main__":
    main()

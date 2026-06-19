"""Total kills par patch : la seule faille kills résiduelle = lignes du book qui traînent
après un changement de méta (patch plus violent -> plus de kills). Si le swing patch->patch
est gros, parier "over" sur les 1ers matchs d'un patch agressif peut avoir un edge temporaire.

Usage : python -m src.models.kills_by_patch
"""
from __future__ import annotations

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config


def main() -> None:
    cfg = load_config()
    m = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "matches.parquet")
    g = m.groupby("patch").agg(games=("y_total_kills", "size"),
                               kills_moy=("y_total_kills", "mean"),
                               kills_std=("y_total_kills", "std")).reset_index()
    g = g[g["games"] >= 5].sort_values("patch")
    print("=" * 56)
    print("  TOTAL KILLS PAR PATCH (LCK 2026, patchs >=5 games)")
    print("=" * 56)
    print(f"  {'patch':<8} {'games':>6} {'kills_moy':>10} {'std':>7}")
    print("  " + "-" * 52)
    for _, r in g.iterrows():
        print(f"  {r['patch']:<8} {int(r['games']):>6} {r['kills_moy']:>10.1f} {r['kills_std']:>7.1f}")
    swing = g["kills_moy"].max() - g["kills_moy"].min()
    print("=" * 56)
    print(f"  Swing moy entre patchs : {swing:.1f} kills  (std intra-patch ~{g['kills_std'].mean():.1f})")
    print(f"  -> swing {'< std' if swing < g['kills_std'].mean() else '>= std'} : "
          f"{'effet patch NOYÉ dans le bruit' if swing < g['kills_std'].mean() else 'effet patch visible'}")
    print("=" * 56)


if __name__ == "__main__":
    main()

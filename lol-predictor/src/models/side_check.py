"""Diagnostic : le modèle sur-pondère-t-il l'avantage de side (bleu) ?

Pour chaque paire d'équipes, on compare la proba que A gagne quand A est BLEU vs
quand A est ROUGE. L'avantage de side implicite du modèle =
    (P(A|A bleu) + P(B|B bleu) - 1) / 2
On le compare à l'avantage de side RÉEL des données (winrate bleu - 50%).

Usage :
    python -m src.models.side_check
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config
from src.models.predict import MatchPredictor


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    m = pd.read_parquet(proc / "matches.parquet")
    real_blue_wr = m["y_winner"].mean()
    real_side_adv = real_blue_wr - 0.5

    mp = MatchPredictor().fit(cfg)
    empty: dict = {}

    diffs = []
    for a, b in combinations(mp.teams, 2):
        pa_blue = mp.predict_match(a, b, empty, empty)["winner"]["blue"]
        pb_blue = mp.predict_match(b, a, empty, empty)["winner"]["blue"]
        diffs.append((pa_blue + pb_blue - 1.0) / 2.0)

    model_side_adv = sum(diffs) / len(diffs)

    print("=" * 60)
    print("  DIAGNOSTIC AVANTAGE DE SIDE (bleu)")
    print("=" * 60)
    print(f"  Réel (données LCK)   : winrate bleu {real_blue_wr*100:.1f}%  -> +{real_side_adv*100:.1f} pts")
    print(f"  Modèle (moyenne paires) : +{model_side_adv*100:.1f} pts")
    print(f"  Sur-pondération        : x{model_side_adv/real_side_adv:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

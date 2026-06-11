"""Mise à jour quotidienne : data fraîche + tables + watchlist pré-match.

Étapes :
  1. Télécharge le CSV Oracle's Elixir 2026 le plus récent (Google Drive).
  2. Régénère les tables d'analyse (matches/team_games parquet) — garde le reste du projet à jour.
  3. Génère WATCHLIST.md : notre proba Elo sur les matchs des prochains jours.

À planifier 1×/jour (Task Scheduler Windows — voir README).

Usage :
    python -m src.update.daily
    python -m src.update.daily --days 5 --no-download   (options)
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="MAJ quotidienne LoL predictor")
    parser.add_argument("--days", type=int, default=7, help="fenêtre watchlist (jours)")
    parser.add_argument("--no-download", action="store_true", help="ne pas retélécharger la data")
    parser.add_argument("--no-rebuild", action="store_true", help="ne pas régénérer les parquet")
    args = parser.parse_args()

    if not args.no_download:
        print("\n=== 1/3  Téléchargement data (Google Drive) ===")
        try:
            from src.update.download_data import download_latest
            download_latest()
        except Exception as exc:  # noqa: BLE001 - on continue avec la data locale
            print(f"[warn] download échoué ({exc}) -> on garde la data locale.")

    if not args.no_rebuild:
        print("\n=== 2/3  Régénération des tables (matches/team_games) ===")
        try:
            from src.ingest import build_match_table
            build_match_table.main()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] rebuild échoué ({exc}).")

    print("\n=== 3/3  Watchlist pré-match ===")
    from src.update.watchlist import generate
    try:
        res = generate(days=args.days)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] calendrier indisponible ({exc}). Watchlist non générée.")
        return 1

    cov, unc = res["covered"], res["uncovered"]
    print(f"\nOK -> {res['path']}  ({len(cov)} matchs couverts, {len(unc)} non couverts)")
    strong = [r for r in cov if r.get("strong")]
    if strong:
        print(f"\n  {len(strong)} matchs a fort penchant FIABLE (*, a surveiller pour la value) :")
        for r in strong[:15]:
            fav, p = (r["team1"], r["p1"]) if r["p1"] >= 0.5 else (r["team2"], r["p2"])
            print(f"   * {r['when']:14} {r['league']:6} {fav} {p*100:.0f}%  (vs "
                  f"{r['team2'] if fav == r['team1'] else r['team1']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

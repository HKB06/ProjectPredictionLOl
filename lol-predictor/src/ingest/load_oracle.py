"""Chargement et filtrage des données Oracle's Elixir.

Lit le CSV brut, filtre selon config.yaml (ligues, année, date_min) et affiche
un résumé qui sert de sanity-check vs Gol.gg (nb games, winrate bleu, durée moyenne).

Usage :
    python -m src.ingest.load_oracle
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_oracle(cfg: dict) -> pd.DataFrame:
    """Charge le CSV Oracle's Elixir et applique les filtres de périmètre."""
    csv_path = ROOT / cfg["data"]["oracle_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {csv_path}\n"
            "-> Télécharge le fichier sur oracleselixir.com (Tools > Downloads) "
            "et place-le dans data/raw/."
        )

    # patch lu en texte pour ne pas perdre le zéro final (16.10 != 16.1)
    df = pd.read_csv(csv_path, low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]

    # Filtres de périmètre
    leagues = cfg["scope"].get("leagues")
    if leagues:
        df = df[df["league"].isin(leagues)]
    year = cfg["scope"].get("year")
    if year:
        df = df[df["year"] == year]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_min = cfg["scope"].get("date_min")
    if date_min:
        df = df[df["date"] >= pd.to_datetime(date_min)]

    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    """Affiche un résumé à comparer aux chiffres Gol.gg."""
    team_rows = df[df["position"].str.lower() == "team"].copy()

    n_games = df["gameid"].nunique()
    games = team_rows.drop_duplicates("gameid")
    avg_len_min = games["gamelength"].mean() / 60 if "gamelength" in games else float("nan")

    blue = team_rows[team_rows["side"].str.lower() == "blue"]
    blue_wr = blue["result"].mean() * 100 if len(blue) else float("nan")

    date_min = df["date"].min()
    date_max = df["date"].max()

    print("=" * 56)
    print("  RÉSUMÉ ORACLE'S ELIXIR (sanity-check vs Gol.gg)")
    print("=" * 56)
    print(f"  Ligues          : {sorted(df['league'].unique())}")
    print(f"  Nombre de games : {n_games}")
    print(f"  Période         : {date_min:%Y-%m-%d} -> {date_max:%Y-%m-%d}")
    print(f"  Durée moyenne   : {int(avg_len_min)}:{int((avg_len_min % 1) * 60):02d}")
    print(f"  Winrate BLEU    : {blue_wr:.1f}%  (rouge : {100 - blue_wr:.1f}%)")
    print(f"  Lignes totales  : {len(df)}  (12/game : {len(df) / max(n_games, 1):.1f})")
    print("=" * 56)
    print("  Rappel cible Gol.gg : 329 games, 56.8% bleu, 31:55")
    print("=" * 56)


def main() -> None:
    cfg = load_config()
    df = load_oracle(cfg)
    summarize(df)


if __name__ == "__main__":
    main()

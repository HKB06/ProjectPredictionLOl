"""Ligues « pariables » : celles où l'on peut miser ~100 € par match sans faire bouger le prix.

Le juge est Polymarket : un match dont le marché a échangé quelques milliers de dollars ne
supporte pas une mise de 100 € sans glisser, et une stratégie réglée sur ces matchs-là ne
tiendrait pas à mise réelle. La liste retenue vit dans `config.yaml` (`betting`), et sert à :
  - apprendre l'Elo (picks 🎯/⭐, rejeu des paris, Plan de jeu) sur ces ligues seulement
    (`betting.elo_scope`) ;
  - n'afficher de pick ou de value que sur un match pariable (`is_bettable`).

Ce module recalcule aussi le classement à partir des données (marchés Polymarket reliés aux
séries jouées) et propose une liste, à recopier dans `config.yaml` si elle a changé.

Usage :
    python -m src.update.bet_universe            # classement + liste proposée
"""
from __future__ import annotations

import sys

import pandas as pd

from src.ingest.load_oracle import load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Critères « tranquille à 100 € » (≈ 115 $ : moins de 0,25 % du volume médian)
MIN_MEDIAN = 50_000       # $ échangés sur le match, médiane de la saison
MIN_P25 = 20_000          # ... et au moins ça pour 3 matchs sur 4
MIN_COVERAGE = 0.70       # part des séries jouées qui avaient un marché
MIN_SERIES = 10           # en dessous, pas assez de matchs pour juger


def universe(cfg: dict | None = None) -> tuple[set[str], set[str]]:
    """(ligues, événements) pariables, d'après `config.yaml`. Vide = pas de restriction."""
    b = (cfg or load_config()).get("betting") or {}
    return set(b.get("leagues") or []), set(b.get("events") or [])


def is_bettable(code: str | None, home_a: str | None, home_b: str | None,
                cfg: dict | None = None, leagues: set | None = None,
                events: set | None = None) -> bool:
    """Un match est pariable si les deux équipes viennent de ligues pariables et que la
    compétition elle-même l'est (ligue ou événement de la liste ; inconnue = on s'en tient
    aux équipes)."""
    if leagues is None or events is None:
        leagues, events = universe(cfg)
    if not leagues:
        return True
    if home_a not in leagues or home_b not in leagues:
        return False
    if code and code != "?" and code not in leagues | events:
        return False
    return True


def ranking() -> pd.DataFrame:
    """Une ligne par ligue/événement de la saison : séries, part cotée, volume médian."""
    from src.v2.market import attach_market
    from src.v2.precompute import get

    eng = get()
    s, g = eng["series"], eng["games"]
    s = attach_market(s, g, int(pd.to_datetime(s["date"]).dt.year.max()))
    s = s[s["winner"].notna()]
    t = s.groupby("league").agg(
        series=("series_id", "size"),
        cotees=("mkt_p", lambda x: int(x.notna().sum())),
        vol_median=("mkt_volume", "median"),
        vol_p25=("mkt_volume", lambda x: x.quantile(0.25)),
    )
    t["couverture"] = t["cotees"] / t["series"]
    t["ok"] = ((t["series"] >= MIN_SERIES) & (t["couverture"] >= MIN_COVERAGE)
               & (t["vol_median"] >= MIN_MEDIAN) & (t["vol_p25"] >= MIN_P25))
    return t.sort_values("vol_median", ascending=False)


def main() -> None:
    from src.update.elo import EVENT_LEAGUES

    t = ranking()
    pd.set_option("display.width", 200)
    show = t.assign(couverture=(100 * t["couverture"]).round(0),
                    vol_median=t["vol_median"].round(0), vol_p25=t["vol_p25"].round(0))
    print(show.to_string())
    ok = set(t.index[t["ok"]])
    leagues, events = universe()
    prop_l = sorted(ok - EVENT_LEAGUES)
    prop_e = sorted(ok & EVENT_LEAGUES)
    print(f"\nCritères : volume médian ≥ {MIN_MEDIAN:,} $, ≥ {MIN_P25:,} $ pour 3 matchs sur 4, "
          f"≥ {MIN_COVERAGE:.0%} des séries cotées")
    print(f"Ligues proposées     : {prop_l}")
    print(f"Événements proposés  : {prop_e}")
    print(f"Ligues en config     : {sorted(leagues)}")
    print(f"Événements en config : {sorted(events)}")
    add, drop = set(prop_l) - leagues, leagues - set(prop_l)
    if add or drop:
        print(f"→ à revoir dans config.yaml : ajouter {sorted(add) or '-'}, retirer {sorted(drop) or '-'}")
    else:
        print("→ la liste des ligues de config.yaml est à jour")


if __name__ == "__main__":
    main()

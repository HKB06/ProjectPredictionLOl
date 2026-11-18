"""Données Leaguepedia (lol.fandom.com) absentes du CSV Oracle : staff, lieu, comptes soloQ.

Ce que ça apporte, par table :
  - `tournaments` : chaque compétition de l'année, avec `EventType` (Online / Offline = LAN),
                    pays, région, niveau.
  - `games`       : chaque game, avec l'identifiant Riot (= `gameid` Oracle) et la page du
                    tournoi -> relie une game Oracle à son tournoi Leaguepedia.
  - `rosters`     : joueurs ET COACHS inscrits par équipe et par tournoi.
  - `coach_roles` : mouvements de staff (Head Coach / Assistant…), pour savoir qui dirige.
  - `players`     : fiche de chaque joueur inscrit, dont ses comptes soloQ connus.

Accès : `Special:CargoExport`. L'API `cargoquery` classique est inutilisable sans compte
(2 requêtes/minute partagées entre TOUS les anonymes) ; l'export, lui, répond normalement.
On reste poli : une requête à la fois, 1,5 s de pause, 5 000 lignes par page.

Cache : `data/external/leaguepedia/<table>_<année>.csv`. Versionné (petit, et l'app
déployée ne doit pas dépendre du site au démarrage).

Usage :
    python -m src.ingest.leaguepedia              # télécharge ce qui manque
    python -m src.ingest.leaguepedia --refresh    # retélécharge tout (nouveaux matchs)
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import sys
import time

import pandas as pd
import requests

from src.ingest.load_oracle import ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "https://lol.fandom.com/wiki/Special:CargoExport"
HEADERS = {"User-Agent": "lol-predictor/0.2 (projet perso, faible volume)"}
PAGE = 5000
PAUSE = 1.5
TIMEOUT = 120
SHRINK_GUARD = 0.9        # un rafraîchissement qui perd plus de 10 % des lignes est refusé
DIR = ROOT / "data" / "external" / "leaguepedia"

_session = requests.Session()
_session.headers.update(HEADERS)


def cargo(tables: str, fields: str, where: str = "", join_on: str = "",
          group_by: str = "", order_by: str = "") -> pd.DataFrame:
    """Requête Cargo paginée -> DataFrame (colonnes = alias demandés)."""
    rows: list[dict] = []
    offset = 0
    while True:
        params = {"tables": tables, "fields": fields, "limit": PAGE, "offset": offset,
                  "format": "json"}
        for key, val in (("where", where), ("join on", join_on),
                         ("group by", group_by), ("order by", order_by)):
            if val:
                params[key] = val
        batch = None
        for attempt in range(4):
            try:
                r = _session.get(URL, params=params, timeout=TIMEOUT)
                batch = r.json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(5 * (attempt + 1))
        time.sleep(PAUSE)
        if not isinstance(batch, list):
            raise RuntimeError(f"Leaguepedia : réponse inattendue pour {tables} ({str(batch)[:200]})")
        rows += batch
        if len(batch) < PAGE:
            break
        offset += PAGE
    df = pd.DataFrame(rows)
    df = df[[c for c in df.columns if "__" not in c]]          # colonnes techniques Cargo
    df.columns = [c.replace(" ", "_") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: html.unescape(v) if isinstance(v, str) else v)
    return df


# ------------------------------------------------------------------ les tables
def _tournaments(year: int) -> pd.DataFrame:
    return cargo("Tournaments",
                 "Name,OverviewPage,League,Region,Country,EventType,DateStart,Date,"
                 "TournamentLevel,IsPlayoffs,IsQualifier,Split",
                 where=f"Year={year}")


def _games(year: int) -> pd.DataFrame:
    return cargo("ScoreboardGames",
                 "OverviewPage,Team1,Team2,WinTeam,DateTime_UTC,RiotPlatformGameId,GameId,"
                 "MatchId,N_GameInMatch",
                 where=f"DateTime_UTC >= '{year}-01-01' AND DateTime_UTC < '{year + 1}-01-01'",
                 order_by="DateTime_UTC")


def _rosters(year: int) -> pd.DataFrame:
    return cargo("TournamentPlayers=TP,Tournaments=T",
                 "TP.OverviewPage=OverviewPage,TP.Team=Team,TP.Player=Player,TP.Role=Role,"
                 "TP.N_PlayerInTeam=N",
                 where=f"T.Year={year}", join_on="TP.OverviewPage=T.OverviewPage",
                 order_by="TP.OverviewPage,TP.Team,TP.N_PlayerInTeam")


def _coach_roles(year: int) -> pd.DataFrame:
    """Arrivées de staff depuis 3 ans : sert à distinguer le head coach des assistants."""
    return cargo("RosterChanges", "Date_Sort,Player,Team,Role,Direction",
                 where=f"Date_Sort >= '{year - 3}-01-01' AND Role LIKE '%Coach%' "
                       "AND Direction='Join'",
                 order_by="Date_Sort")


def _players(year: int) -> pd.DataFrame:
    return cargo("TournamentPlayers=TP,Tournaments=T,Players=P",
                 "P.OverviewPage=Page,P.ID=ID,P.Role=Role,P.Country=Country,"
                 "P.Residency=Residency,P.SoloqueueIds=SoloqueueIds",
                 where=f"T.Year={year} AND P.OverviewPage IS NOT NULL",
                 join_on="TP.OverviewPage=T.OverviewPage,TP.Player=P.OverviewPage",
                 group_by="P.OverviewPage")


TABLES = {"tournaments": _tournaments, "games": _games, "rosters": _rosters,
          "coach_roles": _coach_roles, "players": _players}
# Colonnes attendues : une table sans aucune ligne (tout début de saison) garde son en-tête.
COLUMNS = {
    "tournaments": ["Name", "OverviewPage", "League", "Region", "Country", "EventType",
                    "DateStart", "Date", "TournamentLevel", "IsPlayoffs", "IsQualifier", "Split"],
    "games": ["OverviewPage", "Team1", "Team2", "WinTeam", "DateTime_UTC", "RiotPlatformGameId",
              "GameId", "MatchId", "N_GameInMatch"],
    "rosters": ["OverviewPage", "Team", "Player", "Role", "N"],
    "coach_roles": ["Date_Sort", "Player", "Team", "Role", "Direction"],
    "players": ["Page", "ID", "Role", "Country", "Residency", "SoloqueueIds"],
}


def path(name: str, year: int):
    return DIR / f"{name}_{year}.csv"


def load(name: str, year: int, refresh: bool = False) -> pd.DataFrame:
    """Table en cache ; téléchargée si absente (ou si `refresh`)."""
    p = path(name, year)
    if p.exists() and not refresh:
        try:
            return pd.read_csv(p, low_memory=False)
        except pd.errors.EmptyDataError:          # ancien cache sans en-tête : table vide
            return pd.DataFrame(columns=COLUMNS[name])
    df = TABLES[name](year)
    if df.empty:                                   # 1er janvier : aucun tournoi de l'année
        df = pd.DataFrame(columns=COLUMNS[name])
    if p.exists():
        # Une réponse tronquée du site ne doit jamais écraser un cache sain : ces tables
        # ne font que grossir au fil de la saison.
        old = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
        if len(df) < SHRINK_GUARD * old:
            raise RuntimeError(f"Leaguepedia : {name} ne contient que {len(df)} lignes "
                               f"(cache : {old}). Réponse incomplète, cache conservé.")
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    return df


THIN = 0.5      # une saison qui a moins de la moitié des joueurs de la précédente débute à peine


def season_years(year: int) -> list[int]:
    """[année] en temps normal ; [année - 1, année] tant que la saison débute (janvier-février :
    presque aucun tournoi de l'année n'existe, la liste des joueurs serait presque vide).
    Dès que la saison est lancée, la précédente n'est plus ajoutée (joueurs retraités)."""
    if not path("players", year - 1).exists():
        return [year]
    cur = load("players", year)
    prev = load("players", year - 1)
    return [year - 1, year] if len(cur) < THIN * len(prev) else [year]


def load_season(name: str, year: int) -> pd.DataFrame:
    """Table de la saison, complétée par la précédente en début de saison (`season_years`)."""
    frames = [load(name, y) for y in season_years(year)]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def available(year: int) -> bool:
    """Toutes les tables sont-elles en cache ? (l'app n'appelle jamais le site elle-même)"""
    return all(path(n, year).exists() for n in TABLES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Télécharge les tables Leaguepedia utiles")
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--refresh", action="store_true", help="retélécharge même si en cache")
    ap.add_argument("--only", nargs="*", choices=list(TABLES), help="tables à traiter")
    args = ap.parse_args()
    for name in args.only or TABLES:
        t0 = time.time()
        df = load(name, args.year, refresh=args.refresh)
        print(f"{name:12} {len(df):6} lignes  ({time.time() - t0:.0f}s)  -> {path(name, args.year).name}")


if __name__ == "__main__":
    main()

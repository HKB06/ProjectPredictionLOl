"""Récupère les matchs LoL à venir via l'API officielle lolesports (Riot).

Source fiable (clé publique connue, peu de rate-limit) couvrant les ligues Riot :
majeures (LCK/LPL/LEC/LCS/LCP), CBLOL, EMEA Masters, et toutes les régionales EMEA
(LFL, Prime League, TCL, NLC, LES, HLL, EBL...) + LJL, VCS, PCS, LLA, LTA...

Limite connue : les événements TIERS hors-Riot (ex. *EWC OQ South America*) n'y sont
pas → pour ceux-là, on saisit le match à la main (comme avant).

Usage :
    python -m src.update.lolesports
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "https://esports-api.lolesports.com/persisted/gw"
# Clé publique du site lolesports (lecture seule, utilisée par toutes les apps tierces).
API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
HEADERS = {"x-api-key": API_KEY}
SKIP_SLUGS = {"tft_esports"}  # pas du LoL
# On garde aussi les matchs déjà commencés/terminés AUJOURD'HUI (le front les affiche
# grisés au lieu de les supprimer) : la borne basse de la fenêtre = minuit (Paris).
PARIS_OFFSET = dt.timedelta(hours=2)  # CEST (été)
ALLOWED_STATES = {"unstarted", "inProgress", "completed"}


def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE}/{path}", params={**(params or {}), "hl": "en-US"},
                        headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def get_league_ids() -> list[str]:
    leagues = _get("getLeagues")["leagues"]
    return [lg["id"] for lg in leagues if lg.get("slug") not in SKIP_SLUGS]


def _parse_events(events: list[dict], start_bound: dt.datetime, end: dt.datetime) -> list[dict]:
    out: list[dict] = []
    for e in events:
        if e.get("state") not in ALLOWED_STATES:
            continue
        match = e.get("match")
        if not match:
            continue
        teams = match.get("teams", [])
        if len(teams) != 2:
            continue
        t1, t2 = teams[0].get("name"), teams[1].get("name")
        if not t1 or not t2 or t1.upper() in ("TBD", "TBA") or t2.upper() in ("TBD", "TBA"):
            continue
        try:
            start = dt.datetime.fromisoformat(e["startTime"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if start < start_bound or start > end:
            continue
        out.append({
            "team1": t1,
            "team2": t2,
            "datetime": start.strftime("%Y-%m-%d %H:%M:%S"),
            "bestof": match.get("strategy", {}).get("count", 1),
            "overview": e.get("league", {}).get("name"),
            "tournament": e.get("league", {}).get("name"),
        })
    return out


def _league_events(lid: str) -> list[dict]:
    try:
        return _get("getSchedule", {"leagueId": lid}).get("schedule", {}).get("events", [])
    except Exception:  # noqa: BLE001 - ligue dormante / erreur transitoire -> on saute
        return []


def fetch_upcoming(days: int = 7, league_ids: list[str] | None = None,
                   max_workers: int = 8) -> list[dict]:
    """Matchs `unstarted` programmés entre maintenant et +`days` jours, TOUTES ligues.

    On interroge ligue par ligue (en parallèle) : le getSchedule global est plafonné
    (~80 events) et se fait saturer par les ligues très actives (EMEA Masters), ce qui
    masque LEC/CBLOL/régionales.

    Retour : liste de dicts {team1, team2, datetime (UTC str), bestof, overview, tournament}.
    """
    now = dt.datetime.now(dt.timezone.utc)
    end = now + dt.timedelta(days=days)
    # Borne basse = minuit AUJOURD'HUI (Paris) : garde les matchs du jour déjà lancés/finis.
    today_paris = (now + PARIS_OFFSET).date()
    start_bound = dt.datetime.combine(today_paris, dt.time(0, 0),
                                      tzinfo=dt.timezone.utc) - PARIS_OFFSET
    ids = league_ids or get_league_ids()

    all_events: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for events in ex.map(_league_events, ids):
            all_events.extend(events)

    seen: set[tuple] = set()
    out: list[dict] = []
    for rec in _parse_events(all_events, start_bound, end):
        key = (rec["team1"], rec["team2"], rec["datetime"])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    out.sort(key=lambda r: r["datetime"])
    return out


def main() -> None:
    matches = fetch_upcoming(days=7)
    print(f"{len(matches)} matchs à venir (7 j) :")
    for m in matches:
        print(f"  {m['datetime']}  [{m['overview']}]  {m['team1']} vs {m['team2']}  (BO{m['bestof']})")


if __name__ == "__main__":
    main()

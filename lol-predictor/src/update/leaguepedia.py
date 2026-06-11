"""Récupère les matchs LoL à venir via l'API Cargo de Leaguepedia (lol.fandom.com).

Source gratuite et officielle du calendrier esport LoL (toutes ligues). On lit la
table `MatchSchedule` (équipes, date UTC, BO, tournoi) sur une fenêtre de N jours.

Usage :
    python -m src.update.leaguepedia
"""
from __future__ import annotations

import datetime as dt
import time

import requests

API = "https://lol.fandom.com/api.php"
# Leaguepedia exige un User-Agent identifiable.
HEADERS = {"User-Agent": "lol-predictor/1.0 (projet perso recherche; contact: local)"}


class LeaguepediaError(RuntimeError):
    pass


def _query(params: dict, retries: int = 4) -> list[dict]:
    """Appel Cargo avec gestion d'erreur + backoff sur rate-limit."""
    delay = 5.0
    for attempt in range(retries):
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            code = data["error"].get("code", "?")
            if code == "ratelimited" and attempt < retries - 1:
                print(f"[leaguepedia] rate-limit, nouvelle tentative dans {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise LeaguepediaError(f"{code}: {data['error'].get('info', '')}")
        return data.get("cargoquery", [])
    raise LeaguepediaError("rate-limit persistant après plusieurs tentatives")


def fetch_upcoming(days: int = 7, limit: int = 500) -> list[dict]:
    """Matchs programmés entre maintenant et +`days` jours.

    Retour : liste de dicts {team1, team2, datetime (UTC str), bestof, overview, tournament}.
    """
    now = dt.datetime.now(dt.timezone.utc)
    end = now + dt.timedelta(days=days)
    where = (
        f"MatchSchedule.DateTime_UTC >= '{now:%Y-%m-%d %H:%M:%S}' "
        f"AND MatchSchedule.DateTime_UTC <= '{end:%Y-%m-%d %H:%M:%S}'"
    )
    params = {
        "action": "cargoquery",
        "format": "json",
        "limit": str(limit),
        "tables": "MatchSchedule",
        "fields": (
            "MatchSchedule.Team1=team1,"
            "MatchSchedule.Team2=team2,"
            "MatchSchedule.DateTime_UTC=datetime,"
            "MatchSchedule.BestOf=bestof,"
            "MatchSchedule.OverviewPage=overview,"
            "MatchSchedule.Tournament=tournament"
        ),
        "where": where,
        "order_by": "MatchSchedule.DateTime_UTC ASC",
    }
    rows = _query(params)

    out: list[dict] = []
    for item in rows:
        t = item.get("title", {})
        t1, t2 = t.get("team1"), t.get("team2")
        if not t1 or not t2:
            continue
        if t1.upper() in ("TBD", "TBA") or t2.upper() in ("TBD", "TBA"):
            continue
        out.append({
            "team1": t1,
            "team2": t2,
            "datetime": t.get("datetime"),
            "bestof": t.get("bestof"),
            "overview": t.get("overview"),
            "tournament": t.get("tournament"),
        })
    return out


def main() -> None:
    try:
        matches = fetch_upcoming(days=7)
    except LeaguepediaError as exc:
        print(f"[leaguepedia] ERREUR : {exc}")
        return
    print(f"{len(matches)} matchs à venir (7 j) :")
    for m in matches[:40]:
        print(f"  {m['datetime']}  [{m['overview']}]  {m['team1']} vs {m['team2']}  (BO{m['bestof']})")


if __name__ == "__main__":
    main()

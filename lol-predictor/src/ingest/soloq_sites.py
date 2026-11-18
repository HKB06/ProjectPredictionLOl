"""Comptes soloQ des pros, smurfs compris, lus sur deux sites qui les répertorient.

  - TrackingThePros (trackingthepros.com) : pros actifs d'Europe, Corée, Amérique du Nord,
    Brésil… Chaque fiche liste tous les comptes connus du joueur (actifs et inactifs).
  - lolpros.gg : la plus grosse base européenne, semi-pros des ligues régionales compris,
    tenue par des managers et coachs de chaque ligue (comptes mis à jour toutes les 6 h).

TrackingThePros n'a pas d'API officielle : on lit les mêmes données que ses pages, sans les
charger (une requête toutes les `PAUSE` secondes, une fois par semaine par joueur, et
seulement pour les joueurs dont le nom correspond à un pro connu de Leaguepedia).

lolpros.gg répond 429 à un usage automatisé et demande de les contacter (Twitter
@LOLProsGG ou Discord) pour obtenir une clé API : la source est DÉSACTIVÉE (`SITES`) tant
qu'on n'a pas cette clé. Le code de lecture reste prêt.

Un site qui répond 429 est mis en pause pour le reste du passage, sans bloquer l'autre.
Les rangs et les parties viennent ensuite de l'API Riot, comme pour les autres comptes.

On ne garde que les comptes ACTIFS (marqués actifs sur TrackingThePros, avec des parties
classées cette saison sur lolpros.gg) : un pro a parfois une douzaine de vieux comptes.

Correspondance site -> pro de Leaguepedia (même échelle que `soloq_ladder`) :
  fort    même nom ET même équipe
  moyen   même nom, un seul pro de ce nom, et le site n'indique pas d'équipe
  ambigu  plusieurs pros de ce nom sans équipe pour trancher : ignoré
  faible  même nom mais autre équipe : ignoré
Seuls « fort » et « moyen » rejoignent le relevé (`soloq.tracked_accounts`).

Le travail est découpé (cache sur disque) : chaque passage avance `minutes` puis s'arrête,
le suivant reprend là où il en était.

Usage :
    python -m src.ingest.soloq_sites                 # un passage de 30 min
    python -m src.ingest.soloq_sites --minutes 120
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time

import pandas as pd
import requests

from src.ingest import soloq as sq
from src.ingest import soloq_ladder as sl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = sq.DIR / "sites"
PAUSE = 2.5               # secondes entre deux requêtes vers un même site
TTL_DAYS = 7              # une fiche joueur est relue au plus une fois par semaine
TIMEOUT = 30
HEADERS = {"User-Agent": "lol-predictor/1.0 (personal research; weekly, rate-limited)"}

SITES = ["ttp"]          # + "lolpros" une fois leur clé API obtenue (voir plus haut)

TTP = "https://www.trackingthepros.com"
LOLPROS = "https://api.lolpros.gg/es"
_TTP_ROW = re.compile(r"<tr( class='inactive_account')?\s*><td><b>\[(\w+)\]</b>\s*([^<]+)</td>")
_TTP_SLUG = re.compile(r"/player/([^\"'\\]+)")


def sites_path(year: int):
    return sq.DIR / f"sites_{year}.csv"


class RateLimited(RuntimeError):
    """Le site demande de ralentir (429) : on le laisse tranquille jusqu'au prochain passage."""


class Unavailable(RuntimeError):
    """Le site ne répond pas (erreurs 5xx répétées) : rien n'est enregistré, on réessaiera."""


class _Http:
    """Une session par site, espacée de `PAUSE` secondes par site."""

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.last: dict[str, float] = {}
        self.calls = 0

    def get(self, site: str, url: str, **params):
        wait = PAUSE - (time.monotonic() - self.last.get(site, 0.0))
        if wait > 0:
            time.sleep(wait)
        self.last[site] = time.monotonic()
        self.calls += 1
        for attempt in range(3):
            r = self.s.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                raise RateLimited(f"{site} : 429 ({r.text[:150]})")
            if r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        raise Unavailable(f"{site} : erreurs serveur répétées")


def _load(name: str) -> dict:
    p = DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(name: str, data: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = DIR / f"{name}.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(DIR / f"{name}.json")


def _stale(stamp: str | None) -> bool:
    return not stamp or stamp < (dt.date.today() - dt.timedelta(days=TTL_DAYS)).isoformat()


# ----------------------------------------------------------------- listes de joueurs
def _ttp_list(http: _Http, cache: dict):
    """Liste des joueurs TrackingThePros (≈ 2 300, 5 requêtes). Une requête par étape."""
    if not _stale(cache.get("list_date")):
        return
    players, start = [], 0
    while True:
        r = http.get("ttp", f"{TTP}/d/list_players", draw=1, start=start, length=500)
        yield
        if r is None:                              # page introuvable : on garde l'ancienne liste
            return
        data = r.json().get("data") or []
        for d in data:
            m = _TTP_SLUG.search(d.get("player_name") or "")
            if m:
                players.append({"slug": m.group(1), "name": d.get("name"),
                                "team": d.get("team_name"), "region": d.get("home_region")})
        if len(data) < 500:
            break
        start += 500
    # une liste qui fond d'un coup = réponse incomplète : on garde la précédente
    if len(players) < 0.9 * len(cache.get("list") or []):
        return
    cache["list"], cache["list_date"] = players, dt.date.today().isoformat()


def _lolpros_list(http: _Http, cache: dict):
    """Classement lolpros.gg page par page (20 joueurs par page) ; reprenable."""
    if not _stale(cache.get("list_date")):
        return
    page = cache.get("next_page", 1)
    building = cache.setdefault("building", [])
    while True:
        r = http.get("lolpros", f"{LOLPROS}/ladder", page=page, sort="rank", order="desc")
        yield
        data = r.json() if r else []
        if not data:
            cache["list"], cache["list_date"] = building, dt.date.today().isoformat()
            cache.pop("building", None)
            cache.pop("next_page", None)
            return
        for d in data:
            building.append({"slug": d.get("slug"), "name": d.get("name"),
                             "team": (d.get("team") or {}).get("name"),
                             "leagues": [lg.get("shorthand") for lg in d.get("leagues") or []]})
        page += 1
        cache["next_page"] = page


# ------------------------------------------------------------------ fiches joueurs
def _ttp_accounts(http: _Http, slug: str) -> list[dict] | None:
    r = http.get("ttp", f"{TTP}/player/{slug}")
    if r is None:
        return None
    out = []
    for inactive, region, rid in _TTP_ROW.findall(r.text):
        name, _, tag = rid.strip().rpartition("#")
        out.append({"region": region, "game_name": name, "tag": tag, "active": not inactive})
    return out


def _lolpros_accounts(http: _Http, slug: str) -> list[dict] | None:
    r = http.get("lolpros", f"{LOLPROS}/profiles/{slug}")
    if r is None:
        return None
    accs = ((r.json() or {}).get("league_player") or {}).get("accounts") or []
    seasons = [s.get("id", "") for a in accs for s in a.get("seasons") or []]
    current = max(seasons) if seasons else ""
    out = []
    for a in accs:
        cur = next((s for s in a.get("seasons") or [] if s.get("id") == current), None)
        end = (cur or {}).get("end") or {}
        played = (end.get("wins") or 0) + (end.get("losses") or 0)
        out.append({"region": a.get("server"), "game_name": a.get("gamename"),
                    "tag": a.get("tagline"), "active": played > 0})
    return out


# ------------------------------------------------------------ correspondance pros
def _team_match(site_team, pro: dict) -> bool:
    if not isinstance(site_team, str) or not site_team.strip() or not isinstance(pro.get("Team"), str):
        return False
    a, b = sl.team_tags(site_team, None), pro["tags"]
    return bool(a & b) or sl.norm(site_team) == sl.norm(pro["Team"])


# région affichée par TrackingThePros -> serveurs où l'on s'attend à trouver le joueur
_SITE_REGION = {"EU": {"EUW", "EUNE", "TR", "ME"}, "KR": {"KR"}, "NA": {"NA", "LAN"},
                "BR": {"BR", "LAS"}, "LLA": {"LAN", "LAS"}, "TR": {"TR", "EUW", "EUNE"},
                "VT": {"VN"}, "JP": {"JP"}, "OCE": {"OCE"}, "RU": {"RU", "EUW", "EUNE"},
                "LCP": {"TW", "VN", "JP", "OCE", "SEA"}, "CN": {"KR"}}


def _region_ok(site_region, pro: dict) -> bool:
    """La région du site colle-t-elle aux serveurs de la ligue du pro ? (inconnue = oui)"""
    servers = _SITE_REGION.get(str(site_region or "").upper())
    return not servers or not pro.get("home") or bool(servers & pro["home"])


def _match(name, team, index: dict, region=None) -> tuple[dict | None, str]:
    """Le site tient les équipes à jour à son rythme, le wiki au sien : une équipe différente
    est fréquente pour le même joueur (transfert, nom de sponsor). D'où : même équipe = fort ;
    sinon un seul pro de ce nom, dans une région cohérente = moyen."""
    cands = index.get(sl.norm(name), [])
    if not cands:
        return None, ""
    same_team = [p for p in cands if _team_match(team, p)]
    if len(same_team) == 1:
        return same_team[0], "fort"
    if len(same_team) > 1:
        return None, "ambigu"
    near = [p for p in cands if _region_ok(region, p)]
    if len(near) == 1:
        return near[0], "moyen"
    return None, ("ambigu" if len(near) > 1 else "faible")


# ------------------------------------------------------------------------ passage
def run(year: int, minutes: float = 30) -> dict:
    deadline = time.time() + minutes * 60
    http = _Http()
    pr = sl.pros(year)
    index: dict[str, list[dict]] = {}
    for p in pr.to_dict("records"):
        index.setdefault(p["key"], []).append(p)

    caches = {s: _load(s) for s in SITES}
    lists = {"ttp": _ttp_list, "lolpros": _lolpros_list}
    fetch = {"ttp": _ttp_accounts, "lolpros": _lolpros_accounts}
    left, empty = {}, {}

    def work(site: str):
        """Une requête par étape : la liste du site si elle est à refaire, puis les fiches
        des joueurs dont le nom correspond à un pro (absentes ou vieilles d'une semaine)."""
        c = caches[site]
        yield from lists[site](http, c)
        prof = c.setdefault("profiles", {})
        todo = [p for p in c.get("list") or []
                if p.get("slug") and sl.norm(p.get("name")) in index
                and _stale((prof.get(p["slug"]) or {}).get("date"))]
        left[site] = len(todo)
        for p in todo:
            accs = fetch[site](http, p["slug"])
            before = (prof.get(p["slug"]) or {}).get("accounts") or []
            # fiche introuvable ou illisible (page changée) : on ne remplace pas des comptes
            # connus par « aucun compte », ils sortiraient du suivi pour une semaine
            if not accs and before:
                accs = before
                empty[site] = empty.get(site, 0) + 1
            prof[p["slug"]] = {"date": dt.date.today().isoformat(), "accounts": accs or []}
            left[site] -= 1
            yield

    gens = {s: work(s) for s in caches}
    steps, limited = 0, []
    try:
        while gens and time.time() < deadline:
            # le site qui peut repartir le plus tôt (pause par site)
            site = min(gens, key=lambda s: http.last.get(s, 0.0))
            try:
                next(gens[site])
            except StopIteration:
                del gens[site]
                continue
            except (RateLimited, Unavailable) as exc:
                limited.append(str(exc))
                del gens[site]
                continue
            steps += 1
            if steps % 50 == 0:
                for k, c in caches.items():
                    _save(k, c)
    finally:
        for k, c in caches.items():
            _save(k, c)
    fetched = sum(1 for c in caches.values() for v in (c.get("profiles") or {}).values()
                  if v.get("date") == dt.date.today().isoformat())

    # table des comptes retenus
    rows, levels = [], {}
    for site, c in caches.items():
        prof = c.get("profiles") or {}
        for p in c.get("list") or []:
            got = prof.get(p.get("slug"))
            if not got:
                continue
            pro, conf = _match(p.get("name"), p.get("team"), index, p.get("region"))
            levels[conf] = levels.get(conf, 0) + 1
            if conf not in ("fort", "moyen"):
                continue
            for a in got["accounts"]:
                region = sq._region(str(a.get("region") or ""))
                if not a.get("active") or region not in sq.PLATFORMS \
                        or not a.get("game_name") or not a.get("tag"):
                    continue
                rows.append({"player": pro["Page"], "id": pro["ID"], "role": pro["Role"],
                             "region": region, "game_name": a["game_name"], "tag": a["tag"],
                             "confidence": conf, "source": site, "team": pro.get("Team")})
    df = pd.DataFrame(rows, columns=["player", "id", "role", "region", "game_name", "tag",
                                     "confidence", "source", "team"])
    df = df.drop_duplicates(["region", "game_name", "tag"])
    if len(df):
        sq.write_csv(df, sites_path(year))
    unfinished = sum(left.values()) + sum(1 for c in caches.values() if c.get("next_page"))
    return {"accounts": len(df), "players": int(df["player"].nunique()) if len(df) else 0,
            "by_source": df["source"].value_counts().to_dict() if len(df) else {},
            "matches": levels, "fetched": fetched, "left": unfinished, "calls": http.calls,
            "limited": limited, "empty": sum(empty.values()),
            "lists_ready": all(c.get("list") for c in caches.values())}


def message(res: dict) -> str:
    left = f", reste {res['left']} pages ou fiches à lire" if res["left"] else ""
    lim = f" · freiné : {'; '.join(res['limited'])}" if res.get("limited") else ""
    if res.get("empty"):
        lim += f" · {res['empty']} fiches revenues vides (anciens comptes gardés)"
    return (f"{res['accounts']} comptes actifs pour {res['players']} pros "
            f"({res['by_source']}) · {res['fetched']} fiches lues aujourd'hui{left}{lim}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Comptes soloQ des pros (TrackingThePros, lolpros.gg)")
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--minutes", type=float, default=30)
    args = ap.parse_args()
    res = run(args.year, args.minutes)
    print(message(res))
    print("Correspondances :", res["matches"], f"· {res['calls']} requêtes")


if __name__ == "__main__":
    main()

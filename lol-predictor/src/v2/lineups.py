"""Effectif du jour : le cinq attendu d'une équipe avant un match.

Sans cela, le pronostic d'avant-match prend le dernier cinq ALIGNÉ : après un transfert, il
continue de compter le joueur parti (et d'ignorer l'arrivant) jusqu'au premier match joué.
Ici, le dernier cinq est corrigé par l'effectif suivi (`src.ingest.rosters`, relevé toutes les
6 h sur Leaguepedia) :

  - un joueur du dernier cinq toujours à l'effectif reste en place (même si un concurrent a
    été recruté à son poste : on ne sait pas qui jouera) ;
  - un joueur qui a quitté l'équipe est remplacé par le joueur de l'effectif à ce poste ;
  - s'il est parti sans remplaçant connu, le poste est tenu par un joueur « inconnu », noté
    au niveau moyen de la ligue : on ne garde pas le niveau de celui qui est parti.

Un arrivant déjà vu en compétition garde sa note, son pool de champions et ses face-à-face :
le lien entre sa fiche Leaguepedia et son identifiant Oracle est appris sur les effectifs par
tournoi des saisons passées (même équipe, même tournoi, même pseudo).

Limite : l'effectif Leaguepedia ne dit pas qui est remplaçant. Ce module ne peut pas être
rejoué sur le passé (les effectifs ne sont relevés que depuis novembre 2026).
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.ingest import leaguepedia as lp
from src.ingest.load_oracle import load_config, read_seasons
from src.update.watchlist import norm
from src.v2.external import link_games

ROLES = ["top", "jng", "mid", "bot", "sup"]
ROSTER_ROLE = {"top": "top", "jungle": "jng", "mid": "mid", "bot": "bot", "support": "sup"}
UNKNOWN = "?:"                     # préfixe d'un poste dont le titulaire est inconnu
NEW = "lp:"                        # préfixe d'un joueur jamais vu en compétition

_roster_cache: dict = {}


def _page_id(page: str) -> str:
    return str(page).split(" (")[0]


def build_index(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Lien fiche Leaguepedia -> identifiant Oracle, et noms Leaguepedia de chaque équipe.

    `df` : jeu de données v2 (gameid, date, blue, red, yb)."""
    cfg = cfg or load_config()
    raw = read_seasons(cfg, {"gameid", "position", "playerid", "playername", "teamname"})
    raw = raw[raw["position"].astype(str).str.lower().isin(ROLES)]
    raw = raw.assign(pid=raw["playerid"].fillna(raw["playername"])).dropna(subset=["pid", "teamname"])
    by_game: dict = defaultdict(lambda: defaultdict(list))
    for gid, team, pid, name in zip(raw["gameid"], raw["teamname"], raw["pid"], raw["playername"]):
        by_game[gid][team].append((pid, name))

    page_pid: dict[str, str] = {}
    team_lp: dict[str, set] = defaultdict(set)
    years = [int(y) for y in sorted(df["date"].dt.year.unique()) if lp.available(int(y))]
    for y in years:
        part = df[df["date"].dt.year == y].sort_values("date")
        link = link_games(part, y).dropna(subset=["page"])
        linked = {g: (p, a, b) for g, p, a, b in zip(link["gameid"], link["page"],
                                                     link["lp_blue"], link["lp_red"])}
        ros = lp.load("rosters", y).dropna(subset=["Player", "Team"])
        roster: dict[tuple, dict] = defaultdict(dict)
        for page, team, player in zip(ros["OverviewPage"], ros["Team"], ros["Player"]):
            roster[(page, team)].setdefault(norm(_page_id(player)), []).append(player)
        for g in part.itertuples():
            if g.gameid not in linked:
                continue
            page, lp_blue, lp_red = linked[g.gameid]
            for team, lp_team in ((g.blue, lp_blue), (g.red, lp_red)):
                if not isinstance(lp_team, str):
                    continue
                team_lp[team].add(lp_team)
                cands = roster.get((page, lp_team), {})
                for pid, name in by_game.get(g.gameid, {}).get(team, ()):
                    hit = cands.get(norm(name), [])
                    if len(hit) == 1:                      # un seul joueur de ce pseudo dans l'équipe
                        page_pid[hit[0]] = pid
    names = dict(zip(raw["pid"], raw["playername"]))
    # pseudo -> identifiants Oracle des deux dernières saisons (secours quand la fiche
    # Leaguepedia n'a pas pu être reliée : accepté seulement si le pseudo est unique)
    recent = raw[raw["season"] >= raw["season"].max() - 1]
    by_name: dict[str, set] = defaultdict(set)
    for pid, name in zip(recent["pid"], recent["playername"]):
        by_name[norm(name)].add(pid)
    return {"page_pid": page_pid, "team_lp": {t: sorted(v) for t, v in team_lp.items()},
            "names": names, "by_name": {k: sorted(v) for k, v in by_name.items()}}


def _roster() -> pd.DataFrame:
    """Effectifs actuels, relus seulement quand le fichier change."""
    from src.ingest import rosters as ro
    path = ro.DIR / "current.csv" if hasattr(ro, "DIR") else None
    key = path.stat().st_mtime if path is not None and path.exists() else None
    if _roster_cache.get("key") != key or "df" not in _roster_cache:
        try:
            cur = ro.read("current.csv")
        except Exception:
            cur = pd.DataFrame(columns=["player", "id", "team", "role", "sub"])
        if len(cur):
            cur = cur.assign(tkey=cur["team"].map(norm),
                             pos=cur["role"].astype(str).str.strip().str.lower().map(ROSTER_ROLE))
        _roster_cache.update(key=key, df=cur)
    return _roster_cache["df"]


def expected(index: dict | None, team: str, last: dict) -> dict:
    """Le cinq attendu de `team`. `last` : {poste: identifiant} du dernier cinq aligné.

    Renvoie {"lineup": {poste: identifiant}, "changes": [texte], "names": {identifiant: pseudo},
    "known": l'équipe a-t-elle un effectif suivi ?}."""
    out = {"lineup": dict(last), "changes": [], "names": {}, "known": False, "pages": {}}
    if not index:
        return out
    cur = _roster()
    if not len(cur):
        return out
    keys = {norm(team)} | {norm(t) for t in index["team_lp"].get(team, ())}
    rows = cur[cur["tkey"].isin(keys)]
    if rows.empty:
        return out
    out["known"] = True
    pid_of, names = {}, index["names"]
    last_by_name = {norm(names.get(p, p)): p for p in last.values()}
    for page, pid_name in zip(rows["player"], rows["id"]):
        pid = index["page_pid"].get(page)
        if pid is None:                    # fiche non reliée : même pseudo que le dernier cinq,
            key = norm(pid_name)           # sinon pseudo unique en compétition, sinon inconnu
            same = index.get("by_name", {}).get(key, [])
            pid = last_by_name.get(key) or (same[0] if len(same) == 1 else NEW + str(page))
        pid_of[page] = pid
        out["names"][pid] = names.get(pid, str(pid_name))
    here = set(pid_of.values())
    at_role: dict[str, list] = defaultdict(list)
    for page, pos in zip(rows["player"], rows["pos"]):
        if isinstance(pos, str):
            at_role[pos].append(pid_of[page])
    used = {p for p in last.values() if p in here}
    for role in ROLES:
        old = last.get(role)
        if old in here:
            continue
        free = [p for p in at_role.get(role, []) if p not in used]
        old_name = names.get(old, old) if old else "personne"
        if free:
            new = free[0]
            used.add(new)
            out["lineup"][role] = new
            more = f" (ou {', '.join(out['names'][p] for p in free[1:])})" if len(free) > 1 else ""
            out["changes"].append(f"{role} : {out['names'][new]} remplace {old_name}{more}")
        elif old:
            ghost = f"{UNKNOWN}{team}:{role}"
            out["lineup"][role] = ghost
            out["names"][ghost] = "inconnu"
            out["changes"].append(f"{role} : {old_name} a quitté l'équipe, remplaçant inconnu")
    page_of = {pid: page for page, pid in pid_of.items()}
    out["pages"] = {role: page_of[pid] for role, pid in out["lineup"].items() if pid in page_of}
    return out

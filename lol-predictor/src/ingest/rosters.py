"""Effectifs à jour de toutes les équipes suivies : mercato et changements en cours de saison.

Ce que ça fait, à chaque passage (toutes les 6 h sur le serveur) :
  1. liste les équipes suivies : celles des ligues couvertes par Oracle's Elixir (saison en
     cours, plus les tournois de la saison suivante dès que le wiki les publie) ;
  2. lit sur Leaguepedia l'équipe ACTUELLE de chaque joueur (table `Players`, tenue à jour
     par le wiki au fil des annonces officielles : une signature pour la saison prochaine y
     apparaît dès qu'elle est annoncée) ;
  3. compare au relevé précédent et note chaque arrivée et chaque départ dans un journal ;
  4. garde les annonces officielles récentes (table `RosterChanges`, avec la source).

Les joueurs de ces effectifs sont ajoutés au suivi soloQ (`soloq.accounts_table`) : un
joueur qui change d'équipe, ou qui arrive d'une ligue mineure, est suivi dès son annonce,
sans attendre son premier match officiel.

Fichiers (`data/external/rosters/`) :
  - `current.csv`  : effectif actuel, une ligne par joueur (équipe, ligue, rôle, comptes) ;
  - `log.csv`      : journal des changements détectés (date, équipe, joueur, arrivée/départ) ;
  - `news.csv`     : annonces officielles Leaguepedia des 120 derniers jours ;
  - `teams.csv`    : équipes suivies et leur ligue.

Usage :
    python -m src.ingest.rosters               # relevé + changements
    python -m src.ingest.rosters --team "T1"   # affiche un effectif
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

from src.ingest import leaguepedia as lp
from src.ingest.load_oracle import ROOT, load_config, read_seasons

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = ROOT / "data" / "external" / "rosters"
ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]
NEWS_DAYS = 120
CHUNK = 40                 # équipes par requête (la clause IN reste courte)
MIN_PLAYERS = 1000         # garde-fou : un relevé plus petit = réponse incomplète du wiki

# Compétitions internationales : leurs équipes viennent des ligues, pas l'inverse.
EVENTS = {"WLDs", "WLRQ", "MSI", "EWC", "FST", "AM", "AC", "CCWS", "WSCI",
          "DCup", "DCGI", "KeSPA Cup", "IEM", "MSC"}


def _q(s: str) -> str:
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


# ---------------------------------------------------------------- équipes suivies
# Ligues dont les games n'ont pas d'identifiant Riot sur le wiki (serveur chinois) : la
# correspondance ne peut pas s'apprendre, on la donne.
FIXED = {"Tencent LoL Pro League": "LPL", "LoL Development League": "LDL",
         "Demacia Cup": "DCup", "Demacia Cup Global Invitational": "DCGI"}


def league_map(years: list[int]) -> dict[str, str]:
    """Nom de ligue Leaguepedia -> code Oracle, appris sur les games présentes des deux côtés.
    La saison la plus récente l'emporte (une ligue renommée garde son nouveau code)."""
    cfg = load_config()
    votes = []
    for rank, y in enumerate(years):
        if not lp.path("games", y).exists() or not lp.path("tournaments", y).exists():
            continue
        g = lp.load("games", y)[["OverviewPage", "RiotPlatformGameId"]].dropna()
        t = lp.load("tournaments", y)[["OverviewPage", "League"]]
        try:
            o = read_seasons(cfg, {"gameid", "league"}, history=y != cfg["scope"]["year"])
        except Exception:
            continue
        o = o.drop_duplicates("gameid")
        m = g.merge(t, on="OverviewPage").merge(o, left_on="RiotPlatformGameId", right_on="gameid")
        votes.append(m[["League", "league"]].assign(rank=rank))
    if not votes:
        return dict(FIXED)
    v = pd.concat(votes).value_counts().reset_index()
    v = v.sort_values(["rank", "count"], ascending=False).drop_duplicates("League")
    return {**dict(zip(v["League"], v["league"])), **FIXED}


def _next_rosters(year: int) -> pd.DataFrame:
    """Rosters déjà inscrits aux tournois de la saison suivante (vide tant que rien n'est publié)."""
    try:
        r = lp.cargo("TournamentPlayers=TP,Tournaments=T",
                     "TP.OverviewPage=OverviewPage,TP.Team=Team,T.League=League,T.DateStart=DateStart",
                     where=f"T.Year={year + 1}", join_on="TP.OverviewPage=T.OverviewPage")
    except RuntimeError:
        return pd.DataFrame(columns=["OverviewPage", "Team", "League", "DateStart"])
    return r


def followed_teams(year: int) -> pd.DataFrame:
    """Équipes suivies -> leur ligue (code Oracle), d'après leur tournoi le plus récent.
    Saison passée + saison en cours + saison suivante : en janvier, peu de tournois de
    l'année existent encore, le suivi ne doit pas rétrécir."""
    lmap = league_map([year - 1, year])
    parts = [_next_rosters(year)]
    for y in (year - 1, year):
        if lp.path("rosters", y).exists() and lp.path("tournaments", y).exists():
            t = lp.load("tournaments", y)[["OverviewPage", "League", "DateStart"]]
            parts.append(lp.load("rosters", y)[["OverviewPage", "Team"]].drop_duplicates()
                         .merge(t, on="OverviewPage"))
    r = pd.concat(parts, ignore_index=True)
    r["league"] = r["League"].map(lmap)
    r = r.dropna(subset=["league", "Team"])
    r["event"] = r["league"].isin(EVENTS)
    # ligue « maison » = dernier tournoi hors événement ; à défaut, l'événement
    r = r.sort_values(["event", "DateStart"], ascending=[False, True])
    teams = r.drop_duplicates("Team", keep="last")[["Team", "league", "DateStart"]]
    teams = teams[~teams["league"].isin(EVENTS) | ~teams["Team"].isin(r.loc[~r["event"], "Team"])]
    return teams.rename(columns={"Team": "team", "DateStart": "last_event"}).reset_index(drop=True)


# ---------------------------------------------------------------- effectifs
def fetch_current(teams: list[str]) -> pd.DataFrame:
    """Joueurs (rôles en jeu, non retraités) dont l'équipe actuelle est suivie."""
    parts = []
    for i in range(0, len(teams), CHUNK):
        names = ",".join(_q(t) for t in teams[i:i + CHUNK])
        parts.append(lp.cargo(
            "Players",
            "OverviewPage=player,ID=id,Team=team,Team2=team2,Role=role,IsRetired=retired,"
            "IsSubstitute=sub,IsTrainee=trainee,Residency=residency,SoloqueueIds=SoloqueueIds",
            where=f"(Team IN ({names}) OR Team2 IN ({names}))"))
    p = pd.concat(parts, ignore_index=True)
    for c in ("player", "id", "team", "team2"):          # « 1116 » arrive en nombre du wiki
        p[c] = p[c].map(lambda v: str(v) if v is not None and v == v else None)
    p = p.drop_duplicates("player")
    p = p[p["role"].isin(ROLES) & (pd.to_numeric(p["retired"], errors="coerce").fillna(0) == 0)]
    # un joueur prêté ou partagé compte dans chacune de ses deux équipes suivies
    keep = set(teams)
    rows = []
    for r in p.itertuples(index=False):
        for team in (r.team, r.team2):
            if isinstance(team, str) and team in keep:
                rows.append({**r._asdict(), "team": team})
    out = pd.DataFrame(rows).drop(columns=["team2", "retired"])
    for c in ("sub", "trainee"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def fetch_news(teams: set[str]) -> pd.DataFrame:
    """Annonces officielles récentes (arrivées, départs, prêts, remplaçants temporaires)."""
    since = (dt.date.today() - dt.timedelta(days=NEWS_DAYS)).isoformat()
    n = lp.cargo("RosterChanges", "Date_Sort=date,Player=player,Team=team,Direction=direction,"
                                  "Role=role,Status=status,Source=source",
                 where=f"Date_Sort >= '{since}'", order_by="Date_Sort DESC")
    n = n[n["team"].isin(teams) & n["role"].isin(ROLES)].copy()
    n["date"] = n["date"].astype(str).str[:10]
    # la source wiki est « url;;;titre;;;… » : on garde la première url
    n["source"] = n["source"].fillna("").str.split(";;;").str[0]
    return n.reset_index(drop=True)


# ---------------------------------------------------------------- journal
def _diff(old: pd.DataFrame, new: pd.DataFrame, today: str, teams: set) -> pd.DataFrame:
    """Arrivées et départs, sur les équipes suivies aux deux relevés (une équipe qui entre
    dans le suivi ne compte pas comme cinq arrivées)."""
    cols = ["date", "team", "league", "player", "role", "change"]
    if old.empty:
        return pd.DataFrame(columns=cols)
    old, new = old[old["team"].isin(teams)], new[new["team"].isin(teams)]
    k_old = set(zip(old["team"], old["player"]))
    k_new = set(zip(new["team"], new["player"]))
    rows = []
    for df, keys, label in ((new, k_new - k_old, "arrivée"), (old, k_old - k_new, "départ")):
        sel = df[[k in keys for k in zip(df["team"], df["player"])]]
        for r in sel.itertuples(index=False):
            rows.append({"date": today, "team": r.team, "league": r.league, "player": r.player,
                         "role": r.role, "change": label})
    return pd.DataFrame(rows, columns=cols)


def _write(df: pd.DataFrame, name: str) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = DIR / f"{name}.tmp"
    df.to_csv(tmp, index=False)
    tmp.replace(DIR / name)


def read(name: str) -> pd.DataFrame:
    p = DIR / name
    # tout en texte : un pseudo comme « 1116 » ne doit pas devenir un nombre (sinon le
    # joueur semble partir et revenir d'un relevé à l'autre)
    text = {c: str for c in ("player", "id", "team", "league", "role", "date")}
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p, keep_default_na=False, na_values=[""], dtype=text)


def run(year: int | None = None) -> dict:
    year = year or load_config()["scope"]["year"]
    teams = followed_teams(year)
    cur = fetch_current(teams["team"].tolist())
    if len(cur) < MIN_PLAYERS:
        raise RuntimeError(f"rosters : seulement {len(cur)} joueurs lus, relevé ignoré "
                           "(réponse incomplète du wiki).")
    cur = cur.merge(teams[["team", "league"]], on="team", how="left")
    cur = cur.sort_values(["league", "team", "sub", "role"]).reset_index(drop=True)
    today = dt.date.today().isoformat()
    prev = read("teams.csv")
    both = set(teams["team"]) & set(prev["team"]) if len(prev) else set()
    changes = _diff(read("current.csv"), cur, today, both)
    log = pd.concat([read("log.csv"), changes], ignore_index=True)
    news = fetch_news(set(teams["team"]))
    _write(teams, "teams.csv")
    _write(cur, "current.csv")
    _write(log, "log.csv")
    _write(news, "news.csv")
    return {"teams": len(teams), "players": len(cur), "changes": changes, "news": news}


def message(res: dict) -> str:
    ch = res["changes"]
    msg = f"{res['teams']} équipes · {res['players']} joueurs"
    if len(ch):
        arr = (ch["change"] == "arrivée").sum()
        msg += f" · {arr} arrivées, {len(ch) - arr} départs depuis le dernier relevé"
    return msg


def soloq_tracked(year: int) -> pd.Series:
    """Nombre de comptes soloQ suivis par joueur, d'après les fichiers du relevé (sans requête)."""
    from src.ingest import soloq as sq
    parts = []
    if sq.accounts_path(year).exists():
        a = sq.read_csv(sq.accounts_path(year))
        parts.append(a.loc[a["usable"].astype(str).eq("True") & (a["n"] < sq.MAX_ACCOUNTS), "player"])
    for p in (sq.discovered_path(year), sq.DIR / f"sites_{year}.csv"):
        if p.exists():
            d = sq.read_csv(p)
            parts.append(d.loc[d["confidence"].isin(["fort", "moyen"]), "player"])
    return pd.concat(parts).value_counts() if parts else pd.Series(dtype=int)


def coverage(year: int, leagues: list[str] | None = None) -> pd.DataFrame:
    """Part des titulaires actuels suivis en soloQ, par ligue."""
    cur = read("current.csv")
    cur = cur[cur["sub"] == 0]
    if leagues:
        cur = cur[cur["league"].isin(leagues)]
    n = soloq_tracked(year)
    cur = cur.assign(suivi=cur["player"].map(n).fillna(0) > 0)
    return cur.groupby("league").agg(titulaires=("player", "size"), suivis=("suivi", "mean"))


def roster(team: str) -> pd.DataFrame:
    cur = read("current.csv")
    return cur[cur["team"].str.lower() == team.lower()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Effectifs à jour (Leaguepedia)")
    ap.add_argument("--team", help="affiche l'effectif d'une équipe (sans requête)")
    args = ap.parse_args()
    if args.team:
        print(roster(args.team)[["player", "role", "sub", "league"]].to_string(index=False))
        return
    res = run()
    print(message(res))
    ch = res["changes"]
    if len(ch):
        print(ch.to_string(index=False))


if __name__ == "__main__":
    main()

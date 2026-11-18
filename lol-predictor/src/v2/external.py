"""Signaux tirés de Leaguepedia : le COACH qui dirige l'équipe, le LIEU et le parcours en tournoi.

Passe chronologique stricte, comme les autres moteurs : features calculées AVANT chaque
game, état mis à jour APRÈS.

Bloc « Coach » (le head coach de l'équipe pour ce tournoi) :
  - coach_rating  : l'équipe fait-elle mieux que ce que vaut son cinq (Elo joueur) quand
                    ce coach la dirige ? Note portée par le COACH : elle le suit s'il
                    change d'équipe.
  - coach_draft   : confort moyen qu'il donne à ses joueurs en draft (écart à la ligue),
                    porté par le coach lui aussi.
  - coach_tenure  : games passées ensemble (coach + cette équipe).
  - coach_exp     : games dirigées au total.

Bloc « Lieu & parcours » :
  - venue_adapt   : surperformance de l'équipe dans CE type de lieu (LAN ou online).
  - lan_exp       : expérience de la LAN, comptée seulement si le match est en LAN.
  - tourn_wr      : bilan de l'équipe dans le tournoi en cours.

Non couvert : « équipe déjà qualifiée / déjà éliminée ». Il faudrait les règles de
qualification de chaque ligue, que Leaguepedia ne fournit pas sous forme exploitable.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.ingest import leaguepedia as lp
from src.update.watchlist import norm

K_COACH = 16.0            # points d'Elo gagnés/perdus par le staff pour un écart de 1 victoire
K_SHRINK = 10.0
TOURN_K = 4.0
MATCH_WINDOW = pd.Timedelta(hours=8)   # appariement par date quand l'id Riot manque (LPL)

EXT_FACTORS = {
    "coach_rating": ("Coach", "Note du head coach (surperf. de ses équipes)"),
    "coach_draft": ("Coach", "Confort de draft donné par le coach"),
    "coach_tenure": ("Coach", "Ancienneté coach + équipe"),
    "coach_exp": ("Coach", "Expérience du coach (games dirigées)"),
    "venue_adapt": ("Lieu & parcours", "Surperf. dans ce type de lieu (LAN/online)"),
    "lan_exp": ("Lieu & parcours", "Expérience de la LAN"),
    "tourn_wr": ("Lieu & parcours", "Bilan dans le tournoi en cours"),
}
EXT_COLS = [f"d_{k}" for k in EXT_FACTORS]


def _empty_record() -> list:
    return [0.0, 0]


def _shrunk(values, k: float = K_SHRINK) -> float:
    vals = [v for v in values if v == v]
    return float(sum(vals) / (len(vals) + k))


# ---------------------------------------------------------- lien Oracle <-> LP
def _venue(event_type) -> float:
    """1 = LAN, 0 = online, NaN = inconnu ou mixte (phases online puis finale en LAN)."""
    if not isinstance(event_type, str):
        return np.nan
    t = event_type.strip().lower()
    return 1.0 if t == "offline" else 0.0 if t == "online" else np.nan


def link_games(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Pour chaque game Oracle : tournoi Leaguepedia + noms LP des deux équipes.

    1) par identifiant Riot (`gameid` Oracle = `RiotPlatformGameId`) ;
    2) sinon (LPL, Demacia Cup : ids différents) par paire d'équipes + date la plus proche.
    """
    g = lp.load("games", year).dropna(subset=["Team1", "Team2", "DateTime_UTC"]).copy()
    g["when"] = pd.to_datetime(g["DateTime_UTC"], errors="coerce")
    ids = g.dropna(subset=["RiotPlatformGameId"]).drop_duplicates("RiotPlatformGameId")
    by_id = {gid: (page, t1, t2, win) for gid, page, t1, t2, win in zip(
        ids["RiotPlatformGameId"], ids["OverviewPage"], ids["Team1"], ids["Team2"], ids["WinTeam"])}

    # Le nom Leaguepedia d'une équipe DÉPEND DU TOURNOI (« Anyone's Legend » en LPL,
    # « AG.AL » à l'EWC) : on le lit game par game, jamais une fois pour toutes.
    rows: dict[str, tuple] = {}
    votes: dict[str, Counter] = defaultdict(Counter)       # nom Oracle -> noms LP observés
    for r in df.itertuples():
        if r.gameid not in by_id:
            continue
        page, t1, t2, lp_win = by_id[r.gameid]
        a = b = None
        # Team1 n'est pas toujours le côté bleu : on apparie par le RÉSULTAT (vainqueur
        # Oracle = WinTeam Leaguepedia), seul repère commun aux deux sources.
        if lp_win in (t1, t2):
            lp_lose = t2 if lp_win == t1 else t1
            a, b = (lp_win, lp_lose) if r.yb == 1 else (lp_lose, lp_win)
            votes[r.blue][a] += 1
            votes[r.red][b] += 1
        rows[r.gameid] = (page, a, b)

    lp_norm = {norm(t): t for t in pd.concat([g["Team1"], g["Team2"]]).unique()}

    def aliases(team: str) -> list[str]:
        """Noms LP possibles : le nom identique d'abord, puis ceux déjà observés."""
        exact = lp_norm.get(norm(team))
        seen = [n for n, _ in votes[team].most_common()]
        return ([exact] if exact else []) + [n for n in seen if n != exact]

    by_team: dict[str, pd.DataFrame] = {}
    for col in ("Team1", "Team2"):
        for team, grp in g.groupby(col):
            by_team[team] = pd.concat([by_team[team], grp]) if team in by_team else grp

    def nearest(team: str, when: pd.Timestamp) -> pd.Series | None:
        cand = by_team.get(team)
        if cand is None:
            return None
        gap = (cand["when"] - when).abs()
        return cand.loc[gap.idxmin()] if gap.min() <= MATCH_WINDOW else None

    for r in df.itertuples():
        if r.gameid in rows:
            continue
        # Sans id Riot (LPL) : la game LP la plus proche jouée par l'une des deux équipes,
        # à condition que l'adversaire soit le bon — ou porte un nom encore inconnu chez LP
        # (« Ninjas in Pyjamas.CN »), que l'on apprend alors.
        al_b, al_r = aliases(r.blue), aliases(r.red)
        for mine, theirs, mine_is_blue in ((al_b, al_r, True), (al_r, al_b, False)):
            hit = None
            for cand in mine:
                m = nearest(cand, r.date)
                if m is None:
                    continue
                opp = m["Team2"] if m["Team1"] == cand else m["Team1"]
                if not theirs or opp in theirs:
                    hit = (m["OverviewPage"], cand, opp)
                    break
            if hit:
                page, me, opp = hit
                votes[r.red if mine_is_blue else r.blue][opp] += 1
                rows[r.gameid] = (page, me, opp) if mine_is_blue else (page, opp, me)
                break

    out = pd.DataFrame([(k, *v) for k, v in rows.items()],
                       columns=["gameid", "page", "lp_blue", "lp_red"])
    t = lp.load("tournaments", year).drop_duplicates("OverviewPage")
    t = t.assign(venue=t["EventType"].map(_venue)).rename(columns={"OverviewPage": "page"})
    return out.merge(t[["page", "Name", "venue", "Country", "Region"]], on="page", how="left")


def staff_table(year: int) -> dict[tuple[str, str], list[str]]:
    """(tournoi, équipe LP) -> coachs, le head coach en premier."""
    r = lp.load("rosters", year)
    r = r[r["Role"].fillna("").str.contains("Coach")].sort_values(["OverviewPage", "Team", "N"])
    roles = lp.load("coach_roles", year).dropna(subset=["Player", "Team"])
    # dernier rôle connu de chaque (coach, équipe) : « Head Coach » prime sur l'ordre d'inscription
    last_role = roles.drop_duplicates(["Player", "Team"], keep="last")
    head = {(p, t) for p, t, role in zip(last_role["Player"], last_role["Team"], last_role["Role"])
            if "Head Coach" in str(role)}
    out: dict[tuple[str, str], list[str]] = {}
    for (page, team), grp in r.groupby(["OverviewPage", "Team"], sort=False):
        names = [p for p in grp["Player"] if isinstance(p, str)]
        names.sort(key=lambda p: (p, team) not in head)      # tri stable : head coach devant
        if names:
            out[(page, team)] = names
    return out


# -------------------------------------------------------------------- l'état
class ExtState:
    def __init__(self) -> None:
        self.coach_rating: dict[str, float] = defaultdict(float)
        self.coach_games: Counter = Counter()
        self.coach_team: Counter = Counter()                  # (coach, équipe) -> games
        self.coach_draft: dict[str, list] = defaultdict(list)
        self.venue_resid: dict[tuple, list] = defaultdict(list)   # (équipe, lieu) -> résidus
        self.lan_games: Counter = Counter()
        self.tourn: dict[tuple, list] = defaultdict(_empty_record)      # (tournoi, équipe) -> [v, n]
        self.last_staff: dict[str, list[str]] = {}
        self.last_page: dict[str, str] = {}
        self.page_info: dict[str, dict] = {}                  # tournoi -> {venue, nom}

    def features(self, team: str, staff: list[str] | None, venue: float, page: str | None) -> dict:
        head = staff[0] if staff else None
        known = venue == venue
        w, n = self.tourn[(page, team)] if page else (0.0, 0)
        return {
            "coach_rating": self.coach_rating[head] if head else np.nan,
            "coach_draft": _shrunk(self.coach_draft[head]) if head else np.nan,
            "coach_tenure": math.log1p(self.coach_team[(head, team)]) if head else np.nan,
            "coach_exp": math.log1p(self.coach_games[head]) if head else np.nan,
            "venue_adapt": _shrunk(self.venue_resid[(team, venue)]) if known else 0.0,
            "lan_exp": math.log1p(self.lan_games[team]) if venue == 1.0 else 0.0,
            "tourn_wr": (w + 0.5 * TOURN_K) / (n + TOURN_K),
        }

    def update(self, team: str, staff: list[str] | None, venue: float, page: str | None,
               won: float, p: float, comfort: float) -> None:
        resid = won - p
        for coach in staff or []:
            self.coach_rating[coach] += K_COACH * resid
            self.coach_games[coach] += 1
            self.coach_team[(coach, team)] += 1
        if staff:
            if comfort == comfort:
                self.coach_draft[staff[0]].append(comfort)
            self.last_staff[team] = staff
        if venue == venue:
            self.venue_resid[(team, venue)].append(resid)
            if venue == 1.0:
                self.lan_games[team] += 1
        if page:
            rec = self.tourn[(page, team)]
            rec[0] += won
            rec[1] += 1
            self.last_page[team] = page


def build_external(df: pd.DataFrame) -> tuple[pd.DataFrame, ExtState | None]:
    """Signaux Leaguepedia as-of par game (`d_*` = bleu − rouge) + état final.

    `df` : jeu de données v2 (gameid, date, blue, red, yb, p_player, post_comfort_*).
    Sans cache Leaguepedia (`python -m src.ingest.leaguepedia`), renvoie des colonnes vides :
    le modèle tourne quand même, simplement sans ces deux blocs.
    """
    # Une année = un cache Leaguepedia. Les années sans cache restent sans coach ni lieu
    # (signaux neutres) ; les notes des coachs, elles, se suivent d'une année à l'autre.
    by_year = df["date"].dt.year
    years = [int(y) for y in sorted(by_year.unique()) if lp.available(int(y))]
    if not years:
        return pd.DataFrame({"gameid": df["gameid"], **{c: np.nan for c in EXT_COLS}}), None

    link = pd.concat([link_games(df[by_year == y], y) for y in years], ignore_index=True)
    link = link.drop_duplicates("gameid")
    linked = {gid: (page, venue, a, b) for gid, page, venue, a, b in zip(
        link["gameid"], link["page"], link["venue"], link["lp_blue"], link["lp_red"])}
    staff: dict[tuple[str, str], list[str]] = {}
    for y in years:
        staff.update(staff_table(y))
    stt = ExtState()
    for page, info in link.drop_duplicates("page").set_index("page")[["venue", "Name"]].iterrows():
        stt.page_info[page] = {"venue": info["venue"], "name": info["Name"]}

    rows = []
    for g in df.sort_values("date").itertuples():
        if g.gameid in linked:
            page, venue, lp_blue, lp_red = linked[g.gameid]
            sb, sr = staff.get((page, lp_blue)), staff.get((page, lp_red))
        else:
            page, venue, sb, sr = None, np.nan, None, None
        fb, fr = stt.features(g.blue, sb, venue, page), stt.features(g.red, sr, venue, page)
        rows.append({"gameid": g.gameid, "ext_page": page, "ext_venue": venue,
                     "coach_blue": sb[0] if sb else None, "coach_red": sr[0] if sr else None,
                     **{f"d_{k}": fb[k] - fr[k] for k in EXT_FACTORS}})
        stt.update(g.blue, sb, venue, page, float(g.yb), g.p_player, g.post_comfort_blue)
        stt.update(g.red, sr, venue, page, 1.0 - g.yb, 1.0 - g.p_player, g.post_comfort_red)
    return pd.DataFrame(rows), stt


def external_snapshot(stt: ExtState | None, team: str) -> dict:
    """Signaux pré-match d'une équipe : dernier staff connu, lieu de son dernier tournoi."""
    if stt is None:
        return {**{k: np.nan for k in EXT_FACTORS}, "staff": [], "venue": np.nan, "tournament": None}
    staff = stt.last_staff.get(team)
    page = stt.last_page.get(team)
    info = stt.page_info.get(page, {})
    venue = info.get("venue", np.nan)
    feats = stt.features(team, staff, venue, page)
    feats.update({"staff": staff or [], "venue": venue, "tournament": info.get("name") or page})
    return feats

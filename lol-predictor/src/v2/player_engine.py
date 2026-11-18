"""Moteur v2 : l'équipe vue comme 5 joueurs + un staff qui draft + un style de jeu.

Passe chronologique stricte : features calculées AVANT chaque game, état mis à jour APRÈS.

Blocs de signaux (écarts équipe A − équipe B) :
  - Elo JOUEUR      : chaque joueur a son Elo ; la force d'une équipe = moyenne de ses 5.
                      Un transfert emporte l'Elo du joueur. Un nouveau joueur démarre au
                      niveau moyen des joueurs de sa ligue (pas à 1500 partout).
  - Méta & pool     : les champions joués par chaque joueur sont-ils présents dans la méta
                      actuelle (pick+ban des 21 derniers jours) ? Combien de champions
                      méta dans son pool ?
  - Tempo           : profil early (gold diff @15), late (évolution 15 -> 25 min),
                      capacité de comeback (gagner en étant derrière à 15 min).
  - Draft (staff)   : qualité moyenne des drafts récentes de l'équipe, proxy du staff :
                      confort donné aux joueurs, picks méta, avantage de winrate champion,
                      bans ciblés sur le pool adverse.

Le staff réel (coach / head coach) n'est pas dans le CSV Oracle : ces mesures sont
attribuées à l'ÉQUIPE. L'identité du coach (Leaguepedia) est l'étape suivante.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from functools import partial

import numpy as np
import pandas as pd

from src.ingest.load_oracle import load_config, read_seasons, season_files
from src.update.elo import BASE, _mov_mult, load_games, win_prob

ROLES = ["top", "jng", "mid", "bot", "sup"]
K_PLAYER = 32.0
LEAGUE_PRIOR_MIN = 20     # joueurs connus dans la ligue avant d'utiliser sa moyenne
META_DAYS = 21
POOL_DAYS = 90
CHAMP_WR_DAYS = 180       # fenêtre du winrate d'un champion (au-delà : équilibrages, autre méta)
META_CHAMP = 0.25         # présence pick+ban mini pour qu'un champion compte « méta »
COMFORT_GAMES = 3         # un champion est « confort » s'il a été joué >= 3 fois
TEAM_WINDOW = 20
K_SHRINK = 5.0
K_CHAMP = 30.0
# Signaux de draft mesurés en écart à la moyenne de la ligue (valeur de départ tant
# que la ligue compte moins de DRAFT_BASE_MIN observations).
DRAFT_PRIOR = {"draft_comfort": 0.6, "target_bans": 0.3}
DRAFT_BASE_MIN = 30

SIGNALS = {
    "d_rating": ("Elo joueur", "Écart d'Elo moyen des 5 joueurs"),
    "d_meta_comfort": ("Méta & pool", "Champions joués présents dans la méta"),
    "d_pool_meta": ("Méta & pool", "Champions méta dans le pool"),
    "d_early": ("Tempo", "Profil early (gold diff @15, k)"),
    "d_late": ("Tempo", "Profil late (gold 15 -> 25 min, k)"),
    "d_comeback": ("Tempo", "Comebacks (gagner derrière à 15 min)"),
    "d_draft_comfort": ("Draft (staff)", "Confort donné aux joueurs (vs ligue)"),
    "d_draft_meta": ("Draft (staff)", "Picks méta"),
    "d_draft_edge": ("Draft (staff)", "Avantage winrate champion"),
    "d_target_bans": ("Draft (staff)", "Bans ciblés sur le pool adverse (vs ligue)"),
}


def _team_windows() -> dict:
    """Fenêtres glissantes d'une équipe (nommé, pas de lambda : l'état est enregistré)."""
    return defaultdict(partial(deque, maxlen=TEAM_WINDOW))


def _empty_base() -> tuple:
    return (0.0, 0)


def _shrunk(values, k: float = K_SHRINK, prior: float = 0.0) -> float:
    vals = [v for v in values if v == v]
    return float((sum(vals) + prior * k) / (len(vals) + k))


_rows_cache: dict = {}


def _load_rows(cfg: dict) -> tuple[dict, dict, dict]:
    """`_read_rows`, gardé en mémoire tant que les fichiers ne changent pas : le moteur
    joueurs et la draft relisent les mêmes lignes."""
    key = tuple((str(p), p.stat().st_mtime) for _, p in season_files(cfg, True))
    if _rows_cache.get("key") != key:
        _rows_cache.update(key=key, rows=_read_rows(cfg))
    return _rows_cache["rows"]


def _read_rows(cfg: dict) -> tuple[dict, dict, dict]:
    """(joueurs par game, équipes par game, pseudos) en dicts Python (rapide en boucle)."""
    cols = {"gameid", "position", "playerid", "playername", "teamname", "champion",
            "golddiffat15", "golddiffat25", "ban1", "ban2", "ban3", "ban4", "ban5"}
    df = read_seasons(cfg, cols)
    pos = df["position"].str.lower()

    pl = df[pos.isin(ROLES)].assign(position=pos[pos.isin(ROLES)])
    pl = pl.assign(pid=pl["playerid"].fillna(pl["playername"])).dropna(subset=["pid", "teamname"])
    players: dict = defaultdict(lambda: defaultdict(dict))
    for gid, team, role, pid, champ in zip(pl["gameid"], pl["teamname"], pl["position"],
                                           pl["pid"], pl["champion"]):
        players[gid][team][role] = (pid, champ if isinstance(champ, str) else None)

    tm = df[pos == "team"]
    teams: dict = defaultdict(dict)
    bans = tm[["ban1", "ban2", "ban3", "ban4", "ban5"]].to_numpy()
    for i, (gid, team, g15, g25) in enumerate(zip(tm["gameid"], tm["teamname"],
                                                  tm["golddiffat15"], tm["golddiffat25"])):
        teams[gid][team] = {"gd15": g15, "gd25": g25,
                            "bans": [b for b in bans[i] if isinstance(b, str)]}
    names = dict(zip(pl["pid"], pl["playername"]))
    return players, teams, names


class PlayerState:
    def __init__(self) -> None:
        self.rating: dict[str, float] = {}
        self.player_league: dict[str, str] = {}
        self.league_sum: dict[str, float] = defaultdict(float)
        self.league_cnt: dict[str, int] = defaultdict(int)
        self.hist: dict[str, list] = defaultdict(list)          # pid -> [(date, champ)]
        self.pc: Counter = Counter()                            # (pid, champ) -> games
        self.comfort: dict[str, set] = defaultdict(set)         # pid -> champions « confort »
        self.champ_g: Counter = Counter()                       # sur CHAMP_WR_DAYS glissants
        self.champ_w: Counter = Counter()
        self.champ_q: deque = deque()                           # (date, [(champ, gagné)])
        self.meta_q: deque = deque()                            # (date, champs)
        self.meta_cnt: Counter = Counter()
        self.meta_games = 0
        self.team: dict[str, dict[str, deque]] = defaultdict(_team_windows)
        self.draft_base: dict[tuple, tuple] = defaultdict(_empty_base)       # (ligue, signal)
        self.last_comfort: dict[str, float] = {}                 # confort de la dernière draft
        self.last_lineup: dict[str, dict] = {}
        self.team_league: dict[str, str] = {}
        self.last_date: pd.Timestamp | None = None
        self.names: dict[str, str] = {}

    # ----------------------------------------------------------- notes joueurs --
    def _init_rating(self, league: str) -> float:
        if self.league_cnt[league] >= LEAGUE_PRIOR_MIN:
            return self.league_sum[league] / self.league_cnt[league]
        return BASE

    def get_rating(self, pid: str, league: str) -> float:
        return self.rating.get(pid, self._init_rating(league))

    def _set_rating(self, pid: str, league: str, value: float) -> None:
        old_lg = self.player_league.get(pid)
        if pid in self.rating and old_lg is not None:
            self.league_sum[old_lg] -= self.rating[pid]
            self.league_cnt[old_lg] -= 1
        self.rating[pid] = value
        self.player_league[pid] = league
        self.league_sum[league] += value
        self.league_cnt[league] += 1

    def team_rating(self, lineup: dict, league: str) -> float:
        pids = [v[0] for v in lineup.values()]
        return float(np.mean([self.get_rating(p, league) for p in pids])) if pids else BASE

    # ------------------------------------------------------------------ méta --
    def _advance_meta(self, date: pd.Timestamp) -> None:
        lo = date - pd.Timedelta(days=META_DAYS)
        while self.meta_q and self.meta_q[0][0] < lo:
            _, champs = self.meta_q.popleft()
            self.meta_cnt.subtract(champs)
            self.meta_games -= 1
        lo = date - pd.Timedelta(days=CHAMP_WR_DAYS)
        while self.champ_q and self.champ_q[0][0] < lo:
            for champ, won in self.champ_q.popleft()[1]:
                self.champ_g[champ] -= 1
                self.champ_w[champ] -= won

    def presence(self, champ: str | None) -> float:
        if not champ or not self.meta_games:
            return 0.0
        return self.meta_cnt[champ] / self.meta_games

    def champ_wr(self, champ: str | None) -> float:
        if not champ:
            return 0.5
        return (self.champ_w[champ] + 0.5 * K_CHAMP) / (self.champ_g[champ] + K_CHAMP)

    # ------------------------------------------------------------ features --
    def team_features(self, team: str, lineup: dict, league: str,
                      date: pd.Timestamp) -> dict:
        lo = date - pd.Timedelta(days=POOL_DAYS)
        comfort, pool_meta = [], []
        for pid, _ in lineup.values():
            recent = []                      # du plus récent au plus ancien, dans la fenêtre
            for d, c in reversed(self.hist.get(pid, ())):
                if d < lo:
                    break
                if c:
                    recent.append(c)
            if recent:
                comfort.append(float(np.mean([self.presence(c) for c in recent])))
                cnt = Counter(recent)
                pool_meta.append(sum(1 for c, n in cnt.items()
                                     if n >= 2 and self.presence(c) >= META_CHAMP))
        t = self.team.get(team, {})
        # draft_comfort et target_bans sont déjà centrés sur la ligue (cf. update).
        return {
            "rating": self.team_rating(lineup, league),
            "meta_comfort": float(np.mean(comfort)) if comfort else 0.0,
            "pool_meta": float(np.mean(pool_meta)) if pool_meta else 0.0,
            "early": _shrunk(t.get("gd15", [])) / 1000,
            "late": _shrunk(t.get("late", [])) / 1000,
            "comeback": _shrunk(t.get("comeback", []), prior=0.3),
            "draft_comfort": _shrunk(t.get("draft_comfort", [])),
            "draft_meta": _shrunk(t.get("draft_meta", []), prior=0.3),
            "draft_edge": _shrunk(t.get("draft_edge", [])),
            "target_bans": _shrunk(t.get("target_bans", [])),
        }

    def _vs_league(self, league: str, key: str, value: float) -> float:
        """Écart à la moyenne de la ligue (calculée sur les games déjà jouées).

        « Confort » et « bans ciblés » dépendent du nombre de games connues par joueur :
        en valeur brute, une équipe d'une ligue peu couverte paraît mal drafter. On ne
        garde donc que l'écart à sa ligue, comparable d'une ligue à l'autre.
        """
        if value != value:
            return value
        s, n = self.draft_base[(league, key)]
        base = s / n if n >= DRAFT_BASE_MIN else DRAFT_PRIOR[key]
        self.draft_base[(league, key)] = (s + value, n + 1)
        return value - base

    def _draft_metrics(self, own: dict, opp: dict, own_bans: list) -> dict:
        comfort = [self.pc[(pid, c)] >= COMFORT_GAMES for pid, c in own.values() if c]
        meta = [self.presence(c) for _, c in own.values() if c]
        wr_own = [self.champ_wr(c) for _, c in own.values() if c]
        wr_opp = [self.champ_wr(c) for _, c in opp.values() if c]
        opp_pool = self._pools([p for p, _ in opp.values()])
        targeted = [b in opp_pool for b in own_bans]
        return {
            "draft_comfort": float(np.mean(comfort)) if comfort else np.nan,
            "draft_meta": float(np.mean(meta)) if meta else np.nan,
            "draft_edge": (float(np.mean(wr_own) - np.mean(wr_opp))
                           if wr_own and wr_opp else np.nan),
            "target_bans": float(np.mean(targeted)) if targeted else np.nan,
        }

    def _pools(self, pids: list[str]) -> set:
        out: set = set()
        for pid in pids:
            out |= self.comfort.get(pid, set())
        return out

    # --------------------------------------------------------- mise à jour --
    def update(self, g, lineups: dict, teams: dict, p_blue: float) -> None:
        date, lg = g.date, g.league
        side = {g.blue: (g.red, g.yb, p_blue), g.red: (g.blue, 1 - g.yb, 1 - p_blue)}

        for team, (opp, won, _) in side.items():
            own, other = lineups.get(team, {}), lineups.get(opp, {})
            info = teams.get(team, {})
            dm = self._draft_metrics(own, other, info.get("bans", []))
            t = self.team[team]
            for k, v in dm.items():
                v = self._vs_league(lg, k, v) if k in DRAFT_PRIOR else v
                t[k].append(v)
                if k == "draft_comfort":
                    self.last_comfort[team] = v      # relu par src.v2.external (crédit au coach)
            g15, g25 = info.get("gd15"), info.get("gd25")
            if g15 == g15 and g15 is not None:
                t["gd15"].append(float(g15))
                if g15 < 0:
                    t["comeback"].append(float(won))
                if g25 == g25 and g25 is not None:
                    t["late"].append(float(g25) - float(g15))
            if own:
                self.last_lineup[team] = own
            self.team_league[team] = lg

        mult = _mov_mult(g.kblue, g.kred)
        played = []
        for team, (_, won, p) in side.items():
            for pid, champ in lineups.get(team, {}).values():
                r = self.get_rating(pid, lg)
                self._set_rating(pid, lg, r + K_PLAYER * mult * (won - p))
                self.hist[pid].append((date, champ))
                if champ:
                    self.pc[(pid, champ)] += 1
                    if self.pc[(pid, champ)] >= COMFORT_GAMES:
                        self.comfort[pid].add(champ)
                    self.champ_g[champ] += 1
                    self.champ_w[champ] += int(won)
                    played.append((champ, int(won)))
        self.champ_q.append((date, played))

        picks_bans = [c for lu in lineups.values() for _, c in lu.values() if c]
        picks_bans += [b for t in teams.values() for b in t.get("bans", [])]
        self.meta_q.append((date, picks_bans))
        self.meta_cnt.update(picks_bans)
        self.meta_games += 1
        self.last_date = date


def build_player_table(cfg: dict | None = None) -> tuple[pd.DataFrame, PlayerState]:
    """Rejoue toutes les games : (signaux v2 as-of par game, état final)."""
    cfg = cfg or load_config()
    games = load_games(cfg, history=True)
    players, teams, names = _load_rows(cfg)
    stt = PlayerState()
    stt.names = names
    rows = []
    for g in games.itertuples():
        stt._advance_meta(g.date)
        lineups = players.get(g.gameid, {})
        fb = stt.team_features(g.blue, lineups.get(g.blue, {}), g.league, g.date)
        fr = stt.team_features(g.red, lineups.get(g.red, {}), g.league, g.date)
        p_player = win_prob(fb["rating"], fr["rating"])
        row = {"gameid": g.gameid, "p_player": p_player}
        for key in SIGNALS:
            name = key[2:]
            row[key] = fb[name] - fr[name]
        rows.append(row)
        stt.update(g, lineups, teams.get(g.gameid, {}), p_player)
        # Mesuré SUR cette game (donc connu après) : ce n'est pas un signal pré-match,
        # seulement de quoi créditer le coach dans src.v2.external.
        row["post_comfort_blue"] = stt.last_comfort.get(g.blue, np.nan)
        row["post_comfort_red"] = stt.last_comfort.get(g.red, np.nan)
    return pd.DataFrame(rows), stt


def player_snapshot(stt: PlayerState, team: str, date: pd.Timestamp | None = None,
                    lineup: dict | None = None, names: dict | None = None) -> dict:
    """Signaux v2 pré-match d'une équipe (dernière lineup connue, ou `lineup` = {poste:
    joueur}, le cinq attendu d'après l'effectif suivi ; `names` : pseudos des arrivants).

    La méta et le pool sont mesurés à la date des dernières données : une copie
    locale vieille de plus de 21 jours viderait sinon la fenêtre méta.
    """
    date = date or (stt.last_date or pd.Timestamp.now()).normalize() + pd.Timedelta(days=1)
    stt._advance_meta(date)
    if lineup is None:
        lineup = stt.last_lineup.get(team, {})
    else:
        lineup = {role: (pid, None) for role, pid in lineup.items()}
    names = names or {}
    feats = stt.team_features(team, lineup, stt.team_league.get(team, ""), date)
    lg = stt.team_league.get(team, "")
    feats["players"] = {role: (names.get(pid) or stt.names.get(pid, pid),
                               round(stt.get_rating(pid, lg)))
                        for role, (pid, _) in lineup.items()}
    return feats

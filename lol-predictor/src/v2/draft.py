"""Signaux de DRAFT d'une game : ce que l'on sait une fois les picks et les bans posés.

À la différence des autres blocs, ces signaux ne sont PAS connus avant le match : ils servent
au pari en direct (entre la fin de la draft et le début de la game) et à la page « Draft »,
où l'on saisit les équipes et les champions. En historique, la draft réelle de chaque game
est connue : le modèle « avec draft » s'entraîne dessus, toujours en ne lisant que le passé
(passe chronologique stricte, état mis à jour APRÈS la game).

Signaux (écart bleu − rouge) :
  - dr_champ_wr   : winrate récent des champions pris (toutes ligues, 180 jours) ;
  - dr_presence   : présence pick+ban des champions pris dans la méta (21 jours) ;
  - dr_comfort    : expérience des joueurs sur leur champion (games jouées, en log) ;
  - dr_player_wr  : réussite passée de chaque joueur sur CE champion ;
  - dr_matchup    : duel de champions poste pour poste (365 jours) ;
  - dr_duo        : paires de champions bot+support et jungle+mid (365 jours) ;
  - dr_bans       : bans qui tombent dans le pool de confort des joueurs adverses.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from src.ingest.load_oracle import load_config
from src.update.elo import load_games
from src.v2.player_engine import ROLES, _load_rows

WR_DAYS = 180
META_DAYS = 21
PAIR_DAYS = 365
K_CHAMP = 30.0
K_PLAYER_CHAMP = 5.0
K_PAIR = 10.0
COMFORT_GAMES = 3
DUOS = [("bot", "sup"), ("jng", "mid")]

DRAFT_FACTORS = {
    "dr_champ_wr": "Winrate récent des champions pris",
    "dr_presence": "Champions pris présents dans la méta",
    "dr_comfort": "Expérience des joueurs sur leur champion",
    "dr_player_wr": "Réussite des joueurs sur ce champion",
    "dr_matchup": "Duels de champions, poste pour poste",
    "dr_duo": "Paires de champions (bot+sup, jungle+mid)",
    "dr_bans": "Bans dans le pool de l'adversaire",
}
DRAFT_COLS = [f"d_{k}" for k in DRAFT_FACTORS]


def _mean(vals) -> float:
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else np.nan


def _diff(a: float, b: float) -> float:
    return a - b if a == a and b == b else np.nan


class DraftState:
    def __init__(self) -> None:
        self.champ_g: Counter = Counter()
        self.champ_w: Counter = Counter()
        self.champ_q: deque = deque()                # (date, [(champ, gagné)])
        self.meta_q: deque = deque()                 # (date, champions pris ou bannis)
        self.meta_cnt: Counter = Counter()
        self.meta_games = 0
        self.pc_g: Counter = Counter()               # (joueur, champion) -> games
        self.pc_w: Counter = Counter()
        self.pool: dict[str, set] = defaultdict(set)
        self.pair: dict[tuple, list] = defaultdict(list)     # duel (poste, c1, c2) -> [(date, c1 gagne)]
        self.duo: dict[tuple, list] = defaultdict(list)      # paire (postes, c1, c2) -> [(date, gagné)]
        self.last_lineup: dict[str, dict] = {}       # équipe -> {poste: joueur}

    def advance(self, date: pd.Timestamp) -> None:
        lo = date - pd.Timedelta(days=META_DAYS)
        while self.meta_q and self.meta_q[0][0] < lo:
            self.meta_cnt.subtract(self.meta_q.popleft()[1])
            self.meta_games -= 1
        lo = date - pd.Timedelta(days=WR_DAYS)
        while self.champ_q and self.champ_q[0][0] < lo:
            for champ, won in self.champ_q.popleft()[1]:
                self.champ_g[champ] -= 1
                self.champ_w[champ] -= won

    def _wr(self, champ: str) -> float:
        return (self.champ_w[champ] + 0.5 * K_CHAMP) / (self.champ_g[champ] + K_CHAMP)

    def _recent(self, hist: list, date: pd.Timestamp) -> float:
        """Écart à 50 % d'une liste [(date, gagné)] sur PAIR_DAYS, ramené vers 0."""
        lo = date - pd.Timedelta(days=PAIR_DAYS)
        vals = [w for d, w in hist if d >= lo]
        return (sum(vals) - 0.5 * len(vals)) / (len(vals) + K_PAIR)

    def team_side(self, own: dict, opp: dict, bans: list, date: pd.Timestamp) -> dict:
        """`own` / `opp` : {poste: (joueur, champion)}. Signaux d'UN côté (non différenciés)."""
        picks = [(role, pid, c) for role, (pid, c) in own.items() if c]
        match = []
        for role, _, c in picks:
            oc = opp.get(role, (None, None))[1]
            if oc and oc != c:
                key = (role, c, oc) if c <= oc else (role, oc, c)
                v = self._recent(self.pair.get(key, ()), date)
                match.append(v if key[1] == c else -v)
        duo = []
        for r1, r2 in DUOS:
            c1, c2 = own.get(r1, (None, None))[1], own.get(r2, (None, None))[1]
            if c1 and c2:
                duo.append(self._recent(self.duo.get((r1, c1, c2), ()), date))
        opp_pool: set = set()
        for pid, _ in opp.values():
            opp_pool |= self.pool.get(pid, set())
        return {
            "dr_champ_wr": _mean([self._wr(c) for _, _, c in picks]),
            "dr_presence": _mean([self.meta_cnt[c] / self.meta_games if self.meta_games else 0.0
                                  for _, _, c in picks]),
            "dr_comfort": _mean([np.log1p(self.pc_g[(pid, c)]) for _, pid, c in picks]),
            "dr_player_wr": _mean([(self.pc_w[(pid, c)] - 0.5 * self.pc_g[(pid, c)])
                                   / (self.pc_g[(pid, c)] + K_PLAYER_CHAMP) for _, pid, c in picks]),
            "dr_matchup": _mean(match) if match else 0.0,
            "dr_duo": _mean(duo) if duo else 0.0,
            "dr_bans": _mean([b in opp_pool for b in bans]) if bans else np.nan,
        }

    def features(self, blue: dict, red: dict, bans_blue: list, bans_red: list,
                 date: pd.Timestamp) -> dict:
        fb = self.team_side(blue, red, bans_blue, date)
        fr = self.team_side(red, blue, bans_red, date)
        out = {f"d_{k}": _diff(fb[k], fr[k]) for k in DRAFT_FACTORS}
        out["d_dr_matchup"] = fb["dr_matchup"]          # déjà un duel : bleu contre rouge
        return out

    def update(self, date: pd.Timestamp, blue: dict, red: dict, bans: list, yb: float,
               blue_team: str, red_team: str) -> None:
        played = []
        for own, won, team in ((blue, yb, blue_team), (red, 1 - yb, red_team)):
            for role, (pid, c) in own.items():
                if not c:
                    continue
                self.pc_g[(pid, c)] += 1
                self.pc_w[(pid, c)] += won
                if self.pc_g[(pid, c)] >= COMFORT_GAMES:
                    self.pool[pid].add(c)
                self.champ_g[c] += 1
                self.champ_w[c] += won
                played.append((c, won))
            for r1, r2 in DUOS:
                c1, c2 = own.get(r1, (None, None))[1], own.get(r2, (None, None))[1]
                if c1 and c2:
                    self.duo[(r1, c1, c2)].append((date, won))
            if own:
                self.last_lineup[team] = {role: pid for role, (pid, _) in own.items()}
        for role in ROLES:
            cb, cr = blue.get(role, (None, None))[1], red.get(role, (None, None))[1]
            if cb and cr and cb != cr:
                key = (role, cb, cr) if cb <= cr else (role, cr, cb)
                self.pair[key].append((date, yb if key[1] == cb else 1 - yb))
        self.champ_q.append((date, played))
        seen = [c for c, _ in played] + list(bans)
        self.meta_q.append((date, seen))
        self.meta_cnt.update(seen)
        self.meta_games += 1


def build_draft(cfg: dict | None = None) -> tuple[pd.DataFrame, DraftState]:
    """Signaux de draft as-of de chaque game + état final (pour la page Draft / le direct)."""
    cfg = cfg or load_config()
    games = load_games(cfg, history=True)
    players, teams, _ = _load_rows(cfg)
    stt = DraftState()
    rows = []
    for g in games.itertuples():
        stt.advance(g.date)
        lu, tm = players.get(g.gameid, {}), teams.get(g.gameid, {})
        blue, red = dict(lu.get(g.blue, {})), dict(lu.get(g.red, {}))
        bb, br = tm.get(g.blue, {}).get("bans", []), tm.get(g.red, {}).get("bans", [])
        rows.append({"gameid": g.gameid, **stt.features(blue, red, bb, br, g.date)})
        stt.update(g.date, blue, red, bb + br, float(g.yb), g.blue, g.red)
    return pd.DataFrame(rows), stt


def draft_row(stt: DraftState, blue: str, red: str, picks_blue: dict, picks_red: dict,
              bans_blue: list | None = None, bans_red: list | None = None,
              date: pd.Timestamp | None = None,
              lineup_blue: dict | None = None, lineup_red: dict | None = None) -> dict:
    """Signaux de draft d'un match saisi à la main. `picks_*` : {poste: champion}.
    Les joueurs sont ceux du cinq attendu (`lineup_*`), sinon du dernier cinq aligné."""
    date = date or pd.Timestamp.now().normalize()
    stt.advance(date)
    lb = lineup_blue if lineup_blue is not None else stt.last_lineup.get(blue, {})
    lr = lineup_red if lineup_red is not None else stt.last_lineup.get(red, {})
    b = {role: (lb.get(role, f"?{blue}:{role}"), c) for role, c in picks_blue.items() if c}
    r = {role: (lr.get(role, f"?{red}:{role}"), c) for role, c in picks_red.items() if c}
    return stt.features(b, r, list(bans_blue or []), list(bans_red or []), date)

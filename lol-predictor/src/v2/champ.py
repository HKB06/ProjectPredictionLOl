"""Joueur SUR champion (ajout du 18/11/2026) : la note d'un joueur sur le champion qu'il
prend, et le duel « joueur sur champion contre joueur sur champion ».

Ces signaux dépendent des picks : ils ne sont connus qu'une fois la draft posée (page
« Draft », pari en direct). Passe chronologique stricte : signaux calculés AVANT chaque game,
état mis à jour APRÈS.

Tout est mesuré en ÉCART À L'ATTENDU, pas en victoires brutes :
  - résultat : gagné − proba que l'Elo des joueurs donnait à son équipe ;
  - couloir  : or d'avance à 15 min sur son vis-à-vis − ce que le niveau habituel des deux
               joueurs en couloir laissait attendre.
Un joueur fort ne paraît donc pas « bon sur tout » : seul ce qu'il fait EN PLUS sur ce
champion compte.

La méta change : chaque game passée pèse deux fois moins tous les `HALF_LIFE` jours, et
chaque mesure est ramenée vers 0 tant qu'il y a peu de games (`K_*`).

Signaux (écart bleu − rouge) :
  - dr_pc_elo    : note du joueur sur CE champion (résultat) ;
  - dr_pc_lane   : idem, en or à 15 min ;
  - dr_duel      : duel poste pour poste, du plus précis au plus large :
                     1. ce joueur sur ce champion contre CE joueur sur CE champion ;
                     2. ce joueur sur ce champion contre ce champion (peu importe qui) ;
                     3. ce joueur (n'importe quel champion) contre ce champion.
                   Chaque niveau s'appuie sur le suivant quand il manque de games ;
  - dr_duel_lane : le même duel, en or à 15 min.
Avant la draft (`pool_elo`) : la note moyenne du joueur sur les champions qu'il joue en ce
moment, pondérée par ses picks récents.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.ingest.load_oracle import load_config
from src.v2.extra import _observations
from src.v2.player_engine import ROLES, _load_rows

HALF_LIFE = 365.0        # jours : poids d'une game divisé par 2
K_PC = 6.0               # games « fictives » à 0 pour la note sur champion
K_L0, K_L1, K_L2 = 2.0, 4.0, 8.0     # duel : exact, joueur+champion contre champion, joueur contre champion
K_BASE = 5.0             # niveau habituel du joueur en couloir
POOL_HALF_LIFE = 60.0    # picks récents du joueur (pour la note d'avant draft)
_DAY = 8.64e13           # nanosecondes par jour

CHAMP_FACTORS = {
    "dr_pc_elo": "Note du joueur sur ce champion (résultat, vs Elo)",
    "dr_pc_lane": "Note du joueur sur ce champion (or à 15 min)",
    "dr_duel": "Duel joueur sur champion contre joueur sur champion",
    "dr_duel_lane": "Duel joueur sur champion : or à 15 min",
}
CHAMP_COLS = [f"d_{k}" for k in CHAMP_FACTORS]
POOL_COL = "d_pool_elo"


def _rec() -> list:
    return [0.0, 0.0, 0.0, 0.0, 0.0]          # [jour, somme résultat, n, somme or, n or]


def _pick() -> dict:
    return {}


class ChampState:
    def __init__(self, half_life: float = HALF_LIFE) -> None:
        self.hl = float(half_life)
        self.pc: dict[tuple, list] = defaultdict(_rec)      # (joueur, champion)
        self.l0: dict[tuple, list] = defaultdict(_rec)      # (j1, c1, j2, c2), j1 <= j2
        self.l1: dict[tuple, list] = defaultdict(_rec)      # (joueur, son champion, champion d'en face)
        self.l2: dict[tuple, list] = defaultdict(_rec)      # (joueur, champion d'en face)
        self.base: dict[str, list] = defaultdict(_rec)      # niveau habituel en couloir
        self.picks: dict[str, dict] = defaultdict(_pick)    # joueur -> {champion: [jour, poids]}
        self.last_lineup: dict[str, dict] = {}

    # ----------------------------------------------------------------- lecture
    def _get(self, table: dict, key, t: float) -> tuple[float, float, float, float]:
        r = table.get(key)
        if r is None:
            return 0.0, 0.0, 0.0, 0.0
        f = 0.5 ** ((t - r[0]) / self.hl) if np.isfinite(self.hl) else 1.0
        return r[1] * f, r[2] * f, r[3] * f, r[4] * f

    def _put(self, table: dict, key, t: float, res: float, gd: float) -> None:
        r = table[key]
        f = 0.5 ** ((t - r[0]) / self.hl) if np.isfinite(self.hl) and r[2] + r[4] else 1.0
        r[0] = t
        r[1], r[2] = r[1] * f + res, r[2] * f + 1.0
        if gd == gd:
            r[3], r[4] = r[3] * f + gd, r[4] * f + 1.0
        else:
            r[3], r[4] = r[3] * f, r[4] * f

    def lane_level(self, pid: str, t: float) -> float:
        """Or d'avance habituel du joueur à 15 min (milliers), ramené vers 0."""
        _, _, s, n = self._get(self.base, pid, t)
        return s / (n + K_BASE)

    def note(self, pid: str, champ: str, t: float) -> tuple[float, float, float]:
        """(note résultat, note couloir, games pondérées) du joueur sur ce champion."""
        s, n, sg, ng = self._get(self.pc, (pid, champ), t)
        return s / (n + K_PC), sg / (ng + K_PC), n

    def pool_note(self, pid: str, t: float) -> float:
        """Note moyenne sur les champions joués récemment (connue avant la draft)."""
        tot = w = 0.0
        for champ, (t0, n0) in self.picks.get(pid, {}).items():
            wt = n0 * 0.5 ** ((t - t0) / POOL_HALF_LIFE)
            if wt > 0.05:
                tot += wt * self.note(pid, champ, t)[0]
                w += wt
        return tot / w if w else 0.0

    def duel(self, a: tuple, b: tuple, t: float) -> tuple[float, float, tuple]:
        """Duel de `a` = (joueur, champion) contre `b`, vu de `a` : (résultat, couloir,
        games pondérées aux 3 niveaux)."""
        (pa, ca), (pb, cb) = a, b
        s2a, n2a, g2a, m2a = self._get(self.l2, (pa, cb), t)
        s2b, n2b, g2b, m2b = self._get(self.l2, (pb, ca), t)
        s1a, n1a, g1a, m1a = self._get(self.l1, (pa, ca, cb), t)
        s1b, n1b, g1b, m1b = self._get(self.l1, (pb, cb, ca), t)
        key = (pa, ca, pb, cb) if pa <= pb else (pb, cb, pa, ca)
        sign = 1.0 if key[0] == pa else -1.0
        s0, n0, g0, m0 = self._get(self.l0, key, t)
        e2 = (s2a - s2b) / (n2a + n2b + K_L2)
        e1 = (s1a - s1b + K_L1 * e2) / (n1a + n1b + K_L1)
        e0 = (sign * s0 + K_L0 * e1) / (n0 + K_L0)
        l2 = (g2a - g2b) / (m2a + m2b + K_L2)
        l1 = (g1a - g1b + K_L1 * l2) / (m1a + m1b + K_L1)
        l0 = (sign * g0 + K_L0 * l1) / (m0 + K_L0)
        return e0, l0, (n0, n1a + n1b, n2a + n2b)

    # ---------------------------------------------------------------- signaux
    def features(self, blue: dict, red: dict, date: pd.Timestamp) -> dict:
        """`blue` / `red` : {poste: (joueur, champion)}."""
        t = date.value / _DAY
        side = {}
        for name, own in (("b", blue), ("r", red)):
            notes = [self.note(pid, c, t) for pid, c in own.values() if c]
            side[name] = (float(np.mean([x[0] for x in notes])) if notes else np.nan,
                          float(np.mean([x[1] for x in notes])) if notes else np.nan,
                          float(np.mean([x[2] for x in notes])) if notes else np.nan)
        res, lane, cov = [], [], []
        for role in ROLES:
            a, b = blue.get(role), red.get(role)
            if not a or not b or not a[1] or not b[1] or a[0] == b[0]:
                continue
            e, l, n = self.duel(a, b, t)
            res.append(e)
            lane.append(l)
            cov.append(n)
        cov = np.mean(cov, axis=0) if cov else (0.0, 0.0, 0.0)
        return {
            "d_dr_pc_elo": side["b"][0] - side["r"][0],
            "d_dr_pc_lane": side["b"][1] - side["r"][1],
            "d_dr_duel": float(np.mean(res)) if res else 0.0,
            "d_dr_duel_lane": float(np.mean(lane)) if lane else 0.0,
            "pc_games": float(np.nanmean([side["b"][2], side["r"][2]])) if (blue or red) else 0.0,
            "duel_n0": float(cov[0]), "duel_n1": float(cov[1]), "duel_n2": float(cov[2]),
        }

    def pool_diff(self, la: dict, lb: dict, date: pd.Timestamp) -> float:
        """`la` / `lb` : {poste: joueur}. Écart des notes moyennes sur les picks récents."""
        t = date.value / _DAY
        a = [self.pool_note(p, t) for p in la.values() if p]
        b = [self.pool_note(p, t) for p in lb.values() if p]
        return float(np.mean(a) - np.mean(b)) if a and b else 0.0

    # ------------------------------------------------------------ mise à jour
    def update(self, date: pd.Timestamp, blue: dict, red: dict, gd_blue: dict,
               resid: float, blue_team: str, red_team: str) -> None:
        """`gd_blue` : {poste: or d'avance du joueur BLEU à 15 min, en milliers} ;
        `resid` : résultat du bleu − proba que l'Elo des joueurs lui donnait."""
        t = date.value / _DAY
        for role in ROLES:
            a, b = blue.get(role), red.get(role)
            gd = gd_blue.get(role, np.nan)
            lane = np.nan
            if a and b and gd == gd:
                lane = gd - (self.lane_level(a[0], t) - self.lane_level(b[0], t))
            for (own, opp, r, ln, raw) in ((a, b, resid, lane, gd), (b, a, -resid, -lane, -gd)):
                if not own or not own[1]:
                    continue
                pid, c = own
                self._put(self.pc, (pid, c), t, r, ln)
                pk = self.picks[pid].get(c)
                if pk is None:
                    self.picks[pid][c] = [t, 1.0]
                else:
                    pk[1] = pk[1] * 0.5 ** ((t - pk[0]) / POOL_HALF_LIFE) + 1.0
                    pk[0] = t
                if opp and opp[1] and opp[0] != pid:
                    self._put(self.l1, (pid, c, opp[1]), t, r, ln)
                    self._put(self.l2, (pid, opp[1]), t, r, ln)
            if a and b and a[1] and b[1] and a[0] != b[0]:
                key = (a[0], a[1], b[0], b[1]) if a[0] <= b[0] else (b[0], b[1], a[0], a[1])
                s = 1.0 if key[0] == a[0] else -1.0
                self._put(self.l0, key, t, s * resid, s * lane)
            if a and b and gd == gd:               # niveau en couloir : après les notes
                self._put(self.base, a[0], t, 0.0, gd)
                self._put(self.base, b[0], t, 0.0, -gd)
        for team, own in ((blue_team, blue), (red_team, red)):
            if own:
                self.last_lineup[team] = {role: pid for role, (pid, _) in own.items()}


def build_champ(df: pd.DataFrame, cfg: dict | None = None,
                half_life: float = HALF_LIFE) -> tuple[pd.DataFrame, ChampState]:
    """Signaux as-of par game + état final. `df` : jeu de données v2 (gameid, date, blue,
    red, yb, p_player, p_elo)."""
    cfg = cfg or load_config()
    players, _, _ = _load_rows(cfg)
    _, lineups = _observations(cfg)
    stt = ChampState(half_life)
    rows = []
    for g in df.sort_values(["date", "gameid"]).itertuples():
        lu = players.get(g.gameid, {})
        blue, red = dict(lu.get(g.blue, {})), dict(lu.get(g.red, {}))
        feats = stt.features(blue, red, g.date)
        feats[POOL_COL] = stt.pool_diff({r: p for r, (p, _) in blue.items()},
                                        {r: p for r, (p, _) in red.items()}, g.date)
        rows.append({"gameid": g.gameid, **feats})
        gd = {role: v[1] for role, v in lineups.get(g.gameid, {}).get("blue", {}).items()}
        p = g.p_player if g.p_player == g.p_player else g.p_elo
        stt.update(g.date, blue, red, gd, float(g.yb) - float(p), g.blue, g.red)
    return pd.DataFrame(rows), stt


def champ_detail(stt: ChampState, lineup_blue: dict, lineup_red: dict, picks_blue: dict,
                 picks_red: dict, date: pd.Timestamp | None = None) -> list[dict]:
    """Poste par poste : la note de chaque joueur sur son champion et le duel, avec le
    nombre de games (pondérées par l'ancienneté) sur lequel chaque chiffre repose."""
    date = date or pd.Timestamp.now().normalize()
    t = date.value / _DAY
    out = []
    for role in ROLES:
        row = {"role": role}
        for side, lu, picks in (("blue", lineup_blue, picks_blue), ("red", lineup_red, picks_red)):
            pid, c = lu.get(role), picks.get(role)
            row[f"{side}_pid"], row[f"{side}_champ"] = pid, c
            if pid and c:
                res, lane, n = stt.note(pid, c, t)
                row.update({f"{side}_res": res, f"{side}_lane": lane, f"{side}_games": n})
        a = (lineup_blue.get(role), picks_blue.get(role))
        b = (lineup_red.get(role), picks_red.get(role))
        if all(a) and all(b) and a[0] != b[0]:
            e, l, n = stt.duel(a, b, t)
            row.update({"duel": e, "duel_lane": l, "n_exact": n[0], "n_pc_c": n[1], "n_p_c": n[2]})
        out.append(row)
    return out


def champ_row(stt: ChampState, blue: str, red: str, picks_blue: dict, picks_red: dict,
              date: pd.Timestamp | None = None,
              lineup_blue: dict | None = None, lineup_red: dict | None = None) -> dict:
    """Signaux d'un match saisi à la main. `picks_*` : {poste: champion}."""
    date = date or pd.Timestamp.now().normalize()
    lb = lineup_blue if lineup_blue is not None else stt.last_lineup.get(blue, {})
    lr = lineup_red if lineup_red is not None else stt.last_lineup.get(red, {})
    b = {role: (lb.get(role, f"?{blue}:{role}"), c) for role, c in picks_blue.items() if c}
    r = {role: (lr.get(role, f"?{red}:{role}"), c) for role, c in picks_red.items() if c}
    return stt.features(b, r, date)

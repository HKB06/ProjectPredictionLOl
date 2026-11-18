"""Signaux ajoutés au v2 le 18/11/2026 : objectifs en jeu, face-à-face, préparation soloQ.

Passe chronologique stricte, comme les autres moteurs : signaux calculés AVANT chaque game,
état mis à jour APRÈS.

Bloc « Objectifs » (profil de l'équipe sur ses `OBJ_WINDOW` dernières games) :
  - obj_dragons / obj_barons / obj_towers : part des dragons, barons, tours pris ;
  - obj_first   : premiers objectifs (premier sang, première tour, premier dragon, héraut) ;
  - obj_vision  : score de vision par minute, en part du total des deux équipes ;
  - obj_damage  : dégâts aux champions par minute, en part du total des deux équipes.
  Tout est centré sur 0,5 (« autant que l'adversaire ») et ramené vers 0 quand l'équipe a
  peu de games.

Bloc « Face-à-face » (rencontres des 2 dernières années, en écart à ce que l'Elo prévoyait) :
  - h2h_team    : ces deux équipes l'une contre l'autre ;
  - h2h_player  : poste pour poste, les deux joueurs l'un contre l'autre (résultat) ;
  - h2h_lane    : poste pour poste, l'écart d'or à 15 min entre ces deux joueurs.
  Un joueur garde son historique quand il change d'équipe.

Bloc « SoloQ » (`src.features.soloq_prep`, test du 18/11/2026) : moyenne des joueurs
couverts de l'équipe, si au moins `SQ_MIN_PLAYERS` le sont dans CHAQUE équipe :
  - sq_vol7     : parties classées jouées les 7 jours d'avant ;
  - sq_role14   : parties jouées à son poste les 14 jours d'avant.
  Vide (signal neutre) avant mars 2025, hors ligues pariables, ou quand l'historique soloQ
  d'un joueur est tronqué par Riot.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.ingest.load_oracle import load_config, read_seasons

OBJ_WINDOW = 20
K_OBJ = 5.0
K_H2H = 4.0
K_H2H_PLAYER = 3.0
H2H_DAYS = 730
SQ_MIN_PLAYERS = 4
SQ_SINCE = "2025-03-10"          # début de la collecte soloQ (01/01/2025) + fenêtre de 67 jours
ROLES = ["top", "jng", "mid", "bot", "sup"]
ROSTER_ROLE = {"top": "top", "jungle": "jng", "mid": "mid", "bot": "bot", "support": "sup"}
_SOLOQ_ROLE = {"top": "TOP", "jng": "JUNGLE", "mid": "MIDDLE", "bot": "BOTTOM", "sup": "UTILITY"}

OBJ = ["obj_dragons", "obj_barons", "obj_towers", "obj_first", "obj_vision", "obj_damage"]
EXTRA_FACTORS = {
    "obj_dragons": ("Objectifs", "Part des dragons pris"),
    "obj_barons": ("Objectifs", "Part des barons pris"),
    "obj_towers": ("Objectifs", "Part des tours prises"),
    "obj_first": ("Objectifs", "Premiers objectifs (sang, tour, dragon, héraut)"),
    "obj_vision": ("Objectifs", "Vision (vs adversaire)"),
    "obj_damage": ("Objectifs", "Dégâts aux champions (vs adversaire)"),
    "h2h_team": ("Face-à-face", "Rencontres passées entre ces deux équipes (vs Elo)"),
    "h2h_player": ("Face-à-face", "Joueur contre joueur, poste pour poste (vs Elo)"),
    "h2h_lane": ("Face-à-face", "Joueur contre joueur : écart d'or à 15 min"),
    "sq_vol7": ("SoloQ", "Parties soloQ sur 7 jours"),
    "sq_role14": ("SoloQ", "Parties soloQ à son poste sur 14 jours"),
}
EXTRA_COLS = [f"d_{k}" for k in EXTRA_FACTORS]

_TEAM_COLS = {"gameid", "side", "position", "dragons", "opp_dragons", "barons", "opp_barons",
              "towers", "opp_towers", "firstblood", "firstdragon", "firstherald", "firsttower",
              "vspm", "dpm"}
_PLAYER_COLS = {"gameid", "side", "position", "playername", "playerid", "golddiffat15"}
_SQ_EMPTY = ["gameid", "teamname", "sq_vol7", "sq_role14"]


def _deque() -> deque:
    return deque(maxlen=OBJ_WINDOW)


def _team_hist() -> dict:
    return {k: _deque() for k in OBJ}


def _sides() -> dict:
    return {"blue": {}, "red": {}}


def _share(a, b) -> float:
    """a / (a + b) − 0,5 ; NaN si l'un manque ou si rien n'a été pris."""
    if a != a or b != b or a + b <= 0:
        return np.nan
    return a / (a + b) - 0.5


def _shrunk(vals, k: float) -> float:
    vals = [v for v in vals if v == v]
    return float(sum(vals) / (len(vals) + k)) if vals else 0.0


# ------------------------------------------------------------------- lecture
def _observations(cfg: dict) -> tuple[dict, dict]:
    """(objectifs par game et par côté, joueurs par game et par côté)."""
    raw = read_seasons(cfg, _TEAM_COLS | _PLAYER_COLS, history=True)
    raw["position"] = raw["position"].astype(str).str.lower()
    raw["side"] = raw["side"].astype(str).str.lower()
    t = raw[raw["position"] == "team"]
    num = [c for c in _TEAM_COLS if c not in ("gameid", "side", "position")]
    t = t.assign(**{c: pd.to_numeric(t[c], errors="coerce") for c in num})
    opp = {(g, s): (v, d) for g, s, v, d in zip(t["gameid"], t["side"], t["vspm"], t["dpm"])}
    obs: dict[str, dict] = defaultdict(dict)
    for r in t.itertuples(index=False):
        ov, od = opp.get((r.gameid, "red" if r.side == "blue" else "blue"), (np.nan, np.nan))
        firsts = [v for v in (r.firstblood, r.firstdragon, r.firstherald, r.firsttower) if v == v]
        obs[r.gameid][r.side] = {
            "obj_dragons": _share(r.dragons, r.opp_dragons),
            "obj_barons": _share(r.barons, r.opp_barons),
            "obj_towers": _share(r.towers, r.opp_towers),
            "obj_first": (sum(firsts) / len(firsts) - 0.5) if firsts else np.nan,
            "obj_vision": _share(r.vspm, ov),
            "obj_damage": _share(r.dpm, od),
        }
    p = raw[raw["position"].isin(ROLES)].copy()
    p["pid"] = p["playerid"].fillna(p["playername"])
    p["gd"] = pd.to_numeric(p["golddiffat15"], errors="coerce") / 1000.0
    lineups: dict[str, dict] = defaultdict(_sides)
    for gid, side, pos, pid, gd in zip(p["gameid"], p["side"], p["position"], p["pid"], p["gd"]):
        if isinstance(pid, str) and side in ("blue", "red"):
            lineups[gid][side][pos] = (pid, gd)
    return dict(obs), dict(lineups)


# --------------------------------------------------------------------- l'état
class ExtraState:
    def __init__(self) -> None:
        self.obj: dict[str, dict] = defaultdict(_team_hist)
        self.team_h2h: dict[tuple, list] = defaultdict(list)      # (a, b) triés -> [(date, résidu de a)]
        self.player_h2h: dict[tuple, list] = defaultdict(list)    # (pa, pb) triés -> [(date, résidu, or)]
        self.last_lineup: dict[str, dict] = {}                    # équipe -> {poste: joueur}
        self.soloq_now: dict[str, dict] = {}                      # page -> {vol_7, roles, complete}
        self.soloq_asof: pd.Timestamp | None = None

    def obj_features(self, team: str) -> dict:
        h = self.obj[team]
        return {k: _shrunk(h[k], K_OBJ) for k in OBJ}

    def h2h_features(self, a: str, b: str, la: dict, lb: dict, date: pd.Timestamp) -> dict:
        lo = date - pd.Timedelta(days=H2H_DAYS)
        key = (a, b) if a <= b else (b, a)
        sign = 1.0 if key[0] == a else -1.0
        team = sign * _shrunk([r for d, r in self.team_h2h.get(key, ()) if d >= lo], K_H2H)
        res, lane = [], []
        for role in ROLES:
            pa, pb = la.get(role), lb.get(role)
            if not pa or not pb or pa == pb:
                continue
            k = (pa, pb) if pa <= pb else (pb, pa)
            s = 1.0 if k[0] == pa else -1.0
            past = [(r, g) for d, r, g in self.player_h2h.get(k, ()) if d >= lo]
            res.append(s * _shrunk([r for r, _ in past], K_H2H_PLAYER))
            lane.append(s * _shrunk([g for _, g in past], K_H2H_PLAYER))
        return {"h2h_team": team,
                "h2h_player": float(np.mean(res)) if res else 0.0,
                "h2h_lane": float(np.mean(lane)) if lane else 0.0}

    def update(self, g, obs: dict, lu: dict) -> None:
        for side, team in (("blue", g.blue), ("red", g.red)):
            o = obs.get(side)
            if o:
                h = self.obj[team]
                for k in OBJ:
                    h[k].append(o[k])
            if lu.get(side):
                self.last_lineup[team] = {role: pid for role, (pid, _) in lu[side].items()}
        key = (g.blue, g.red) if g.blue <= g.red else (g.red, g.blue)
        resid = float(g.yb) - float(g.p_elo)
        self.team_h2h[key].append((g.date, resid if key[0] == g.blue else -resid))
        resid_p = float(g.yb) - float(g.p_player) if g.p_player == g.p_player else resid
        for role in ROLES:
            b, r = lu.get("blue", {}).get(role), lu.get("red", {}).get(role)
            if not b or not r or b[0] == r[0]:
                continue
            k = (b[0], r[0]) if b[0] <= r[0] else (r[0], b[0])
            s = 1.0 if k[0] == b[0] else -1.0
            gd = s * b[1] if b[1] == b[1] else np.nan
            self.player_h2h[k].append((g.date, s * resid_p, gd))


# ---------------------------------------------------------------------- soloQ
def _soloq_history(gameids: pd.Series) -> pd.DataFrame:
    """(gameid, teamname) -> sq_vol7, sq_role14 ; équipes avec >= SQ_MIN_PLAYERS joueurs couverts."""
    empty = pd.DataFrame(columns=_SQ_EMPTY)
    try:
        from src.features import soloq_prep as sp
        from src.ingest import soloq_history as sh
        if not (sh.DIR / "ids.csv").exists():
            return empty
        f = sp.build(since=SQ_SINCE)
    except Exception as e:                                   # pas de données soloQ : bloc neutre
        print(f"[extra] soloQ indisponible ({type(e).__name__}: {e}) : bloc laissé vide")
        return empty
    c = f[f["covered"].fillna(False).astype(bool) & f["gameid"].isin(set(gameids))]
    t = c.groupby(["gameid", "teamname"]).agg(n=("vol_7", "size"), sq_vol7=("vol_7", "mean"),
                                              sq_role14=("role_14", "mean"))
    return t[t["n"] >= SQ_MIN_PLAYERS].drop(columns="n").reset_index()


def _soloq_now() -> tuple[dict, pd.Timestamp | None]:
    """Pour chaque joueur suivi : parties des 7 derniers jours et, par poste, des 14 derniers."""
    try:
        from src.features import soloq_prep as sp
        from src.ingest import soloq_history as sh
        if not (sh.DIR / "ids.csv").exists():
            return {}, None
        vol, det = sp.soloq_tables()
    except Exception as e:
        print(f"[extra] soloQ indisponible ({type(e).__name__}: {e})")
        return {}, None
    if vol.empty:
        return {}, None
    now = vol["time"].max()
    valid = vol.attrs.get("valid_from") or {}
    t_now = np.datetime64(now.tz_convert(None).to_datetime64())
    v7 = vol[vol["time"] >= now - pd.Timedelta(days=7)].groupby("player").size().to_dict()
    roles: dict[str, dict] = defaultdict(dict)
    d14 = det[det["time"] >= now - pd.Timedelta(days=14)]
    for (player, pos), n in d14.groupby(["player", "position"]).size().items():
        roles[player][pos] = int(n)
    out = {}
    for page in vol["player"].unique():
        vf = valid.get(page)
        out[page] = {"vol_7": int(v7.get(page, 0)), "roles": roles.get(page, {}),
                     "complete": bool(vf is None or t_now - np.timedelta64(67, "D") >= vf)}
    return out, now.tz_convert(None)


def soloq_team_now(stt: "ExtraState", team: str, pages: dict | None = None) -> dict:
    """Préparation soloQ ACTUELLE d'une équipe. `pages` : {poste: fiche Leaguepedia} du cinq
    attendu (`src.v2.lineups`) ; sans lui, tous les joueurs de l'effectif suivi (`rosters`)."""
    out = {"sq_vol7": np.nan, "sq_role14": np.nan, "sq_players": 0}
    if not stt.soloq_now:
        return out
    if pages:
        v7, r14 = [], []
        for pos, page in pages.items():
            rec = stt.soloq_now.get(page)
            if pos in _SOLOQ_ROLE and rec and rec["complete"]:
                v7.append(rec["vol_7"])
                r14.append(rec["roles"].get(_SOLOQ_ROLE[pos], 0))
        if len(v7) >= SQ_MIN_PLAYERS:
            out.update(sq_vol7=float(np.mean(v7)), sq_role14=float(np.mean(r14)),
                       sq_players=len(v7))
        return out
    try:
        from src.ingest import rosters as ro
        from src.update.watchlist import norm
        cur = ro.read("current.csv")
    except Exception:
        return out
    if cur.empty:
        return out
    cur = cur[(cur["team"].map(norm) == norm(team)) & (cur["sub"].astype(str) == "0")]
    v7, r14 = [], []
    for player, role in zip(cur["player"], cur["role"]):
        pos = ROSTER_ROLE.get(str(role).strip().lower())
        rec = stt.soloq_now.get(player)
        if not pos or not rec or not rec["complete"]:
            continue
        v7.append(rec["vol_7"])
        r14.append(rec["roles"].get(_SOLOQ_ROLE[pos], 0))
    if len(v7) >= SQ_MIN_PLAYERS:
        out.update(sq_vol7=float(np.mean(v7)), sq_role14=float(np.mean(r14)), sq_players=len(v7))
    return out


# ------------------------------------------------------------------- la table
def build_extra(df: pd.DataFrame, cfg: dict | None = None,
                soloq: bool = True) -> tuple[pd.DataFrame, ExtraState]:
    """Signaux as-of par game (`d_*` = bleu − rouge) + état final.

    `df` : jeu de données v2 (gameid, date, blue, red, yb, p_elo, p_player)."""
    cfg = cfg or load_config()
    obs, lineups = _observations(cfg)
    stt = ExtraState()
    rows = []
    for g in df.sort_values(["date", "gameid"]).itertuples():
        lu = lineups.get(g.gameid, {})
        la = {role: pid for role, (pid, _) in lu.get("blue", {}).items()}
        lb = {role: pid for role, (pid, _) in lu.get("red", {}).items()}
        fb, fr = stt.obj_features(g.blue), stt.obj_features(g.red)
        row = {"gameid": g.gameid, **{f"d_{k}": fb[k] - fr[k] for k in OBJ}}
        row.update({f"d_{k}": v for k, v in stt.h2h_features(g.blue, g.red, la, lb, g.date).items()})
        rows.append(row)
        stt.update(g, obs.get(g.gameid, {}), lu)
    out = pd.DataFrame(rows)

    sq = _soloq_history(df["gameid"]) if soloq else pd.DataFrame(columns=_SQ_EMPTY)
    for k in ("sq_vol7", "sq_role14"):
        m = {(g, t): v for g, t, v in zip(sq["gameid"], sq["teamname"], sq[k])}
        b = np.array([m.get(x, np.nan) for x in zip(df["gameid"], df["blue"])], dtype=float)
        r = np.array([m.get(x, np.nan) for x in zip(df["gameid"], df["red"])], dtype=float)
        out = out.merge(pd.DataFrame({"gameid": df["gameid"].to_numpy(), f"d_{k}": b - r}),
                        on="gameid", how="left")
    if soloq:
        stt.soloq_now, stt.soloq_asof = _soloq_now()
    return out, stt


def extra_pair(stt: ExtraState | None, a: str, b: str, date: pd.Timestamp | None = None,
               la: dict | None = None, lb: dict | None = None,
               pa: dict | None = None, pb: dict | None = None) -> dict:
    """Signaux pré-match de A contre B (déjà en écart A − B). `la` / `lb` : {poste: joueur}
    du cinq attendu (sinon le dernier cinq aligné) ; `pa` / `pb` : leurs fiches Leaguepedia,
    pour le soloQ."""
    if stt is None:
        return {f"d_{k}": np.nan for k in EXTRA_FACTORS}
    date = date or pd.Timestamp.now().normalize()
    fa, fb = stt.obj_features(a), stt.obj_features(b)
    row = {f"d_{k}": fa[k] - fb[k] for k in OBJ}
    la = la if la is not None else stt.last_lineup.get(a, {})
    lb = lb if lb is not None else stt.last_lineup.get(b, {})
    h = stt.h2h_features(a, b, la, lb, date)
    row.update({f"d_{k}": v for k, v in h.items()})
    sa, sb = soloq_team_now(stt, a, pa), soloq_team_now(stt, b, pb)
    for k in ("sq_vol7", "sq_role14"):
        row[f"d_{k}"] = sa[k] - sb[k]
    return row

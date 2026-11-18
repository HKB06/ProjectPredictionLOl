"""Note de chaque joueur sur chaque game, et résumé écrit de chaque match (18/11/2026).

Tout vient des statistiques officielles déjà collectées (Oracle's Elixir) : aucune note
n'est reprise d'un autre site. Pour chaque joueur, huit mesures sont comparées à ce que
font les joueurs du MÊME POSTE (moyenne et écart-type appris sur les seules games
antérieures) :

    kda     kills et assists rapportés aux morts          dmg     part des dégâts de l'équipe
    kp      participation aux kills de l'équipe            gold    part de l'or gagné par l'équipe
    gd15    or d'avance sur son vis-à-vis à 15 min         xpd15   expérience d'avance à 15 min
    csd15   sbires d'avance à 15 min                       vision  score de vision par minute

Le poids de chaque mesure dépend du poste (`WEIGHTS`) : la vision compte pour un support,
les dégâts pour un bot. Deux scores en sortent :
  - `score`       : niveau de la partie du joueur, tel quel (un vainqueur a de meilleures
                    stats : c'est la note affichée, sur 10) ;
  - `score_resid` : le même, comparé aux joueurs du même poste AYANT EU LE MÊME RÉSULTAT.
                    C'est ce que le joueur a fait en plus ou en moins de son équipe ; c'est
                    la mesure testée dans le modèle.

Usage :
    python -m src.v2.perf            # reconstruit data/cache/perf.pkl et montre un exemple
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from src.ingest.load_oracle import ROOT, load_config, read_seasons, season_files
from src.update.elo import series_ids

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROLES = ["top", "jng", "mid", "bot", "sup"]
ROLE_FR = {"top": "top", "jng": "jungle", "mid": "mid", "bot": "bot", "sup": "support"}
STATS = ["kda", "kp", "dmg", "gold", "gd15", "xpd15", "csd15", "vision"]
WEIGHTS = {
    "top": {"kda": 1.0, "kp": 0.5, "dmg": 1.0, "gold": 0.5, "gd15": 1.0, "xpd15": 0.5, "csd15": 0.5, "vision": 0.25},
    "jng": {"kda": 1.0, "kp": 1.0, "dmg": 0.75, "gold": 0.5, "gd15": 1.0, "xpd15": 0.5, "csd15": 0.25, "vision": 0.5},
    "mid": {"kda": 1.0, "kp": 0.5, "dmg": 1.0, "gold": 0.5, "gd15": 1.0, "xpd15": 0.5, "csd15": 0.5, "vision": 0.25},
    "bot": {"kda": 1.0, "kp": 0.5, "dmg": 1.25, "gold": 0.75, "gd15": 1.0, "xpd15": 0.25, "csd15": 0.75, "vision": 0.25},
    "sup": {"kda": 1.0, "kp": 1.25, "dmg": 0.25, "gold": 0.0, "gd15": 0.75, "xpd15": 0.5, "csd15": 0.0, "vision": 1.25},
}
MIN_OBS = 300            # games du poste avant de noter (début de l'historique)
NOTE_MID, NOTE_SCALE = 6.0, 2.4      # note sur 10 = 6 + 2,4 × score, bornée à [1 ; 10]
CACHE = ROOT / "data" / "cache" / "perf.pkl"
DISPLAY_SEASONS = 2      # saisons gardées pour l'affichage

# Bloc « Niveau individuel » du modèle : moyenne des notes passées de chaque joueur, une game
# pesant deux fois moins tous les PERF_HALF_LIFE jours, ramenée vers 0 par K_PERF games
# fictives. Réglé sur 2023-2025 (demi-vie 1 an, 2 ans ou sans oubli : même résultat ;
# K = 5 meilleur que 15), puis jugé sur 2026.
PERF_HALF_LIFE = 730.0
K_PERF = 5.0
PERF_FACTORS = {
    "perf_raw": ("Niveau individuel", "Note moyenne des cinq joueurs (statistiques en jeu)"),
    "perf_res": ("Niveau individuel", "Note des joueurs au-delà du résultat de leur équipe"),
}
PERF_COLS = [f"d_{k}" for k in PERF_FACTORS]
_DAY = 8.64e13

_P_COLS = {"gameid", "date", "league", "side", "position", "playername", "playerid", "teamname",
           "champion", "result", "gamelength", "kills", "deaths", "assists", "teamkills",
           "damageshare", "earnedgoldshare", "vspm", "golddiffat15", "xpdiffat15", "csdiffat15",
           "dragons", "barons", "towers", "firstblood", "firstdragon", "firsttower", "game"}


def _expanding_z(x: pd.Series, groups: list[pd.Series]) -> pd.Series:
    """Écart à la moyenne des lignes PRÉCÉDENTES du même groupe, en écarts-types."""
    ok = x.notna().astype(float)
    v = x.fillna(0.0)
    g = [s.to_numpy() for s in groups]
    n = ok.groupby(g).cumsum() - ok
    s1 = v.groupby(g).cumsum() - v
    s2 = (v * v).groupby(g).cumsum() - v * v
    mean = s1 / n.where(n > 0)
    var = (s2 / n.where(n > 0) - mean ** 2).clip(lower=1e-9)
    z = (x - mean) / np.sqrt(var)
    return z.where(n >= MIN_OBS).clip(-3, 3)


_tables_cache: dict = {}


def build_tables(cfg: dict | None = None, history: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`_read_tables`, gardé en mémoire tant que les fichiers ne changent pas (le bloc du
    modèle et le cache d'affichage lisent les mêmes lignes)."""
    cfg = cfg or load_config()
    key = (history, tuple((str(p), p.stat().st_mtime) for _, p in season_files(cfg, history)))
    if _tables_cache.get("key") != key:
        _tables_cache.update(key=key, tables=_read_tables(cfg, history))
    return _tables_cache["tables"]


def _read_tables(cfg: dict, history: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(une ligne par joueur et par game avec ses notes, une ligne par équipe et par game)."""
    raw = read_seasons(cfg, _P_COLS, history=history)
    raw["position"] = raw["position"].astype(str).str.lower()
    raw["side"] = raw["side"].astype(str).str.lower()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["gameid", "date"]).sort_values(["date", "gameid"], kind="stable")

    t = raw[raw["position"] == "team"].copy()
    for c in ("result", "gamelength", "kills", "golddiffat15", "dragons", "barons", "towers",
              "firstblood", "firstdragon", "firsttower"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    teams = t[["gameid", "date", "league", "season", "side", "teamname", "result", "gamelength",
               "kills", "golddiffat15", "dragons", "barons", "towers", "firstblood",
               "firstdragon", "firsttower", "game"]].reset_index(drop=True)

    p = raw[raw["position"].isin(ROLES)].copy()
    num = ["result", "kills", "deaths", "assists", "teamkills", "damageshare", "earnedgoldshare",
           "vspm", "golddiffat15", "xpdiffat15", "csdiffat15"]
    p[num] = p[num].apply(pd.to_numeric, errors="coerce")
    p["pid"] = p["playerid"].fillna(p["playername"])
    p = p.dropna(subset=["pid", "teamname", "result"]).reset_index(drop=True)
    p["kda"] = np.log((p["kills"] + p["assists"] + 1.0) / (p["deaths"] + 1.0))
    p["kp"] = (p["kills"] + p["assists"]) / p["teamkills"].clip(lower=1)
    p["dmg"], p["gold"] = p["damageshare"], p["earnedgoldshare"]
    p["gd15"], p["xpd15"] = p["golddiffat15"] / 1000.0, p["xpdiffat15"] / 1000.0
    p["csd15"], p["vision"] = p["csdiffat15"], p["vspm"]

    w = pd.DataFrame(WEIGHTS).T.reindex(p["position"].to_numpy())[STATS].to_numpy()
    za = np.column_stack([_expanding_z(p[s], [p["position"]]) for s in STATS])
    zr = np.column_stack([_expanding_z(p[s], [p["position"], p["result"]]) for s in STATS])
    for name, z in (("score", za), ("score_resid", zr)):
        have = ~np.isnan(z)
        tot = (w * have).sum(axis=1)
        val = np.nansum(w * np.where(have, z, 0.0), axis=1) / np.where(tot > 0, tot, np.nan)
        p[name] = np.where(tot >= 2.0, val, np.nan)       # trop de mesures manquantes : pas de note
    p["note"] = (NOTE_MID + NOTE_SCALE * p["score"]).clip(1, 10).round(1)
    keep = ["gameid", "date", "league", "season", "side", "position", "pid", "playername",
            "teamname", "champion", "result", "kills", "deaths", "assists", "dmg", "kp", "gd15",
            "csd15", "vision", "score", "score_resid", "note"]
    return p[keep], teams


# --------------------------------------------------------- le bloc du modèle
def _prec() -> list:
    return [0.0, 0.0, 0.0, 0.0, 0.0]          # [jour, somme score, n, somme score_resid, n]


class PerfState:
    """Note moyenne (pondérée par l'ancienneté) de chaque joueur."""

    def __init__(self) -> None:
        self.p: dict[str, list] = defaultdict(_prec)

    def player(self, pid: str, t: float) -> tuple[float, float, float]:
        """(note brute, note au-delà du résultat, games pondérées), en écarts-types."""
        r = self.p.get(pid)
        if r is None:
            return 0.0, 0.0, 0.0
        f = 0.5 ** ((t - r[0]) / PERF_HALF_LIFE)
        return r[1] * f / (r[2] * f + K_PERF), r[3] * f / (r[4] * f + K_PERF), r[2] * f

    def team(self, pids: list, date: pd.Timestamp) -> dict:
        t = date.value / _DAY
        v = [self.player(p, t) for p in pids if p]
        if not v:
            return {"perf_raw": np.nan, "perf_res": np.nan}
        return {"perf_raw": float(np.mean([x[0] for x in v])),
                "perf_res": float(np.mean([x[1] for x in v]))}

    def update(self, date: pd.Timestamp, entries: list) -> None:
        """`entries` : [(joueur, score, score_resid)] de la game qui vient de se jouer."""
        t = date.value / _DAY
        for pid, s, sr in entries:
            r = self.p[pid]
            f = 0.5 ** ((t - r[0]) / PERF_HALF_LIFE) if r[2] + r[4] else 1.0
            r[0] = t
            r[1], r[2], r[3], r[4] = r[1] * f, r[2] * f, r[3] * f, r[4] * f
            if s == s:
                r[1] += s
                r[2] += 1.0
            if sr == sr:
                r[3] += sr
                r[4] += 1.0


def build_perf(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, PerfState]:
    """Signaux as-of par game (`d_perf_*` = bleu − rouge) + état final.
    `df` : jeu de données v2 (gameid, date)."""
    cfg = cfg or load_config()
    players, _ = build_tables(cfg)
    by_game: dict = defaultdict(lambda: {"blue": [], "red": []})
    for gid, side, pid, s, sr in zip(players["gameid"], players["side"], players["pid"],
                                     players["score"], players["score_resid"]):
        if side in ("blue", "red"):
            by_game[gid][side].append((pid, s, sr))
    stt = PerfState()
    rows = []
    for g in df.sort_values(["date", "gameid"]).itertuples():
        lu = by_game.get(g.gameid)
        row = {"gameid": g.gameid}
        if lu:
            fb = stt.team([p for p, _, _ in lu["blue"]], g.date)
            fr = stt.team([p for p, _, _ in lu["red"]], g.date)
            row.update({f"d_{k}": fb[k] - fr[k] for k in PERF_FACTORS})
            stt.update(g.date, lu["blue"] + lu["red"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["gameid", *PERF_COLS]), stt


def perf_team(stt: "PerfState | None", lineup: dict | None,
              date: pd.Timestamp | None = None) -> dict:
    """Niveau individuel d'un cinq attendu (`lineup` = {poste: joueur}) + la note sur 10 de
    chaque joueur, pour l'affichage. Un joueur inconnu vaut la moyenne de son poste."""
    out = {"perf_raw": np.nan, "perf_res": np.nan, "notes": {}}
    if stt is None or not lineup:
        return out
    date = date or pd.Timestamp.now().normalize()
    out.update(stt.team(list(lineup.values()), date))
    t = date.value / _DAY
    for role, pid in lineup.items():
        raw, res, n = stt.player(pid, t)
        out["notes"][role] = (round(NOTE_MID + NOTE_SCALE * raw, 1), round(res, 2), round(n))
    return out


# ------------------------------------------------------------------- le cache
def _key(cfg: dict) -> list:
    return [[str(p), int(p.stat().st_mtime)] for _, p in season_files(cfg, True)]


def load(cfg: dict | None = None, rebuild: bool = False) -> dict:
    """Notes et fiches d'équipe des `DISPLAY_SEASONS` dernières saisons, relues du disque
    tant que les fichiers Oracle ne changent pas."""
    cfg = cfg or load_config()
    key = _key(cfg)
    if not rebuild and CACHE.exists():
        try:
            with open(CACHE, "rb") as f:
                d = pickle.load(f)
            if d.get("key") == key:
                return d
        except Exception:  # noqa: BLE001 - cache d'une autre version
            pass
    players, teams = build_tables(cfg)
    keep = sorted(players["season"].unique())[-DISPLAY_SEASONS:]
    players = players[players["season"].isin(keep)].reset_index(drop=True)
    teams = teams[teams["season"].isin(keep)].reset_index(drop=True)
    # une série = mêmes équipes, même ligue, games qui se suivent
    g = teams.pivot_table(index=["gameid", "date", "league"], columns="side", values="teamname",
                          aggfunc="first").reset_index().sort_values(["date", "gameid"])
    g = g.dropna(subset=["blue", "red"])
    g["series"] = series_ids(g["league"], g["blue"], g["red"], g["date"])
    d = {"key": key, "players": players, "teams": teams,
         "games": g[["gameid", "date", "league", "blue", "red", "series"]].reset_index(drop=True)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE)
    return d


# ------------------------------------------------------------------ le résumé
def _fr(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def _line(r) -> str:
    extra = []
    if r.dmg == r.dmg:
        extra.append(f"{r.dmg * 100:.0f} % des dégâts")
    if r.gd15 == r.gd15 and abs(r.gd15) >= 0.5:
        extra.append(f"{'+' if r.gd15 > 0 else '−'}{_fr(abs(r.gd15))} k d'or à 15 min")
    kda = f"{int(r.kills)}/{int(r.deaths)}/{int(r.assists)}"
    return (f"{r.playername} ({r.champion}, {ROLE_FR[r.position]}, {r.teamname}) "
            f"{_fr(r.note)}/10 — {kda}" + (", " + ", ".join(extra) if extra else ""))


def game_summary(players: pd.DataFrame, teams: pd.DataFrame, gameid: str) -> dict:
    """Résumé d'UNE game : les faits, les phrases, et les dix notes."""
    t = teams[teams["gameid"] == gameid]
    p = players[players["gameid"] == gameid]
    if len(t) != 2 or p.empty:
        return {"lines": [], "notes": p}
    w, l = t[t["result"] == 1], t[t["result"] == 0]
    if len(w) != 1 or len(l) != 1:
        return {"lines": [], "notes": p}
    w, l = w.iloc[0], l.iloc[0]
    lines = []
    head = f"**{w.teamname}** bat {l.teamname}"
    if w.gamelength == w.gamelength:
        head += f" en {int(round(w.gamelength / 60))} min"
    if w.kills == w.kills and l.kills == l.kills:
        head += f" ({int(w.kills)} kills à {int(l.kills)})"
    lines.append(head + ".")
    gd = w.golddiffat15
    if gd == gd:
        if gd <= -1500:
            lines.append(f"Retournement : {l.teamname} menait de {_fr(-gd / 1000)} k d'or à 15 min.")
        elif gd >= 1500:
            lines.append(f"{w.teamname} menait déjà de {_fr(gd / 1000)} k d'or à 15 min.")
        else:
            lines.append(f"Début de partie équilibré ({'+' if gd >= 0 else '−'}{_fr(abs(gd) / 1000)} k "
                         f"d'or à 15 min pour {w.teamname}).")
    obj = [f"{int(a)} {name} à {int(b)}" for name, a, b in (
        ("dragons", w.dragons, l.dragons), ("barons", w.barons, l.barons),
        ("tours", w.towers, l.towers)) if a == a and b == b and a + b > 0]
    if obj:
        lines.append("Objectifs : " + ", ".join(obj) + ".")
    n = p.dropna(subset=["note"])
    if len(n) >= 6:
        best = n.sort_values("note", ascending=False).iloc[0]
        worst = n.sort_values("note").iloc[0]
        lines.append("Meilleur joueur : " + _line(best) + ".")
        lw = n[n["result"] == 1].sort_values("note")
        ll = n[n["result"] == 0].sort_values("note", ascending=False)
        if len(ll) and ll.iloc[0].note >= 6.5:
            lines.append("Le meilleur du côté perdant : " + _line(ll.iloc[0]) + ".")
        lines.append("Le plus en difficulté : " + _line(worst) + ".")
        if len(lw) and lw.iloc[0].pid != worst.pid and lw.iloc[0].note <= 5.0:
            lines.append("Le moins bon du côté gagnant : " + _line(lw.iloc[0]) + ".")
    lanes = {w.teamname: [], l.teamname: []}
    for r in p[p["side"] == w.side].itertuples():
        if r.gd15 == r.gd15 and abs(r.gd15) >= 0.7 and r.position != "sup":
            lanes[w.teamname if r.gd15 > 0 else l.teamname].append(ROLE_FR[r.position])
    won = [f"{', '.join(v)} pour {k}" for k, v in lanes.items() if v]
    if won:
        lines.append("Couloirs gagnés à 15 min : " + " ; ".join(won) + ".")
    return {"lines": lines, "notes": p, "winner": w.teamname, "loser": l.teamname}


def series_notes(players: pd.DataFrame, gameids: list) -> pd.DataFrame:
    """Note moyenne de chaque joueur sur les games d'une série."""
    p = players[players["gameid"].isin(gameids)]
    out = p.groupby(["teamname", "position", "pid", "playername"], as_index=False).agg(
        games=("note", "size"), note=("note", "mean"), kills=("kills", "sum"),
        deaths=("deaths", "sum"), assists=("assists", "sum"), gd15=("gd15", "mean"),
        champions=("champion", lambda s: ", ".join(dict.fromkeys(s.dropna()))))
    out["position"] = pd.Categorical(out["position"], ROLES, ordered=True)
    return out.sort_values(["teamname", "position"]).reset_index(drop=True)


def ranking(players: pd.DataFrame, season: int | None = None, leagues: list | None = None,
            min_games: int = 15) -> pd.DataFrame:
    """Classement des joueurs par note moyenne (saison et ligues au choix)."""
    p = players if season is None else players[players["season"] == season]
    if leagues:
        p = p[p["league"].isin(leagues)]
    last = p.sort_values("date").groupby("pid").agg(playername=("playername", "last"),
                                                    teamname=("teamname", "last"),
                                                    position=("position", "last"))
    agg = p.groupby("pid").agg(games=("note", "count"), note=("note", "mean"),
                               plus=("score_resid", "mean"), wins=("result", "mean"),
                               gd15=("gd15", "mean"))
    out = last.join(agg)
    out = out[out["games"] >= min_games].sort_values("note", ascending=False)
    return out.reset_index(drop=True)


def main() -> None:
    d = load(rebuild=True)
    pl, tm, g = d["players"], d["teams"], d["games"]
    print(f"{len(pl)} lignes joueur · {len(g)} games · notes : moyenne {pl['note'].mean():.2f}, "
          f"écart-type {pl['note'].std():.2f}, vainqueurs {pl.loc[pl['result'] == 1, 'note'].mean():.2f}, "
          f"perdants {pl.loc[pl['result'] == 0, 'note'].mean():.2f}")
    print(pl["note"].describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_string())
    last = g.sort_values("date").iloc[-1]
    print(f"\nDernière game : {last.blue} – {last.red} ({last.league}, {last.date:%d/%m/%Y})")
    for line in game_summary(pl, tm, last.gameid)["lines"]:
        print("  " + line)


if __name__ == "__main__":
    main()

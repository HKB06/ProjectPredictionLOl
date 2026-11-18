"""Facteurs de contexte d'un BO : tout ce que l'Elo d'équipe ne voit pas.

Passe chronologique stricte (même principe que `build_features`) : pour chaque game,
on calcule les facteurs des 2 équipes à partir de l'historique ANTÉRIEUR uniquement,
PUIS on met à jour l'état. Aucun résultat de la game courante ne fuite.

Facteurs (par équipe, as-of) :
  1. Fearless draft  : la ligue interdit-elle de rejouer un champion dans la série ?
                       + profondeur du champion pool des 5 joueurs (decisive en Fearless).
  2. Stabilité       : games joués par ce 5 exact, par le duo bot/sup, par le duo mid/jungle.
  3. Patch           : jours depuis la sortie du patch + surperformance historique de
                       l'équipe dans les 7 jours suivant un nouveau patch (adaptation).
  4. Forme           : surperformance vs Elo sur les 5 dernières games + forme de lane
                       des joueurs (gold diff @15 récent vs leur moyenne).
  5. Remplaçants     : joueurs alignés absents du 5 habituel des 10 dernières games.
  6. Calendrier      : jours de repos, games jouées sur 7 jours, expérience internationale.
  7. Grands matchs   : expérience playoffs des joueurs + surperformance de l'équipe en playoffs.

Non couverts (données absentes du CSV Oracle) : LAN vs online, voyages, enjeu du match
(nécessite les classements -> Leaguepedia).

La lineup utilisée en historique est celle réellement alignée (annoncée avant la game,
ce n'est pas une fuite de résultat). En pré-match (`team_snapshot`), on prend la
dernière lineup connue de l'équipe.

« Surperformance » = résultat − proba Elo K32+MOV (identique à `src.update.elo`) :
un facteur n'a de valeur prédictive que s'il explique ce que l'Elo rate.

Usage :
    python -m src.features.context_factors "T1" "Gen.G"
    python -m src.features.context_factors --eval
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict, deque
from functools import partial

import numpy as np
import pandas as pd

from src.ingest.load_oracle import load_config, read_seasons
from src.update.elo import BASE, EVENT_LEAGUES, K, _mov_mult, load_games, win_prob  # noqa: F401

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROLES = ["top", "jng", "mid", "bot", "sup"]

POOL_DAYS = 90          # fenêtre du champion pool
POOL_MIN_GAMES = 2      # un champion compte dans le pool s'il a été joué >= 2 fois
PATCH_NEW_DAYS = 7      # « nouveau patch » = 7 premiers jours
REST_CAP_DAYS = 14.0    # au-delà, c'est une intersaison : on ne distingue plus 20 j de 80 j
FORM_N = 5              # fenêtre de forme
MAIN_LINEUP_N = 10      # fenêtre du 5 habituel
SHRINK_K = 10.0         # lissage des surperformances vers 0
FEARLESS_MIN_OBS = 20   # games 2+ de série observées avant de trancher Fearless
FEARLESS_MAX_REPEAT = 0.02

_PLAYER_COLS = ["gameid", "date", "league", "playoffs", "patch", "position",
                "playerid", "playername", "teamname", "champion", "golddiffat15"]

FACTORS = {
    "pool_min": "Pool le plus court (champions)",
    "pool_mean": "Pool moyen (champions)",
    "lineup_games": "Games joués par ce 5 exact",
    "duo_bot": "Games du duo bot/sup",
    "duo_mid_jng": "Games du duo mid/jungle",
    "days_patch": "Jours depuis le patch",
    "patch_adapt": "Surperf. nouveaux patchs",
    "form_elo": "Surperf. vs Elo (5 dern.)",
    "form_lane": "Forme de lane (GD@15)",
    "n_subs": "Joueurs hors 5 habituel",
    "days_rest": "Jours de repos",
    "games_7d": "Games sur 7 jours",
    "intl_games": "Exp. internationale (games/joueur)",
    "playoff_games": "Exp. playoffs (games/joueur)",
    "playoff_adapt": "Surperf. en playoffs",
}


def _base_elo() -> float:
    return BASE


def _zero_pair() -> list:
    return [0, 0]


def _shrunk_mean(values) -> float:
    return float(sum(values) / (len(values) + SHRINK_K)) if values else 0.0


def load_player_games(cfg: dict | None = None, history: bool = True) -> pd.DataFrame:
    """Lignes joueur utiles (5 par équipe et par game), identifiant joueur stable."""
    cfg = cfg or load_config()
    df = read_seasons(cfg, set(_PLAYER_COLS), dtype={"patch": "string"}, history=history)
    df = df[df["position"].str.lower().isin(ROLES)].copy()
    df["position"] = df["position"].str.lower()
    df["pid"] = df["playerid"].fillna(df["playername"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["playoffs"] = pd.to_numeric(df["playoffs"], errors="coerce").fillna(0).astype(int)
    # Texte Python ordinaire (None si absent) : bien plus rapide à parcourir ligne à ligne.
    df["patch"] = df["patch"].astype(object).where(df["patch"].notna(), None)
    return df.dropna(subset=["date", "pid", "teamname"])


class _State:
    """État accumulé au fil des games (tout ce qui est connu « avant »)."""

    def __init__(self) -> None:
        # Valeurs par défaut nommées (pas de lambda) : l'état doit pouvoir être enregistré.
        self.elo: dict[str, float] = defaultdict(_base_elo)
        self.n: dict[str, int] = defaultdict(int)
        # joueur -> liste (date, champion, gd15)
        self.player_hist: dict[str, list] = defaultdict(list)
        # joueur -> gd15 connus, dans l'ordre, et leur somme (forme de lane sans tout relire)
        self.player_gd: dict[str, list] = defaultdict(list)
        self.player_gd_sum: dict[str, float] = defaultdict(float)
        self.player_playoffs: dict[str, int] = defaultdict(int)
        self.player_intl: dict[str, int] = defaultdict(int)
        # équipe
        self.lineups: dict[str, deque] = defaultdict(partial(deque, maxlen=MAIN_LINEUP_N))
        self.last_lineup: dict[str, dict] = {}
        self.lineup_count: Counter = Counter()
        self.duo_count: Counter = Counter()
        self.team_dates: dict[str, list] = defaultdict(list)
        self.last_date: pd.Timestamp | None = None           # dernière game rejouée
        self.resid: dict[str, deque] = defaultdict(partial(deque, maxlen=FORM_N))
        self.resid_newpatch: dict[str, list] = defaultdict(list)
        self.resid_playoffs: dict[str, list] = defaultdict(list)
        self.team_league: dict[str, str] = {}
        # patch -> 1re date vue ; Fearless : ligue -> [obs, répétitions]
        self.patch_start: dict[str, pd.Timestamp] = {}
        # Compté par saison : le Fearless est arrivé en cours de route, les répétitions
        # des saisons précédentes ne disent rien du format actuel.
        self.fearless: dict[str, list] = defaultdict(_zero_pair)
        self.fearless_season: dict[str, int] = {}
        self.series_champs: dict[tuple, set] = {}

    # ------------------------------------------------------------ lecture --
    def is_fearless(self, league: str) -> bool | None:
        obs, rep = self.fearless.get(league, (0, 0))
        if obs < FEARLESS_MIN_OBS:
            return None
        return rep / obs <= FEARLESS_MAX_REPEAT

    def team_features(self, team: str, lineup: dict, date: pd.Timestamp,
                      patch: str | None) -> dict:
        pids = [lineup.get(r) for r in ROLES if lineup.get(r)]
        lo = date - pd.Timedelta(days=POOL_DAYS)
        pools, lane_form = [], []
        for pid in pids:
            # Historique en ordre chronologique : on remonte depuis la fin et on s'arrête
            # dès qu'on sort de la fenêtre (plusieurs saisons = des centaines de games).
            recent: Counter = Counter()
            for d, c, _ in reversed(self.player_hist.get(pid, ())):
                if d < lo:
                    break
                if isinstance(c, str):
                    recent[c] += 1
            pools.append(sum(1 for v in recent.values() if v >= POOL_MIN_GAMES))
            gd = self.player_gd.get(pid, ())
            if len(gd) > FORM_N:
                last = sum(gd[-FORM_N:])
                lane_form.append(last / FORM_N
                                 - (self.player_gd_sum[pid] - last) / (len(gd) - FORM_N))

        main = Counter()
        for lu in self.lineups.get(team, []):
            main.update(lu.values())
        usual = {p for p, c in main.items() if c >= len(self.lineups.get(team, [])) / 2}
        n_subs = sum(1 for p in pids if usual and p not in usual)

        dates = self.team_dates.get(team, [])
        last = dates[-1] if dates else None
        start = self.patch_start.get(patch) if patch else None

        return {
            "pool_min": min(pools) if pools else 0,
            "pool_mean": float(np.mean(pools)) if pools else 0.0,
            "lineup_games": self.lineup_count[(team, frozenset(pids))],
            "duo_bot": self.duo_count[(lineup.get("bot"), lineup.get("sup"))],
            "duo_mid_jng": self.duo_count[(lineup.get("mid"), lineup.get("jng"))],
            "days_patch": (date - start).days if start is not None else np.nan,
            "patch_adapt": _shrunk_mean(self.resid_newpatch.get(team, [])),
            "form_elo": float(np.mean(self.resid[team])) if self.resid.get(team) else 0.0,
            "form_lane": float(np.mean(lane_form)) if lane_form else 0.0,
            "n_subs": n_subs,
            "days_rest": (min((date - last).total_seconds() / 86400, REST_CAP_DAYS)
                          if last is not None else np.nan),
            "games_7d": self._count_since(dates, date - pd.Timedelta(days=7)),
            "intl_games": float(np.mean([self.player_intl[p] for p in pids])) if pids else 0.0,
            "playoff_games": float(np.mean([self.player_playoffs[p] for p in pids])) if pids else 0.0,
            "playoff_adapt": _shrunk_mean(self.resid_playoffs.get(team, [])),
        }

    @staticmethod
    def _count_since(dates: list, lo: pd.Timestamp) -> int:
        n = 0
        for d in reversed(dates):
            if d < lo:
                break
            n += 1
        return n

    # --------------------------------------------------------- mise à jour --
    def update(self, g, lineups: dict, players: list[tuple], p_blue: float) -> None:
        """`players` : (équipe, poste, joueur, champion ou None, gold diff @15) de la game."""
        date, lg = g.date, g.league
        patch = g.patch if isinstance(g.patch, str) else None     # patch absent de certaines games
        self.last_date = date
        if self.fearless_season.get(lg) != g.season:
            self.fearless_season[lg] = g.season
            self.fearless[lg] = [0, 0]
        if patch and patch not in self.patch_start:
            self.patch_start[patch] = date
        new_patch = (patch in self.patch_start
                     and (date - self.patch_start[patch]).days <= PATCH_NEW_DAYS)
        day = date.normalize()
        pair = frozenset((g.blue, g.red))

        for team, won, p in ((g.blue, g.yb, p_blue), (g.red, 1 - g.yb, 1 - p_blue)):
            lu = lineups.get(team, {})
            res = float(won) - p
            self.resid[team].append(res)
            if new_patch:
                self.resid_newpatch[team].append(res)
            if g.playoffs:
                self.resid_playoffs[team].append(res)
            self.team_dates[team].append(date)
            if lu:
                pids = frozenset(v for v in lu.values() if v)
                self.lineup_count[(team, pids)] += 1
                self.duo_count[(lu.get("bot"), lu.get("sup"))] += 1
                self.duo_count[(lu.get("mid"), lu.get("jng"))] += 1
                self.lineups[team].append(lu)
                self.last_lineup[team] = lu
            self.team_league[team] = lg

            # Fearless : un champion déjà joué par CETTE équipe dans la série ?
            champs = {c for t, _, _, c, _ in players if t == team and c}
            key = (lg, day, pair, team)
            seen = self.series_champs.get(key)
            if seen is not None:
                self.fearless[lg][0] += 1
                self.fearless[lg][1] += int(bool(champs & seen))
                seen |= champs
            else:
                self.series_champs[key] = set(champs)

        for _, _, pid, champ, gd in players:
            self.player_hist[pid].append((date, champ, gd))
            if gd is not None and gd == gd:
                self.player_gd[pid].append(gd)
                self.player_gd_sum[pid] += gd
            if g.playoffs:
                self.player_playoffs[pid] += 1
            if lg in EVENT_LEAGUES:
                self.player_intl[pid] += 1

        mult = _mov_mult(g.kblue, g.kred)
        self.elo[g.blue] += K * mult * (g.yb - p_blue)
        self.elo[g.red] += K * mult * ((1 - g.yb) - (1 - p_blue))
        self.n[g.blue] += 1
        self.n[g.red] += 1


def build_context(cfg: dict | None = None) -> tuple[pd.DataFrame, _State]:
    """Rejoue toutes les games : (table des facteurs as-of par game, état final)."""
    cfg = cfg or load_config()
    games = load_games(cfg, history=True)
    pl = load_player_games(cfg)
    meta = pl.drop_duplicates("gameid").set_index("gameid")["patch"]
    games["patch"] = games["gameid"].map(meta).astype(object)
    # Lignes joueur regroupées par game en structures Python simples : la boucle
    # ci-dessous tourne sur des dizaines de milliers de games.
    by_game: dict[str, list[tuple]] = defaultdict(list)
    for gid, team, pos, pid, champ, gd in zip(pl["gameid"], pl["teamname"], pl["position"],
                                              pl["pid"], pl["champion"], pl["golddiffat15"]):
        by_game[gid].append((team, pos, pid, champ if isinstance(champ, str) else None, gd))

    stt = _State()
    rows = []
    for g in games.itertuples():
        players = by_game.get(g.gameid, [])
        lineups: dict[str, dict] = defaultdict(dict)
        for team, pos, pid, _, _ in players:
            lineups[team][pos] = pid
        patch = g.patch if isinstance(g.patch, str) else None
        p_blue = win_prob(stt.elo[g.blue], stt.elo[g.red])
        fb = stt.team_features(g.blue, lineups.get(g.blue, {}), g.date, patch)
        fr = stt.team_features(g.red, lineups.get(g.red, {}), g.date, patch)
        row = {"gameid": g.gameid, "date": g.date, "season": g.season, "league": g.league,
               "blue": g.blue, "red": g.red, "yb": g.yb, "p_elo": p_blue,
               "nmin": min(stt.n[g.blue], stt.n[g.red]),
               "fearless": stt.is_fearless(g.league)}
        for k in FACTORS:
            row[f"d_{k}"] = fb[k] - fr[k]
        rows.append(row)
        stt.update(g, lineups, players, p_blue)
    return pd.DataFrame(rows), stt


def team_snapshot(stt: _State, team: str, date: pd.Timestamp | None = None,
                  lineup: dict | None = None) -> dict:
    """Facteurs pré-match d'une équipe, avec sa DERNIÈRE lineup connue — ou `lineup`
    ({poste: joueur}), le cinq attendu d'après l'effectif suivi (`src.v2.lineups`).

    Par défaut, mesurés au lendemain des dernières données (comme `player_snapshot`) et
    non à la date du jour : avec des données en retard, les fenêtres glissantes (pool sur
    90 jours, games sur 7 jours) se videraient et le prono changerait d'un jour à l'autre
    sans qu'aucun match n'ait été joué.
    """
    date = date or (stt.last_date or pd.Timestamp.now()).normalize() + pd.Timedelta(days=1)
    patch = max(stt.patch_start, key=stt.patch_start.get) if stt.patch_start else None
    lu = lineup if lineup is not None else stt.last_lineup.get(team, {})
    feats = stt.team_features(team, lu, date, patch)
    feats["league"] = stt.team_league.get(team, "?")
    feats["fearless"] = stt.is_fearless(feats["league"])
    feats["lineup"] = lu
    feats["elo"] = stt.elo.get(team, BASE)
    feats["patch"] = patch
    return feats


def evaluate(table: pd.DataFrame, min_games: int = 15, train_frac: float = 0.7) -> pd.DataFrame:
    """Chaque facteur explique-t-il ce que l'Elo rate ? Test chronologique.

    Pour chaque facteur : corrélation (écart bleu−rouge, résultat − proba Elo), puis
    régression logistique `logit(p_elo) + facteur` apprise sur les 70 % premiers
    matchs et jugée sur les 30 % suivants. Gain = baisse de log-loss vs Elo seul
    recalibré sur le même échantillon. Gain > 0 = le facteur apporte de l'info.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    d = table[table["nmin"] >= min_games].dropna(subset=["p_elo"]).sort_values("date")
    p = d["p_elo"].clip(1e-4, 1 - 1e-4)
    d = d.assign(logit=np.log(p / (1 - p)), resid=d["yb"] - d["p_elo"])
    cut = int(len(d) * train_frac)
    tr, te = d.iloc[:cut], d.iloc[cut:]

    base = LogisticRegression(C=1e6).fit(tr[["logit"]], tr["yb"])
    ll_base = log_loss(te["yb"], base.predict_proba(te[["logit"]])[:, 1])
    rng = np.random.default_rng(42)

    def _ll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _score(cols: list[str], label: str, corr: float | None, coef: float | None) -> dict:
        sub_tr, sub_te = tr.dropna(subset=cols), te.dropna(subset=cols)
        mu, sd = sub_tr[cols].mean(), sub_tr[cols].std().replace(0, 1)
        Xtr = np.column_stack([sub_tr["logit"], (sub_tr[cols] - mu) / sd])
        Xte = np.column_stack([sub_te["logit"], (sub_te[cols] - mu) / sd])
        m = LogisticRegression(C=1.0).fit(Xtr, sub_tr["yb"])
        y = sub_te["yb"].to_numpy()
        diff = (_ll(y, base.predict_proba(sub_te[["logit"]])[:, 1])
                - _ll(y, m.predict_proba(Xte)[:, 1]))
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        lo, hi = np.percentile(boots, [5, 95])
        return {
            "facteur": label,
            "corr_vs_residu": corr,
            "coef": round(float(m.coef_[0][1]), 3) if coef is None and len(cols) == 1 else coef,
            "gain_logloss_x1000": round(diff.mean() * 1000, 2),
            "ic90_x1000": f"[{lo*1000:+.2f} ; {hi*1000:+.2f}]",
            "significatif": bool(lo > 0),
            "n_test": len(sub_te),
        }

    rows = []
    usable = []
    for k, label in FACTORS.items():
        col = f"d_{k}"
        if tr[col].notna().sum() < 200 or tr[col].std() == 0:
            continue
        usable.append(col)
        rows.append(_score([col], label, round(float(d[col].corr(d["resid"])), 3), None))
    rows.append(_score(usable, "TOUS LES FACTEURS COMBINÉS", None, "—"))
    out = pd.DataFrame(rows).sort_values("gain_logloss_x1000", ascending=False)
    out.attrs["logloss_elo"] = ll_base
    out.attrs["n_test"] = len(te)
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:+.3f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Facteurs de contexte d'un BO")
    ap.add_argument("teams", nargs="*", help="deux équipes (noms Oracle exacts)")
    ap.add_argument("--eval", action="store_true", help="test chronologique de chaque facteur")
    args = ap.parse_args()

    print("Rejeu chronologique des games (facteurs as-of)…")
    table, stt = build_context()
    print(f"  {len(table)} games · {len(stt.player_hist)} joueurs · "
          f"{len(stt.patch_start)} patchs")
    fl = sorted(lg for lg in stt.fearless if stt.is_fearless(lg))
    print(f"  Ligues en Fearless draft détectées : {fl or 'aucune'}")

    if len(args.teams) == 2:
        a, b = (team_snapshot(stt, t) for t in args.teams)
        print(f"\n{'Facteur':38} {args.teams[0]:>16} {args.teams[1]:>16}")
        for k, label in FACTORS.items():
            print(f"{label:38} {_fmt(a[k]):>16} {_fmt(b[k]):>16}")

    if args.eval:
        ev = evaluate(table)
        print(f"\n=== Test chronologique (log-loss Elo seul : {ev.attrs['logloss_elo']:.4f}, "
              f"{ev.attrs['n_test']} games test) ===")
        print(ev.to_string(index=False))


if __name__ == "__main__":
    main()

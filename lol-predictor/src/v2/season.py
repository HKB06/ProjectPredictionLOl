"""Table de la saison : toutes les séries jouées, leurs stats et les prédictions.

Trois tables, construites à partir du CSV Oracle complet (toutes ligues) :
  - `games`  : 1 ligne par game (stats des deux côtés + proba pré-game des 2 modèles) ;
  - `series` : 1 ligne par série/BO (score, vainqueur, prono de chaque modèle, juste ou faux) ;
  - `team_table` / `league_table` : agrégats par équipe et par ligue.

Prédictions (toutes hors-échantillon, calculées AVANT la game) :
  - Ancien  : Elo d'équipe K32+MOV calibré par ligue (celui de l'ancien site).
  - Nouveau : v2 complet en walk-forward. Sans saisons passées (`data.history_years`),
    le v2 n'a pas assez de games pour être entraîné pendant les `SEASON_WARMUP_DAYS`
    premiers jours : on affiche alors l'Elo joueur seul, signalé par `rodage`.

Le prono d'une série est celui de sa PREMIÈRE game (seul instant pré-match),
converti en proba de série.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.load_oracle import ROOT, load_config
from src.update.elo import series_prob, series_prob_first

OLD_COL = "p_elo_cal"
NEW_COL = "Nouveau · v2 complet"
SERIES_GAP = pd.Timedelta(hours=10)   # au-delà : nouvelle série entre les mêmes équipes
MIN_GAMES = 15                        # en dessous : équipe en cold-start
BLUE_PRIOR, BLUE_PRIOR_N = 0.53, 200  # avantage du côté bleu supposé avant d'avoir des games

_PICKS = [f"pick{i}" for i in range(1, 6)]
_STATS = ["kills", "dragons", "barons", "towers", "firstblood", "firstdragon", "firsttower"]
_COLS = {"gameid", "date", "league", "playoffs", "patch", "game", "side", "position",
         "teamname", "result", "gamelength", "golddiffat15", *_STATS, *_PICKS}


def load_game_stats(cfg: dict | None = None) -> pd.DataFrame:
    """1 ligne par game : stats des deux côtés, picks, durée, patch."""
    cfg = cfg or load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False,
                     usecols=lambda c: c.strip() in _COLS, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    t = df[df["position"].str.lower() == "team"].copy()
    t["picks"] = t[_PICKS].apply(lambda r: ", ".join(c for c in r if isinstance(c, str)), axis=1)
    side = t["side"].str.lower()
    keep = ["gameid", "teamname", "result", "golddiffat15", "picks", *_STATS]
    blue = t[side == "blue"][["date", "league", "playoffs", "patch", "game",
                              "gamelength", *keep]]
    red = t[side == "red"][keep]
    g = blue.merge(red, on="gameid", suffixes=("_b", "_r")).dropna(
        subset=["teamname_b", "teamname_r", "result_b"])
    out = pd.DataFrame({
        "gameid": g["gameid"],
        "date": pd.to_datetime(g["date"], errors="coerce"),
        "league": g["league"],
        "playoffs": pd.to_numeric(g["playoffs"], errors="coerce").fillna(0).astype(int),
        "patch": g["patch"],
        "game_no": pd.to_numeric(g["game"], errors="coerce").fillna(1).astype(int),
        "blue": g["teamname_b"], "red": g["teamname_r"],
        "winner": np.where(g["result_b"].astype(int) == 1, g["teamname_b"], g["teamname_r"]),
        "length_min": g["gamelength"] / 60.0,
        "gd15_blue": g["golddiffat15_b"],
        "blue_picks": g["picks_b"], "red_picks": g["picks_r"],
    })
    for s in _STATS:
        out[f"blue_{s}"] = g[f"{s}_b"]
        out[f"red_{s}"] = g[f"{s}_r"]
    return out.dropna(subset=["date"]).reset_index(drop=True)


def _series_ids(games: pd.DataFrame) -> list[int]:
    """Numéro de série : mêmes équipes + même ligue, games qui se suivent."""
    last: dict = {}
    ids, n = [], 0
    for lg, b, r, d, no in zip(games["league"], games["blue"], games["red"],
                               games["date"], games["game_no"]):
        key = (lg, frozenset((b, r)))
        prev = last.get(key)
        if prev is None or d - prev[1] > SERIES_GAP or no <= prev[2]:
            n += 1
            cur = n
        else:
            cur = prev[0]
        last[key] = (cur, d, no)
        ids.append(cur)
    return ids


def build_season(df: pd.DataFrame, res: pd.DataFrame,
                 cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, series) à partir du jeu de données v2 (`df`) et du walk-forward (`res`)."""
    preds = df[["gameid", "p_elo", OLD_COL, "p_player", "nmin"]].merge(
        res[["gameid", NEW_COL]], on="gameid", how="left")
    preds["rodage"] = preds[NEW_COL].isna()
    preds["p_new"] = preds[NEW_COL].fillna(preds["p_player"])
    preds = preds.rename(columns={OLD_COL: "p_old"})

    games = load_game_stats(cfg).merge(
        preds[["gameid", "p_old", "p_new", "nmin", "rodage"]], on="gameid", how="inner")
    games = games.sort_values(["date", "gameid"]).reset_index(drop=True)
    games["series_id"] = _series_ids(games)
    # Avantage du côté bleu connu AVANT chaque game (part des victoires bleues jusque-là).
    # Le nouveau modèle prédit « le côté bleu gagne » : la proba de la game 1 contient cet
    # avantage, mais les games suivantes d'une série ne se jouent pas toutes du même côté.
    blue_won = (games["winner"] == games["blue"]).astype(float)
    before = blue_won.cumsum() - blue_won
    rate = (BLUE_PRIOR * BLUE_PRIOR_N + before) / (BLUE_PRIOR_N + np.arange(len(games)))
    games["blue_edge"] = np.log(rate / (1 - rate))

    rows = []
    for sid, g in games.groupby("series_id", sort=False):
        first = g.iloc[0]
        a, b = first["blue"], first["red"]
        wa, wb = int((g["winner"] == a).sum()), int((g["winner"] == b).sum())
        complete = wa != wb
        wn = max(wa, wb, 1)
        winner = (a if wa > wb else b) if complete else None
        row = {
            "series_id": sid, "date": first["date"], "league": first["league"],
            "playoffs": int(g["playoffs"].max()), "patch": first["patch"],
            "team_a": a, "team_b": b, "score": f"{wa}-{wb}",
            "bo": wn * 2 - 1 if complete else len(g), "games": len(g),
            "winner": winner, "nmin": int(first["nmin"]),
            "low_data": bool(first["nmin"] < MIN_GAMES), "rodage": bool(first["rodage"]),
            "length_min": float(g["length_min"].mean()),
            "kills": float((g["blue_kills"] + g["red_kills"]).mean()),
        }
        for tag in ("old", "new"):
            p1 = float(first[f"p_{tag}"])
            if tag == "new":
                # game 1 : A est côté bleu ; games suivantes : sans avantage de côté
                pc = min(max(p1, 1e-6), 1 - 1e-6)
                p_next = 1 / (1 + np.exp(-(np.log(pc / (1 - pc)) - first["blue_edge"])))
                ps = series_prob_first(p1, float(p_next), wn)
            else:                                                 # Elo : déjà sans côté
                ps = series_prob(p1, wn)                          # P(A gagne la série)
            fav = a if ps >= 0.5 else b
            row[f"fav_{tag}"] = fav
            row[f"p_{tag}"] = max(ps, 1 - ps)
            row[f"ok_{tag}"] = (fav == winner) if complete else None
        rows.append(row)
    series = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    return games, series


def accuracy(series: pd.DataFrame, tag: str) -> float:
    """Part des séries terminées où le favori du modèle a gagné."""
    ok = series[f"ok_{tag}"].dropna()
    return float(ok.astype(bool).mean()) if len(ok) else float("nan")


def team_table(games: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Stats de la saison par équipe (toutes ses games, des deux côtés)."""
    parts = []
    for side, opp, sign in (("blue", "red", 1), ("red", "blue", -1)):
        parts.append(pd.DataFrame({
            "team": games[side], "league": games["league"],
            "win": (games["winner"] == games[side]).astype(float),
            "kills": games[f"{side}_kills"], "deaths": games[f"{opp}_kills"],
            "gd15": sign * games["gd15_blue"], "length": games["length_min"],
            "fb": games[f"{side}_firstblood"], "fd": games[f"{side}_firstdragon"],
            "ft": games[f"{side}_firsttower"],
            "dragons": games[f"{side}_dragons"], "barons": games[f"{side}_barons"],
        }))
    long = pd.concat(parts, ignore_index=True)
    out = long.groupby("team").agg(
        ligue=("league", lambda s: s.mode().iat[0]), games=("win", "size"),
        winrate=("win", "mean"), kills=("kills", "mean"), deaths=("deaths", "mean"),
        gd15=("gd15", "mean"), duree=("length", "mean"), first_blood=("fb", "mean"),
        first_dragon=("fd", "mean"), first_tower=("ft", "mean"),
        dragons=("dragons", "mean"), barons=("barons", "mean"),
    )
    done = series.dropna(subset=["winner"])
    played = pd.concat([done["team_a"], done["team_b"]]).value_counts()
    won = done["winner"].value_counts()
    out["series"] = played.reindex(out.index).fillna(0).astype(int)
    out["series_won"] = won.reindex(out.index).fillna(0).astype(int)
    return out.reset_index().sort_values("games", ascending=False)


def league_table(games: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Par ligue : volume, stats de jeu et réussite des deux modèles sur les séries."""
    g = games.groupby("league").agg(
        games=("gameid", "size"), duree=("length_min", "mean"),
        kills=("blue_kills", "mean"), kills_r=("red_kills", "mean"),
        blue_win=("winner", lambda s: float((s == games.loc[s.index, "blue"]).mean())),
    )
    g["kills"] = g["kills"] + g.pop("kills_r")
    done = series.dropna(subset=["winner"])
    s = done.groupby("league").agg(
        series=("series_id", "size"),
        ok_old=("ok_old", lambda x: float(x.astype(bool).mean())),
        ok_new=("ok_new", lambda x: float(x.astype(bool).mean())),
    )
    return g.join(s, how="left").reset_index().sort_values("games", ascending=False)

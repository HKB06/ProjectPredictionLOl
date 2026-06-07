"""P3 : feature engineering SANS FUITE de données.

Passe chronologique stricte : on parcourt les matchs dans l'ordre du temps ; pour
chaque match, on calcule les features des 2 équipes à partir de leur historique
*antérieur* uniquement, PUIS on met à jour cet historique avec le résultat du match.
=> aucune information du match courant (ni de l'avenir) ne fuite dans les features.

Features par équipe (as-of) :
- elo (mis à jour match par match)
- winrate global / 10 derniers / sur le side courant  (lissés = anti-variance)
- taux first blood / first tower / first dragon (lissés)
- moyenne GD@15 (force early), moyenne kills/deaths, durée moyenne
- nombre de games joués (exposition)
Puis on ajoute les écarts (bleu - rouge) et le H2H bleu vs rouge.

Usage :
    python -m src.features.build_features
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.features.champion_priors import (ChampionWinrateIndex,
                                          load_full_champion_results)
from src.ingest.load_oracle import ROOT, load_config

K_RATE = 5.0      # force du lissage des taux (vers 0.5)
K_SIDE = 3.0
K_H2H = 2.0
ELO_K = 20.0
ELO_BASE = 1500.0

ROLES = ["top", "jng", "mid", "bot", "sup"]
DELTA_KEYS = ["elo", "wr", "wr_l10", "wr_side", "fb_rate", "ft_rate", "fd_rate", "gd15_avg"]


def _shrink(success: float, n: int, k: float = K_RATE, prior: float = 0.5) -> float:
    return (success + prior * k) / (n + k)


def _mean(values: list[float], default: float = 0.0) -> float:
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(clean)) if clean else default


def _team_features(hist: list[dict], elo: float, side: str) -> dict:
    n = len(hist)
    if n == 0:
        return {
            "elo": elo, "n_games": 0,
            "wr": 0.5, "wr_l10": 0.5, "wr_side": 0.5,
            "fb_rate": 0.5, "ft_rate": 0.5, "fd_rate": 0.5,
            "gd15_avg": 0.0, "kills_avg": 0.0, "deaths_avg": 0.0, "len_avg": 0.0,
        }
    results = [h["result"] for h in hist]
    last10 = hist[-10:]
    side_hist = [h for h in hist if h["side"] == side]
    return {
        "elo": elo,
        "n_games": n,
        "wr": _shrink(sum(results), n),
        "wr_l10": _shrink(sum(h["result"] for h in last10), len(last10)),
        "wr_side": _shrink(sum(h["result"] for h in side_hist), len(side_hist), k=K_SIDE),
        "fb_rate": _shrink(sum(h["firstblood"] for h in hist), n),
        "ft_rate": _shrink(sum(h["firsttower"] for h in hist), n),
        "fd_rate": _shrink(sum(h["firstdragon"] for h in hist), n),
        "gd15_avg": _mean([h["golddiffat15"] for h in hist]),
        "kills_avg": _mean([h["kills"] for h in hist]),
        "deaths_avg": _mean([h["deaths"] for h in hist]),
        "len_avg": _mean([h["gamelength"] for h in hist]),
    }


def _comp_champ_wr(m, side: str, champ_idx: ChampionWinrateIndex) -> float:
    """Winrate moyen (as-of) des 5 champions draftés d'un side pour ce match."""
    wrs = []
    for role in ROLES:
        champ = getattr(m, f"{side}_{role}", None)
        if champ is not None and not (isinstance(champ, float) and np.isnan(champ)):
            wrs.append(champ_idx.asof(champ, m.date))
    return float(np.mean(wrs)) if wrs else 0.5


def build_features(matches: pd.DataFrame, team_games: pd.DataFrame,
                   champ_idx: ChampionWinrateIndex | None = None,
                   return_state: bool = False):
    tg = team_games.set_index(["gameid", "teamname"])
    matches = matches.sort_values(["date", "gameid"]).reset_index(drop=True)

    history: dict[str, list[dict]] = defaultdict(list)
    elo: dict[str, float] = defaultdict(lambda: ELO_BASE)
    h2h: dict[frozenset, list[str]] = defaultdict(list)

    delta_keys = DELTA_KEYS
    label_cols = [c for c in matches.columns if c.startswith("y_")]
    rows = []

    for m in matches.itertuples():
        b, r = m.blue_team, m.red_team
        fb = _team_features(history[b], elo[b], "Blue")
        fr = _team_features(history[r], elo[r], "Red")

        pair = frozenset((b, r))
        meetings = h2h[pair]
        nh = len(meetings)
        blue_wins = sum(1 for w in meetings if w == b)

        row = {"gameid": m.gameid, "date": m.date,
               "is_playoffs": int(getattr(m, "playoffs", 0) or 0)}
        for key, val in fb.items():
            row[f"blue_{key}"] = val
        for key, val in fr.items():
            row[f"red_{key}"] = val
        for key in delta_keys:
            row[f"d_{key}"] = fb[key] - fr[key]
        row["h2h_blue_wr"] = _shrink(blue_wins, nh, k=K_H2H)
        row["h2h_n"] = nh

        # --- features DRAFT (priors champion as-of, sans fuite) ---
        if champ_idx is not None:
            bcw = _comp_champ_wr(m, "blue", champ_idx)
            rcw = _comp_champ_wr(m, "red", champ_idx)
            row["blue_champ_wr"] = bcw
            row["red_champ_wr"] = rcw
            row["d_champ_wr"] = bcw - rcw

        for c in label_cols:
            row[c] = getattr(m, c)
        rows.append(row)

        # --- mise à jour APRÈS calcul des features (anti-fuite) ---
        for team, side in ((b, "Blue"), (r, "Red")):
            try:
                s = tg.loc[(m.gameid, team)]
            except KeyError:
                continue
            history[team].append({
                "result": float(s["result"]),
                "firstblood": float(s.get("firstblood", 0) or 0),
                "firsttower": float(s.get("firsttower", 0) or 0),
                "firstdragon": float(s.get("firstdragon", 0) or 0),
                "kills": float(s.get("kills", 0) or 0),
                "deaths": float(s.get("deaths", 0) or 0),
                "golddiffat15": s.get("golddiffat15"),
                "gamelength": float(s.get("gamelength", 0) or 0),
                "side": side,
            })

        exp_b = 1.0 / (1.0 + 10 ** ((elo[r] - elo[b]) / 400.0))
        res_b = float(m.y_winner)
        elo[b] += ELO_K * (res_b - exp_b)
        elo[r] += ELO_K * ((1.0 - res_b) - (1.0 - exp_b))

        winner = b if m.y_winner == 1 else r
        h2h[pair].append(winner)

    feats = pd.DataFrame(rows)
    if return_state:
        return feats, {"history": history, "elo": elo, "h2h": h2h}
    return feats


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    matches = pd.read_parquet(proc / "matches.parquet")
    team_games = pd.read_parquet(proc / "team_games.parquet")

    # Priors de draft : index winrate champion (source de ligues configurable)
    prior_leagues = cfg.get("features", {}).get("champ_prior_leagues")
    src_label = ", ".join(prior_leagues) if prior_leagues else "toutes ligues"
    print(f"Construction de l'index winrate champion (priors : {src_label})...")
    champ_idx = ChampionWinrateIndex(load_full_champion_results(cfg, prior_leagues))

    feats = build_features(matches, team_games, champ_idx=champ_idx)
    feats.to_parquet(proc / "features.parquet", index=False)

    print(f"features -> {proc / 'features.parquet'}  ({feats.shape})")
    feature_cols = [c for c in feats.columns if not c.startswith("y_") and c not in ("gameid", "date")]
    print(f"\n{len(feature_cols)} features :")
    print("  " + ", ".join(feature_cols))

    # Contrôle de cohérence : écarts corrélés positivement à la victoire bleue
    sub = feats[feats["blue_n_games"] >= 5]
    corr = sub["d_elo"].corr(sub["y_winner"])
    print(f"\nContrôle (matchs avec historique >=5) : corr(d_elo, y_winner) = {corr:+.3f}")
    if "d_champ_wr" in feats.columns:
        corr_c = sub["d_champ_wr"].corr(sub["y_winner"])
        print(f"Contrôle draft : corr(d_champ_wr, y_winner) = {corr_c:+.3f}")
    print(f"Matchs exploitables (>=5 games d'historique) : {len(sub)}/{len(feats)}")


if __name__ == "__main__":
    main()

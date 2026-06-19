"""Backtest de VALEUR : nos probas de série vs les cotes du bookmaker.

Question clé du projet "paris" : est-ce qu'on BAT le prix (pas juste : a-t-on raison) ?
On mesure ça en comparant, série par série, notre proba estimée à la proba implicite
de la cote, en ne pariant que là où on a un avantage (edge / EV > seuil), puis en
simulant le ROI.

MÉTHODE (sans fuite, honnête) :
- Walk-forward strict : pour une série au jour D, le modèle vainqueur est (ré)entraîné
  UNIQUEMENT sur les games antérieures à D. Les features de prédiction sont l'état des
  équipes (Elo/forme/H2H) au DÉBUT du jour D (snapshot avant les games du jour).
- SANS DRAFT : les cotes "série" sont posées AVANT les drafts -> on prédit sans les
  features de draft (comparaison équitable, et pas de fuite intra-série).
- Side neutralisé : une série alterne les sides -> on moyenne (log-odds) la proba quand
  l'équipe est bleue et rouge, puis on convertit en proba de série (BO3/BO5).

PIÈGE ASSUMÉ (cold-start) : en début de saison, l'Elo n'a pas convergé (pas de data
2025) -> on n'évalue que les séries où les 2 équipes ont >= MIN_GAMES games passées.
Sinon on croit voir de la "value" alors que c'est notre ignorance.

Usage :
    python -m src.models.value_backtest                 # seuil EV 5%, warmup 5 games
    python -m src.models.value_backtest --edge 0.10 --min-games 8
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import (DELTA_KEYS, ELO_BASE, ELO_K, K_H2H,
                                         _shrink, _team_features, build_features)
from src.ingest.load_oracle import ROOT, load_config
from src.models.predict import WINNER_C, bo_series_prob

# Cotes (nom du book) -> nom Oracle's Elixir
TEAM_MAP = {
    "FearX": "BNK FEARX",
    "BNK FearX": "BNK FEARX",
    "KRX": "Kiwoom DRX",
    "DRX": "Kiwoom DRX",
    "OKSavingsBank Brion": "HANJIN BRION",
    "OKSavingsBank BRION": "HANJIN BRION",
    "BRION": "HANJIN BRION",
    "Dplus Kia": "Dplus Kia",
    "Dplus KIA": "Dplus Kia",
    "DK": "Dplus Kia",
    "KT Rolster": "KT Rolster",
    "DN SOOPers": "DN SOOPers",
    "Gen.G": "Gen.G",
    "Nongshim RedForce": "Nongshim RedForce",
    "Hanwha Life": "Hanwha Life Esports",
    "Hanwha Life Esports": "Hanwha Life Esports",
    "HLE": "Hanwha Life Esports",
    "T1": "T1",
}


def map_team(name: str) -> str:
    return TEAM_MAP.get(name.strip(), name.strip())


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=WINNER_C))


def reconstruct_series(matches: pd.DataFrame) -> pd.DataFrame:
    """Games -> séries (jour + paire d'équipes), avec vainqueur, score, format."""
    m = matches.copy()
    m["day"] = pd.to_datetime(m["date"]).dt.date
    m["pair"] = m.apply(lambda r: tuple(sorted((r["blue_team"], r["red_team"]))), axis=1)
    rows = []
    for (day, pair), g in m.groupby(["day", "pair"]):
        t1, t2 = pair
        wins = {t1: 0, t2: 0}
        for r in g.itertuples():
            wins[(r.blue_team if r.y_winner == 1 else r.red_team)] += 1
        wn = max(wins.values())
        rows.append({
            "day": day, "teamA": t1, "teamB": t2,
            "scoreA": wins[t1], "scoreB": wins[t2],
            "winner": t1 if wins[t1] > wins[t2] else t2,
            "wins_needed": wn,  # 2 -> BO3, 3 -> BO5
            "is_playoffs": int(g["playoffs"].fillna(0).max()) if "playoffs" in g else 0,
        })
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def day_snapshots(matches: pd.DataFrame, team_games: pd.DataFrame, target_days: set):
    """Rejoue l'historique et capture l'état (history/elo/h2h) au DÉBUT de chaque
    jour cible (avant les games du jour). Mêmes règles de MAJ que build_features."""
    tg = team_games.set_index(["gameid", "teamname"])
    m = matches.sort_values(["date", "gameid"]).reset_index(drop=True)
    m["day"] = pd.to_datetime(m["date"]).dt.date

    from collections import defaultdict
    history: dict[str, list[dict]] = defaultdict(list)
    elo: dict[str, float] = defaultdict(lambda: ELO_BASE)
    h2h: dict[frozenset, list[str]] = defaultdict(list)

    snaps: dict = {}
    seen_days = set()
    for r in m.itertuples():
        if r.day in target_days and r.day not in seen_days:
            snaps[r.day] = (
                {k: list(v) for k, v in history.items()},
                dict(elo),
                {k: list(v) for k, v in h2h.items()},
            )
            seen_days.add(r.day)
        # MAJ historique APRÈS snapshot
        for team, side in ((r.blue_team, "Blue"), (r.red_team, "Red")):
            try:
                s = tg.loc[(r.gameid, team)]
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
        exp_b = 1.0 / (1.0 + 10 ** ((elo[r.red_team] - elo[r.blue_team]) / 400.0))
        res_b = float(r.y_winner)
        elo[r.blue_team] += ELO_K * (res_b - exp_b)
        elo[r.red_team] += ELO_K * ((1.0 - res_b) - (1.0 - exp_b))
        h2h[frozenset((r.blue_team, r.red_team))].append(
            r.blue_team if r.y_winner == 1 else r.red_team)
    return snaps


def _row(hist, elo, h2h, A, B, fcols, is_playoffs):
    """Vecteur de features (SANS draft) pour 'A bleu vs B rouge', à partir d'un snapshot."""
    fb = _team_features(hist.get(A, []), elo.get(A, ELO_BASE), "Blue")
    fr = _team_features(hist.get(B, []), elo.get(B, ELO_BASE), "Red")
    meetings = h2h.get(frozenset((A, B)), [])
    nh = len(meetings)
    a_wins = sum(1 for w in meetings if w == A)
    row: dict = {"is_playoffs": int(is_playoffs)}
    for k, v in fb.items():
        row[f"blue_{k}"] = v
    for k, v in fr.items():
        row[f"red_{k}"] = v
    for k in DELTA_KEYS:
        row[f"d_{k}"] = fb[k] - fr[k]
    row["h2h_blue_wr"] = _shrink(a_wins, nh, k=K_H2H)
    row["h2h_n"] = nh
    return pd.DataFrame([row]).reindex(columns=fcols)


def _neutral_game_prob(model, hist, elo, h2h, A, B, fcols, is_playoffs) -> float:
    """Proba que A gagne UNE game, side neutralisé (moyenne log-odds bleu/rouge)."""
    p_blue = float(model.predict_proba(_row(hist, elo, h2h, A, B, fcols, is_playoffs))[:, 1][0])
    p_red = 1.0 - float(model.predict_proba(_row(hist, elo, h2h, B, A, fcols, is_playoffs))[:, 1][0])
    eps = 1e-6
    lb = np.log(np.clip(p_blue, eps, 1 - eps) / np.clip(1 - p_blue, eps, 1 - eps))
    lr = np.log(np.clip(p_red, eps, 1 - eps) / np.clip(1 - p_red, eps, 1 - eps))
    return float(1.0 / (1.0 + np.exp(-(lb + lr) / 2.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.05, help="seuil d'EV minimal pour parier")
    ap.add_argument("--max-edge", type=float, default=0.30, help="EV au-delà = erreur modèle probable -> ignoré")
    ap.add_argument("--min-games", type=int, default=5, help="games passées mini par équipe (anti cold-start)")
    ap.add_argument("--kelly", type=float, default=0.25, help="fraction de Kelly")
    args = ap.parse_args()

    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    matches = pd.read_parquet(proc / "matches.parquet")
    team_games = pd.read_parquet(proc / "team_games.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    odds = pd.read_csv(ROOT / "data" / "odds" / "lck_2026_odds.csv")
    odds["team1"] = odds["team1"].map(map_team)
    odds["team2"] = odds["team2"].map(map_team)
    odds["day"] = pd.to_datetime(odds["date"]).dt.date

    # Features as-of SANS draft (champ_idx=None) = pool d'entraînement walk-forward
    feats = build_features(matches, team_games, champ_idx=None)
    feats["date"] = pd.to_datetime(feats["date"])
    fcols = [c for c in feats.columns if not c.startswith("y_") and c not in ("gameid", "date")]

    series = reconstruct_series(matches)
    snaps = day_snapshots(matches, team_games, set(odds["day"]))

    # Vérif mapping
    known = set(matches["blue_team"]) | set(matches["red_team"])
    for _, o in odds.iterrows():
        for t in (o["team1"], o["team2"]):
            if t not in known:
                print(f"  [!] équipe non trouvée dans les données : {t!r}")

    results = []
    for _, o in odds.iterrows():
        day, t1, t2 = o["day"], o["team1"], o["team2"]
        odd1, odd2 = float(o["odd1"]), float(o["odd2"])

        # série réelle correspondante
        pair = tuple(sorted((t1, t2)))
        srow = series[(series["day"] == day) &
                      (series["teamA"] == pair[0]) & (series["teamB"] == pair[1])]
        if srow.empty:
            print(f"  [!] série introuvable : {day} {t1} vs {t2}")
            continue
        srow = srow.iloc[0]
        wins_needed = int(srow["wins_needed"])
        winner = srow["winner"]
        is_po = int(srow["is_playoffs"])

        snap = snaps.get(day)
        if snap is None:
            continue
        hist, elo, h2h = snap
        n1 = len(hist.get(t1, []))
        n2 = len(hist.get(t2, []))

        train = feats[feats["date"].dt.date < day].dropna(subset=["y_winner"])
        train = train[(train["blue_n_games"] >= 3) & (train["red_n_games"] >= 3)]

        rec = {"day": day, "t1": t1, "t2": t2, "odd1": odd1, "odd2": odd2,
               "n1": n1, "n2": n2, "winner": winner, "wins_needed": wins_needed}

        if len(train) < 30 or train["y_winner"].nunique() < 2 or min(n1, n2) < args.min_games:
            rec["status"] = "skip (cold-start)"
            results.append(rec)
            continue

        model = _logreg().fit(train[fcols], train["y_winner"].astype(int))
        p1_game = _neutral_game_prob(model, hist, elo, h2h, t1, t2, fcols, is_po)
        p1 = bo_series_prob(p1_game, wins_needed)   # proba que t1 gagne la SÉRIE
        p2 = 1.0 - p1

        impl1, impl2 = 1 / odd1, 1 / odd2
        over = impl1 + impl2
        rec.update({
            "status": "ok",
            "p1": p1, "p2": p2,
            "impl1": impl1 / over, "impl2": impl2 / over,  # implicite "fair" (sans marge)
            "ev1": p1 * odd1 - 1, "ev2": p2 * odd2 - 1,
        })
        results.append(rec)

    # ----- affichage détaillé -----
    print("=" * 100)
    print(f"  BACKTEST DE VALEUR — seuil EV {args.edge:.0%}, warmup {args.min_games} games, Kelly {args.kelly:g}")
    print("=" * 100)
    bets = []
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['day']}  {r['t1']:<20} vs {r['t2']:<20}  [{r['status']}]  "
                  f"(games passées : {r['n1']}/{r['n2']})")
            continue
        bo = "BO5" if r["wins_needed"] == 3 else "BO3"
        print(f"  {r['day']}  {r['t1']:<20} vs {r['t2']:<20} {bo}")
        print(f"      modèle  : {r['t1']} {r['p1']*100:5.1f}%  | {r['t2']} {r['p2']*100:5.1f}%")
        print(f"      implicite(fair): {r['impl1']*100:5.1f}% | {r['impl2']*100:5.1f}%   "
              f"(cotes {r['odd1']:.2f}/{r['odd2']:.2f})")
        print(f"      EV      : {r['t1']} {r['ev1']*100:+5.1f}%  | {r['t2']} {r['ev2']*100:+5.1f}%")
        for side, p, odd, ev in ((r["t1"], r["p1"], r["odd1"], r["ev1"]),
                                 (r["t2"], r["p2"], r["odd2"], r["ev2"])):
            if ev <= args.edge:
                continue
            if ev > args.max_edge:
                print(f"      [ignoré] {side} EV {ev*100:+.1f}% > {args.max_edge*100:.0f}% "
                      f"(edge implausible = erreur modèle probable)")
                continue
            won = (side == r["winner"])
            profit = (odd - 1) if won else -1.0
            kelly_f = max(0.0, (p * odd - 1) / (odd - 1)) * args.kelly
            bets.append({"side": side, "odd": odd, "ev": ev, "won": won,
                         "profit": profit, "kelly_f": kelly_f,
                         "kelly_profit": kelly_f * (odd - 1) if won else -kelly_f})
            print(f"      >>> PARI {side} @ {odd:.2f} (EV {ev*100:+.1f}%) -> "
                  f"{'GAGNÉ' if won else 'perdu'}")

    # ----- calibration : NOUS vs LE MARCHÉ (sur toutes les séries évaluables) -----
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        y = np.array([1.0 if r["winner"] == r["t1"] else 0.0 for r in ok])
        pm = np.array([r["p1"] for r in ok])          # notre proba (t1)
        pk = np.array([r["impl1"] for r in ok])       # proba marché "fair" (t1)
        brier_model = float(np.mean((pm - y) ** 2))
        brier_market = float(np.mean((pk - y) ** 2))
        acc_model = float(np.mean((pm > 0.5) == (y > 0.5)))
        acc_market = float(np.mean((pk > 0.5) == (y > 0.5)))
        print("-" * 100)
        print(f"  CALIBRATION (sur {len(ok)} séries, plus robuste que le ROI) :")
        print(f"    Brier   NOUS {brier_model:.3f}  vs  MARCHÉ {brier_market:.3f}   "
              f"({'on est MEILLEUR' if brier_model < brier_market else 'le marché est meilleur'})")
        print(f"    Accuracy NOUS {acc_model*100:.0f}%   vs  MARCHÉ {acc_market*100:.0f}%")

    n_ok = len(ok)
    n_skip = sum(1 for r in results if r["status"] != "ok")
    print("-" * 100)
    print(f"  Séries avec cote : {len(results)}  | évaluables : {n_ok}  | skip cold-start : {n_skip}")
    if bets:
        b = pd.DataFrame(bets)
        roi = b["profit"].sum() / len(b)
        hit = b["won"].mean()
        k_roi = b["kelly_profit"].sum() / b["kelly_f"].sum() if b["kelly_f"].sum() > 0 else 0
        print(f"  PARIS PLACÉS     : {len(b)}  | gagnés {b['won'].sum()} ({hit*100:.0f}%)  | cote moy {b['odd'].mean():.2f}")
        print(f"  FLAT (1u/pari)   : profit {b['profit'].sum():+.2f}u  -> ROI {roi*100:+.1f}%")
        print(f"  KELLY {args.kelly:g}        : profit {b['kelly_profit'].sum():+.3f}u  -> ROI {k_roi*100:+.1f}%")
        print("  (échantillon minuscule -> indicatif. Ajouter des cotes mid/fin de saison.)")
    else:
        print("  Aucun pari placé (pas de value au-dessus du seuil, ou tout en cold-start).")
    print("=" * 100)


if __name__ == "__main__":
    main()

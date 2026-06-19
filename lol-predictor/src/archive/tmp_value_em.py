"""Backtest valeur EMEA Masters (EM) : biais de marche (back/fade, sans modele)
+ calibration NOUS vs MARCHE + paris de value (walk-forward sans fuite)."""
from __future__ import annotations

import difflib
import re
import sys

import numpy as np
import pandas as pd

from src.features.build_features import build_features
from src.ingest.load_oracle import ROOT, load_config
from src.models.predict import bo_series_prob
from src.models.value_backtest import _logreg, _neutral_game_prob, day_snapshots

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EDGE = 0.05
MAX_EDGE = 0.30
MIN_GAMES = 5


def norm(s: str) -> str:
    s = s.lower()
    for w in ("esports", "e-sports", " club", " gaming"):
        s = s.replace(w, "")
    return re.sub(r"[^a-z0-9]", "", s)


def resolve(name: str, known: list[str], nmap: dict) -> str | None:
    n = norm(name)
    if n in nmap:
        return nmap[n]
    cand = difflib.get_close_matches(n, list(nmap), n=1, cutoff=0.84)
    return nmap[cand[0]] if cand else None


def main() -> None:
    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]
    matches = pd.read_parquet(proc / "matches.parquet")
    team_games = pd.read_parquet(proc / "team_games.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    odds = pd.read_csv(ROOT / "data" / "odds" / "em_2026_odds.csv")
    odds["day"] = pd.to_datetime(odds["date"]).dt.date
    odds[["sA", "sB"]] = odds["score"].str.split("-", expand=True).astype(int)
    odds["winner_side"] = np.where(odds["sA"] > odds["sB"], 1, 2)
    odds["wins_needed"] = odds[["sA", "sB"]].max(axis=1)

    # ---------- PARTIE 1 : BIAIS DE MARCHE (cotes + score seulement) ----------
    print("=" * 90)
    print(f"  PARTIE 1 — BIAIS DE MARCHE EM (pur, {len(odds)} matchs, sans modele)")
    print("=" * 90)
    fav_odd = odds[["odd1", "odd2"]].min(axis=1)
    fav_side = np.where(odds["odd1"] <= odds["odd2"], 1, 2)
    fav_won = fav_side == odds["winner_side"]
    dog_odd = odds[["odd1", "odd2"]].max(axis=1)
    dog_won = ~fav_won

    back_fav = np.where(fav_won, fav_odd - 1, -1.0)
    fade = np.where(dog_won, dog_odd - 1, -1.0)
    print(f"  Favori du book gagne : {fav_won.mean()*100:.1f}% des series (n={len(odds)})")
    print(f"  BACK FAVORI (1u/match) : ROI {back_fav.mean()*100:+.1f}%  (profit {back_fav.sum():+.1f}u)")
    print(f"  FADE  (back outsider)  : ROI {fade.mean()*100:+.1f}%  (profit {fade.sum():+.1f}u)")

    print("\n  Par tranche de cote du FAVORI :")
    print(f"    {'tranche':<12}{'n':>4}{'fav_gagne':>11}{'implicite':>11}{'ROI_back':>10}{'ROI_fade':>10}")
    bins = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 99)]
    for lo, hi in bins:
        m = (fav_odd >= lo) & (fav_odd < hi)
        if m.sum() == 0:
            continue
        impl = (1 / fav_odd[m]).mean()
        print(f"    [{lo:.1f}-{hi:.1f}){'':<3}{m.sum():>4}{fav_won[m].mean()*100:>10.1f}%"
              f"{impl*100:>10.1f}%{back_fav[m].mean()*100:>9.1f}%{fade[m].mean()*100:>9.1f}%")

    # ---------- PARTIE 2 : MODELE (calibration + value, walk-forward) ----------
    known = sorted(set(matches["blue_team"]) | set(matches["red_team"]))
    nmap = {norm(t): t for t in known}
    feats = build_features(matches, team_games, champ_idx=None)
    feats["date"] = pd.to_datetime(feats["date"])
    fcols = [c for c in feats.columns if not c.startswith("y_") and c not in ("gameid", "date")]
    snaps = day_snapshots(matches, team_games, set(odds["day"]))

    unresolved = set()
    rows = []
    for o in odds.itertuples():
        A = resolve(o.team1, known, nmap)
        B = resolve(o.team2, known, nmap)
        if A is None or B is None:
            if A is None:
                unresolved.add(o.team1)
            if B is None:
                unresolved.add(o.team2)
            continue
        snap = snaps.get(o.day)
        if snap is None:
            continue
        hist, elo, h2h = snap
        n1, n2 = len(hist.get(A, [])), len(hist.get(B, []))
        train = feats[feats["date"].dt.date < o.day].dropna(subset=["y_winner"])
        train = train[(train["blue_n_games"] >= 3) & (train["red_n_games"] >= 3)]
        if len(train) < 30 or train["y_winner"].nunique() < 2 or min(n1, n2) < MIN_GAMES:
            continue
        model = _logreg().fit(train[fcols], train["y_winner"].astype(int))
        p1g = _neutral_game_prob(model, hist, elo, h2h, A, B, fcols, 0)
        p1 = bo_series_prob(p1g, int(o.wins_needed))
        rows.append({
            "day": o.day, "t1": o.team1, "t2": o.team2,
            "odd1": o.odd1, "odd2": o.odd2, "p1": p1, "p2": 1 - p1,
            "t1_won": o.winner_side == 1,
        })

    print("\n" + "=" * 90)
    print(f"  PARTIE 2 — MODELE walk-forward (resolus+chauds : {len(rows)} series ; "
          f"warmup {MIN_GAMES}g)")
    if unresolved:
        print(f"  [!] equipes non trouvees dans Oracle ({len(unresolved)}) : "
              f"{', '.join(sorted(unresolved)[:12])}{'...' if len(unresolved) > 12 else ''}")
    print("=" * 90)
    if not rows:
        print("  Aucune serie evaluable.")
        return

    df = pd.DataFrame(rows)
    y = df["t1_won"].astype(float).to_numpy()
    pm = df["p1"].to_numpy()
    over = 1 / df["odd1"] + 1 / df["odd2"]
    pk = (1 / df["odd1"]) / over
    print(f"  CALIBRATION : Brier NOUS {np.mean((pm-y)**2):.3f} vs MARCHE {np.mean((pk-y)**2):.3f}")
    print(f"  ACCURACY    : NOUS {np.mean((pm>0.5)==(y>0.5))*100:.0f}% vs "
          f"MARCHE {np.mean((pk>0.5)==(y>0.5))*100:.0f}%")

    bets = []
    for r in df.itertuples():
        for side_t1, p, odd, won in ((True, r.p1, r.odd1, r.t1_won),
                                     (False, r.p2, r.odd2, not r.t1_won)):
            ev = p * odd - 1
            if ev <= EDGE or ev > MAX_EDGE:
                continue
            bets.append({"pick": r.t1 if side_t1 else r.t2, "odd": odd, "ev": ev,
                         "won": won, "profit": (odd - 1) if won else -1.0,
                         "is_dog": odd > (r.odd2 if side_t1 else r.odd1)})
    if bets:
        b = pd.DataFrame(bets)
        print(f"\n  PARIS VALUE (EV {EDGE:.0%}-{MAX_EDGE:.0%}) : {len(b)} | gagnes "
              f"{int(b['won'].sum())} ({b['won'].mean()*100:.0f}%) | cote moy {b['odd'].mean():.2f}"
              f" | outsiders {int(b['is_dog'].sum())}/{len(b)}")
        print(f"  ROI flat : {b['profit'].mean()*100:+.1f}%  (profit {b['profit'].sum():+.2f}u)")
    else:
        print("\n  Aucun pari de value au-dessus du seuil.")
    print("=" * 90)


if __name__ == "__main__":
    main()

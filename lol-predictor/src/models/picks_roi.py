"""Rentabilité des picks ciblés : mise FIXE de 10 € par match, ROI réel.

Le juge de paix du projet : l'accuracy ne paie pas les factures, le ROI oui. Un pick
🎯 à 85 % joué à cote 1.10 est **perdant** sur la durée (il faut > 1.18). Ce module
mesure ça noir sur blanc.

Deux briques indépendantes :

1. **Journal des picks** (`data/picks_ledger.csv`) — suivi RÉEL.
   On fige, match par match : la date, notre prévision, notre proba, et **la cote du
   book au moment de la capture**. Puis on règle automatiquement (gagné/perdu) depuis
   la data Oracle, et on applique `STAKE` € à plat sur chaque pick.

2. **Simulation historique** (`simulate`) — réponse IMMÉDIATE.
   Rejoue les picks 🎯 du passé en walk-forward (zéro fuite, via `production_records`)
   pour sortir le taux de réussite réel, et surtout la **cote moyenne minimale** qui
   rend la stratégie rentable (= 1 / taux de réussite). C'est LE chiffre qui dit si
   ça vaut le coup avant d'avoir 30 paris au compteur.

Usage :
    python -m src.models.picks_roi                  # bilan + simulation 60 j
    python -m src.models.picks_roi --days 30
    python -m src.models.picks_roi --capture        # ajoute les picks à venir au journal
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd

from src.ingest.load_oracle import ROOT, load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAKE = 10.0                 # mise FIXE par pick (€) — jamais de mise variable
LEDGER_PATH = ROOT / "data" / "picks_ledger.csv"
COLS = [
    "match_date", "when", "league", "team1", "team2", "bestof",
    "tier", "pick", "our_proba", "breakeven", "odds", "odds_source",
    "stake", "result", "winner", "score", "profit", "captured_at",
]
RESULTS = ["open", "won", "lost", "void"]
TIER_HC, TIER_STRONG = "🎯", "⭐"

# Typage EXPLICITE des colonnes. Indispensable pour `st.data_editor` : une colonne
# vide relue depuis le CSV arrive en float64 (que des NaN), et Streamlit refuse une
# TextColumn posée sur du numérique (StreamlitAPIException).
STR_COLS = ["match_date", "when", "league", "team1", "team2", "tier", "pick",
            "odds_source", "result", "winner", "score", "captured_at"]
NUM_COLS = ["bestof", "our_proba", "breakeven", "odds", "stake", "profit"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Force les dtypes du journal (texte vs numérique) et complète les colonnes."""
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = np.nan
    for c in STR_COLS:
        d[c] = d[c].astype("string")
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["result"] = d["result"].fillna("open")
    d["stake"] = d["stake"].fillna(STAKE)
    d["tier"] = d["tier"].fillna(TIER_HC)   # journaux d'avant l'ajout du tier
    return d[COLS]


def value_flag(df: pd.DataFrame) -> pd.Series:
    """Colonne d'affichage : la cote prise couvre-t-elle la cote mini (1/proba) ?

    C'est le juge de paix d'un pick : un favori sûr à cote trop courte est perdant.
    """
    o = pd.to_numeric(df["odds"], errors="coerce")
    b = pd.to_numeric(df["breakeven"], errors="coerce")
    out = pd.Series("— cote à saisir", index=df.index, dtype="string")
    known = o.notna() & b.notna()
    out[known & (o > b)] = "✅ value"
    out[known & (o <= b)] = "❌ sous la cote mini"
    return out


# --------------------------------------------------------------------- journal
def load_ledger() -> pd.DataFrame:
    """Charge le journal des picks (crée un DataFrame vide si absent)."""
    if not LEDGER_PATH.exists():
        return normalize(pd.DataFrame(columns=COLS))
    return normalize(pd.read_csv(LEDGER_PATH))


def save_ledger(df: pd.DataFrame) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(LEDGER_PATH, index=False)


def _key(team1: str, team2: str, date: str) -> tuple:
    """Clé d'unicité d'un pick : paire d'équipes + jour (ordre des équipes ignoré)."""
    return (frozenset((str(team1), str(team2))), str(date)[:10])


# ------------------------------------------------------------------ candidats
def candidates(days: int = 7, cfg: dict | None = None, with_odds: bool = True,
               include_passed: bool = False, include_strong: bool = False) -> list[dict]:
    """Picks de la fenêtre, prêts à être ajoutés au journal.

    Reprend EXACTEMENT les filtres de la watchlist :
      - 🎯 `high_conf` : ligue fiable + favori ≥70 %/**game** + data ≥15 g + pas de x-ligue ;
      - ⭐ `strong` (si `include_strong`) : proba de **série** ≥62 %, fiable + data OK.

    ⚠️ Les ⭐ sont nettement plus fragiles : leur cote mini est haute (1.3-1.6) et le
    book cote souvent en dessous, donc beaucoup sont perdants. La colonne `tier`
    permet de comparer leur ROI à celui des 🎯 une fois l'échantillon constitué.

    Cotes : odds-api.io en primaire (vraies cotes book, nécessite une clé), puis
    Polymarket en secours (API publique sans clé, couvre d'autres ligues, mais
    volumes parfois dérisoires → le volume est indiqué dans `odds_source`).

    Par défaut on **exclut les matchs déjà commencés** : enregistrer une cote après
    le coup d'envoi fausserait le forward test (on connaîtrait déjà le déroulé).
    """
    from src.update.watchlist import build_rows
    covered, _ = build_rows(days, cfg)
    picks = [r for r in covered
             if r.get("high_conf") or (include_strong and r.get("strong"))]
    if not include_passed:
        picks = [r for r in picks if not r.get("passed")]

    odds_rows = _fetch_odds(days) if with_odds else []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    out: list[dict] = []
    for r in picks:
        fav, p_fav = ((r["team1"], r["p1"]) if r["p1"] >= r["p2"]
                      else (r["team2"], r["p2"]))
        match_date = (r.get("paris_date") or str(r.get("datetime", ""))[:10])
        odds, src = _lookup_odds(odds_rows, r["team1"], r["team2"], fav)
        if with_odds and not (odds == odds):    # rien chez le book -> Polymarket
            odds, src = _polymarket_odds(r["team1"], r["team2"], match_date, fav)
        out.append({
            "match_date": match_date,
            "when": r.get("when", ""),
            "league": r.get("league", "?"),
            "team1": r["team1"], "team2": r["team2"],
            "bestof": r.get("bestof", 1),
            "tier": TIER_HC if r.get("high_conf") else TIER_STRONG,
            "pick": fav,
            "our_proba": round(float(p_fav), 4),
            "breakeven": round(1.0 / p_fav, 2) if p_fav > 0 else np.nan,
            "odds": odds,
            "odds_source": src,
            "stake": STAKE,
            "result": "open",
            "winner": np.nan, "score": np.nan, "profit": np.nan,
            "captured_at": now,
        })
    return out


def _fetch_odds(days: int) -> list[dict]:
    """Cotes book (odds-api.io) — best-effort : sans clé API on renvoie [] et la
    cote sera à saisir à la main dans la page."""
    try:
        from src.update.oddsapi import load_key, scan
        if not load_key():
            return []
        return scan(days=days)
    except Exception:  # noqa: BLE001 - l'absence de cote ne doit jamais bloquer
        return []


def _polymarket_odds(team1: str, team2: str, date: str, pick: str) -> tuple[float, str]:
    """Cote implicite Polymarket pour notre pick — **API publique, sans clé**.

    Complément indispensable à odds-api.io, pour deux raisons : ça marche en ligne
    sans secret à configurer, et Polymarket cote des ligues que les books grand
    public ignorent (Arabian League, Road of Legends...).

    Le prix en cents EST une probabilité, quasi sans marge : la cote équivalente
    vaut 1/prix. ⚠️ Les volumes sont souvent minuscules (quelques dollars) : le prix
    est alors du bruit, donc on remonte le volume dans la source pour que ce soit
    visible au moment de décider.
    """
    try:
        from src.update.polymarket import find_market
        pm = find_market(team1, team2, date)
    except Exception:  # noqa: BLE001 - l'absence de cote ne doit jamais bloquer
        return np.nan, ""
    if not pm or pm.get("closed"):
        return np.nan, ""
    imp = pm["imp1"] if pick == team1 else pm["imp2"]
    if not imp or imp <= 0:
        return np.nan, ""
    return round(1.0 / imp, 2), f"polymarket (${pm['vol']:,.0f})"


def _lookup_odds(odds_rows: list[dict], team1: str, team2: str,
                 pick: str) -> tuple[float, str]:
    """Retrouve la meilleure cote dispo POUR notre pick dans le scan book."""
    pair = frozenset((team1, team2))
    for o in odds_rows:
        if frozenset((o.get("team1") or "", o.get("team2") or "")) != pair:
            continue
        if pick == o.get("team1") and o.get("best_home"):
            return float(o["best_home"]), "odds-api.io"
        if pick == o.get("team2") and o.get("best_away"):
            return float(o["best_away"]), "odds-api.io"
    return np.nan, ""


def add_candidates(ledger: pd.DataFrame, cands: list[dict]) -> tuple[pd.DataFrame, int]:
    """Ajoute les picks absents du journal (dédoublonnage sur paire+jour)."""
    known = {_key(r.team1, r.team2, r.match_date) for r in ledger.itertuples()}
    new = [c for c in cands if _key(c["team1"], c["team2"], c["match_date"]) not in known]
    if not new:
        return normalize(ledger), 0
    out = pd.concat([normalize(ledger), normalize(pd.DataFrame(new))], ignore_index=True)
    return normalize(out), len(new)


# ------------------------------------------------------------------ règlement
def _load_matches(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    m = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "matches.parquet")
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    return m


def series_result(matches: pd.DataFrame, team1: str, team2: str,
                  date: str, tol_days: int = 1) -> tuple[str | None, str | None]:
    """Vainqueur de la SÉRIE (agrège les games du jour) depuis la data Oracle.

    Renvoie (vainqueur, score) ou (None, None) si la série n'est pas (encore) dans
    le CSV. `tol_days` absorbe les décalages de fuseau (match tard le soir).
    """
    d = pd.to_datetime(date, errors="coerce")
    if pd.isna(d):
        return None, None
    pair = ((matches["blue_team"] == team1) & (matches["red_team"] == team2)) | \
           ((matches["blue_team"] == team2) & (matches["red_team"] == team1))
    tol = pd.Timedelta(days=tol_days)
    sub = matches[pair & (matches["date"] >= d - tol) & (matches["date"] <= d + tol)]
    if sub.empty:
        return None, None
    w1 = w2 = 0
    for row in sub.itertuples():
        winner = row.blue_team if row.y_winner == 1 else row.red_team
        if winner == team1:
            w1 += 1
        else:
            w2 += 1
    if w1 == w2:                       # série incomplète dans la data
        return None, f"{w1}-{w2}"
    return (team1 if w1 > w2 else team2), f"{w1}-{w2}"


def settle(ledger: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Règle les picks ouverts depuis la data Oracle et calcule le P/L à mise fixe."""
    if ledger.empty:
        return normalize(ledger)
    df = normalize(ledger)
    matches = _load_matches(cfg)

    for i, row in df.iterrows():
        if str(row.get("result")) in ("won", "lost", "void"):
            continue
        winner, score = series_result(matches, row["team1"], row["team2"],
                                      row["match_date"])
        if not winner:
            continue
        df.at[i, "winner"] = winner
        df.at[i, "score"] = score
        df.at[i, "result"] = "won" if winner == row["pick"] else "lost"

    return _compute_profit(df)


def _compute_profit(df: pd.DataFrame) -> pd.DataFrame:
    """P/L à mise fixe : gagné -> stake*(cote-1) ; perdu -> -stake ; sans cote -> NaN."""
    d = normalize(df)
    d["profit"] = np.nan
    won = d["result"] == "won"
    lost = d["result"] == "lost"
    d.loc[won & d["odds"].notna(), "profit"] = (
        d.loc[won & d["odds"].notna(), "stake"] * (d.loc[won & d["odds"].notna(), "odds"] - 1))
    d.loc[lost, "profit"] = -d.loc[lost, "stake"]
    d.loc[d["result"] == "void", "profit"] = 0.0
    return d


# --------------------------------------------------------------------- bilan
def summary(ledger: pd.DataFrame) -> dict:
    """KPIs du journal : réussite, ROI à mise fixe, cote moyenne, seuil de rentabilité."""
    if ledger.empty:
        return {"n": 0, "n_settled": 0}
    d = _compute_profit(ledger)
    settled = d[d["result"].isin(["won", "lost"])]
    if settled.empty:
        return {"n": len(d), "n_settled": 0, "n_open": int((d["result"] == "open").sum())}

    hit = float((settled["result"] == "won").mean())
    priced = settled[settled["odds"].notna()]
    staked = float(priced["stake"].sum())
    profit = float(priced["profit"].sum())
    return {
        "n": len(d),
        "n_settled": len(settled),
        "n_open": int((d["result"] == "open").sum()),
        "n_priced": len(priced),
        "hit_rate": hit,
        "avg_odds": float(priced["odds"].mean()) if len(priced) else float("nan"),
        "odds_needed": (1.0 / hit) if hit > 0 else float("inf"),
        "staked": staked,
        "profit": profit,
        "roi": (profit / staked) if staked else float("nan"),
        "avg_proba": float(pd.to_numeric(settled["our_proba"], errors="coerce").mean()),
    }


# ------------------------------------------------------- simulation historique
def simulate(days: int = 60, conf: float = 0.70, cfg: dict | None = None,
             stake: float = STAKE) -> dict:
    """Rejoue les picks 🎯 des `days` derniers jours (walk-forward, zéro fuite).

    Donne le taux de réussite RÉEL et la **cote moyenne minimale** de rentabilité.
    ⚠️ Granularité = la GAME (la data Oracle est par game), pas la série.
    """
    from src.models.eval_models import BURN_IN, production_records
    from src.update.elo import RELIABLE_ACC, compute_elo
    from src.update.watchlist import MIN_GAMES_CONF

    cfg = cfg or load_config()
    rec = production_records(cfg)
    rec = rec[rec["nmin"] >= BURN_IN]
    max_date = rec["date"].max()
    cutoff = max_date - pd.Timedelta(days=days - 1)
    rec = rec[rec["date"] >= cutoff]

    reliability = compute_elo(cfg)["reliability"]
    reliable = {lg for lg, acc in reliability.items() if acc >= RELIABLE_ACC}

    # Le filtre 🎯 : ligue fiable + confiance >= conf + data suffisante.
    hc = rec[rec["league"].isin(reliable)
             & (rec["proba_fav"] >= conf)
             & (rec["nmin"] >= MIN_GAMES_CONF)].copy()

    if hc.empty:
        return {"n": 0, "days": days, "cutoff": cutoff, "max_date": max_date}

    hit = float(hc["correct"].mean())
    odds_needed = 1.0 / hit if hit > 0 else float("inf")

    by_league = (hc.groupby("league")
                 .agg(n=("correct", "size"), hit=("correct", "mean"),
                      proba=("proba_fav", "mean"))
                 .reset_index().sort_values("n", ascending=False))
    by_league["cote_mini"] = 1.0 / by_league["hit"].replace(0, np.nan)

    return {
        "n": len(hc), "days": days, "conf": conf,
        "cutoff": cutoff, "max_date": max_date,
        "hit_rate": hit,
        "avg_proba": float(hc["proba_fav"].mean()),
        "odds_needed": odds_needed,
        "stake": stake,
        "by_league": by_league,
        "records": hc,
    }


def past_picks(days: int = 15, conf: float = 0.70, cfg: dict | None = None,
               stake: float = STAKE, odds: float | None = None) -> pd.DataFrame:
    """Les picks 🎯 **déjà joués**, au format EXACT du journal (mêmes colonnes).

    Pendant du journal réel, mais tourné vers le passé : on remet le détail du rejeu
    walk-forward (`simulate`) dans la structure du ledger, résultat réel déjà rempli.

    ⚠️ La cote est une **hypothèse** (`odds`), pas une cote réellement prise : les
    cotes historiques du book ne sont pas archivées. Le P/L est donc *simulé*, alors
    que celui du journal réel est *constaté*. Ne pas confondre les deux.
    """
    sim = simulate(days=days, conf=conf, cfg=cfg, stake=stake)
    rec = sim.get("records")
    if rec is None or len(rec) == 0:
        return normalize(pd.DataFrame(columns=COLS))

    d = rec.sort_values("date", ascending=False)
    out = pd.DataFrame({
        "match_date": d["date"].dt.strftime("%Y-%m-%d"),
        "when": d["date"].dt.strftime("%a %d/%m %H:%M"),
        "league": d["league"],
        "team1": d["blue"], "team2": d["red"],
        "bestof": 1,                                   # granularité = la game
        "tier": TIER_HC,                               # `simulate` ne garde que les 🎯
        "pick": d["favori"],
        "our_proba": d["proba_fav"].round(4),
        "breakeven": (1.0 / d["proba_fav"]).round(2),
        "odds": (float(odds) if odds else np.nan),
        "odds_source": ("hypothèse" if odds else ""),
        "stake": stake,
        "result": np.where(d["correct"], "won", "lost"),
        "winner": d["vainqueur"],
        "score": pd.NA,                                # pas de score de série par game
        "profit": np.nan,
        "captured_at": "walk-forward",
    })
    return _compute_profit(normalize(out))


def apply_odds(df: pd.DataFrame, odds: float, stake: float = STAKE) -> pd.DataFrame:
    """Rejoue le P/L d'une table de picks à une cote uniforme (hypothèse de travail).

    Permet de faire varier la cote sans relancer le rejeu walk-forward (coûteux).
    """
    d = normalize(df)
    d["odds"] = float(odds)
    d["odds_source"] = "hypothèse"
    d["stake"] = float(stake)
    return _compute_profit(d)


def roi_at_odds(hit_rate: float, odds: float, n: int, stake: float = STAKE) -> dict:
    """ROI théorique si TOUS les picks étaient joués à `odds` (mise fixe)."""
    staked = n * stake
    profit = n * (hit_rate * stake * (odds - 1) - (1 - hit_rate) * stake)
    return {"staked": staked, "profit": profit,
            "roi": (profit / staked) if staked else float("nan")}


# ----------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Rentabilité des picks ciblés (mise fixe)")
    ap.add_argument("--days", type=int, default=60, help="fenêtre de simulation (def. 60 j)")
    ap.add_argument("--conf", type=float, default=0.70, help="seuil de confiance (def. 0.70)")
    ap.add_argument("--capture", action="store_true",
                    help="ajoute les picks à venir au journal")
    args = ap.parse_args()

    if args.capture:
        led = load_ledger()
        led, added = add_candidates(led, candidates(days=7))
        led = settle(led)
        save_ledger(led)
        print(f"[capture] {added} pick(s) ajouté(s) -> {LEDGER_PATH}")

    print(f"\n=== SIMULATION : picks 🎯 des {args.days} derniers jours "
          f"(confiance >= {args.conf*100:.0f}%/game) ===")
    sim = simulate(days=args.days, conf=args.conf)
    if not sim.get("n"):
        print("  (aucun pick sur la fenêtre)")
    else:
        print(f"  Data jusqu'au {sim['max_date']:%Y-%m-%d} · {sim['n']} picks")
        print(f"  Proba moyenne annoncée : {sim['avg_proba']*100:.1f}%")
        print(f"  Taux de reussite REEL  : {sim['hit_rate']*100:.1f}%")
        print(f"  >>> COTE MOYENNE MINIMALE POUR ETRE RENTABLE : {sim['odds_needed']:.2f}")
        print(f"\n  ROI simule a mise fixe de {STAKE:.0f} EUR :")
        for o in (1.10, 1.20, 1.30, 1.40, 1.50):
            r = roi_at_odds(sim["hit_rate"], o, sim["n"])
            flag = "OK" if r["roi"] > 0 else "PERTE"
            print(f"    cote {o:.2f} -> {r['profit']:+8.2f} EUR  (ROI {r['roi']*100:+6.1f}%)  [{flag}]")
        print("\n  Par ligue :")
        for row in sim["by_league"].itertuples():
            print(f"    {row.league:8} n={row.n:3d}  reussite {row.hit*100:5.1f}%  "
                  f"cote mini {row.cote_mini:.2f}")

    led = settle(load_ledger())
    s = summary(led)
    print(f"\n=== JOURNAL REEL ({LEDGER_PATH.name}) ===")
    if not s.get("n"):
        print("  (vide — lance --capture pour enregistrer les picks a venir)")
        return
    print(f"  {s['n']} picks · {s.get('n_settled', 0)} regles · {s.get('n_open', 0)} en attente")
    if s.get("n_settled"):
        print(f"  Reussite : {s['hit_rate']*100:.1f}%  (proba annoncee {s['avg_proba']*100:.1f}%)")
        print(f"  Cote mini de rentabilite : {s['odds_needed']:.2f}")
        if s.get("n_priced"):
            print(f"  Cote moyenne prise : {s['avg_odds']:.2f}")
            print(f"  Mise totale {s['staked']:.0f} EUR -> P/L {s['profit']:+.2f} EUR "
                  f"(ROI {s['roi']*100:+.1f}%)")
        else:
            print("  (aucune cote saisie -> ROI incalculable ; renseigne les cotes)")


if __name__ == "__main__":
    main()

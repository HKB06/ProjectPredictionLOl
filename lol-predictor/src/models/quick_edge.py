"""Edge rapide (Elo toutes ligues) sur des matchs a venir, vs cotes book.

Pre-game, SANS draft (drafts pas encore connues) -> signal = force d'equipe (Elo).
Sert a reperer ou le book pourrait se tromper (ligues peu suivies).

Modifier MATCHUPS : (label, sous-chaine equipe1, cote1, sous-chaine equipe2, cote2).
Usage : python -m src.models.quick_edge
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config

K = 24.0
BASE = 1500.0

# (label, equipe1 (sous-chaine), cote1, equipe2 (sous-chaine), cote2)
# NB : matching par sous-chaine du nom Oracle (sensible aux accents/mojibake, ex. "LOS"="L\ufffdS").
MATCHUPS = [
    ("EMEA Masters play-in", "einfach", 1.52, "nightbird", 2.35),
]


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    teams = df[df["position"].str.lower() == "team"].copy()
    teams["tl"] = teams["teamname"].astype(str).str.lower()

    tb = teams[teams["side"].str.lower() == "blue"][["gameid", "tl", "result", "date", "league"]]
    tr = teams[teams["side"].str.lower() == "red"][["gameid", "tl"]]
    g = tb.merge(tr, on="gameid", suffixes=("_b", "_r")).dropna().sort_values("date")

    elo: dict[str, float] = defaultdict(lambda: BASE)
    n: dict[str, int] = defaultdict(int)
    last_league: dict[str, str] = {}
    for a, yb, lg, b in zip(g["tl_b"], g["result"], g["league"], g["tl_r"]):
        ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400))
        elo[a] += K * (yb - ea)
        elo[b] += K * ((1 - yb) - (1 - ea))
        n[a] += 1
        n[b] += 1
        last_league[a] = lg
        last_league[b] = lg

    def find(key: str):
        cands = [t for t in n if key in t]
        if not cands:
            return None
        name = max(cands, key=lambda t: n[t])  # le plus de games = le bon
        return name

    print("=" * 92)
    print(f"  EDGE RAPIDE (Elo toutes ligues, {len(g)} games) — pre-game, SANS draft")
    print("=" * 92)
    header = f"  {'Match':22} {'Equipe':26} {'Elo':>5} {'n':>4} {'p_model':>8} {'p_book':>7} {'edge':>7}"
    for label, k1, o1, k2, o2 in MATCHUPS:
        t1, t2 = find(k1), find(k2)
        print("-" * 92)
        if not t1 or not t2:
            print(f"  {label}: equipe introuvable ({k1 if not t1 else k2})")
            continue
        e1, e2 = elo[t1], elo[t2]
        p1 = 1.0 / (1.0 + 10 ** ((e2 - e1) / 400))  # P(team1 gagne) selon Elo
        # devig book
        inv1, inv2 = 1 / o1, 1 / o2
        vig = inv1 + inv2 - 1
        b1, b2 = inv1 / (inv1 + inv2), inv2 / (inv1 + inv2)
        print(f"  {label}  (marge book = {vig*100:.1f}%)")
        print(header)
        for name, e, nn, pm, pb, cote in [
            (t1, e1, n[t1], p1, b1, o1),
            (t2, e2, n[t2], 1 - p1, b2, o2),
        ]:
            edge = pm - pb
            flag = "  <== value" if edge > 0.05 else ""
            print(f"  {label[:20]:22} {name[:26]:26} {e:5.0f} {nn:4d} {pm*100:7.1f}% {pb*100:6.1f}% {edge*100:+6.1f}%  cote {cote}{flag}")
    print("=" * 92)
    print("  Rappel : Elo-only (pas de draft), 1 map = grosse variance, cold-start possible si roster recent.")
    print("  Heavy favoris (cote < ~1.2) : a eviter (gain minime, risque de ruine sur upset).")
    print("=" * 92)


if __name__ == "__main__":
    main()

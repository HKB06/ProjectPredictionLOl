"""Les modèles face au MARCHÉ : le seul juge de la rentabilité.

Relie chaque série jouée à son prix Polymarket d'avant-match (`src.update.polymarket_history`)
puis répond à trois questions :
  1. Qui prédit le mieux : le marché, l'ancien modèle ou le nouveau ?
  2. Un modèle sait-il quelque chose que le marché ignore ? (mélange marché + modèle,
     appris sur la première moitié de la saison, jugé sur la seconde)
  3. Qu'aurait rapporté le fait de parier chaque fois que le modèle voit plus de
     `edge` points d'écart avec le prix ?

Un prix Polymarket est une probabilité (pas de marge de bookmaker). Pour le pari simulé on
ajoute `SLIPPAGE` au prix : on n'achète jamais exactement au prix affiché.

Usage :
    python -m src.v2.market
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.update.polymarket_history import load_cache
from src.update.watchlist import alias_of, core_tokens, fuzzy_match, norm, norm_core

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MATCH_WINDOW = pd.Timedelta(hours=18)     # écart toléré entre l'heure annoncée et la game 1
DRAFT = pd.Timedelta(minutes=10)          # le prix doit dater d'au moins 10 min avant le début
                                          # EN JEU : plus tard, il a déjà vu la draft (ou la game)
MIN_VOLUME = 1000.0                       # $ échangés : en dessous, le prix ne veut rien dire
SLIPPAGE = 0.01
EDGES = (0.03, 0.05, 0.08, 0.12)
SOURCES = {"Marché (Polymarket)": "mkt_p", "Ancien modèle": "old_p", "Nouveau modèle": "new_p"}


class _Resolver:
    """Nom d'équipe Polymarket -> nom Oracle (mêmes règles que la watchlist)."""

    def __init__(self, teams) -> None:
        teams = list(teams)
        self.full = {norm(t): t for t in teams}
        self.core = {norm_core(t): t for t in teams}
        self.tokens = {t: core_tokens(t) for t in teams}
        self.cache: dict[str, str | None] = {}

    def __call__(self, name: str) -> str | None:
        if name not in self.cache:
            self.cache[name] = (alias_of(name, self.tokens) or self.full.get(norm(name))
                                or self.core.get(norm_core(name))
                                or fuzzy_match(core_tokens(name), self.tokens))
        return self.cache[name]


def _pre_match(row, actual: pd.Timestamp, prefix: str = "") -> float:
    """Dernier prix relevé AVANT le vrai coup d'envoi (la game 1 du CSV Oracle).

    L'heure annoncée par Polymarket peut être postérieure au vrai début : on recule alors
    au relevé précédent plutôt que de lire un prix déjà influencé par la game.
    """
    early = row.start - actual                 # > 0 : le marché annonce un début postérieur
    if early <= -DRAFT:
        key = "p_close"
    elif early <= pd.Timedelta(minutes=30) - DRAFT:
        key = "p_30m"
    elif early <= pd.Timedelta(hours=1) - DRAFT:
        key = "p_1h"
    else:
        key = "p_24h"
    return getattr(row, prefix + key)


def attach_market(series: pd.DataFrame, games: pd.DataFrame, year: int,
                  snapshot: str | None = None) -> pd.DataFrame:
    """Ajoute à `series` : mkt_p (P(team_a gagne la série) selon le marché), mkt_volume,
    mkt_g1_p (P(team_a gagne la game 1)), et les probas des modèles côté team_a.

    `snapshot` force un relevé (« p_24h », « p_1h »…) au lieu du dernier prix d'avant-match :
    sert à vérifier que le résultat ne dépend pas d'un prix relevé trop tard.
    """
    s = series.copy()
    for tag in ("old", "new"):
        s[f"{tag}_p"] = np.where(s[f"fav_{tag}"] == s["team_a"], s[f"p_{tag}"], 1 - s[f"p_{tag}"])
    first = games.sort_values("date").drop_duplicates("series_id").set_index("series_id")
    s["old_g1"] = s["series_id"].map(first["p_old"])       # team_a = côté bleu de la game 1
    s["new_g1"] = s["series_id"].map(first["p_new"])
    s["a_won_g1"] = s["series_id"].map((first["winner"] == first["blue"]).astype(float))
    for c in ("mkt_p", "mkt_volume", "mkt_g1_p", "mkt_lead_min"):
        s[c] = np.nan
    s["mkt_title"] = None

    mk = load_cache(year)
    if mk.empty:
        return s
    mk = mk.assign(start=pd.to_datetime(mk["start"]))
    resolve = _Resolver(pd.concat([s["team_a"], s["team_b"]]).unique())
    index: dict[frozenset, list] = {}
    for i, a, b, d in zip(s.index, s["team_a"], s["team_b"], s["date"]):
        index.setdefault(frozenset((a, b)), []).append((d, i))

    used: set = set()
    for r in mk.itertuples():
        a, b = resolve(r.team_a), resolve(r.team_b)
        if not a or not b or a == b:
            continue
        cand = [(abs(d - r.start), i, d) for d, i in index.get(frozenset((a, b)), [])]
        if not cand:
            continue
        gap, i, actual = min(cand)
        if gap > MATCH_WINDOW or i in used:
            continue
        used.add(i)
        flip = s.at[i, "team_a"] != a                       # ordre Polymarket ≠ ordre de la série
        p = getattr(r, snapshot) if snapshot else _pre_match(r, actual)
        g1 = getattr(r, "g1_" + snapshot) if snapshot else _pre_match(r, actual, "g1_")
        s.at[i, "mkt_p"] = (1 - p) if flip else p
        s.at[i, "mkt_g1_p"] = (1 - g1) if flip else g1
        s.at[i, "mkt_volume"] = r.volume
        s.at[i, "mkt_lead_min"] = (actual - r.start).total_seconds() / 60   # > 0 : joué après l'heure annoncée
        s.at[i, "mkt_title"] = r.title
    return s


# ------------------------------------------------------------------ évaluation
def _ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sample(s: pd.DataFrame, solid: bool = True) -> pd.DataFrame:
    """Séries jugeables : terminées, cotées avec un vrai volume, et (si `solid`) où le
    nouveau modèle était entraîné et les deux équipes connues."""
    d = s[s["winner"].notna() & s["mkt_p"].notna() & (s["mkt_volume"] >= MIN_VOLUME)]
    if solid:
        d = d[~d["rodage"] & ~d["low_data"]]
    return d.assign(y=(d["winner"] == d["team_a"]).astype(float)).sort_values("date")


def scoreboard(d: pd.DataFrame) -> pd.DataFrame:
    """Qualité des probabilités de série, sur exactement les mêmes séries."""
    y = d["y"].to_numpy()
    ref = _ll(y, d["mkt_p"].to_numpy())
    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    rows = []
    for name, col in SOURCES.items():
        p = d[col].to_numpy()
        diff = _ll(y, p) - ref
        lo, hi = np.percentile([diff[i].mean() for i in idx], [5, 95])
        rows.append({
            "source": name, "log-loss": round(float(_ll(y, p).mean()), 4),
            "Brier": round(float(np.mean((p - y) ** 2)), 4),
            "bon vainqueur %": round(float(((p >= 0.5) == (y == 1)).mean() * 100), 1),
            "écart au marché (x1000)": round(float(diff.mean() * 1000), 1),
            "IC 90 %": f"[{lo * 1000:+.1f} ; {hi * 1000:+.1f}]",
        })
    out = pd.DataFrame(rows)
    out.attrs["n"] = len(d)
    out.attrs["span"] = (d["date"].min(), d["date"].max())
    return out


def blend(d: pd.DataFrame) -> pd.DataFrame:
    """Le modèle apporte-t-il de l'information EN PLUS du marché ?

    Régression (marché + modèle) apprise sur la première moitié, jugée sur la seconde,
    contre le marché seul recalibré de la même façon. Gain > 0 avec un intervalle qui ne
    touche pas zéro = le modèle sait quelque chose que le prix n'intègre pas.
    """
    cut = len(d) // 2
    tr, te = d.iloc[:cut], d.iloc[cut:]
    y = te["y"].to_numpy()
    base = LogisticRegression(C=1e6).fit(_logit(tr[["mkt_p"]]), tr["y"])
    ref = _ll(y, base.predict_proba(_logit(te[["mkt_p"]]))[:, 1])
    rng = np.random.default_rng(42)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    rows = []
    for name, col in list(SOURCES.items())[1:]:
        m = LogisticRegression(C=1e6).fit(_logit(tr[["mkt_p", col]]), tr["y"])
        diff = ref - _ll(y, m.predict_proba(_logit(te[["mkt_p", col]]))[:, 1])
        lo, hi = np.percentile([diff[i].mean() for i in idx], [5, 95])
        rows.append({"modèle ajouté au marché": name,
                     "poids du marché": round(float(m.coef_[0][0]), 2),
                     "poids du modèle": round(float(m.coef_[0][1]), 2),
                     "gain (x1000)": round(float(diff.mean() * 1000), 1),
                     "IC 90 %": f"[{lo * 1000:+.1f} ; {hi * 1000:+.1f}]",
                     "verdict": "apporte de l'info" if lo > 0 else "rien de prouvé"})
    out = pd.DataFrame(rows)
    out.attrs["n_test"] = len(te)
    return out


def bets(d: pd.DataFrame, col: str, edge: float) -> pd.DataFrame:
    """Paris simulés : 1 unité sur le côté où le modèle dépasse le prix d'au moins `edge`."""
    p, mkt, y = d[col].to_numpy(), d["mkt_p"].to_numpy(), d["y"].to_numpy()
    on_a = (p - mkt) >= edge
    on_b = (mkt - p) >= edge
    price = np.where(on_a, mkt, 1 - mkt) + SLIPPAGE
    won = np.where(on_a, y == 1, y == 0)
    out = d.loc[on_a | on_b, ["date", "league", "team_a", "team_b", "winner"]].copy()
    sel = on_a | on_b
    out["pick"] = np.where(on_a, d["team_a"], d["team_b"])[sel]
    out["price"] = price[sel]
    out["model_p"] = np.where(on_a, p, 1 - p)[sel]
    out["won"] = won[sel]
    out["profit"] = np.where(won[sel], 1 / price[sel] - 1, -1.0)
    return out[(out["price"] > 0.02) & (out["price"] < 0.99)]


def betting_board(d: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for name, col in list(SOURCES.items())[1:]:
        for edge in EDGES:
            b = bets(d, col, edge)
            if b.empty:
                continue
            pr = b["profit"].to_numpy()
            boot = [pr[rng.integers(0, len(pr), len(pr))].mean() for _ in range(1000)]
            lo, hi = np.percentile(boot, [5, 95])
            rows.append({"modèle": name, "écart mini": f"{edge * 100:.0f} pts", "paris": len(b),
                         "gagnés %": round(float(b["won"].mean() * 100), 1),
                         "prix moyen": round(float(b["price"].mean()), 2),
                         "rendement %": round(float(pr.mean() * 100), 1),
                         "IC 90 %": f"[{lo * 100:+.1f} ; {hi * 100:+.1f}]",
                         "verdict": "rentable" if lo > 0 else "perdant" if hi < 0 else "pas prouvé"})
    return pd.DataFrame(rows)


def game1_board(s: pd.DataFrame) -> pd.DataFrame:
    """Même comparaison sur la GAME 1 seule (marché « Game 1 Winner ») : nos probas sont
    faites par game, celle-ci ne passe par aucune conversion game -> série."""
    d = s[s["mkt_g1_p"].notna() & s["a_won_g1"].notna() & ~s["rodage"] & ~s["low_data"]]
    y = d["a_won_g1"].to_numpy()
    rows = []
    for name, col in (("Marché (Polymarket)", "mkt_g1_p"), ("Ancien modèle", "old_g1"),
                      ("Nouveau modèle", "new_g1")):
        p = d[col].to_numpy()
        rows.append({"source": name, "log-loss": round(float(_ll(y, p).mean()), 4),
                     "bon vainqueur %": round(float(((p >= 0.5) == (y == 1)).mean() * 100), 1)})
    out = pd.DataFrame(rows)
    out.attrs["n"] = len(d)
    return out


def main() -> None:
    from src.v2.model import SEASON_WARMUP_DAYS, build_dataset, walk_forward
    from src.v2.season import build_season

    df, _ = build_dataset()
    games, series = build_season(df, walk_forward(df, SEASON_WARMUP_DAYS))
    s = attach_market(series, games, int(df["date"].dt.year.max()))
    done = s["winner"].notna()
    print(f"Séries : {int(done.sum())} · cotées sur Polymarket : {int(s.loc[done, 'mkt_p'].notna().sum())}")
    d = sample(s)
    sb = scoreboard(d)
    print(f"\n=== QUI PRÉDIT LE MIEUX ? ({sb.attrs['n']} séries cotées, données solides, "
          f"{sb.attrs['span'][0]:%d/%m} -> {sb.attrs['span'][1]:%d/%m/%Y}) ===")
    print(sb.to_string(index=False))
    g1 = game1_board(s)
    print(f"\n=== SUR LA GAME 1 SEULE ({g1.attrs['n']} games) ===")
    print(g1.to_string(index=False))
    bl = blend(d)
    print(f"\n=== LE MODÈLE AJOUTE-T-IL DE L'INFO AU MARCHÉ ? (test sur {bl.attrs['n_test']} séries) ===")
    print(bl.to_string(index=False))
    print(f"\n=== PARIS SIMULÉS (1 unité, prix + {SLIPPAGE:.2f}) ===")
    print(betting_board(d).to_string(index=False))
    cov = (s[done].assign(cote=s["mkt_p"].notna()).groupby("league")["cote"].agg(["sum", "size"])
           .sort_values("size", ascending=False))
    print("\nCouverture par ligue (cotées / jouées) :",
          ", ".join(f"{lg} {int(r['sum'])}/{int(r['size'])}" for lg, r in cov.head(25).iterrows()))


if __name__ == "__main__":
    main()

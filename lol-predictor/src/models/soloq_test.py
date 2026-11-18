"""LE TEST : la préparation soloQ apporte-t-elle de l'information que le marché n'a pas ?

Pour chaque game pro, on connaît les 10 joueurs qui l'ont RÉELLEMENT jouée (feuille de match
Oracle) et leur soloQ des jours d'AVANT (`src.features.soloq_prep`) : volume, écart à leur
habitude, travail sur les champions méta de leur ligue. On en tire, par équipe, la moyenne
des joueurs couverts, puis l'écart entre les deux équipes.

Trois juges, du plus facile au plus dur :
  1. vs l'ancien modèle (Elo)          — par game
  2. vs le nouveau modèle (joueurs)    — par game
  3. vs le prix Polymarket d'avant-match — par série (préparation avant la game 1)

Méthode, fixée AVANT de voir les données complètes :
  - walk-forward : chaque mois est prédit par une régression apprise uniquement sur les
    mois d'avant (jamais sur le futur) ; rien n'est jugé avant `MIN_TRAIN` games
    (`MIN_TRAIN_SERIES` séries) d'apprentissage ;
  - la référence est recalibrée de la même façon (même régression, sans les indicateurs) :
    le gain mesuré vient des indicateurs, pas d'un simple recalage ;
  - intervalle de confiance par bootstrap SUR LES SÉRIES (les games d'une série se ressemblent) ;
  - le modèle jugé en premier est « tous les indicateurs » ; les indicateurs un par un sont
    donnés pour comprendre, pas pour choisir le meilleur après coup.

Critère d'adoption (les trois à la fois) :
  a. gain de log-loss contre le marché avec un IC 95 % qui ne touche pas zéro ;
  b. gain positif sur la première ET la seconde moitié de la période jugée ;
  c. paris simulés (prix + glissement) non perdants.
Sinon : « pas prouvé », et rien n'entre dans le modèle.

Usage :
    python -m src.models.soloq_test            # test complet
    python -m src.models.soloq_test --save     # + écrit reports/soloq_test.txt
"""
from __future__ import annotations

import argparse
import io
import sys
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import soloq_prep as sp
from src.ingest.load_oracle import ROOT
from src.update.bet_universe import universe
from src.v2 import market as mk
from src.v2.precompute import get

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FEATS = {"lvr": "volume vs habitude", "vol_7": "volume 7 j", "role_14": "parties à son poste 14 j",
         "meta_champs_14": "champions méta travaillés", "meta_share_14": "part de méta",
         "meta_wr_14": "victoires sur la méta"}
MIN_PLAYERS = 4          # joueurs couverts par équipe pour qu'une game compte
MIN_TRAIN = 300          # games d'apprentissage avant de juger quoi que ce soit
MIN_TRAIN_SERIES = 150   # idem par série (3 fois moins de lignes que de games)
C = 1.0                  # régularisation (indicateurs centrés-réduits sur l'apprentissage)
N_BOOT = 2000
MIN_DETAIL = 0.93        # lignes de détail du joueur / parties comptées, 14 j avant match
MIN_EXACT = 0.97         # en dessous, l'historique est incomplet : pas de conclusion
EDGES = (0.03, 0.05, 0.08)


# ------------------------------------------------------------------ tables
def team_features(f: pd.DataFrame) -> pd.DataFrame:
    """Moyenne des joueurs couverts, par (game, équipe). Une équipe sous MIN_PLAYERS est écartée."""
    f = f[f["covered"].fillna(False).astype(bool)].copy()
    f["lvr"] = np.log(f["vol_ratio"])
    t = f.groupby(["gameid", "teamname"]).agg(n=("lvr", "size"), **{k: (k, "mean") for k in FEATS})
    return t[t["n"] >= MIN_PLAYERS].drop(columns="n")


def _diff(d: pd.DataFrame, tf: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Ajoute d_<indicateur> = équipe `a` − équipe `b` ; écarte les lignes sans les deux équipes."""
    fa = tf.add_prefix("a_").reset_index().rename(columns={"teamname": a})
    fb = tf.add_prefix("b_").reset_index().rename(columns={"teamname": b})
    n0 = len(d)
    d = d.merge(fa, on=["gameid", a]).merge(fb, on=["gameid", b])
    assert not d.duplicated("gameid").any(), "une game apparaît deux fois après la jointure"
    for k in FEATS:
        d[f"d_{k}"] = d[f"a_{k}"] - d[f"b_{k}"]
    d.attrs["kept"] = (len(d), n0)
    return d


def game_table(eng: dict, tf: pd.DataFrame, leagues: set) -> pd.DataFrame:
    g = eng["games"]
    g = g[g["league"].isin(leagues) & ~g["rodage"]].copy()
    g["y"] = (g["winner"] == g["blue"]).astype(float)
    d = _diff(g, tf, "blue", "red")
    return d.sort_values("date").reset_index(drop=True)


def series_table(eng: dict, tf: pd.DataFrame, leagues: set, snapshot: str | None = None) -> pd.DataFrame:
    """Une ligne par série cotée ; indicateurs relevés avant la GAME 1 (= avant le pari)."""
    year = int(pd.to_datetime(eng["series"]["date"]).dt.year.max())
    s = mk.attach_market(eng["series"], eng["games"], year, snapshot=snapshot)
    s = mk.sample(s[s["league"].isin(leagues)])
    first = eng["games"].sort_values("date").drop_duplicates("series_id")[["series_id", "gameid"]]
    s = s.merge(first, on="series_id")
    d = _diff(s, tf, "team_a", "team_b")
    kept = d.attrs["kept"]
    # `team_a` est l'équipe côté bleu de la game 1, qui gagne un peu plus souvent : sans
    # précaution, la régression apprendrait « parie team_a », une information de côté qui
    # n'a rien à voir avec la soloQ. On retourne une série sur deux (tirage fixe).
    flip = np.array([zlib.crc32(str(i).encode()) % 2 == 1 for i in d["series_id"]])
    d.loc[flip, ["team_a", "team_b"]] = d.loc[flip, ["team_b", "team_a"]].to_numpy()
    d.loc[flip, "mkt_p"] = 1 - d.loc[flip, "mkt_p"]
    d.loc[flip, "y"] = 1 - d.loc[flip, "y"]
    for k in FEATS:
        d.loc[flip, f"d_{k}"] = -d.loc[flip, f"d_{k}"]
    d = d.sort_values("date").reset_index(drop=True)
    d.attrs["kept"] = kept
    return d


# ------------------------------------------------------------------ walk-forward
def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _ll(y, p) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def walk_forward(d: pd.DataFrame, base: str, extra: list[str], min_train: int = MIN_TRAIN) -> np.ndarray:
    """Proba pour chaque ligne, apprise sur les mois STRICTEMENT antérieurs. NaN avant `min_train`."""
    month = d["date"].dt.to_period("M")
    x0 = _logit(d[base])
    y = d["y"].to_numpy()
    pred = np.full(len(d), np.nan)
    for m in month.unique():
        te = (month == m).to_numpy()
        tr = (month < m).to_numpy()
        if tr.sum() < min_train:
            continue
        cols = [x0]
        for k in extra:
            v = d[k].to_numpy(dtype=float)
            mu, sd = np.nanmean(v[tr]), np.nanstd(v[tr])
            cols.append(np.nan_to_num((v - mu) / (sd if sd > 0 else 1.0)))   # manquant -> moyenne
        X = np.column_stack(cols)
        model = LogisticRegression(C=C).fit(X[tr], y[tr])
        pred[te] = model.predict_proba(X[te])[:, 1]
    return pred


def _ci(diff: np.ndarray, groups: np.ndarray, rng) -> tuple[float, float]:
    """IC 95 % de la moyenne, en rééchantillonnant des séries entières."""
    codes, _ = pd.factorize(groups)
    sums = np.bincount(codes, weights=diff)
    cnts = np.bincount(codes)
    n = len(sums)
    idx = rng.integers(0, n, (N_BOOT, n))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def compare(d: pd.DataFrame, base: str, label: str,
            min_train: int = MIN_TRAIN) -> tuple[pd.DataFrame, np.ndarray]:
    """Gain de log-loss de (référence + indicateurs) contre la référence recalibrée."""
    rng = np.random.default_rng(0)
    y = d["y"].to_numpy()
    ref = walk_forward(d, base, [], min_train)
    ok = ~np.isnan(ref)
    half = np.zeros(len(d), dtype=bool)
    half[np.flatnonzero(ok)[: ok.sum() // 2]] = True
    groups = d["series_id"].to_numpy()
    rows, primary = [], None
    models = [("TOUS les indicateurs", [f"d_{k}" for k in FEATS])] + \
             [(lab, [f"d_{k}"]) for k, lab in FEATS.items()]
    for name, cols in models:
        p = walk_forward(d, base, cols, min_train)
        if primary is None:
            primary = p
        diff = _ll(y[ok], ref[ok]) - _ll(y[ok], p[ok])
        lo, hi = _ci(diff, groups[ok], rng)
        h1 = diff[half[ok]].mean()
        h2 = diff[~half[ok]].mean()
        rows.append({"ajouté à " + label: name, "gain (x1000)": round(diff.mean() * 1000, 1),
                     "IC 95 %": f"[{lo * 1000:+.1f} ; {hi * 1000:+.1f}]",
                     "1re moitié": round(h1 * 1000, 1), "2e moitié": round(h2 * 1000, 1),
                     "verdict": "PROUVÉ" if lo > 0 and h1 > 0 and h2 > 0
                     else "nuisible" if hi < 0 else "pas prouvé"})
    out = pd.DataFrame(rows)
    out.attrs.update(n=int(ok.sum()), raw=float(_ll(y[ok], d[base].to_numpy()[ok]).mean()),
                     ref=float(_ll(y[ok], ref[ok]).mean()))
    return out, primary


def betting(d: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Paris simulés sur les séries jugées : 1 unité quand (marché + soloQ) s'écarte du prix."""
    rng = np.random.default_rng(0)
    ok = ~np.isnan(p)
    dd = d[ok].assign(prep_p=p[ok])
    rows = []
    for edge in EDGES:
        b = mk.bets(dd, "prep_p", edge)
        if b.empty:
            rows.append({"écart mini": f"{edge * 100:.0f} pts", "paris": 0})
            continue
        pr = b["profit"].to_numpy()
        lo, hi = np.percentile([pr[rng.integers(0, len(pr), len(pr))].mean()
                                for _ in range(N_BOOT)], [2.5, 97.5])
        rows.append({"écart mini": f"{edge * 100:.0f} pts", "paris": len(b),
                     "gagnés %": round(b["won"].mean() * 100, 1),
                     "prix moyen": round(b["price"].mean(), 2),
                     "rendement %": round(pr.mean() * 100, 1),
                     "IC 95 %": f"[{lo * 100:+.1f} ; {hi * 100:+.1f}]",
                     "verdict": "rentable" if lo > 0 else "perdant" if hi < 0 else "pas prouvé"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rapport
def _show(title: str, t: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    a = t.attrs
    if "n" in a:
        print(f"{a['n']} exemples jugés · log-loss référence brute {a['raw']:.4f}, "
              f"recalibrée {a['ref']:.4f}")
    print(t.to_string(index=False))


def run() -> None:
    pd.set_option("display.width", 220)
    eng = get()
    leagues = set().union(*universe())
    since = pd.to_datetime(eng["games"]["date"]).min().strftime("%Y-%m-%d")
    f = sp.build(since=since)
    cov = f["covered"].fillna(False).astype(bool)
    exact = f.attrs.get("exact", 0.0)
    print(f"Historique soloQ : {exact:.0%} des parties listées ont leur détail (date exacte, "
          "champion, poste). Seules celles-là sont comptées.")
    if exact < MIN_EXACT:
        print(f"!!! HISTORIQUE INCOMPLET (< {MIN_EXACT:.0%}) : les volumes sont sous-estimés là où "
              "le détail manque. Résultats à ne PAS interpréter, essai technique seulement.")
    done = f.loc[cov, "det_14"].sum() / max(f.loc[cov, "vol_14"].sum(), 1)
    print(f"Détail du joueur présent pour {done:.0%} de ses parties des 14 jours d'avant-match "
          f"(attendu ≈ 97 % : les parties abandonnées ne comptent pas). "
          f"{f.attrs.get('impostors', 0)} comptes écartés (mal attribués).")
    if done < MIN_DETAIL:
        print(f"!!! DÉTAIL INCOMPLET (< {MIN_DETAIL:.0%}) : les indicateurs de poste et de méta sont "
              "sous-estimés. Seuls les indicateurs de VOLUME sont interprétables.")
    cut = f["complete"].eq(False)
    print(f"Historique tronqué par Riot (1 000 dernières parties par compte) : "
          f"{f.attrs.get('truncated', 0)} joueurs concernés · {cut.mean():.0%} des lignes joueur x "
          "game écartées, la fenêtre d'avant-match n'étant pas entièrement connue.")
    print("Lignes écartées par mois (%) :", (f.assign(c=cut).groupby(f["date"].dt.strftime("%m"))["c"]
          .mean() * 100).round(0).astype(int).to_dict())
    print(f"Joueurs x games (ligues pariables, depuis {since}) : {len(f)} · reliés à un joueur "
          f"suivi {f['page'].notna().mean():.0%} · avec soloQ dans les 60 j {cov.mean():.0%}")
    print("Couverture par ligue (%) :", (f.assign(c=cov).groupby("league")["c"].mean() * 100)
          .round(0).sort_values(ascending=False).astype(int).to_dict())
    tf = team_features(f)

    g = game_table(eng, tf, leagues)
    print(f"\nGames avec ≥ {MIN_PLAYERS} joueurs couverts dans CHAQUE équipe : "
          f"{g.attrs['kept'][0]} sur {g.attrs['kept'][1]} "
          f"({g['date'].min():%d/%m/%Y} -> {g['date'].max():%d/%m/%Y})")
    for base, label in (("p_old", "l'ancien modèle (Elo)"), ("p_new", "le nouveau modèle")):
        t, _ = compare(g, base, label)
        _show(f"PAR GAME — contre {label}", t)

    s = series_table(eng, tf, leagues)
    print(f"\nSéries cotées avec les deux équipes couvertes : {s.attrs['kept'][0]} sur "
          f"{s.attrs['kept'][1]}")
    t, p = compare(s, "mkt_p", "Polymarket", MIN_TRAIN_SERIES)
    _show("PAR SÉRIE — contre le prix Polymarket d'avant-match (LE JUGE)", t)
    _show("PARIS SIMULÉS (marché + tous les indicateurs soloQ, prix + glissement)", betting(s, p))

    s24 = series_table(eng, tf, leagues, snapshot="p_24h")
    s24 = s24[s24["mkt_p"].notna()].reset_index(drop=True)
    if len(s24) > 2 * MIN_TRAIN_SERIES:
        t, p = compare(s24, "mkt_p", "Polymarket J-1", MIN_TRAIN_SERIES)
        _show("PAR SÉRIE — contre le prix Polymarket 24 h avant", t)
        _show("PARIS SIMULÉS au prix de la veille", betting(s24, p))
    print("\nNote : les indicateurs d'une série sont ceux d'avant la game 1 ; au prix J-1, le "
          "pari serait posé la veille alors que la soloQ du dernier jour n'est pas encore jouée "
          "(résultat à lire comme un maximum).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Test de la préparation soloQ contre le marché")
    ap.add_argument("--save", action="store_true", help="écrit aussi reports/soloq_test.txt")
    args = ap.parse_args()
    if not args.save:
        run()
        return
    buf, out = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        run()
    finally:
        sys.stdout = out
    print(buf.getvalue())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "soloq_test.txt").write_text(buf.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()

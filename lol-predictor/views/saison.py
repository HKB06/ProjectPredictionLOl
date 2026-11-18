"""Page SAISON : tous les matchs joués, leurs stats, et ce que chaque modèle avait prédit.

Une ligne par série (BO). Chaque prono est celui d'AVANT le match (aucune triche) :
l'ancien modèle (Elo d'équipe) et le nouveau (joueurs, tempo, draft, méta).
Cliquer sur une ligne affiche les stats de chaque game de la série.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.engine import load_engine, load_market, paris_series
from src.v2.market import MIN_VOLUME
from src.v2.season import MIN_GAMES, accuracy, league_table, team_table

PCT = st.column_config.NumberColumn(format="%.0f %%")


def _with_market(s: pd.DataFrame) -> pd.DataFrame:
    """Favori, proba et verdict du marché, au même format que ceux des deux modèles."""
    p = s["mkt_p"].where(s["mkt_volume"] >= MIN_VOLUME)      # prix sans volume = pas un prix
    fav = pd.Series(np.where(p >= 0.5, s["team_a"], s["team_b"]), index=s.index).where(p.notna())
    ok = (fav == s["winner"]).astype(object).where(p.notna() & s["winner"].notna(), None)
    return s.assign(fav_mkt=fav, p_mkt=np.maximum(p, 1 - p), ok_mkt=ok)


def _mark(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "—" if v is None or v != v else ("✅" if v else "❌"))


def _mmss(minutes: float) -> str:
    if minutes != minutes:
        return "—"
    return f"{int(minutes)}:{int(round((minutes % 1) * 60)) % 60:02d}"


def _filters(series: pd.DataFrame) -> pd.DataFrame:
    f = st.columns([3, 2.4, 2.2, 1.6, 1.6])
    leagues = f[0].multiselect("Ligues", sorted(series["league"].unique()),
                               placeholder="Toutes les ligues")
    teams = sorted(set(series["team_a"]) | set(series["team_b"]))
    team = f[1].selectbox("Équipe", ["Toutes"] + teams)
    day = paris_series(series["date"]).dt.date   # même jour que celui affiché dans le tableau
    dmin, dmax = day.min(), day.max()
    period = f[2].date_input("Période", (dmin, dmax), min_value=dmin, max_value=dmax,
                             format="DD/MM/YYYY")
    solid = f[3].toggle("Données solides", value=False,
                        help=f"Ne garder que les matchs où les deux équipes avaient déjà "
                             f"joué {MIN_GAMES} games (prono fiable).")
    errors = f[4].toggle("Pronos ratés", value=False,
                         help="Séries où au moins un des deux modèles s'est trompé.")

    d = series
    if leagues:
        d = d[d["league"].isin(leagues)]
    if team != "Toutes":
        d = d[(d["team_a"] == team) | (d["team_b"] == team)]
    if isinstance(period, tuple) and len(period) == 2:
        d = d[(day[d.index] >= period[0]) & (day[d.index] <= period[1])]
    if solid:
        d = d[~d["low_data"]]
    if errors:
        d = d[(d["ok_old"] == False) | (d["ok_new"] == False)]  # noqa: E712 - None exclus
    return d


def _kpis(d: pd.DataFrame, games: pd.DataFrame) -> None:
    acc_old, acc_new = accuracy(d, "old"), accuracy(d, "new")
    k = st.columns(6)
    k[0].metric("Séries", f"{len(d):,}".replace(",", " "))
    k[1].metric("Games", f"{len(games):,}".replace(",", " "))
    k[2].metric("Ancien modèle", f"{acc_old * 100:.1f} %" if acc_old == acc_old else "—",
                help="Part des séries où son favori a gagné.")
    k[3].metric("Nouveau modèle", f"{acc_new * 100:.1f} %" if acc_new == acc_new else "—",
                delta=f"{(acc_new - acc_old) * 100:+.1f} pts" if acc_new == acc_new else None,
                help="Part des séries où son favori a gagné. Écart = nouveau − ancien.")
    k[4].metric("Durée moyenne", _mmss(games["length_min"].mean()) if len(games) else "—")
    k[5].metric("Kills par game",
                f"{(games['blue_kills'] + games['red_kills']).mean():.1f}" if len(games) else "—")


def _series_detail(s: pd.Series, games: pd.DataFrame) -> None:
    g = games[games["series_id"] == s["series_id"]].sort_values("date")
    title = f"{s['team_a']} {s['score']} {s['team_b']}"
    st.subheader(f"{title} · {s['league']}{' · playoffs' if s['playoffs'] else ''}")
    c = st.columns(5)
    c[0].metric("Vainqueur", s["winner"] or "—")
    # Nom du favori dans le libellé : dans la valeur, les noms longs étaient tronqués.
    for col, tag, name in ((c[1], "old", "Ancien"), (c[2], "new", "Nouveau"),
                           (c[3], "mkt", "Marché")):
        if s[f"p_{tag}"] != s[f"p_{tag}"]:
            col.metric("Marché", "—", help="Match non coté sur Polymarket (ou trop peu échangé).")
            continue
        ok = s[f"ok_{tag}"]
        col.metric(f"{name} : {s[f'fav_{tag}']}", f"{s[f'p_{tag}'] * 100:.0f} %",
                   delta=None if ok is None or ok != ok else ("juste" if ok else "raté"),
                   delta_color="normal" if ok else "inverse", delta_arrow="off",
                   help=f"Prix Polymarket d'avant-match · {s['mkt_volume']:,.0f} $ échangés."
                   .replace(",", " ") if tag == "mkt" else None)
    c[4].metric("Patch", s["patch"] if isinstance(s["patch"], str) else "—")

    def pair(col: str) -> pd.Series:
        return (g[f"blue_{col}"].fillna(0).astype(int).astype(str) + " – "
                + g[f"red_{col}"].fillna(0).astype(int).astype(str))

    show = pd.DataFrame({
        "Game": g["game_no"],
        "Côté bleu": g["blue"], "Côté rouge": g["red"],
        "Vainqueur": g["winner"],
        "Durée": g["length_min"].map(_mmss),
        "Kills": pair("kills"),
        "Or à 15 min (bleu)": g["gd15_blue"],
        "Dragons": pair("dragons"), "Barons": pair("barons"), "Tours": pair("towers"),
        "P(bleu) ancien": g["p_old"] * 100, "P(bleu) nouveau": g["p_new"] * 100,
        "Picks bleu": g["blue_picks"], "Picks rouge": g["red_picks"],
    })
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Or à 15 min (bleu)": st.column_config.NumberColumn(format="%+d"),
        "P(bleu) ancien": PCT, "P(bleu) nouveau": PCT})
    st.caption("Stats au format bleu – rouge. Les probas par game sont celles d'avant chaque "
               "game ; le prono de la série est celui d'avant la game 1.")


def _tab_matches(d: pd.DataFrame, games: pd.DataFrame) -> None:
    note = np.select([d["rodage"], d["low_data"]], ["début de saison", "peu de données"], "")
    show = pd.DataFrame({
        "Date": paris_series(d["date"]),
        "Ligue": d["league"],
        "Match": d["team_a"] + " vs " + d["team_b"],
        "Score": d["score"],
        "Vainqueur": d["winner"].fillna("—"),
        "Prono ancien": d["fav_old"], "% ancien": d["p_old"] * 100, "Ancien": _mark(d["ok_old"]),
        "Prono nouveau": d["fav_new"], "% nouveau": d["p_new"] * 100, "Nouveau": _mark(d["ok_new"]),
        "Favori marché": d["fav_mkt"], "% marché": d["p_mkt"] * 100, "Marché": _mark(d["ok_mkt"]),
        "Note": note,
    })
    event = st.dataframe(
        show, hide_index=True, width="stretch", height=480,
        on_select="rerun", selection_mode=["single-row", "single-cell"], key="season_table",
        column_config={"Date": st.column_config.DatetimeColumn(format="DD/MM HH:mm"),
                       "% ancien": PCT, "% nouveau": PCT, "% marché": PCT})
    st.caption("« % » = probabilité que le favori gagne la série, annoncée avant le match. "
               "**Marché** = prix Polymarket juste avant le début (vide si le match n'était pas "
               "coté). « début de saison » (sans les saisons passées seulement) : le nouveau "
               "modèle n'était pas encore entraîné, il utilise l'Elo des joueurs seul.")
    # Un clic n'importe où sur la ligne suffit : case à cocher ou cellule.
    sel = event.selection if event else None
    rows = list(sel.rows) if sel and sel.rows else [c[0] for c in (sel.cells if sel else [])]
    if rows and rows[0] < len(d):
        _series_detail(d.iloc[rows[0]], games)
    else:
        st.info("Clique sur une ligne du tableau pour voir les stats de chaque game de la série.")


def _tab_teams(d: pd.DataFrame, games: pd.DataFrame, states: dict) -> None:
    if games.empty:
        st.info("Aucun match sur cette sélection.")
        return
    t = team_table(games, d)
    ctx, pl = states["ctx"], states["player"]
    show = pd.DataFrame({
        "Équipe": t["team"], "Ligue": t["ligue"],
        "Séries": t["series_won"].astype(str) + " / " + t["series"].astype(str),
        "Games": t["games"], "Victoires": t["winrate"] * 100,
        "Elo équipe": [round(ctx.elo[x]) if x in ctx.elo else None for x in t["team"]],
        "Elo joueurs": [round(pl.team_rating(pl.last_lineup.get(x, {}), pl.team_league.get(x, "")))
                        for x in t["team"]],
        "Or à 15 min": t["gd15"], "Durée": t["duree"].map(_mmss),
        "Kills": t["kills"], "Morts": t["deaths"],
        "1er sang": t["first_blood"] * 100, "1er dragon": t["first_dragon"] * 100,
        "1re tour": t["first_tower"] * 100,
        "Dragons": t["dragons"], "Barons": t["barons"],
    })
    one = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(show, hide_index=True, width="stretch", height=560, column_config={
        "Victoires": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100),
        "Or à 15 min": st.column_config.NumberColumn(format="%+.0f"),
        "Kills": one, "Morts": one, "Dragons": one, "Barons": one,
        "1er sang": PCT, "1er dragon": PCT, "1re tour": PCT})
    st.caption("Séries = gagnées / jouées. Moyennes par game sur la sélection. « Elo » = niveau "
               "actuel (saisons passées comprises) : Elo équipe pour l'ancien modèle, moyenne des 5 joueurs "
               "pour le nouveau.")


def _tab_leagues(d: pd.DataFrame, games: pd.DataFrame) -> None:
    if games.empty:
        st.info("Aucun match sur cette sélection.")
        return
    t = league_table(games, d)
    show = pd.DataFrame({
        "Ligue": t["league"], "Séries": t["series"].fillna(0).astype(int), "Games": t["games"],
        "Ancien modèle": t["ok_old"] * 100, "Nouveau modèle": t["ok_new"] * 100,
        "Écart": (t["ok_new"] - t["ok_old"]) * 100,
        "Durée": t["duree"].map(_mmss), "Kills par game": t["kills"],
        "Victoires côté bleu": t["blue_win"] * 100,
    })
    st.dataframe(show, hide_index=True, width="stretch", height=560, column_config={
        "Ancien modèle": PCT, "Nouveau modèle": PCT,
        "Écart": st.column_config.NumberColumn(format="%+.1f pts"),
        "Kills par game": st.column_config.NumberColumn(format="%.1f"),
        "Victoires côté bleu": PCT})
    st.caption("Ancien / nouveau modèle = part des séries où le favori du modèle a gagné. "
               "Sur une petite ligue (peu de séries), un écart de quelques points ne veut rien dire.")


def main() -> None:
    st.title("📊 Saison — tous les matchs")
    eng = load_engine()
    games, series = eng["games"], _with_market(load_market())
    n_series = f"{len(series):,}".replace(",", " ")
    n_games = f"{len(games):,}".replace(",", " ")
    when = paris_series(series["date"])
    st.caption(f"Toute la saison : **{n_series} séries**, {n_games} games, "
               f"{series['league'].nunique()} ligues, du {when.min():%d/%m/%Y} au "
               f"{when.max():%d/%m/%Y}. Pour chaque match : le résultat, les stats, "
               "et le prono que chaque modèle avait fait **avant** le match.")

    d = _filters(series)
    g = games[games["series_id"].isin(d["series_id"])]
    _kpis(d, g)

    t1, t2, t3 = st.tabs(["📋 Tous les matchs", "👥 Équipes", "🏆 Ligues"])
    with t1:
        _tab_matches(d, g)
    with t2:
        _tab_teams(d, g, eng["states"])
    with t3:
        _tab_leagues(d, g)


main()

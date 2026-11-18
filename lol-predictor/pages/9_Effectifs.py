"""Page Streamlit — EFFECTIFS : roster actuel de chaque équipe, mercato et changements.

Lit les fichiers de `src.ingest.rosters` (relevé Leaguepedia toutes les 6 h sur le serveur).
Aucune requête au chargement.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

st.set_page_config(page_title="LoL — Effectifs", page_icon="👥", layout="wide")

from src.ui.auth import require_password  # noqa: E402 - doit suivre set_page_config

require_password()

from src.ingest import rosters as ro  # noqa: E402
from src.update.bet_universe import universe  # noqa: E402

st.title("👥 Effectifs")
cur = ro.read("current.csv")
if cur.empty:
    st.info("Aucun relevé d'effectifs pour l'instant (`python -m src.ingest.rosters`).")
    st.stop()

leagues_bet, _ = universe()
tracked = ro.soloq_tracked(dt.date.today().year)
cur["soloq"] = cur["player"].map(tracked).fillna(0).astype(int)
stamp = dt.datetime.fromtimestamp((ro.DIR / "current.csv").stat().st_mtime)
st.caption(f"Relevé Leaguepedia du {stamp:%d/%m %H:%M} · {cur['team'].nunique()} équipes · "
           f"{len(cur)} joueurs. Une signature apparaît dès son annonce officielle.")

all_leagues = sorted(cur["league"].dropna().unique())
default = [lg for lg in all_leagues if lg in leagues_bet]
sel = st.multiselect("Ligues", all_leagues, default=default)
view = cur[cur["league"].isin(sel)] if sel else cur

tab_mov, tab_teams, tab_cov = st.tabs(["Mouvements", "Rosters", "Suivi soloQ"])

with tab_mov:
    days = st.slider("Sur les derniers jours", 1, 120, 14)
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    teams = ro.read("teams.csv")
    teams_sel = set(teams.loc[teams["league"].isin(sel), "team"]) if sel else None
    news = ro.read("news.csv")
    if len(news):
        news = news[news["date"] >= since]
        if teams_sel is not None:
            news = news[news["team"].isin(teams_sel)]
        news = news.assign(mouvement=news["direction"].map({"Join": "➕ arrivée",
                                                              "Leave": "➖ départ"}))
        st.subheader(f"Annonces officielles ({len(news)})")
        st.dataframe(news[["date", "team", "player", "role", "mouvement", "status", "source"]],
                     hide_index=True, width="stretch",
                     column_config={"source": st.column_config.LinkColumn("source")})
    log = ro.read("log.csv")
    if len(log):
        log = log[log["date"] >= since]
        if sel:
            log = log[log["league"].isin(sel)]
        st.subheader(f"Changements détectés entre deux relevés ({len(log)})")
        st.dataframe(log, hide_index=True, width="stretch")

with tab_teams:
    starters = view[view["sub"] == 0]
    count = starters.groupby("team")["role"].nunique()
    incomplete = sorted(count[count < 5].index)
    if incomplete:
        st.warning(f"{len(incomplete)} équipes sans 5 rôles titulaires (mercato en cours) : "
                   + ", ".join(incomplete))
    for lg in sel or all_leagues:
        sub = view[view["league"] == lg]
        if sub.empty:
            continue
        st.subheader(lg)
        order = {r: i for i, r in enumerate(ro.ROLES)}
        for team, g in sub.groupby("team"):
            g = g.assign(o=g["role"].map(order)).sort_values(["sub", "o"])
            line = " · ".join(f"{r.role[:3]} **{r.id}**" + (" (remp.)" if r.sub else "")
                              + ("" if r.soloq else " ⚠️")
                              for r in g.itertuples())
            st.markdown(f"**{team}** — {line}")
    st.caption("⚠️ = aucun compte soloQ suivi pour ce joueur.")

with tab_cov:
    cov = ro.coverage(dt.date.today().year, sel or None)
    st.dataframe(cov.sort_values("suivis").style.format({"suivis": "{:.0%}"}), width="stretch")
    miss = view[(view["sub"] == 0) & (view["soloq"] == 0)]
    st.caption(f"{len(miss)} titulaires sans compte suivi : leur compte n'a pas de tag sur le "
               "wiki et n'a pas encore été trouvé dans les classements ni sur TrackingThePros.")
    st.dataframe(miss[["league", "team", "id", "role"]], hide_index=True, width="stretch")

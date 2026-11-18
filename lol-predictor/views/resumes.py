"""Page RÉSUMÉS : ce qui s'est passé dans chaque match, et qui a bien ou mal joué.

Pour chaque série jouée : le résumé écrit de chaque game (vainqueur, avance à 15 min,
objectifs, meilleur joueur, joueur en difficulté, couloirs gagnés) et la note sur 10 des
dix joueurs. Second onglet : le classement des joueurs par note moyenne.

Les notes sont calculées par nous à partir des statistiques officielles de chaque game
(`src.v2.perf`) : elles ne viennent d'aucun autre site.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ingest.load_oracle import load_config, season_files
from src.ui.engine import paris_series
from src.v2 import perf

ROLE_FR = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
NOTE = st.column_config.NumberColumn(format="%.1f")


def _version() -> float:
    return max((p.stat().st_mtime for _, p in season_files(load_config(), True)), default=0.0)


@st.cache_resource(max_entries=1, show_spinner="Calcul des notes de match…")
def _load(version: float) -> dict:
    d = perf.load()
    g, t = d["games"], d["teams"]
    won = t[t["result"] == 1].set_index("gameid")["teamname"]
    g = g.assign(winner=g["gameid"].map(won), day=paris_series(g["date"]).dt.date,
                 season=g["gameid"].map(t.drop_duplicates("gameid").set_index("gameid")["season"]))
    rows = []
    for sid, s in g.groupby("series"):
        s = s.sort_values("date")
        a, b = s["blue"].iloc[0], s["red"].iloc[0]
        wa, wb = int((s["winner"] == a).sum()), int((s["winner"] == b).sum())
        rows.append({"series": sid, "date": s["date"].iloc[0], "day": s["day"].iloc[0],
                     "league": s["league"].iloc[0], "season": s["season"].iloc[0],
                     "a": a, "b": b, "wa": wa, "wb": wb, "gameids": list(s["gameid"])})
    series = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    return {**d, "series": series}


def _notes_table(p: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    rows = []
    for role, label in ROLE_FR.items():
        row = {"Poste": label}
        for team, suf in ((a, ""), (b, " ")):
            r = p[(p["teamname"] == team) & (p["position"] == role)]
            if len(r):
                r = r.iloc[0]
                row[team] = f"{r.playername} · {r.champion}"
                row["K/D/A" + suf] = f"{int(r.kills)}/{int(r.deaths)}/{int(r.assists)}"
                row["Note" + suf] = r.note
        rows.append(row)
    return pd.DataFrame(rows)


def _matches(d: dict) -> None:
    series = d["series"]
    cfg = load_config()
    bettable = set((cfg.get("betting") or {}).get("leagues") or []) | set(
        (cfg.get("betting") or {}).get("events") or [])
    f = st.columns([3, 2.4, 1.6])
    leagues = f[0].multiselect("Ligues", sorted(series["league"].unique()),
                               placeholder="Toutes les ligues")
    only = f[2].toggle("Ligues pariables", value=True, disabled=bool(leagues))
    s = series
    if leagues:
        s = s[s["league"].isin(leagues)]
    elif only and bettable:
        s = s[s["league"].isin(bettable)]
    teams = sorted(set(s["a"]) | set(s["b"]))
    team = f[1].selectbox("Équipe", ["Toutes"] + teams)
    if team != "Toutes":
        s = s[(s["a"] == team) | (s["b"] == team)]
    s = s.head(200)
    if s.empty:
        st.info("Aucun match pour ce filtre.")
        return
    labels = {r.series: f"{r.day:%d/%m/%Y} · {r.league} · {r.a} {r.wa}–{r.wb} {r.b}"
              for r in s.itertuples()}
    sid = st.selectbox(f"Match ({len(s)} plus récents)", list(labels), format_func=labels.get)
    r = s[s["series"] == sid].iloc[0]
    pl, tm = d["players"], d["teams"]

    if len(r.gameids) > 1:
        st.markdown(f"#### {r.a} {r.wa} – {r.wb} {r.b}")
        sn = perf.series_notes(pl, r.gameids)
        if not sn.empty and sn["note"].notna().any():
            best, worst = sn.sort_values("note", ascending=False).iloc[0], sn.sort_values("note").iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Meilleur joueur de la série", f"{best.playername} ({best.teamname})",
                      f"{best.note:.1f} / 10".replace(".", ","), delta_color="off")
            c2.metric("Le plus en difficulté", f"{worst.playername} ({worst.teamname})",
                      f"{worst.note:.1f} / 10".replace(".", ","), delta_color="off")
            show = sn.assign(Poste=sn["position"].map(ROLE_FR),
                             KDA=sn["kills"].astype(int).astype(str) + "/" + sn["deaths"].astype(int).astype(str)
                             + "/" + sn["assists"].astype(int).astype(str),
                             gd15=(sn["gd15"] * 1000).round())
            show = show.rename(columns={"teamname": "Équipe", "playername": "Joueur", "games": "Games",
                                        "note": "Note moyenne", "gd15": "Or à 15 min (moy.)",
                                        "champions": "Champions", "KDA": "K/D/A"})
            st.dataframe(show[["Équipe", "Poste", "Joueur", "Note moyenne", "K/D/A",
                               "Or à 15 min (moy.)", "Champions"]],
                         hide_index=True, width="stretch", column_config={"Note moyenne": NOTE})

    for i, gid in enumerate(r.gameids, 1):
        g = perf.game_summary(pl, tm, gid)
        if not g["lines"]:
            continue
        st.markdown(f"##### Game {i}")
        st.markdown("\n".join(f"- {line}" for line in g["lines"]))
        st.dataframe(_notes_table(g["notes"], r.a, r.b), hide_index=True, width="stretch",
                     column_config={"Note": NOTE, "Note ": NOTE})
    st.caption(
        "Note sur 10 : KDA, participation aux kills, part des dégâts et de l'or, or, expérience "
        "et sbires d'avance à 15 min, vision — chaque mesure comparée aux joueurs du même "
        "poste, avec un poids qui dépend du poste. 6 = partie moyenne ; un vainqueur tourne "
        "autour de 6,7, un perdant autour de 5,4. Source : statistiques officielles "
        "(Oracle's Elixir), calcul maison.")


def _ranking(d: dict) -> None:
    pl = d["players"]
    cfg = load_config()
    bettable = sorted((cfg.get("betting") or {}).get("leagues") or [])
    f = st.columns([1.2, 3, 1.6, 1.6])
    seasons = sorted(pl["season"].unique(), reverse=True)
    season = f[0].selectbox("Saison", seasons)
    all_lg = sorted(pl.loc[pl["season"] == season, "league"].unique())
    leagues = f[1].multiselect("Ligues", all_lg, default=[x for x in bettable if x in all_lg][:4])
    role = f[2].selectbox("Poste", ["Tous"] + list(ROLE_FR.values()))
    min_games = f[3].number_input("Games minimum", 5, 100, 15)
    rk = perf.ranking(pl, int(season), leagues or None, int(min_games))
    if role != "Tous":
        rk = rk[rk["position"].map(ROLE_FR) == role]
    if rk.empty:
        st.info("Aucun joueur pour ce filtre.")
        return
    show = pd.DataFrame({
        "Joueur": rk["playername"], "Équipe": rk["teamname"], "Poste": rk["position"].map(ROLE_FR),
        "Games": rk["games"].astype(int), "Note moyenne": rk["note"].round(2),
        "Au-delà du résultat": rk["plus"].round(2),
        "Victoires %": (rk["wins"] * 100).round(0),
        "Or à 15 min": (rk["gd15"] * 1000).round(),
    })
    st.dataframe(show, hide_index=True, width="stretch", height=560,
                 column_config={"Note moyenne": st.column_config.NumberColumn(format="%.2f")})
    st.caption(
        "« Au-delà du résultat » : ce que le joueur fait en plus (ou en moins) des joueurs de "
        "son poste qui ont eu le même résultat que lui. Un joueur fort dans une équipe qui perd "
        "y ressort ; un joueur porté par son équipe y descend. Ces deux mesures entrent dans le "
        "calcul du nouveau modèle depuis le 18/11/2026 (bloc « Niveau individuel »).")


def main() -> None:
    st.markdown("### 📝 Résumés des matchs")
    d = _load(_version())
    t1, t2 = st.tabs(["Matchs", "Classement des joueurs"])
    with t1:
        _matches(d)
    with t2:
        _ranking(d)


main()

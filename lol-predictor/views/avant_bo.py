"""Page AVANT LE BO : les matchs à venir et le prono des deux modèles, avant qu'ils soient joués.

Une carte par match : l'heure (Paris), les deux équipes avec la proba de gagner la
série selon le NOUVEAU modèle, l'ancien modèle en rappel, et un verdict lisible
(favori, cote minimale, accord ou désaccord entre les deux modèles).
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

from src.ui.components import NEW, breakeven, matchup_detail
from src.ui.engine import (PARIS, data_version, day_label, load_engine, refresh_data, to_paris,
                           upcoming_prices)

GAP_PTS = 10   # écart de proba (points) à partir duquel on signale un écart entre modèles
TOSS_UP = 0.55  # en dessous, aucun des deux modèles n'a de vrai favori : pas un « désaccord »


@st.cache_data(ttl=900, show_spinner="Récupération du calendrier (lolesports)…")
def load_calendar(days: int, version: float):
    from src.update.watchlist import build_rows
    return build_rows(days)


def _bestof(r: dict) -> int:
    try:
        return int(r.get("bestof") or 1)
    except (TypeError, ValueError):
        return 1


def _toss_up(x: dict) -> bool:
    return max(x["p_new"], x["pa_old"], 1 - x["pa_old"]) < TOSS_UP


def _verdict(r: dict, x: dict) -> None:
    a, b = r["team1"], r["team2"]
    if r.get("passed"):
        winner = r.get("result_winner")
        if not winner:
            st.markdown("⏳ **En cours** ou résultat pas encore disponible")
            return
        score = f" ({r['score']})" if r.get("score") else ""
        st.markdown(f"🏁 **{winner}** a gagné{score}")
        st.caption(f"Nouveau {'✅' if x['fav_new'] == winner else '❌'} · "
                   f"Ancien {'✅' if x['fav_old'] == winner else '❌'}")
        return

    st.markdown(f"Favori : **{x['fav_new']}** · cote mini **{breakeven(x['p_new']):.2f}**")
    gap = abs(x["pa_new"] - x["pa_old"]) * 100
    if _toss_up(x):
        st.markdown("⚖️ **Match serré** : aucun favori net pour les deux modèles")
    elif x["fav_new"] != x["fav_old"]:
        st.markdown(f"⚠️ **Désaccord** : l'ancien modèle voit {x['fav_old']}")
    elif gap >= GAP_PTS:
        st.markdown(f"🟡 Même favori, écart de {gap:.0f} pts entre les modèles")
    else:
        st.markdown("✅ Les deux modèles sont d'accord")
    flags = []
    if not r.get("conf"):
        flags.append("peu de données")
    if r.get("xleague"):
        flags.append("équipes de régions différentes")
    if not r.get("reliable", True):
        flags.append("ligue imprévisible")
    if flags:
        st.caption("⚠️ " + " · ".join(flags))


def _card(r: dict, x: dict, key: str) -> None:
    a, b, bo = r["team1"], r["team2"], _bestof(r)
    with st.container(border=True):
        c = st.columns([1.1, 2.2, 2.6, 2.2, 3])
        c[0].markdown(f"**{x['when']:%H:%M}**  \n{r.get('league', '?')}  \nBO{bo}")
        c[1].metric(a, f"{x['pa_new'] * 100:.0f} %")
        with c[2]:
            st.progress(float(x["pa_new"]), text="Nouveau modèle")
            st.caption(f"Ancien modèle : {x['pa_old'] * 100:.0f} % – {(1 - x['pa_old']) * 100:.0f} %")
            pm = x.get("pm")
            if pm:
                st.caption(f"💰 [Marché]({pm['url']}) : **{pm['imp1'] * 100:.0f} % – "
                           f"{pm['imp2'] * 100:.0f} %** · {pm['vol']:,.0f} $".replace(",", " "))
        c[3].metric(b, f"{(1 - x['pa_new']) * 100:.0f} %")
        with c[4]:
            _verdict(r, x)
        if st.toggle("Voir le détail", key=key):
            matchup_detail(x["m"], a, b, bo)


def main() -> None:
    st.title("📅 Avant le BO")
    st.caption("Les matchs à venir, avec la probabilité de gagner la **série** avant qu'elle "
               "soit jouée. Grand chiffre = **nouveau modèle** (joueurs, tempo, draft, méta). "
               "L'ancien modèle (Elo d'équipe) est rappelé en dessous, ainsi que le **prix du "
               "marché** (Polymarket) quand le match est coté : sur la saison, c'est lui qui "
               "se trompe le moins (page Modèles → Face au marché).")

    eng = load_engine()
    pred = eng["predictor"]
    last = eng["games"]["date"].max()

    top = st.columns([2, 1.4, 4])
    if top[0].button("🔄 Actualiser les données", type="primary", width="stretch"):
        refresh_data()
        st.rerun()
    days = top[1].selectbox("Fenêtre", [3, 5, 7, 10, 14], index=2,
                            format_func=lambda d: f"{d} jours", label_visibility="collapsed")
    age = (dt.datetime.now() - last.to_pydatetime()).days
    top[2].caption(f"Résultats connus jusqu'au **{last:%d/%m/%Y}**.")
    if age > 3:
        st.warning(f"Les données ont **{age} jours de retard** : les pronos ignorent les matchs "
                   "joués depuis. Clique sur « Actualiser les données ».")

    try:
        covered, uncovered = load_calendar(days, data_version())
        from src.update.watchlist import mark_started
        covered = mark_started(covered)        # le cache ne doit pas figer « à venir »
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible de récupérer le calendrier : {exc}")
        st.stop()

    items = []
    for r in covered:
        when = to_paris(r.get("datetime"))
        if when is None:
            continue
        m = pred.matchup(r["team1"], r["team2"], _bestof(r), league=r.get("league_code"))
        pa_new, pa_old = float(m["series"][NEW]), float(r.get("p1_elo", r["p1"]))
        items.append((r, {
            "m": m, "when": when, "pa_new": pa_new, "pa_old": pa_old,
            "fav_new": r["team1"] if pa_new >= 0.5 else r["team2"],
            "fav_old": r["team1"] if pa_old >= 0.5 else r["team2"],
            "p_new": max(pa_new, 1 - pa_new),
        }))
    items.sort(key=lambda it: it[1]["when"])

    todo = [it for it in items if not it[0].get("passed")]
    # Prix du marché pour les matchs pas encore joués : c'est la meilleure référence connue
    # (cf. page Modèles > Face au marché). Un prix déjà « live » n'aurait aucun sens ici.
    wanted = tuple((r["team1"], r["team2"], r.get("datetime", "")) for r, _ in todo)
    prices = upcoming_prices(wanted) if wanted else {}
    for r, x in todo:
        pm = prices.get((r["team1"], r["team2"], r.get("datetime", "")))
        if pm and not pm.get("closed"):
            x["pm"] = pm
    k = st.columns(4)
    k[0].metric("Matchs à venir", len(todo))
    close = [_toss_up(x) for _, x in todo]
    same = [x["fav_new"] == x["fav_old"] for _, x in todo]
    k[1].metric("Modèles d'accord", sum(s and not c for s, c in zip(same, close)))
    k[2].metric("Désaccords", sum(not s and not c for s, c in zip(same, close)),
                help=f"Favori différent, hors matchs serrés (les deux modèles sous "
                     f"{TOSS_UP * 100:.0f} %) : {sum(close)} match(s) serré(s).")
    k[3].metric("Équipes inconnues", len(uncovered),
                help="Matchs dont au moins une équipe n'est pas dans nos données.")

    if not items:
        st.info("Aucun match avec deux équipes connues sur cette fenêtre. Élargis la fenêtre "
                "ou réessaie plus tard (les affiches de playoffs ne sont pas toujours fixées).")

    by_day: dict[dt.date, list] = {}
    for it in items:
        by_day.setdefault(it[1]["when"].date(), []).append(it)
    today = dt.datetime.now(PARIS).date()
    for day, lst in by_day.items():
        st.subheader(day_label(day))
        for i, (r, x) in enumerate(lst):
            _card(r, x, key=f"det_{day}_{i}_{r['team1']}_{r['team2']}")
        if day == today and all(r.get("passed") for r, _ in lst):
            st.caption("Tous les matchs du jour ont déjà commencé.")

    if uncovered:
        with st.expander(f"Matchs sans prono ({len(uncovered)}) — équipe absente des données"):
            for u in uncovered:
                when = to_paris(u.get("datetime"))
                stamp = f"{day_label(when.date())} {when:%H:%M}" if when else "?"
                st.write(f"{stamp} · {u.get('overview', '?')} · {u['team1']} vs {u['team2']}")


main()

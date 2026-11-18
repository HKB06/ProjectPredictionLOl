"""Page MODÈLES : le nouveau modèle fait-il mieux que l'ancien ?

Les deux sont jugés sur les mêmes games, le nouveau étant réentraîné tous les
14 jours sur le passé seulement (cf. src.v2.model). On y trouve aussi une
confrontation libre entre deux équipes.
"""
from __future__ import annotations

import streamlit as st

from src.ui.components import NEW, breakeven, matchup_detail
from src.ui.engine import (data_status, data_version, load_engine, load_market,
                           load_season_board, market_version)
from src.v2 import market as mk
from src.v2.model import CONF, EVAL_MIN_GAMES, REFIT_DAYS, ablation, scoreboard, series_board


@st.cache_data(show_spinner="Comparaison des modèles…")
def _boards(version: float):
    cmp = load_engine()["cmp"]
    sb = scoreboard(cmp)
    return sb, sb.attrs["n"], sb.attrs["span"], series_board(cmp), ablation(cmp)


def _verdict(sb) -> None:
    row = sb.set_index("modèle").loc[NEW]
    lo, hi = (float(x) for x in row["IC 90 %"].strip("[]").split(";"))
    gain = row["gain vs baseline (x1000)"]
    if lo > 0:
        st.success(f"✅ **Le nouveau modèle est meilleur** : gain {gain:+.1f} "
                   f"(intervalle de confiance [{lo:+.1f} ; {hi:+.1f}]).")
    elif gain > 0:
        st.info(f"🟡 **Léger avantage au nouveau modèle, pas encore prouvé** : gain {gain:+.1f}, "
                f"mais l'intervalle de confiance touche zéro ([{lo:+.1f} ; {hi:+.1f}]).")
    else:
        st.warning(f"❌ **Le nouveau modèle ne bat pas l'ancien** (gain {gain:+.1f}).")


@st.cache_data(show_spinner="Comparaison au marché…")
def _market_boards(version: float, mversion: float):
    s = load_market()
    d = mk.sample(s)
    if len(d) < 200:
        return None
    done = s[s["winner"].notna()]
    cov = (done.assign(cote=done["mkt_p"].notna()).groupby("league")["cote"]
           .agg(cotées="sum", jouées="size").reset_index().rename(columns={"league": "Ligue"})
           .sort_values("jouées", ascending=False))
    cov["cotées"] = cov["cotées"].astype(int)
    sb = mk.scoreboard(d)
    g1 = mk.game1_board(s)
    bl = mk.blend(d)
    return {"sb": sb, "n": sb.attrs["n"], "span": sb.attrs["span"], "g1": g1,
            "n_g1": g1.attrs["n"], "blend": bl, "n_blend": bl.attrs["n_test"],
            "bets": mk.betting_board(d), "cov": cov,
            "quoted": int(done["mkt_p"].notna().sum()), "played": len(done)}


def _tab_market() -> None:
    b = _market_boards(data_version(), market_version())
    if b is None:
        st.info("Pas encore assez de matchs cotés. L'archive des prix se remplit avec "
                "`python -m src.update.refresh_all` (ou `python -m src.update.polymarket_history`).")
        return
    sb = b["sb"].set_index("source")
    models = sb.drop("Marché (Polymarket)")
    best = models["écart au marché (x1000)"].min()
    # borne basse de l'intervalle : > 0 pour les deux modèles = écart au marché certain
    certain = all(float(ic.strip("[]").split(";")[0]) > 0 for ic in models["IC 90 %"])
    st.caption(f"{b['n']} séries cotées · {b['span'][0]:%d/%m} → {b['span'][1]:%d/%m/%Y} · "
               f"prix Polymarket relevé juste avant le vrai début du match · au moins "
               f"{mk.MIN_VOLUME:,.0f} $ échangés · équipes ayant déjà joué 15 games."
               .replace(",", " "))
    rates = (f"Il trouve le vainqueur de la série "
             f"{sb.loc['Marché (Polymarket)', 'bon vainqueur %']:.1f} % du temps, contre "
             f"{sb.loc['Ancien modèle', 'bon vainqueur %']:.1f} % (ancien) et "
             f"{sb.loc['Nouveau modèle', 'bon vainqueur %']:.1f} % (nouveau).")
    if certain:
        st.error(f"❌ **Le marché prédit mieux que nos deux modèles, sans doute possible.** "
                 f"{rates} Parier parce que notre modèle n'est pas d'accord avec le prix n'est "
                 "**pas** rentable aujourd'hui.")
    elif best > 0:
        st.warning(f"🟡 **Le marché fait mieux que nos modèles, mais l'écart n'est pas certain.** "
                   f"{rates}")
    else:
        st.success(f"✅ **Au moins un modèle fait aussi bien que le marché sur cette période.** "
                   f"{rates}")
    st.subheader("Qui prédit le mieux le vainqueur de la série ?")
    st.dataframe(b["sb"], hide_index=True, width="stretch")
    st.caption("**Écart au marché** : hausse de log-loss ×1000 par rapport au prix Polymarket, "
               "sur exactement les mêmes séries. Positif = moins bon que le marché ; un intervalle "
               "entièrement positif = l'écart est certain.")

    c = st.columns(2)
    with c[0]:
        st.subheader("Sur la game 1 seule")
        st.dataframe(b["g1"], hide_index=True, width="stretch")
        st.caption(f"{b['n_g1']} games avec un marché « Game 1 Winner ». Nos probas sont faites "
                   "par game : ici aucune conversion game → série ne peut être en cause.")
    with c[1]:
        st.subheader("Nos modèles savent-ils quelque chose que le marché ignore ?")
        st.dataframe(b["blend"], hide_index=True, width="stretch")
        st.caption(f"Mélange marché + modèle appris sur la première moitié de la saison, jugé "
                   f"sur la seconde ({b['n_blend']} séries). Un poids du modèle proche de 0 = "
                   "une fois le prix connu, le modèle n'ajoute rien.")

    st.subheader("Et si on avait parié ?")
    st.dataframe(b["bets"], hide_index=True, width="stretch")
    st.caption(f"1 unité sur chaque série où le modèle voit au moins « écart mini » points de "
               f"plus que le prix, achetée au prix + {mk.SLIPPAGE * 100:.0f} cent. Rendement "
               "négatif = perte par unité misée.")
    with st.expander(f"Couverture : {b['quoted']} séries cotées sur {b['played']} jouées"):
        st.dataframe(b["cov"], hide_index=True, width="stretch", height=360)


def _tab_data() -> None:
    st.caption("Collecte automatique (`python -m src.update.refresh_all`, à planifier sur le "
               "serveur toutes les 30 minutes). Heures de Paris.")
    st.dataframe(data_status(), hide_index=True, width="stretch")
    st.caption("**oracle** : toutes les games (Oracle's Elixir) · **history** : saisons passées "
               "(téléchargées une fois) · **leaguepedia** : tournois, "
               "coachs, LAN/online, comptes soloQ · **polymarket** : prix d'avant-match · "
               "**odds** : cotes des bookmakers (clé odds-api.io) · **soloq** : rang soloQ "
               "des pros (clé Riot) · **model** : précalcul des deux modèles pour le site, "
               "refait dès que les données changent. « ignorée » = clé absente.")


def main() -> None:
    st.title("🧬 Nouveau modèle contre ancien")
    eng = load_engine()
    pred = eng["predictor"]
    sb, n, span, sboard, abl = _boards(data_version())

    tab_cmp, tab_mkt, tab_match, tab_data = st.tabs(
        ["📊 Qui est le meilleur ?", "💰 Face au marché", "⚔️ Confrontation de 2 équipes",
         "🗄️ Données"])
    with tab_mkt:
        _tab_market()
    with tab_data:
        _tab_data()

    with tab_cmp:
        past = sorted(int(y) for y in eng["df"]["season"].unique())[:-1]
        history = (f" · saisons {past[0]}-{past[-1]} rejouées avant celle-ci (notes des joueurs "
                   "et des équipes, entraînement)" if past else "")
        st.caption(f"{n} games de test · {span[0]:%d/%m} → {span[1]:%d/%m/%Y} · équipes ayant "
                   f"déjà joué {EVAL_MIN_GAMES} games · nouveau modèle réentraîné tous les "
                   f"{REFIT_DAYS} jours sur le passé seulement{history}.")
        _verdict(sb)
        seasons = load_season_board()
        if seasons is not None:
            st.subheader("Saison par saison")
            st.dataframe(seasons, hide_index=True, width="stretch", column_config={
                "gain (x1000)": st.column_config.NumberColumn(format="%+.2f")})
            st.caption("Chaque saison est un test à part : le modèle n'y connaît que le passé. "
                       "**Gain** : baisse de log-loss ×1000 du nouveau modèle face à l'ancien "
                       "recalibré, qui profite lui aussi des saisons passées. La première saison "
                       "sert surtout de rodage : sans passé derrière lui, le nouveau modèle ne "
                       "fait pas mieux que l'ancien.")
        st.subheader("Par game")
        st.dataframe(sb, hide_index=True, width="stretch")
        st.caption(
            "**Log-loss / Brier** : plus bas = meilleures probabilités. **Gain vs baseline** : "
            "baisse de log-loss ×1000 face à « Ancien · recalibré walk-forward », qui reçoit "
            "les mêmes traitements que le nouveau. « Ancien · calibré par ligue » est celui de "
            "l'ancien site : sa calibration a été apprise sur toutes les games, celles du test "
            "comprises, ce qui l'avantage."
        )
        st.subheader("Par série (BO)")
        st.dataframe(sboard, hide_index=True, width="stretch")
        st.caption(f"Proba de la game 1 convertie en proba de série. « ≥{CONF * 100:.0f} % » = "
                   "séries où le modèle donnait au moins 70 % à son favori.")
        st.subheader("Ce que chaque bloc apporte au nouveau modèle")
        st.dataframe(abl, hide_index=True, width="stretch", column_config={
            "apport (x1000)": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Perte de log-loss (×1000) quand on retire le bloc. Positif = le bloc aide ; "
                   "négatif = il ajoute surtout du bruit pour l'instant. Les signaux de draft "
                   "« confort » et « bans ciblés » sont mesurés par rapport à la moyenne de la "
                   "ligue : en valeur brute ils reflétaient surtout le nombre de games connues.")
        st.info("**Pas encore dans le modèle** : soloQ des joueurs (la "
                "collecte démarre dès qu'une clé Riot est fournie : il faut plusieurs semaines "
                "de relevés avant de pouvoir mesurer un signal), enjeu du match (équipe déjà "
                "qualifiée ou éliminée). Le lieu est inconnu pour LCK, LCS, LCP et PRM.")

    with tab_match:
        teams = pred.teams
        c = st.columns([3, 3, 1.6])
        a = c[0].selectbox("Équipe A", teams, index=0)
        b = c[1].selectbox("Équipe B", teams, index=1 if len(teams) > 1 else 0)
        bo = c[2].radio("Format", [1, 3, 5], index=1, horizontal=True,
                        format_func=lambda x: f"BO{x}")
        if a == b:
            st.warning("Choisis deux équipes différentes.")
            return
        m = pred.matchup(a, b, bo)
        pa = m["series"][NEW]
        po = m["series"]["Ancien · calibré par ligue"]
        k = st.columns(4)
        k[0].metric(a, f"{pa * 100:.0f} %", help="Nouveau modèle, série complète.")
        k[1].metric(b, f"{(1 - pa) * 100:.0f} %")
        k[2].metric("Cote mini du favori", f"{breakeven(max(pa, 1 - pa)):.2f}")
        k[3].metric(f"Ancien modèle ({a})", f"{po * 100:.0f} %")
        matchup_detail(m, a, b, bo)


main()

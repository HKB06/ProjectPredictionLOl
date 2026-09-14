"""Page Streamlit — RENTABILITÉ des picks ciblés : mise fixe de 10 €, ROI réel.

Répond à UNE question : les picks 🎯 qu'on affiche dans « Matchs du jour / semaine »
rapportent-ils de l'argent, ou juste des bonnes prédictions ?

Deux blocs :
1. **Simulation historique** (réponse immédiate) : réussite réelle des picks 🎯 du
   passé en walk-forward + la **cote moyenne minimale** de rentabilité.
2. **Journal réel** : on fige date / prévision / cote du book, on règle
   automatiquement depuis la data Oracle, et on applique 10 € à plat sur chaque pick.

Lancer : depuis l'app principale (menu de gauche).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models.picks_roi import (COLS, RESULTS, STAKE, add_candidates, candidates,
                                  load_ledger, normalize, roi_at_odds, save_ledger,
                                  settle, summary, value_flag)

st.set_page_config(page_title="LoL — Rentabilité des picks", page_icon="💵", layout="wide")

ODDS_GRID = (1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.75, 2.00)
VERDICT = {"won": "✅ gagné", "lost": "❌ perdu", "void": "➖ annulé", "open": "⏳ en attente"}


@st.cache_data(ttl=1800, show_spinner="Rejeu walk-forward des picks passés…")
def load_sim(days: int, conf: float) -> dict:
    from src.models.picks_roi import simulate
    return simulate(days=days, conf=conf)


@st.cache_data(ttl=1800, show_spinner="Reconstruction des picks passés…")
def load_past(days: int, conf: float) -> pd.DataFrame:
    from src.models.picks_roi import past_picks
    return past_picks(days=days, conf=conf)


def _ledger_columns() -> dict:
    """Config des colonnes, partagée par la table du passé et le journal réel."""
    return {
        "match_date": st.column_config.TextColumn("Date", width="small"),
        "when": st.column_config.TextColumn("Quand", width="small"),
        "league": st.column_config.TextColumn("Ligue", width="small"),
        "team1": st.column_config.TextColumn("Équipe 1"),
        "team2": st.column_config.TextColumn("Équipe 2"),
        "bestof": st.column_config.NumberColumn("BO", width="small"),
        "tier": st.column_config.TextColumn(
            "Tier", width="small",
            help="🎯 = favori ≥70 %/game (le filtre validé à 79,8 %). "
                 "⭐ = proba de série ≥62 %, plus fragile."),
        "pick": st.column_config.TextColumn("Notre prévision"),
        "our_proba": st.column_config.NumberColumn("Notre p", format="%.2f"),
        "breakeven": st.column_config.NumberColumn(
            "Cote mini", format="%.2f",
            help="1/proba : il faut une cote STRICTEMENT au-dessus."),
        "odds": st.column_config.NumberColumn("Cote prise", format="%.2f"),
        "odds_source": st.column_config.TextColumn("Source cote", width="small"),
        "stake": st.column_config.NumberColumn("Mise (€)", format="%.0f"),
        "winner": st.column_config.TextColumn("Vainqueur réel"),
        "score": st.column_config.TextColumn("Score", width="small"),
        "profit": st.column_config.NumberColumn("P/L (€)", format="%.2f"),
        "captured_at": st.column_config.TextColumn("Capturé le", width="small"),
    }


def _simulation_block() -> None:
    st.subheader("1️⃣ Réponse immédiate — les picks 🎯 du passé étaient-ils rentables ?")
    st.caption(
        "Rejeu **walk-forward** (le modèle ne voit jamais le futur) des picks 🎯 : "
        "ligue fiable + favori ≥ seuil + data ≥15 games. On en tire le **taux de réussite "
        "réel**, puis la **cote minimale** qui rend la mise fixe rentable. "
        "⚠️ Granularité = la *game* (la data Oracle est par game), pas la série."
    )

    c = st.columns([1.2, 1.2, 1.4])
    days = c[0].slider("Fenêtre (jours)", 15, 180, 60, 15)
    conf = c[1].slider("Seuil de confiance (%/game)", 60, 90, 70, 5) / 100
    stake = c[2].number_input("Mise fixe par match (€)", 1.0, 100.0, STAKE, 1.0)

    sim = load_sim(days, conf)
    if not sim.get("n"):
        st.info("Aucun pick 🎯 sur cette fenêtre. Élargis la fenêtre ou baisse le seuil.")
        return

    hit, needed = sim["hit_rate"], sim["odds_needed"]
    k = st.columns(4)
    k[0].metric("Picks 🎯", sim["n"])
    k[1].metric("Proba annoncée", f"{sim['avg_proba']*100:.1f}%")
    k[2].metric("Réussite réelle", f"{hit*100:.1f}%",
                delta=f"{(hit - sim['avg_proba'])*100:+.1f} pts vs annoncé",
                help="Proche de 0 = modèle bien calibré : il dit la vérité sur ses probas.")
    k[3].metric("🎯 Cote mini pour gagner", f"{needed:.2f}",
                help="= 1 / taux de réussite. En dessous de cette cote, on perd sur la durée, "
                     "même en trouvant le bon vainqueur.")

    st.markdown(
        f"**À retenir :** sur les {sim['days']} derniers jours, les picks gagnent "
        f"**{hit*100:.1f}%** du temps. Il faut donc une cote **supérieure à "
        f"{needed:.2f}** en moyenne. En dessous, la stratégie est perdante — "
        "c'est exactement le piège des gros favoris à cote courte."
    )

    rows = []
    for o in ODDS_GRID:
        r = roi_at_odds(hit, o, sim["n"], stake)
        rows.append({"Cote moyenne": f"{o:.2f}", "Mise totale (€)": round(r["staked"], 0),
                     "P/L (€)": round(r["profit"], 0), "ROI": r["roi"] * 100,
                     "Verdict": "✅ rentable" if r["roi"] > 0 else "❌ perdant"})
    grid = pd.DataFrame(rows)
    st.dataframe(
        grid, width="stretch", hide_index=True,
        column_config={"ROI": st.column_config.NumberColumn("ROI", format="%.1f%%")},
    )
    st.caption(f"ROI simulé si **tous** les {sim['n']} picks avaient été joués à "
               f"{stake:.0f} € et à la cote de la ligne.")

    st.markdown("**Par ligue** — où la stratégie tient vraiment")
    by_lg = sim["by_league"].copy()
    by_lg = by_lg[by_lg["n"] >= 10]
    by_lg["Réussite"] = by_lg["hit"] * 100          # NumberColumn ne convertit pas les
    by_lg["Proba annoncée"] = by_lg["proba"] * 100   # fractions : on passe en points de %
    show = by_lg[["league", "n", "Proba annoncée", "Réussite", "cote_mini"]].rename(
        columns={"league": "Ligue", "n": "Picks", "cote_mini": "Cote mini"})
    st.dataframe(
        show, width="stretch", hide_index=True,
        column_config={
            "Proba annoncée": st.column_config.NumberColumn(format="%.1f%%"),
            "Réussite": st.column_config.NumberColumn(format="%.1f%%"),
            "Cote mini": st.column_config.NumberColumn(
                format="%.2f", help="Cote à dépasser pour être rentable sur cette ligue."),
        },
    )
    st.caption("Ligues avec ≥10 picks. Une **cote mini basse** = ligue où le modèle est "
               "très fiable ; mais le book y cote souvent court, donc vérifie la cote réelle.")

    _past_table(days, conf, stake, needed)


def _past_table(days: int, conf: float, stake: float, needed: float) -> None:
    """Le détail pick par pick du passé, au même format que le journal réel."""
    st.markdown("### 📜 Détail des picks passés — même table que le journal, mais en arrière")
    past = load_past(days, conf)
    if past.empty:
        return

    hyp = st.slider("Cote hypothétique appliquée à tous les picks", 1.00, 2.50,
                    float(round(max(needed + 0.05, 1.05), 2)), 0.05,
                    help="Les cotes historiques du book ne sont pas archivées : on simule "
                         "un P/L en supposant la même cote partout.")
    from src.models.picks_roi import apply_odds
    past = apply_odds(past, hyp, stake)

    won = int((past["result"] == "won").sum())
    pl = float(pd.to_numeric(past["profit"], errors="coerce").sum())
    staked = len(past) * stake
    k = st.columns(4)
    k[0].metric("Picks joués", len(past))
    k[1].metric("Gagnés / perdus", f"{won} / {len(past) - won}")
    k[2].metric(f"P/L à cote {hyp:.2f}", f"{pl:+.0f} €")
    k[3].metric("ROI", f"{pl / staked * 100:+.1f}%" if staked else "—",
                delta=f"cote mini requise {needed:.2f}")

    only_err = st.toggle("Erreurs seulement", value=False,
                         help="Utile pour voir sur quelles ligues/équipes on se trompe.")
    show = past[past["result"] == "lost"] if only_err else past
    show = show.assign(result=show["result"].map(VERDICT))
    st.dataframe(
        show, width="stretch", hide_index=True, height=420,
        column_config={**_ledger_columns(),
                       "result": st.column_config.TextColumn("Résultat", width="small")},
    )
    st.caption(
        f"Les {len(past)} picks 🎯 des {days} derniers jours, du plus récent au plus ancien. "
        "**Cote et P/L sont hypothétiques** (une seule cote pour tous) — c'est une "
        "projection, pas un résultat constaté. Le bloc 2 ci-dessous, lui, fige la vraie "
        "cote du book pick par pick : c'est la seule preuve de rentabilité réelle."
    )


def _capture_report(cands: list[dict]) -> None:
    """Tri immédiat des picks capturés : jouables / sous la cote mini / sans cote."""
    df = pd.DataFrame(cands)
    o = pd.to_numeric(df["odds"], errors="coerce")
    b = pd.to_numeric(df["breakeven"], errors="coerce")

    def _lines(sub, fmt):
        return "\n".join(fmt(r) for r in sub.itertuples())

    good = df[o.notna() & (o > b)]
    under = df[o.notna() & (o <= b)]
    none_ = df[o.isna()]

    if len(good):
        st.markdown("**✅ Jouables — la cote offerte dépasse la cote mini :**\n\n" + _lines(
            good, lambda r: f"- {r.tier} **{r.pick}** ({r.league}) · "
                            f"cote {r.odds:.2f} > mini {r.breakeven:.2f}"))
    if len(under):
        st.warning("**❌ Sous la cote mini — à ne PAS jouer.** Le book cote trop court : "
                   "même en gagnant souvent, ces paris perdent sur la durée.\n\n" + _lines(
                       under, lambda r: f"- {r.tier} {r.pick} ({r.league}) · "
                                        f"cote {r.odds:.2f} ≤ mini {r.breakeven:.2f}"))
    if len(none_):
        st.info(f"**{len(none_)} pick(s) sans cote** — ligue non couverte par les 2 books "
                "du plan gratuit. Saisis la cote à la main, sinon le ROI de la ligne reste "
                "incalculable.\n\n" + _lines(
                    none_, lambda r: f"- {r.tier} {r.pick} ({r.league}) · "
                                     f"mini {r.breakeven:.2f}"))


def _ledger_block() -> None:
    st.subheader("2️⃣ Journal réel — mise fixe de 10 € sur chaque pick")
    st.caption(
        "On **fige** chaque pick : date du match, notre prévision, notre proba, et la "
        "**cote du book**. Le résultat est réglé automatiquement depuis la data Oracle. "
        "C'est la seule preuve qui compte : le ROI réel, pas l'accuracy."
    )

    if "led" not in st.session_state:
        st.session_state.led = settle(load_ledger())
    led = st.session_state.led

    c = st.columns([1.2, 1.3, 1.5, 1.4], vertical_alignment="bottom")
    days = c[0].selectbox("Fenêtre de capture", [1, 2, 3, 7, 14], index=3,
                          format_func=lambda d: f"{d} jour(s)")
    with_stars = c[1].checkbox(
        "Inclure les ⭐", value=False,
        help="Les ⭐ (proba de série ≥62 %) ont une cote mini élevée (1.3-1.6) que le "
             "book dépasse rarement : attends-toi à beaucoup de ❌ sous la cote mini.")
    # Libellé qui reflète la case : sinon on croit que cocher suffit à capturer.
    if c[2].button(f"📥 Capturer {'🎯 + ⭐' if with_stars else 'les 🎯'}", type="primary",
                   help="Cocher la case ne suffit pas : c'est ce bouton qui va chercher "
                        "les matchs et les ajoute au journal."):
        with st.spinner("Calcul Elo + récupération des matchs et des cotes…"):
            try:
                cands = candidates(days=days, include_strong=with_stars)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Impossible de récupérer les picks : {exc}")
                cands = []
        if cands:
            led, added = add_candidates(led, cands)
            led = settle(led)
            save_ledger(led)
            st.session_state.led = led
            n_hc = sum(1 for c_ in cands if c_.get("tier") == "🎯")
            st.success(f"{added} pick(s) ajouté(s) sur {len(cands)} candidat(s) "
                       f"({n_hc} 🎯, {len(cands) - n_hc} ⭐).")
            _capture_report(cands)
        elif cands is not None:
            st.info("Aucun pick dans cette fenêtre — c'est normal, mieux vaut 0 pick "
                    "qu'un faux favori.")

    if c[3].button("🔄 Régler les résultats (data Oracle)"):
        led = settle(led)
        save_ledger(led)
        st.session_state.led = led
        st.success("Résultats mis à jour depuis la data Oracle.")

    if led.empty:
        st.info("Journal vide. Clique sur **Capturer** pour commencer le suivi. "
                "Pense à mettre la data à jour avant, sinon les résultats ne se règlent pas.")
        return

    st.markdown("**Saisis / corrige les cotes**, puis sauvegarde. La colonne **Value ?** "
                "se recalcule à la sauvegarde et te dit si la cote couvre le seuil.")
    disp = normalize(led)
    disp.insert(disp.columns.get_loc("odds_source"), "value", value_flag(disp))
    edited = st.data_editor(
        disp, width="stretch", hide_index=True, num_rows="dynamic",
        column_config={**_ledger_columns(),
                       "result": st.column_config.SelectboxColumn(
                           "Résultat", options=RESULTS, width="small"),
                       "value": st.column_config.TextColumn(
                           "Value ?", width="small", disabled=True,
                           help="✅ = cote > cote mini (jouable). ❌ = cote trop courte : "
                                "perdant sur la durée même en gagnant souvent.")},
    )
    if st.button("💾 Sauvegarder le journal", type="primary"):
        saved = settle(edited[COLS])
        save_ledger(saved)
        st.session_state.led = saved
        st.success("Journal sauvegardé et recalculé.")
        led = saved

    s = summary(led)
    st.markdown("### Bilan")
    if not s.get("n_settled"):
        st.info(f"{s['n']} pick(s) enregistré(s), aucun réglé pour l'instant. "
                "Les KPIs apparaîtront dès que les matchs auront été joués "
                "et la data Oracle mise à jour.")
        return

    k = st.columns(5)
    k[0].metric("Picks réglés", s["n_settled"], help=f"{s.get('n_open', 0)} encore en attente")
    k[1].metric("Réussite", f"{s['hit_rate']*100:.1f}%",
                delta=f"{(s['hit_rate'] - s['avg_proba'])*100:+.1f} pts vs annoncé")
    k[2].metric("Cote mini requise", f"{s['odds_needed']:.2f}")
    if s.get("n_priced"):
        k[3].metric("Cote moyenne prise", f"{s['avg_odds']:.2f}",
                    delta=f"{s['avg_odds'] - s['odds_needed']:+.2f} vs requise",
                    help="Positif = tu prends des cotes assez hautes pour être rentable.")
        k[4].metric("ROI réel", f"{s['roi']*100:+.1f}%",
                    delta=f"{s['profit']:+.2f} € sur {s['staked']:.0f} € misés")
    else:
        k[3].metric("Cote moyenne prise", "—")
        k[4].metric("ROI réel", "—", help="Saisis les cotes pour calculer le ROI.")

    if led["tier"].nunique() > 1:
        rows = []
        for tier, sub in led.groupby("tier"):
            t = summary(sub)
            if not t.get("n_settled"):
                continue
            rows.append({"Tier": tier, "Réglés": t["n_settled"],
                         "Réussite": t["hit_rate"] * 100,
                         "Cote mini requise": t["odds_needed"],
                         "Cote moyenne prise": t.get("avg_odds"),
                         "ROI": (t["roi"] * 100) if t.get("n_priced") else None})
        if rows:
            st.markdown("**Par tier** — les ⭐ tiennent-ils face aux 🎯 ?")
            st.dataframe(
                pd.DataFrame(rows), width="stretch", hide_index=True,
                column_config={
                    "Réussite": st.column_config.NumberColumn(format="%.1f%%"),
                    "Cote mini requise": st.column_config.NumberColumn(format="%.2f"),
                    "Cote moyenne prise": st.column_config.NumberColumn(format="%.2f"),
                    "ROI": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            st.caption("Si le ROI des ⭐ reste négatif quand celui des 🎯 est positif, "
                       "la réponse est claire : arrête de capturer les ⭐.")

    if s.get("n_priced"):
        d = led[led["result"].isin(["won", "lost"]) & led["odds"].notna()].copy()
        d = d.sort_values("match_date")
        d["Bankroll (€)"] = pd.to_numeric(d["profit"], errors="coerce").cumsum()
        st.line_chart(d.set_index("match_date")["Bankroll (€)"], height=260)
        st.caption(f"Cumul du P/L à {STAKE:.0f} € par pick. Si la courbe monte, "
                   "la sélection **et** les cotes prises sont bonnes.")

        verdict = ("✅ Rentable sur cet échantillon." if s["roi"] > 0
                   else "❌ Perdant : la réussite est bonne mais les cotes prises sont trop courtes.")
        st.markdown(f"**Verdict : {verdict}**")
        if s["n_settled"] < 25:
            st.warning(f"⚠️ Seulement {s['n_settled']} paris réglés : **trop peu pour conclure**. "
                       "Il faut ~25-30 picks minimum avant de tirer une conclusion.")


def main() -> None:
    st.title("💵 Les picks ciblés sont-ils rentables ?")
    st.caption(
        f"Mise **fixe de {STAKE:.0f} € par match**, uniquement sur les picks 🎯 "
        "(marché vainqueur). Principe : **trouver le bon vainqueur ne suffit pas** — "
        "il faut que la cote paie plus que le risque."
    )
    _simulation_block()
    st.divider()
    _ledger_block()


if __name__ == "__main__":
    main()

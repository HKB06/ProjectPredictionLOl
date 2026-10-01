"""Page Streamlit — RENTABILITÉ des picks ciblés : mise fixe de 10 €, ROI réel.

Répond à UNE question : les picks 🎯 qu'on affiche dans « Matchs du jour / semaine »
rapportent-ils de l'argent, ou juste des bonnes prédictions ?

Deux blocs :
1. **Simulation historique** (réponse immédiate) : réussite réelle des picks 🎯 du
   passé en walk-forward + la **cote moyenne minimale** de rentabilité.
2. **Simulation bankroll** : on rejoue chronologiquement les picks à mise fixe
   depuis une bankroll de départ — courbe, drawdown, séries de gains/pertes.

Lancer : depuis l'app principale (menu de gauche).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.models.picks_roi import STAKE, roi_at_odds

st.set_page_config(page_title="LoL — Rentabilité des picks", page_icon="💵", layout="wide")

from src.ui.auth import require_password  # noqa: E402 - doit suivre set_page_config

require_password()

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


@st.cache_data(ttl=1800, show_spinner="Reconstruction des séries passées…")
def load_past_series(days: int, conf: float) -> pd.DataFrame:
    """Picks passés au niveau série + vraie cote de clôture quand elle est archivée."""
    from src.models.picks_roi import past_series
    return past_series(days=days, conf=conf)


@st.cache_data(ttl=600, show_spinner=False)
def load_odds_cache() -> pd.DataFrame:
    """Cache des cotes de clôture (rempli par src.update.odds_history en local)."""
    from src.update.odds_history import load_cache
    return load_cache()


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
    # 300 j ≈ décembre 2025 : les cotes de clôture odds-api.io ne remontent pas plus loin.
    days = c[0].slider("Fenêtre (jours)", 15, 300, 60, 15)
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


def _league_label(slug: str) -> str:
    """`league-of-legends-lec-summer` -> `Lec Summer` (lisible dans un tableau)."""
    return str(slug).replace("league-of-legends-", "").replace("-", " ").title()


def _odds_progress() -> None:
    """Avancement du remplissage des cotes réelles : barre + détail par ligue.

    Le cache est rempli en local (plan gratuit odds-api.io : 100 req/h, 500/jour)
    puis poussé sur git — l'app en ligne voit donc l'état du dernier push.
    """
    cache = load_odds_cache()
    if cache.empty:
        return
    n = len(cache)
    ok = int((cache["status"] == "ok").sum())
    none = int((cache["status"] == "none").sum())
    pend = n - ok - none
    st.progress(
        (ok + none) / n,
        text=f"📡 Cotes réelles : **{ok} récupérées** · {pend} en attente · "
             f"{none} non cotées par les books — {n} matchs suivis",
    )
    with st.expander("Avancement détaillé par ligue"):
        det = (cache.assign(Ligue=cache["league_slug"].map(_league_label),
                            ok=cache["status"] == "ok",
                            pend=cache["status"] == "pending",
                            none=cache["status"] == "none")
               .groupby("Ligue", as_index=False)
               .agg(Total=("status", "size"), **{"Avec cote": ("ok", "sum"),
                    "En attente": ("pend", "sum"), "Sans cote": ("none", "sum")})
               .sort_values(["En attente", "Total"], ascending=False))
        det["Traité"] = (det["Total"] - det["En attente"]) / det["Total"] * 100
        st.dataframe(
            det, width="stretch", hide_index=True,
            column_config={"Traité": st.column_config.NumberColumn(format="%.0f%%")},
        )
        last = cache["fetched_at"].dropna().max()
        st.caption(
            "« Sans cote » = 1xbet/GG.bet ne proposaient rien sur ce match (petites "
            "ligues surtout). Le remplissage tourne en boucle en local et le cache est "
            f"poussé au fil de l'eau — dernière cote relevée le {last}."
        )


def _past_table(days: int, conf: float, stake: float, needed: float) -> None:
    """Le détail du passé, au niveau SÉRIE, avec la vraie cote de clôture du book."""
    st.markdown("### 📜 Détail des picks passés — même table que le journal, mais en arrière")

    past = load_past_series(days, conf)
    if past.empty:
        st.info("Aucune série exploitable sur cette fenêtre.")
        return

    real = pd.to_numeric(past["odds"], errors="coerce")
    n_real = int(real.notna().sum())
    st.caption(
        "Une ligne = **une série** (un pari réellement plaçable), et non une game : "
        "les bookmakers cotent le vainqueur de série. La colonne *Cote prise* affiche la "
        "**vraie cote de clôture** relevée par odds-api.io juste avant le coup d'envoi, "
        f"quand elle est archivée — actuellement **{n_real} séries sur {len(past)}**."
    )

    _odds_progress()

    if n_real:
        _real_odds_summary(past[real.notna()], stake)
    else:
        st.warning(
            "Aucune cote réelle en cache pour l'instant. Lance le remplissage en local :\n\n"
            "`python -m src.update.odds_history --days 60`\n\n"
            "Le plan gratuit est limité à 100 requêtes/heure : le script est **reprenable**, "
            "relance-le d'heure en heure jusqu'à ce qu'il n'y ait plus d'attente."
        )

    mode = st.radio(
        "P/L calculé sur", ("Cotes réelles archivées", "Cote hypothétique (toutes séries)"),
        horizontal=True, index=0 if n_real else 1,
        help="Les cotes réelles ne couvrent que les séries déjà récupérées. "
             "L'hypothèse s'applique à toutes, mais reste une projection.",
    )

    from src.models.picks_roi import apply_odds

    if mode.startswith("Cotes réelles"):
        view = past[real.notna()].copy()
        label = "cotes réelles"
    else:
        hyp = st.slider("Cote hypothétique appliquée à toutes les séries", 1.00, 2.50,
                        float(round(max(needed + 0.05, 1.05), 2)), 0.05)
        view = apply_odds(past, hyp, stake)
        label = f"cote {hyp:.2f}"

    if view.empty:
        return

    won = int((view["result"] == "won").sum())
    pl = float(pd.to_numeric(view["profit"], errors="coerce").sum())
    staked = len(view) * stake
    hit = won / len(view)
    k = st.columns(4)
    k[0].metric("Séries jouées", len(view))
    k[1].metric("Gagnées / perdues", f"{won} / {len(view) - won}")
    k[2].metric(f"P/L ({label})", f"{pl:+.0f} €")
    k[3].metric("ROI", f"{pl / staked * 100:+.1f}%" if staked else "—",
                delta=f"cote mini requise {1 / hit:.2f}" if hit else None)

    only_err = st.toggle("Erreurs seulement", value=False,
                         help="Utile pour voir sur quelles ligues/équipes on se trompe.")
    show = view[view["result"] == "lost"] if only_err else view
    show = show.assign(result=show["result"].map(VERDICT))
    st.dataframe(
        show, width="stretch", hide_index=True, height=420,
        column_config={**_ledger_columns(),
                       "result": st.column_config.TextColumn("Résultat", width="small")},
    )


def _real_odds_summary(real: pd.DataFrame, stake: float) -> None:
    """Le verdict qui compte : sur les séries dont on connaît la vraie cote, ROI réel."""
    pl = float(pd.to_numeric(real["profit"], errors="coerce").sum())
    staked = len(real) * stake
    roi = pl / staked if staked else 0.0
    o = pd.to_numeric(real["odds"], errors="coerce")
    b = pd.to_numeric(real["breakeven"], errors="coerce")
    value = int((o > b).sum())

    if roi > 0:
        st.success(
            f"**Sur les {len(real)} séries à cote réelle, le ROI est de {roi * 100:+.1f} %** "
            f"({pl:+.0f} € pour {staked:.0f} € misés). La cote moyenne du book était "
            f"{o.mean():.2f}, et {value} séries offraient une cote au-dessus de notre seuil "
            "de rentabilité."
        )
    else:
        st.error(
            f"**Sur les {len(real)} séries à cote réelle, le ROI est de {roi * 100:+.1f} %** "
            f"({pl:+.0f} € pour {staked:.0f} € misés). La cote moyenne du book était "
            f"{o.mean():.2f} alors qu'il nous fallait {b.mean():.2f} en moyenne : le book "
            f"cote trop court sur nos favoris. Seules {value} séries passaient le seuil."
        )


def _bankroll_block() -> None:
    """Rejeu chronologique à mise fixe : la courbe de caisse et ses trous."""
    st.subheader("2️⃣ Simulation bankroll — encaisse-t-on les séries de pertes ?")
    st.caption(
        "On rejoue les picks 🎯 passés **dans l'ordre chronologique**, mise fixe, en "
        "partant d'une bankroll de départ. La courbe montre le chemin réellement "
        "parcouru : les séries de victoires, les trous (**drawdown**) et le pire creux. "
        "C'est le test psychologique : sait-on traverser une mauvaise passe sans paniquer ?"
    )

    c = st.columns([1.3, 1.1, 1.2, 1.8], vertical_alignment="bottom")
    bank0 = c[0].number_input("Bankroll de départ (€)", 50.0, 10_000.0, 500.0, 50.0)
    stake = c[1].number_input("Mise par série (€)", 1.0, 200.0, STAKE, 1.0, key="bk_stake")
    days = c[2].slider("Fenêtre (jours)", 15, 300, 60, 15, key="bk_days")
    mode = c[3].radio(
        "Cotes utilisées", ("Réelles archivées", "Hypothétique (toutes séries)"),
        horizontal=True, key="bk_mode",
        help="Réelles = uniquement les séries dont la cote de clôture est en cache "
             "(fidèle mais échantillon partiel). Hypothétique = toutes les séries, "
             "à une cote unique de ton choix.")

    past = load_past_series(days, 0.70)
    if past.empty:
        st.info("Aucune série sur cette fenêtre.")
        return

    if mode.startswith("Réelles"):
        view = past[pd.to_numeric(past["odds"], errors="coerce").notna()].copy()
        label = "cotes réelles"
        if view.empty:
            st.info("Aucune cote réelle en cache sur cette fenêtre — le remplissage "
                    "tourne encore, réessaie plus tard ou passe en hypothétique.")
            return
    else:
        hyp = st.slider("Cote hypothétique", 1.00, 2.50, 1.20, 0.05, key="bk_hyp")
        view = past.copy()
        view["odds"] = hyp
        label = f"cote uniforme {hyp:.2f}"

    # L'ordre chronologique strict est ce qui fait apparaître les séries.
    view = view.sort_values("match_date").reset_index(drop=True)
    o = pd.to_numeric(view["odds"], errors="coerce")
    won = (view["result"] == "won").to_numpy()
    profit = np.where(won, stake * (o - 1.0), -stake)
    bank = bank0 + np.cumsum(profit)
    peak = np.maximum.accumulate(np.r_[bank0, bank])[1:]
    dd = bank - peak                      # ≤ 0 : distance au dernier sommet
    n = len(view)

    # Plus longues chaînes de victoires / défaites consécutives.
    best_win = best_lose = cur = 0
    prev = None
    for w in won:
        cur = cur + 1 if w == prev else 1
        prev = w
        if w:
            best_win = max(best_win, cur)
        else:
            best_lose = max(best_lose, cur)

    k = st.columns(5)
    k[0].metric("Bankroll finale", f"{bank[-1]:.0f} €", delta=f"{bank[-1] - bank0:+.0f} €")
    k[1].metric("Plus bas touché", f"{bank.min():.0f} €",
                help="Le pire moment de la courbe : là où il fallait tenir.")
    k[2].metric("Max drawdown", f"{dd.min():.0f} €",
                help="La plus grosse descente depuis un sommet. C'est LE chiffre de risque.")
    k[3].metric("Pire série de défaites", f"{best_lose} de suite",
                delta=f"{-best_lose * stake:.0f} €")
    k[4].metric("Meilleure série de victoires", f"{best_win} de suite")

    idx = pd.RangeIndex(1, n + 1, name="pari n°")
    st.line_chart(pd.DataFrame({"Bankroll (€)": bank, "Départ (€)": bank0}, index=idx),
                  height=280)
    st.area_chart(pd.DataFrame({"Drawdown (€)": dd}, index=idx), height=160)
    st.caption(
        f"{n} séries rejouées ({label}), mise {stake:.0f} € — le graphique du bas montre "
        "à chaque instant la distance au dernier sommet : plus c'est profond, plus il "
        "fallait de sang-froid (et de caisse) pour continuer."
    )

    if bool((bank <= 0).any()):
        st.error(f"💀 Bankroll à zéro en cours de route : {bank0:.0f} € ne suffisent pas "
                 f"pour une mise de {stake:.0f} €. Baisse la mise ou augmente la caisse.")
    else:
        worst_pct = abs(float(dd.min())) / bank0 * 100
        st.markdown(
            f"**Lecture :** le pire trou a représenté **{worst_pct:.0f} %** d'une bankroll "
            f"de {bank0:.0f} € (mise = {stake / bank0 * 100:.1f} % de la caisse). Règle "
            "simple : si le max drawdown dépasse ~30 % de la bankroll, la mise est trop "
            "grosse pour tes nerfs — vise une mise qui laisse le pire trou sous 15-20 %."
        )


def main() -> None:
    st.title("💵 Les picks ciblés sont-ils rentables ?")
    st.caption(
        f"Mise **fixe de {STAKE:.0f} € par match**, uniquement sur les picks 🎯 "
        "(marché vainqueur). Principe : **trouver le bon vainqueur ne suffit pas** — "
        "il faut que la cote paie plus que le risque."
    )
    _simulation_block()
    st.divider()
    _bankroll_block()


if __name__ == "__main__":
    main()

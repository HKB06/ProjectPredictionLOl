"""Page Streamlit — PLAN DE JEU : sur quels matchs se concentrer, aux bons prix.

Trois profils de règles PRÉ-MOTIVÉES (seuil de proba série, value, plancher de
cote, règle BO1, exclusion événements), mesurés sur cotes de clôture réelles en
walk-forward — mesure CORRIGÉE le 18/11 par l'audit croisé avec la branche Seb
(séries reconstruites sur toutes leurs games, fiabilité de ligue jugée au fil de
l'eau, séries chaînées à cheval sur minuit, appariement de cotes strict).

État honnête depuis cette correction : AUCUN profil n'est prouvé rentable.
Les points sont positifs (socle ≈ +1 %, cœur ≈ +5 %, satellite ≈ +9 %) mais tous
les intervalles contiennent zéro et la 2e moitié de fenêtre est négative. Le plan
se joue donc À BLANC (papier) tant que la preuve n'est pas faite — c'est notre
propre règle, elle s'applique à nous.

Ce qui reste solide : la calibration du modèle par game (70 % annoncé ≈ 73 %
réalisé), l'exclusion des événements internationaux (ROI négatif confirmé sur les
Worlds), et la doctrine du PRIX : un pari se décide À LA COTE, jamais au prono.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.models.picks_roi import FAM_EVENT, STAKE, family_of

st.set_page_config(page_title="LoL — Plan de jeu", page_icon="🧭", layout="wide")

from src.ui.auth import require_password  # noqa: E402 - doit suivre set_page_config

require_password()

DAYS = 300          # profondeur d'historique (plancher API : décembre 2025)
POOL_CONF = 0.60    # pool élargi ; chaque profil remet son propre seuil ensuite


@st.cache_data(ttl=1800, show_spinner="Rejeu walk-forward du pool de picks…")
def load_pool() -> pd.DataFrame:
    from src.models.picks_roi import past_series
    d = past_series(days=DAYS, conf=POOL_CONF, real_odds=True)
    d = d[d["result"].isin(["won", "lost"])].copy()
    for c in ("odds", "breakeven", "our_proba", "profit", "stake"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


@st.cache_data(ttl=600, show_spinner="Matchs à venir + cotes du moment…")
def load_today(days: int = 3) -> pd.DataFrame:
    from src.models.picks_roi import candidates
    return pd.DataFrame(candidates(days=days))


# ------------------------------------------------------------------ les profils
def _mask_socle(d: pd.DataFrame) -> pd.Series:
    """≥0.65 de proba série · hors événements · BO1 seulement à cote ≥1.35."""
    return ((d["our_proba"] >= 0.65)
            & (d["family"] != FAM_EVENT)
            & ((d["bestof"] >= 3) | (d["odds"] >= 1.35)))


def _mask_coeur(d: pd.DataFrame) -> pd.Series:
    """≥0.65 · hors événements · cote ≥1.25 (tous formats)."""
    return ((d["our_proba"] >= 0.65)
            & (d["family"] != FAM_EVENT)
            & (d["odds"] >= 1.25))


def _mask_satellite(d: pd.DataFrame) -> pd.Series:
    """≥0.65 · hors événements · VALUE (cote > cote mini) · cote ≥1.35."""
    return ((d["our_proba"] >= 0.65)
            & (d["family"] != FAM_EVENT)
            & (d["odds"] > d["breakeven"])
            & (d["odds"] >= 1.35))


PROFILES = {
    "🏗️ Socle volume": {
        "mask": _mask_socle,
        "rule": "Proba série ≥ 65 % · ligues fiables · hors événements internationaux · "
                "BO1 joué UNIQUEMENT si cote ≥ 1.35 · BO3/BO5 à toute cote.",
        "why": "Le profil « volume » : BO3/BO5 réguliers + BO1 à cote longue seulement. "
               "Depuis la mesure corrigée du 18/11 : point ≈ +1 %, intervalle ±7, "
               "2e moitié négative — NON PROUVÉ. À suivre à blanc.",
    },
    "🎯 Cœur rentable": {
        "mask": _mask_coeur,
        "rule": "Proba série ≥ 65 % · hors événements · cote ≥ 1.25 (tous formats).",
        "why": "Le plancher de prix coupe les favoris courts, là où le book gagne. "
               "Mesure corrigée du 18/11 : point ≈ +5 %, intervalle ±13, 2e moitié "
               "négative — NON PROUVÉ. Le meilleur candidat, pas encore une preuve.",
    },
    "💎 Satellite value": {
        "mask": _mask_satellite,
        "rule": "Proba série ≥ 65 % · hors événements · value (cote > cote mini) · cote ≥ 1.35.",
        "why": "Désaccords favorables avec le book sur matchs serrés de régionales. "
               "Mesure corrigée du 18/11 : point ≈ +9 %, intervalle ±24 à n≈60, "
               "2e moitié négative — NON PROUVÉ, échantillon trop petit pour trancher.",
    },
}


def _stats(sub: pd.DataFrame) -> dict | None:
    if sub.empty:
        return None
    staked = float(sub["stake"].sum())
    pl = float(sub["profit"].sum())
    return {"n": len(sub), "n_an": len(sub) * 365 / DAYS,
            "hit": float((sub["result"] == "won").mean()),
            "odds": float(sub["odds"].mean()),
            "pl": pl, "roi": pl / staked if staked else float("nan")}


def main() -> None:
    st.title("🧭 Plan de jeu — sur quoi se concentrer, aux bons prix")
    st.caption(
        "Trois profils emboîtés, mesurés sur 300 j de **cotes de clôture réelles** en "
        "walk-forward. Règle d'or transverse : un pari se décide **à la cote**, pas au "
        "prono — sans cote affichée, on attend le prix."
    )
    st.warning(
        "**Mesure corrigée le 18/11 (audit croisé avec la branche Seb).** Quatre bugs "
        "de mesure corrigés — séries reconstruites sur toutes leurs games, séries "
        "chaînées au-delà de minuit UTC, fiabilité de ligue jugée sans voir le futur, "
        "appariement de cotes strict. Depuis : **aucun profil n'est prouvé rentable** "
        "(points positifs mais intervalles contenant zéro, 2e moitié de fenêtre "
        "négative). Le plan se suit **à blanc** (sans argent) jusqu'à preuve — "
        "idéalement le test à blanc commun de janvier 2027."
    )

    pool = load_pool()
    priced = pool[pool["odds"].notna()].copy()
    if priced.empty:
        st.info("Aucune série à cote réelle dans le pool — lance la collecte des cotes.")
        return
    priced["date"] = pd.to_datetime(priced["match_date"])
    mid = priced["date"].sort_values().iloc[len(priced) // 2]

    choice = st.radio("Profil", list(PROFILES), horizontal=True)
    prof = PROFILES[choice]
    sel = priced[prof["mask"](priced)]
    s = _stats(sel)
    if not s:
        st.info("Aucune série ne colle à ce profil sur la fenêtre.")
        return

    st.markdown(f"**Règle :** {prof['rule']}")
    st.caption(prof["why"])

    h1, h2 = sel[sel["date"] <= mid], sel[sel["date"] > mid]
    s1, s2 = _stats(h1), _stats(h2)
    k = st.columns(6)
    k[0].metric("Paris (300 j)", s["n"], delta=f"≈{s['n_an']:.0f}/an")
    k[1].metric("Réussite", f"{s['hit']*100:.1f}%")
    k[2].metric("Cote moyenne", f"{s['odds']:.2f}")
    k[3].metric("P/L (mise 10 €)", f"{s['pl']:+.0f} €")
    k[4].metric("ROI", f"{s['roi']*100:+.1f}%")
    k[5].metric("Stabilité (2 moitiés)",
                f"{s1['roi']*100:+.0f}% / {s2['roi']*100:+.0f}%" if s1 and s2 else "—",
                help="ROI sur la 1re et la 2e moitié de la fenêtre : les deux doivent "
                     "être positifs pour qu'une règle soit retenue.")

    # Courbe de P/L cumulé (ordre chronologique)
    curve = sel.sort_values("date")
    cum = np.cumsum(pd.to_numeric(curve["profit"], errors="coerce").fillna(0.0).to_numpy())
    st.line_chart(pd.DataFrame({"P/L cumulé (€)": cum},
                               index=pd.RangeIndex(1, len(cum) + 1, name="pari n°")),
                  height=240)

    with st.expander("📊 Où le profil gagne — par compétition × BO"):
        rows = []
        for (lg, bo), sub in sel.groupby([sel["league"], sel["bestof"].astype(int)]):
            st_ = _stats(sub)
            if st_ and st_["n"] >= 3:
                rows.append({"Compétition": lg, "Format": f"BO{bo}", "Paris": st_["n"],
                             "Réussite": st_["hit"] * 100, "Cote moy.": st_["odds"],
                             "P/L (€)": round(st_["pl"], 0), "ROI": st_["roi"] * 100})
        if rows:
            df = pd.DataFrame(rows).sort_values("ROI", ascending=False)
            st.dataframe(df, width="stretch", hide_index=True, column_config={
                "Réussite": st.column_config.NumberColumn(format="%.1f%%"),
                "Cote moy.": st.column_config.NumberColumn(format="%.2f"),
                "ROI": st.column_config.NumberColumn(format="%+.1f%%"),
            })
        else:
            st.caption("Pas encore 3 paris par case sur ce profil.")

    with st.expander("📜 Toutes les séries du profil (vérifiable ligne à ligne)"):
        verdict = {"won": "✅ gagné", "lost": "❌ perdu"}
        st.dataframe(
            sel.sort_values("match_date", ascending=False)
               .assign(result=sel["result"].map(verdict))
               [["match_date", "league", "bestof", "team1", "team2", "pick", "our_proba",
                 "breakeven", "odds", "result", "winner", "score", "profit"]],
            width="stretch", hide_index=True, height=380,
            column_config={
                "our_proba": st.column_config.NumberColumn("Notre p", format="%.2f"),
                "breakeven": st.column_config.NumberColumn("Cote mini", format="%.2f"),
                "odds": st.column_config.NumberColumn("Cote", format="%.2f"),
                "profit": st.column_config.NumberColumn("P/L (€)", format="%.2f"),
            },
        )

    # ------------------------------------------------------------- à venir
    st.divider()
    st.subheader("📅 À venir : ce qui colle au plan (dès que la cote est là)")
    try:
        today = load_today()
    except Exception as exc:  # noqa: BLE001
        st.info(f"Calendrier/cotes indisponibles ({exc}).")
        today = pd.DataFrame()
    if today.empty:
        st.caption("Aucun pick sur les 3 prochains jours.")
    else:
        rows = []
        for r in today.itertuples():
            bo = int(r.bestof) if r.bestof == r.bestof else 1
            # `our_proba` des candidats est DÉJÀ une proba de série (watchlist la
            # convertit) : la reconvertir ici la gonflait une 2e fois — bug attrapé
            # par l'audit croisé du 18/11 (« conversion BO appliquée deux fois »).
            p_serie = float(r.our_proba)
            odds = float(r.odds) if r.odds == r.odds else float("nan")
            fam = family_of(str(r.league), None)
            checks = {
                "p série ≥ 0.65": p_serie >= 0.65,
                "hors événement": fam != FAM_EVENT,
            }
            if odds == odds:
                if choice.startswith("🏗️"):      # socle : BO1 seulement à cote longue
                    price_ok = (odds >= 1.35) if bo == 1 else True
                elif choice.startswith("🎯"):    # cœur : plancher 1.25 partout
                    price_ok = odds >= 1.25
                else:                            # satellite : value + plancher 1.35
                    price_ok = odds > 1.0 / p_serie and odds >= 1.35
                checks["prix OK (profil)"] = bool(price_ok)
                verdict = "✅ À JOUER" if all(checks.values()) else "❌ hors plan"
            else:
                verdict = ("⏳ attendre la cote" if all(checks.values())
                           else "❌ hors plan")
            rows.append({"Quand": getattr(r, "when", ""), "Ligue": r.league,
                         "Format": f"BO{bo}", "Match": f"{r.team1} vs {r.team2}",
                         "Pick": r.pick, "p série": p_serie * 100,
                         "Cote mini": 1.0 / p_serie,
                         "Cote actuelle": odds if odds == odds else None,
                         "Verdict plan": verdict})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "p série": st.column_config.NumberColumn(format="%.0f%%"),
            "Cote mini": st.column_config.NumberColumn(format="%.2f"),
            "Cote actuelle": st.column_config.NumberColumn(format="%.2f"),
        })
        st.caption(
            "⚠️ La watchlist actuelle ne remonte que les picks ≥70 %/game : les picks "
            "65-70 % du plan n'apparaissent pas encore ici (ticket ouvert). Un match "
            "« ⏳ » se rejoue au moment où le book ouvre sa ligne : c'est le prix qui "
            "décide, pas le prono."
        )

    # ------------------------------------------------------------- honnêteté
    st.divider()
    st.markdown("### 🧪 Lecture d'ingénieur — pourquoi y croire, et jusqu'où")
    st.markdown(
        "- **Pourquoi ces règles et pas d'autres :** grille restreinte à 5 paramètres "
        "motivés par des résultats antérieurs (sélectivité de juin, piège des cotes "
        "courtes d'octobre) — pas de data mining libre.\n"
        "- **Audit croisé du 18/11 :** la branche Seb a audité notre mesure, nous avons "
        "vérifié chaque accusation sur pièce, quatre étaient fondées et sont corrigées "
        "(détail dans `AUDIT_2026-11-18.md`). Conséquence assumée : les rendements "
        "« stables sur deux moitiés » affichés avant étaient un artefact de mesure. "
        "Les chiffres actuels sont les corrigés.\n"
        "- **Incertitude :** ±2 écarts-types ≈ ±7 pts de ROI sur le socle, ±13 sur le "
        "cœur, ±24 sur le satellite. Tant que l'intervalle contient zéro, le profil "
        "n'est pas prouvé et ne se joue pas en argent réel.\n"
        "- **Ce qui est solide :** la calibration par game (70 % annoncé ≈ 73 % réalisé "
        "sur 8 600 games), l'exclusion des événements internationaux (ROI négatif sur "
        "les deux moitiés, Worlds compris), l'intégrité des cotes (chaque cote tracée "
        "vers une ligne brute : id d'événement, book, horodatage).\n"
        "- **Base de mesure :** cotes de CLÔTURE, les plus dures à battre. Prendre les "
        "lignes plus tôt et comparer 2-3 books gagne typiquement 2-5 pts — c'est le "
        "levier n° 1 si un profil finit par être prouvé.\n"
        "- **Discipline :** mise fixe, jamais de rattrapage, suivi à blanc jusqu'à "
        "preuve — un pick sans prix n'est pas un pari."
    )


if __name__ == "__main__":
    main()

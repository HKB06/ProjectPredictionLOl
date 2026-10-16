"""Page Streamlit — PLAN DE JEU : sur quels matchs se concentrer, aux bons prix.

Le résultat d'une recherche de paramètres sous contraintes (16/10, fenêtre 300 j,
cotes de clôture réelles, walk-forward) avec garde-fous anti-surapprentissage :
  - grille de règles PRÉ-MOTIVÉES (seuil de proba série, value, plancher de cote,
    règle BO1, exclusion événements) — pas de data mining libre ;
  - validation par STABILITÉ : une règle n'est retenue que si son ROI est positif
    sur les DEUX moitiés temporelles de la fenêtre ;
  - le socle du projet reste inchangé en amont : ligues fiables (accuracy ≥62 %)
    et ≥15 games de données par équipe.

La découverte centrale : le levier de rentabilité n'est PAS la confiance du
modèle, c'est le PRIX. Les favoris très sûrs (p ≥ 0,75) vivent à cote courte,
notre zone morte ; les picks à p 0,65-0,80 vivent à cote 1,3-1,8, là où les books
sont paresseux sur les ligues régionales. Un pari se décide À LA COTE, jamais au
prono seul.
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
    "🏗️ Socle volume — ~315 paris/an": {
        "mask": _mask_socle,
        "rule": "Proba série ≥ 65 % · ligues fiables · hors événements internationaux · "
                "BO1 joué UNIQUEMENT si cote ≥ 1.35 · BO3/BO5 à toute cote.",
        "why": "Le profil « 200+ matchs/an au meilleur ROI » : le volume vient des BO3/BO5 "
               "réguliers, et la règle BO1 asymétrique coupe le poison mesuré (BO1 courts).",
    },
    "🎯 Cœur rentable — ~146 paris/an": {
        "mask": _mask_coeur,
        "rule": "Proba série ≥ 65 % · hors événements · cote ≥ 1.25 (tous formats).",
        "why": "Le plancher de prix fait le travail : on ne joue plus aucun favori court. "
               "Moitié moins de paris, ROI environ doublé.",
    },
    "💎 Satellite value — ~72 paris/an": {
        "mask": _mask_satellite,
        "rule": "Proba série ≥ 65 % · hors événements · value (cote > cote mini) · cote ≥ 1.35.",
        "why": "La crème : désaccords favorables avec le book sur des matchs serrés de "
               "ligues régionales (LFL/PRM en tête). Peu de paris, gros ROI mesuré.",
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
        "walk-forward et retenus seulement s'ils sont **rentables sur les deux moitiés** "
        "de la fenêtre. Règle d'or transverse : un pari se décide **à la cote**, pas au "
        "prono — sans cote affichée, on attend le prix."
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
        from src.update.elo import series_prob
        rows = []
        for r in today.itertuples():
            bo = int(r.bestof) if r.bestof == r.bestof else 1
            p_game = float(r.our_proba)
            p_serie = series_prob(p_game, (bo + 1) // 2)
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
        "courtes d'octobre), validation par stabilité temporelle. Les règles instables "
        "ont été jetées même quand leur ROI global était meilleur.\n"
        "- **La découverte contre-intuitive :** monter le seuil de confiance à 75 % "
        "DÉGRADE le ROI (+1,3 %, instable). Haute confiance = cote courte = zone où le "
        "book est plus fort que nous. Notre argent se fait à p 65-80 % et cote 1,3-1,8 "
        "dans les régionales (LFL, PRM…), pas sur les gros favoris.\n"
        "- **Incertitude :** ±2 écarts-types ≈ ±6 pts de ROI sur le socle (n=259), "
        "±11 pts sur le cœur (n=120), ±21 pts sur le satellite (n=59). Les signes sont "
        "solides, les grandeurs sont des ordres de grandeur.\n"
        "- **Base de mesure :** cotes de CLÔTURE. En prenant les lignes plus tôt et en "
        "comparant 2-3 books, on gagne typiquement 2-5 pts de mieux.\n"
        "- **Discipline :** mise fixe, jamais de rattrapage, et le plan s'applique à la "
        "cote — un pick sans prix n'est pas un pari."
    )


if __name__ == "__main__":
    main()

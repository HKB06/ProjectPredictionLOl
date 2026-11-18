"""Page Streamlit — PLAN DE JEU : sur quels matchs se concentrer, aux bons prix.

Le résultat d'une recherche de paramètres sous contraintes (16/10, fenêtre 300 j,
cotes de clôture réelles, walk-forward) avec garde-fous anti-surapprentissage :
  - grille de règles PRÉ-MOTIVÉES (seuil de proba série, value, plancher de cote,
    règle BO1, exclusion événements) — pas de data mining libre ;
  - validation par STABILITÉ : une règle n'est retenue que si son ROI est positif
    sur les DEUX moitiés temporelles de la fenêtre ;
  - le socle du projet reste inchangé en amont : ligues fiables (accuracy ≥62 %)
    et ≥15 games de données par équipe.

⚠️ Correction du 17/11/2026 : ces trois profils avaient été retenus sur un bilan dont
les séries étaient mal reconstituées (formats et résultats faux sur une série sur cinq,
ligues « fiables » choisies en connaissant la période testée). Remesurés proprement, aucun
n'est rentable de façon prouvée. La page affiche maintenant le chiffre du jour, son
intervalle de confiance et un verdict calculé — plus aucun rendement écrit en dur.
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
        "why": "Le volume vient des BO3/BO5 réguliers ; les BO1 ne sont joués qu'à cote "
               "longue.",
    },
    "🎯 Cœur (cote ≥ 1,25)": {
        "mask": _mask_coeur,
        "rule": "Proba série ≥ 65 % · hors événements · cote ≥ 1.25 (tous formats).",
        "why": "Un plancher de prix : plus aucun favori court.",
    },
    "💎 Satellite value": {
        "mask": _mask_satellite,
        "rule": "Proba série ≥ 65 % · hors événements · value (cote > cote mini) · cote ≥ 1.35.",
        "why": "Les désaccords avec le book sur des matchs serrés. Très peu de paris : le "
               "chiffre bouge beaucoup d'un mois à l'autre.",
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
        "Trois profils emboîtés, mesurés sur 300 j de **cotes de clôture réelles** (GG.bet, "
        "1xbet) en walk-forward. Un profil n'est « prouvé » que si son rendement est positif "
        "sur les deux moitiés de la fenêtre ET que son intervalle de confiance ne touche pas "
        "zéro. Règle transverse : un pari se décide **à la cote**, pas au prono."
    )
    st.warning(
        "**Remesuré le 17/11/2026 après correction du bilan** (séries reconstituées à partir "
        "de toutes leurs games, ligues fiables jugées sans connaître la suite) : **aucun des "
        "trois profils n'est rentable de façon prouvée.** Les rendements annoncés avant "
        "(+6 %, +14 %, +27 %) venaient d'un bilan faux sur une série sur cinq."
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
    r = (sel["profit"] / sel["stake"]).to_numpy()
    rng = np.random.default_rng(0)
    lo, hi = np.percentile([r[rng.integers(0, len(r), len(r))].mean() for _ in range(2000)],
                           [2.5, 97.5])
    stable = bool(s1 and s2 and s1["roi"] > 0 and s2["roi"] > 0)
    ci = f"intervalle de confiance à 95 % : {lo*100:+.1f} % à {hi*100:+.1f} %"
    if lo > 0 and stable:
        st.success(f"✅ Profil prouvé rentable sur cette fenêtre ({ci}).")
    elif hi < 0:
        st.error(f"❌ Profil perdant ({ci}).")
    else:
        st.warning(f"⚠️ Rentabilité non prouvée ({ci}"
                   + ("" if stable else " ; une des deux moitiés est négative") + ").")

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
            # `our_proba` est déjà la proba de gagner la SÉRIE (picks_roi.candidates) : la
            # reconvertir en BO3/BO5 la gonflait (0,78 affiché 0,88) et baissait la cote mini.
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
    st.markdown("### 🧪 Ce qu'on sait, et ce qu'on ne sait pas")
    st.markdown(
        "- **Ces règles ont été choisies sur un bilan erroné.** Elles sont gardées pour "
        "suivre leur rendement réel, pas parce qu'elles sont validées.\n"
        "- **Ce qui a été corrigé le 17/11/2026 :** une série est reconstituée à partir de "
        "toutes ses games (avant, une game moins nette disparaissait et un 2-3 devenait un "
        "« BO1 gagné ») ; le pick est celui d'avant la game 1 ; une ligue n'est « fiable » "
        "que sur ce qu'on savait avant le match ; les cotes des deux équipes ne peuvent "
        "plus être inversées.\n"
        "- **Peu de paris, donc beaucoup d'incertitude :** l'intervalle de confiance "
        "affiché plus haut est le chiffre à regarder, pas le rendement seul.\n"
        "- **Base de mesure :** cotes de CLÔTURE de bookmakers qui limitent les comptes "
        "gagnants. Sur Polymarket, nos modèles ne battent pas le prix.\n"
        "- **Discipline :** mise fixe, jamais de rattrapage — un pick sans prix n'est pas "
        "un pari."
    )


if __name__ == "__main__":
    main()

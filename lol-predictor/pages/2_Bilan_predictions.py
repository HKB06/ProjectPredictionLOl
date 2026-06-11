"""Page Streamlit — BILAN : ce que le modèle avait prédit vs la réalité.

Rejoue tout l'historique en walk-forward strict (modèle retenu K32 + MOV) : pour chaque
game on prédit AVEC l'Elo d'AVANT (zéro fuite), puis on compare au vrai résultat.
On voit où on a eu bon, où on s'est trompé, par jour / ligue / niveau de confiance,
et si nos probabilités sont bien calibrées.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.models.eval_models import production_records
from src.update.elo import RELIABLE_ACC

st.set_page_config(page_title="LoL — Bilan prédictions", page_icon="✅", layout="wide")

WINDOWS = {"3 jours": 3, "7 jours": 7, "14 jours": 14, "30 jours": 30, "Toute la saison": None}


@st.cache_data(ttl=1800, show_spinner="Rejoue tout l'historique (walk-forward, K32 + MOV)…")
def load_records() -> pd.DataFrame:
    rec = production_records()
    return rec[rec["nmin"] >= 5].reset_index(drop=True)  # hors cold-start


def _calib(view: pd.DataFrame) -> pd.DataFrame:
    pf = view["proba_fav"].to_numpy()
    won = view["correct"].to_numpy().astype(float)
    rows = []
    for lo in np.arange(0.5, 1.0, 0.1):
        hi = lo + 0.1
        m = (pf >= lo) & (pf < hi + (1e-9 if lo > 0.85 else 0))
        if m.sum() == 0:
            continue
        rows.append({"Tranche de confiance": f"{lo*100:.0f}–{hi*100:.0f}%",
                     "Games": int(m.sum()),
                     "Proba annoncée": round(pf[m].mean() * 100, 1),
                     "Gagné réel": round(won[m].mean() * 100, 1),
                     "Écart": round((pf[m].mean() - won[m].mean()) * 100, 1)})
    return pd.DataFrame(rows)


def main() -> None:
    st.title("✅ Bilan — nos prédictions vs la réalité")
    st.caption(
        "**Walk-forward strict** : pour chaque game, prédiction avec l'Elo d'**avant** la game "
        "(zéro triche), modèle **K32 + marge de victoire**. *Correct* = on a désigné le bon "
        "vainqueur (favori côté-neutre). La calibration n'affecte pas qui est favori : "
        "l'accuracy est donc le juge de paix."
    )

    rec = load_records()
    maxd = rec["date"].max()

    c = st.columns([3, 2])
    opt = c[0].radio("Période", list(WINDOWS), horizontal=True, index=1)
    days = WINDOWS[opt]
    view = rec if days is None else rec[rec["date"] >= (maxd.normalize() - pd.Timedelta(days=days - 1))]
    c[1].metric("Data jusqu'au", f"{maxd:%Y-%m-%d}")

    if view.empty:
        st.warning("Aucune game sur cette période.")
        st.stop()

    # --- KPIs ---
    n = len(view)
    acc = view["correct"].mean()
    brier = float(((view["p"] - view["yb"]) ** 2).mean())
    strong = view[view["proba_fav"] >= 0.62]
    k = st.columns(4)
    k[0].metric("Games évaluées", n)
    k[1].metric("Bon vainqueur trouvé", f"{acc*100:.1f}%")
    k[2].metric("Favoris confiants (≥62 %)",
                f"{strong['correct'].mean()*100:.0f}%" if len(strong) else "—",
                help=f"{len(strong)} games où on annonçait ≥62 %")
    k[3].metric("Brier (qualité proba)", f"{brier:.3f}", help="plus bas = mieux ; 0.25 = hasard")

    # --- Par jour ---
    st.subheader("Par jour")
    by_day = (view.assign(jour=view["date"].dt.date)
                  .groupby("jour")
                  .agg(games=("correct", "size"), bons=("correct", "sum"))
                  .reset_index())
    by_day["précision"] = (by_day["bons"] / by_day["games"] * 100).round(0)
    st.dataframe(
        by_day.rename(columns={"jour": "Jour", "games": "Games", "bons": "Bons", "précision": "Précision %"}),
        width="stretch", hide_index=True,
        column_config={"Précision %": st.column_config.ProgressColumn(
            "Précision %", format="%d%%", min_value=0, max_value=100)},
    )

    # --- Par ligue ---
    st.subheader("Par ligue")
    by_lg = (view.groupby("league")
                 .agg(games=("correct", "size"), bons=("correct", "sum"))
                 .reset_index())
    by_lg = by_lg[by_lg["games"] >= 3]
    by_lg["précision"] = (by_lg["bons"] / by_lg["games"] * 100).round(0)
    by_lg["fiabilité"] = np.where(by_lg["précision"] >= RELIABLE_ACC * 100, "✅ fiable", "🌪️ chaotique")
    by_lg = by_lg.sort_values("games", ascending=False)
    st.dataframe(
        by_lg.rename(columns={"league": "Ligue", "games": "Games", "bons": "Bons",
                              "précision": "Précision %", "fiabilité": "Fiabilité"}),
        width="stretch", hide_index=True,
        column_config={"Précision %": st.column_config.ProgressColumn(
            "Précision %", format="%d%%", min_value=0, max_value=100)},
    )

    # --- Calibration ---
    st.subheader("Calibration — nos « X % » sont-ils honnêtes ?")
    st.caption("Si on annonce 70 %, le favori doit gagner ~70 % du temps. Écart positif = sur-confiance.")
    st.dataframe(_calib(view), width="stretch", hide_index=True)

    # --- Détail match par match ---
    st.subheader("Détail — où on a eu bon / faux")
    f = st.columns([3, 1.4, 1.4])
    leagues = sorted(view["league"].unique())
    sel = f[0].multiselect("Ligues", leagues, default=leagues)
    only_err = f[1].toggle("Erreurs seulement", value=False)
    only_strong = f[2].toggle("Confiants ≥62 % seulement", value=False)

    d = view[view["league"].isin(sel)].copy()
    if only_err:
        d = d[~d["correct"]]
    if only_strong:
        d = d[d["proba_fav"] >= 0.62]
    d = d.sort_values("date", ascending=False)

    show = pd.DataFrame({
        "Résultat": np.where(d["correct"], "✅", "❌"),
        "Date": d["date"].dt.strftime("%m-%d %H:%M"),
        "Ligue": d["league"],
        "Match": d["blue"] + " 🔵 vs 🔴 " + d["red"],
        "Notre favori": d["favori"],
        "P(favori)": (d["proba_fav"] * 100).round(0),
        "Vainqueur réel": d["vainqueur"],
    })
    st.dataframe(
        show, width="stretch", hide_index=True, height=460,
        column_config={"P(favori)": st.column_config.ProgressColumn(
            "P(favori)", format="%d%%", min_value=0, max_value=100)},
    )
    st.caption(f"{len(d)} games affichées · ✅ {int(d['correct'].sum())} bons / ❌ "
               f"{int((~d['correct']).sum())} erreurs.")


if __name__ == "__main__":
    main()

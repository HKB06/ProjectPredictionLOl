"""Page Streamlit — JOURNAL DE PARIS : la seule preuve qui compte (ROI réel).

Chaque pari posé est loggé ici (cote, mise, notre proba au moment du pari, cote de
clôture si dispo). La page calcule P/L, ROI, winrate et CLV — c'est la version
structurée du SUIVI_PARIS.md, et le juge de paix du projet : accuracy ≠ profit.

Stockage : data/bets.csv (simple, versionnable, éditable à la main si besoin).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.ingest.load_oracle import ROOT

st.set_page_config(page_title="LoL — Journal de paris", page_icon="💰", layout="wide")

BETS_PATH = ROOT / "data" / "bets.csv"
COLS = ["placed_at", "league", "match", "bet_on", "market", "odds", "stake",
        "our_proba", "closing_odds", "result", "notes"]
RESULTS = ["open", "won", "lost", "void"]


def load_bets() -> pd.DataFrame:
    if not BETS_PATH.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(BETS_PATH)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS]


def save_bets(df: pd.DataFrame) -> None:
    df.to_csv(BETS_PATH, index=False)


def settle(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute payout et profit (unités) pour les paris réglés."""
    d = df.copy()
    d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
    d["stake"] = pd.to_numeric(d["stake"], errors="coerce")
    d["closing_odds"] = pd.to_numeric(d["closing_odds"], errors="coerce")
    d["payout"] = 0.0
    d.loc[d["result"] == "won", "payout"] = d["stake"] * d["odds"]
    d.loc[d["result"] == "void", "payout"] = d["stake"]
    d["profit"] = d["payout"] - d["stake"]
    d.loc[d["result"] == "open", "profit"] = float("nan")
    return d


def main() -> None:
    st.title("💰 Journal de paris — ROI réel")
    st.caption(
        "**La preuve finale n'est ni l'accuracy ni l'edge théorique : c'est cette page.** "
        "Logge chaque pari AVANT le match (cote + mise + notre proba). Renseigne la cote de "
        "clôture quand tu peux : un **CLV positif** (tu as pris mieux que la cote finale) est le "
        "meilleur prédicteur de profit long terme. Objectif : 20-30 paris pour conclure."
    )

    df = load_bets()

    # --- saisie d'un nouveau pari ---
    with st.expander("➕ Ajouter un pari", expanded=df.empty):
        with st.form("add_bet", clear_on_submit=True):
            c1 = st.columns([1.4, 1.2, 2.4, 1.6])
            placed = c1[0].date_input("Date", value=dt.date.today())
            lg = c1[1].text_input("Ligue", placeholder="LJL")
            match = c1[2].text_input("Match", placeholder="Rising Gaming vs Arneb")
            bet_on = c1[3].text_input("Pari sur", placeholder="Rising Gaming")
            c2 = st.columns([1.6, 1, 1, 1.2, 1.2])
            market = c2[0].selectbox("Marché", ["série (moneyline)", "map", "handicap maps", "autre"])
            odds = c2[1].number_input("Cote", 1.01, 51.0, 2.00, 0.01)
            stake = c2[2].number_input("Mise (u)", 0.1, 100.0, 1.0, 0.1)
            our_p = c2[3].number_input("Notre proba %", 1, 99, 60)
            closing = c2[4].number_input("Cote clôture (0 = inconnue)", 0.0, 51.0, 0.0, 0.01)
            notes = st.text_input("Notes (pattern, contexte…)",
                                  placeholder="🎯 pick 75% / fade momentum / value PM +6 pts…")
            ok = st.form_submit_button("Enregistrer", type="primary")
        if ok:
            if not match.strip() or not bet_on.strip():
                st.error("Match et 'Pari sur' sont obligatoires.")
            else:
                row = {"placed_at": placed.isoformat(), "league": lg.strip() or "?",
                       "match": match.strip(), "bet_on": bet_on.strip(), "market": market,
                       "odds": odds, "stake": stake, "our_proba": our_p / 100,
                       "closing_odds": closing or None, "result": "open", "notes": notes}
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                save_bets(df)
                st.success(f"Pari enregistré : {bet_on} @ {odds:.2f} ({stake:.1f} u). "
                           f"EV au moment du pari : {(our_p/100 - 1/odds)*100:+.1f} pts.")

    if df.empty:
        st.info("Aucun pari loggé. Ajoute ton premier pari ci-dessus — y compris les *paper bets* "
                "(mise fictive) pour valider la stratégie sans risque.")
        return

    # --- édition (résultats, clôture) ---
    st.subheader("Paris (édite résultat / cote de clôture puis Sauvegarder)")
    edited = st.data_editor(
        df, width="stretch", hide_index=True, num_rows="dynamic",
        column_config={
            "placed_at": st.column_config.TextColumn("Date"),
            "league": st.column_config.TextColumn("Ligue", width="small"),
            "match": st.column_config.TextColumn("Match", width="medium"),
            "bet_on": st.column_config.TextColumn("Pari sur"),
            "market": st.column_config.TextColumn("Marché", width="small"),
            "odds": st.column_config.NumberColumn("Cote", format="%.2f"),
            "stake": st.column_config.NumberColumn("Mise (u)", format="%.1f"),
            "our_proba": st.column_config.NumberColumn("Notre p", format="%.2f"),
            "closing_odds": st.column_config.NumberColumn("Clôture", format="%.2f"),
            "result": st.column_config.SelectboxColumn("Résultat", options=RESULTS, width="small"),
            "notes": st.column_config.TextColumn("Notes", width="medium"),
        },
    )
    if st.button("💾 Sauvegarder les modifications", type="primary"):
        save_bets(edited)
        st.success("Journal sauvegardé.")
        df = edited

    # --- bilan ---
    d = settle(df)
    closed = d[d["result"].isin(["won", "lost", "void"])]
    real = closed[closed["result"] != "void"]
    st.subheader("Bilan")
    if real.empty:
        st.info("Aucun pari réglé pour l'instant — les KPIs apparaîtront dès le premier résultat.")
        return

    staked = real["stake"].sum()
    profit = real["profit"].sum()
    roi = profit / staked if staked else 0.0
    wr = (real["result"] == "won").mean()
    has_clv = real["closing_odds"].notna() & (real["closing_odds"] > 1)
    clv = (real.loc[has_clv, "odds"] / real.loc[has_clv, "closing_odds"] - 1).mean() if has_clv.any() else None

    k = st.columns(5)
    k[0].metric("Paris réglés", len(real))
    k[1].metric("Winrate", f"{wr*100:.0f}%")
    k[2].metric("P/L", f"{profit:+.2f} u")
    k[3].metric("ROI", f"{roi*100:+.1f}%")
    k[4].metric("CLV moyen", f"{clv*100:+.1f}%" if clv is not None else "—",
                help="cote prise vs cote de clôture ; >0 = on bat le marché de clôture "
                     "(meilleur prédicteur de profit long terme)")

    curve = real.sort_values("placed_at").assign(cumul=lambda x: x["profit"].cumsum())
    st.line_chart(curve.set_index("placed_at")["cumul"], height=260)
    st.caption("Courbe de bankroll (unités cumulées, paris réglés par date de prise).")

    by_note = real.copy()
    by_note["tag"] = by_note["notes"].fillna("").str.extract(r"(🎯|fade|value PM|paper)", expand=False).fillna("autre")
    agg = (by_note.groupby("tag")
           .agg(n=("profit", "size"), pl=("profit", "sum"), wr=("result", lambda s: (s == "won").mean()))
           .reset_index())
    if len(agg) > 1:
        st.subheader("Par pattern (via les notes)")
        agg["wr"] = (agg["wr"] * 100).round(0)
        st.dataframe(agg.rename(columns={"tag": "Pattern", "n": "Paris", "pl": "P/L (u)", "wr": "Win %"}),
                     width="stretch", hide_index=True)


if __name__ == "__main__":
    main()

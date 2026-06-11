"""Front Streamlit — ACCUEIL : matchs LoL à venir (toutes équipes) + notre proba Elo.

Page principale du projet. Affiche le calendrier des prochains jours (API lolesports)
avec NOTRE probabilité (Elo toutes-ligues), pour repérer à l'avance un favori que le
book va peut-être sur-coter (pattern KC / VKS / Heretics). Bouton 1-clic pour
rafraîchir la data (Google Drive) et le calendrier.

Lancer :
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.ingest.load_oracle import ROOT, load_config

st.set_page_config(page_title="LoL — Matchs à venir", page_icon="📅", layout="wide")


@st.cache_data(ttl=1800, show_spinner="Calcul Elo + récupération des matchs (lolesports)...")
def load_watchlist(days: int):
    from src.update.watchlist import build_rows
    return build_rows(days)


@st.cache_data(ttl=300)
def data_info():
    cfg = load_config()
    p = ROOT / cfg["data"]["oracle_csv"]
    if not p.exists():
        return None, 0.0
    mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m %H:%M")
    return mtime, p.stat().st_size / 1e6


def _fav(r: dict):
    if r["p1"] >= r["p2"]:
        return r["team1"], r["p1"], r["team2"], r["p2"], r["elo1"], r["elo2"]
    return r["team2"], r["p2"], r["team1"], r["p1"], r["elo2"], r["elo1"]


def refresh_data():
    with st.status("Mise à jour…", expanded=True) as status:
        st.write("Téléchargement de la data (Google Drive)…")
        try:
            from src.update.download_data import download_latest
            download_latest()
            st.write("✅ Data Oracle's Elixir à jour.")
        except Exception as exc:  # noqa: BLE001
            st.write(f"⚠️ Drive indisponible ({exc}). On garde la data locale.")
        load_watchlist.clear()
        data_info.clear()
        status.update(label="Terminé.", state="complete")


def main() -> None:
    st.title("📅 Matchs LoL à venir — notre lecture (Elo)")

    top = st.columns([2.2, 1.4, 1.4, 3])
    with top[0]:
        if st.button("🔄 Actualiser données + matchs", type="primary", width="stretch"):
            refresh_data()
            st.rerun()
    with top[1]:
        days = st.selectbox("Fenêtre", [3, 5, 7, 10, 14], index=2, format_func=lambda d: f"{d} jours")
    with top[2]:
        mtime, size = data_info()
        st.metric("Data OE", mtime or "—", help=f"{size:.0f} Mo" if size else "fichier absent")

    try:
        covered, uncovered = load_watchlist(days)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible de récupérer le calendrier : {exc}")
        st.stop()

    if not covered:
        st.warning("Aucun match couvert sur la fenêtre (beaucoup de playoffs 'TBD' en ce moment, "
                   "ou équipes hors data). Élargis la fenêtre ou réessaie plus tard.")
    strong = [r for r in covered if r.get("strong")]

    k = st.columns(4)
    k[0].metric("Matchs couverts", len(covered))
    k[1].metric("Penchants forts ⭐", len(strong), help="favori ≥62 %, data fiable ET ligue prévisible")
    k[2].metric("Non couverts", len(uncovered), help="équipe(s) hors de notre data")
    k[3].metric("Ligues", len({r["league"] for r in covered}))

    st.caption(
        "**Elo toutes-ligues K32 + marge de victoire (MOV)**, proba **calibrée par ligue** "
        "(aplatie là où le modèle est chaotique, ex. EM ≈ 56 % d'accuracy → un « 70 % » brut "
        "n'y vaut ~55 %). Signal *partiel* (sans draft). Idée : **fader un favori sur-coté** que "
        "le modèle voit plus serré (cf. KC @2.45, VKS @2.7, Heretics @2.60). On ne fade **jamais** "
        "un favori à cote < ~1,2. 🌪️ = ligue chaotique (proba peu fiable). Poser le pari **tôt** = l'enjeu."
    )

    # --- 🎯 À chasser : nos favoris confiants = candidats value ---
    if strong:
        with st.container(border=True):
            st.markdown("### 🎯 À chasser — nos favoris confiants (candidats *value*)")
            st.caption(
                "Le pattern gagnant = le **book met NOTRE favori en outsider** sur une ligne "
                "**équilibrée** (jamais < 1,2). Ex. **Heretics 0-2 KCB** : book KCB favori (1.45), "
                "nous Heretics 65 % → on était sur l'outsider gagnant. "
                "⚠️ = matchup **cross-ligue** (Elo moins comparable, à vérifier)."
            )
            for r in sorted(strong, key=lambda x: x["datetime"]):
                fav, p_fav, und, _pu, _ea, _eb = _fav(r)
                x = " · ⚠️ **cross-ligue**" if r["xleague"] else ""
                st.markdown(
                    f"- **{r['when']}** · {r['league']} · BO{r['bestof']} — "
                    f"notre favori **{fav} ({p_fav*100:.0f}%)** vs {und}{x}  \n"
                    f"  → *value SI le book donne {fav} perdant ou trop proche*"
                )

    # --- Filtres ---
    leagues = sorted({r["league"] for r in covered})
    f = st.columns([3, 1.3, 1.3])
    sel_leagues = f[0].multiselect("Ligues", leagues, default=leagues)
    only_strong = f[1].toggle("Penchants forts ⭐", value=False)
    only_conf = f[2].toggle("Data fiable seulement", value=False)

    rows = []
    for r in covered:
        if r["league"] not in sel_leagues:
            continue
        if only_conf and not r["conf"]:
            continue
        is_strong = r.get("strong", False)
        if only_strong and not is_strong:
            continue
        fav, p_fav, und, _p_und, elo_fav, elo_und = _fav(r)
        rows.append({
            "⭐": "⭐" if is_strong else ("🌪️" if not r.get("reliable", True) else ""),
            "Quand (Paris)": r["when"],
            "Ligue": r["league"],
            "Match": f"{r['team1']} vs {r['team2']}",
            "BO": f"BO{r['bestof']}",
            "Notre favori": fav,
            "P(favori)": round(p_fav * 100),
            "Fiab. ligue": round(r.get("rel", 0) * 100),
            "Elo (fav / autre)": f"{elo_fav:.0f} / {elo_und:.0f}",
            "X-ligue": "⚠️" if r["xleague"] else "",
            "Data": "✅" if r["conf"] else f"⚠️ {min(r['n1'], r['n2'])}g",
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df, width="stretch", hide_index=True,
            column_config={
                "⭐": st.column_config.TextColumn(width="small"),
                "P(favori)": st.column_config.ProgressColumn(
                    "P(favori)", help="Proba calibrée de notre favori",
                    format="%d%%", min_value=0, max_value=100),
                "Fiab. ligue": st.column_config.ProgressColumn(
                    "Fiab. ligue", help="Accuracy historique du modèle dans cette ligue "
                    "(≥62 % = fiable, sinon 🌪️ chaotique)",
                    format="%d%%", min_value=0, max_value=100),
            },
        )
    else:
        st.info("Aucun match avec ces filtres.")

    _value_calculator(covered)

    if uncovered:
        with st.expander(f"⚠️ {len(uncovered)} matchs non couverts (équipes hors data)"):
            seen = set()
            for m in uncovered:
                key = (m["team1"], m["team2"])
                if key in seen:
                    continue
                seen.add(key)
                st.write(f"- {m.get('overview', '?')} · **{m['team1']}** vs **{m['team2']}**")
            st.caption("Causes : nom ≠ Oracle (suffixe sponsor…), ligue non collectée, ou event tiers (EWC).")


def _value_calculator(covered: list[dict]) -> None:
    st.divider()
    st.subheader("🧮 Calculateur de value (saisis les cotes du book)")
    if not covered:
        return
    labels = [f"{r['when']} · {r['team1']} vs {r['team2']} (BO{r['bestof']})" for r in covered]
    i = st.selectbox("Match", range(len(covered)), format_func=lambda j: labels[j])
    r = covered[i]

    if not r.get("reliable", True):
        st.warning(
            f"🌪️ **Ligue chaotique** ({r.get('league_code', r['league'])} ≈ {r.get('rel', 0)*100:.0f}% "
            "d'accuracy historique) : même nos « favoris » y gagnent à peine plus que pile/face. "
            "La proba est déjà **aplatie** (calibration), mais reste prudent — c'est là que le book "
            "se trompe (value possible) MAIS aussi là qu'on se trompe le plus. **Value franche only.**"
        )
    if r.get("xleague"):
        st.warning(
            f"⚠️ Matchup **cross-ligue** ({r['league_a']} vs {r['league_b']}) : les Elo sont peu "
            "comparables → fiabilité réduite. C'est le piège **KCB 53 % vs PCIFIC** (book @1.02 avait "
            "raison). N'y vois une value que sur un **gros** désaccord + cote équilibrée (jamais < 1,2)."
        )

    c = st.columns(2)
    o1 = c[0].number_input(f"Cote {r['team1']}", min_value=1.01, max_value=51.0, value=2.00, step=0.05)
    o2 = c[1].number_input(f"Cote {r['team2']}", min_value=1.01, max_value=51.0, value=2.00, step=0.05)

    inv1, inv2 = 1 / o1, 1 / o2
    book1 = inv1 / (inv1 + inv2)
    for side, team, p_model, o, book in (
        ("1", r["team1"], r["p1"], o1, book1),
        ("2", r["team2"], r["p2"], o2, 1 - book1),
    ):
        edge = p_model - 1 / o  # EV vs cote brute
        cols = st.columns([2, 1, 1, 1, 2])
        cols[0].markdown(f"**{team}**")
        cols[1].metric("Notre proba", f"{p_model*100:.0f}%")
        cols[2].metric("Book (dévig.)", f"{book*100:.0f}%")
        cols[3].metric("Edge (EV)", f"{edge*100:+.0f}%")
        # Verdict (règles du journal)
        if o < 1.20:
            verdict = "⛔ favori court — on ne fade jamais (piège)"
        elif edge > 0.03:
            verdict = "✅ VALUE — pari défendable (poser tôt !)"
        elif edge > 0:
            verdict = "🟡 léger +EV — couvre à peine la vig, prudence"
        else:
            verdict = "❌ −EV — pas de pari"
        cols[4].markdown(f"<div style='margin-top:.6rem'>{verdict}</div>", unsafe_allow_html=True)

    st.caption(
        "Edge = notre proba − (1 / cote). Value si > +3 %. ⚠️ Elo-only (sans draft) : "
        "sur une ligue molle où le book sur-cote un favori après une grosse game, l'écart est exploitable ; "
        "sinon le book est souvent juste. La **draft en live** reste le complément le plus fort."
    )


if __name__ == "__main__":
    main()

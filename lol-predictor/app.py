"""Front Streamlit — ACCUEIL : matchs LoL à venir (toutes équipes) + notre proba Elo.

Page principale du projet. Affiche le calendrier des prochains jours (API lolesports)
avec NOTRE probabilité (Elo toutes-ligues), pour repérer à l'avance un favori que le
book va peut-être sur-coter (pattern KC / VKS / Heretics). Bouton 1-clic pour
rafraîchir la data (Google Drive) et le calendrier.

Lancer :
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import os
import datetime as dt

import pandas as pd
import streamlit as st

from src.ingest.load_oracle import ROOT, load_config

st.set_page_config(page_title="LoL — Matchs à venir", page_icon="📅", layout="wide")

# Déploiement cloud : expose les secrets Streamlit via os.environ (load_key lit l'env).
try:
    for _k in ("ANTHROPIC_API_KEY", "ODDS_API_KEY"):
        if not os.environ.get(_k) and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass


@st.cache_data(ttl=1800, show_spinner="Calcul Elo + récupération des matchs (lolesports)...")
def load_watchlist(days: int):
    from src.update.watchlist import build_rows
    return build_rows(days)


@st.cache_data(ttl=900, show_spinner="Recherche des marchés Polymarket (1re fois ~20 s)...")
def load_polymarket(covered: list[dict]) -> dict:
    """Cotes Polymarket par match : {(team1, team2, datetime): {value_team, edge, url, actionable}}."""
    from src.update.polymarket import enrich
    out = {}
    for r in enrich(covered):
        if "pm" in r:
            out[(r["team1"], r["team2"], r.get("datetime", ""))] = {
                "value_team": r["value_team"], "edge": r["edge"],
                "our_p": r["our_p"], "mkt_p": r["mkt_p"],
                "url": r["pm"]["url"], "actionable": r.get("actionable", False),
            }
    return out


@st.cache_data(ttl=900, show_spinner="Scan des cotes book (odds-api.io)…")
def load_oddsapi(days: int) -> list[dict]:
    """Matchs LoL cotés chez les books (odds-api.io) enrichis de notre proba + edge."""
    from src.update.oddsapi import load_key, scan
    if not load_key():
        return []
    try:
        return scan(days=days)
    except Exception:  # noqa: BLE001
        return []


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
    f = st.columns([2.6, 1.2, 1.0, 1.2, 1.4])
    sel_leagues = f[0].multiselect("Ligues", leagues, default=leagues)
    only_pick = f[1].toggle("🎯 Picks 75 %", value=False,
                            help="Règle mesurée : ligue fiable + data ≥15 g + favori ≥65 % "
                                 "(hors cross-ligue) → ~81 % de bons vainqueurs (n≈1 170, walk-forward)")
    only_strong = f[2].toggle("⭐ forts", value=False)
    only_conf = f[3].toggle("Data fiable", value=False)
    pm_on = f[4].toggle("Cotes Polymarket", value=True,
                        help="Croise avec les marchés Polymarket (matchs réellement pariables)")

    pm_map = {}
    if pm_on:
        try:
            pm_map = load_polymarket(covered)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Polymarket indisponible ({exc}) — tableau sans cotes.")

    rows = []
    for r in covered:
        if r["league"] not in sel_leagues:
            continue
        if only_conf and not r["conf"]:
            continue
        fav, p_fav, und, _p_und, elo_fav, elo_und = _fav(r)
        is_strong = r.get("strong", False)
        is_pick = (r.get("reliable", False) and r["conf"]
                   and not r["xleague"] and p_fav >= 0.65)
        if only_pick and not is_pick:
            continue
        if only_strong and not is_strong:
            continue
        sig = "🎯" if is_pick else ("⭐" if is_strong else ("🌪️" if not r.get("reliable", True) else ""))
        row = {
            "Signal": sig,
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
        }
        if pm_on:
            pm = pm_map.get((r["team1"], r["team2"], r.get("datetime", "")))
            if pm:
                row["Value PM"] = f"{pm['value_team']} {pm['edge']*100:+.1f} pts"
                row["PM"] = "✅" if pm.get("actionable") else "—"
                row["Lien PM"] = pm["url"]
            else:
                row["Value PM"], row["PM"], row["Lien PM"] = "non coté", "", None
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df, width="stretch", hide_index=True,
            column_config={
                "Signal": st.column_config.TextColumn(width="small"),
                "P(favori)": st.column_config.ProgressColumn(
                    "P(favori)", help="Proba calibrée de notre favori",
                    format="%d%%", min_value=0, max_value=100),
                "Fiab. ligue": st.column_config.ProgressColumn(
                    "Fiab. ligue", help="Accuracy historique du modèle dans cette ligue "
                    "(≥62 % = fiable, sinon 🌪️ chaotique)",
                    format="%d%%", min_value=0, max_value=100),
                "Lien PM": st.column_config.LinkColumn("Lien PM", display_text="ouvrir"),
            },
        )
        st.caption(
            "🎯 = **règle 75 %** (mesurée) : ligue fiable + data ≥15 g + favori ≥65 % hors cross-ligue "
            "→ **~81 %** de bons vainqueurs historiquement (1 174 games walk-forward). "
            "**Value PM** = côté où notre proba dépasse le plus le prix Polymarket ; "
            "✅ = edge ≥4 pts ET ligue fiable ET data OK (cf. `WATCHLIST_PARIABLE.md`)."
        )
    else:
        st.info("Aucun match avec ces filtres.")

    _oddsapi_section(days)
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


def _oddsapi_section(days: int) -> None:
    """Cotes book réelles (odds-api.io) croisées avec notre modèle."""
    from src.update.oddsapi import BOOKMAKERS, load_key
    st.divider()
    with st.container(border=True):
        st.markdown("### 💰 Cotés chez les books (odds-api.io)")
        if not load_key():
            st.info("Pas de clé odds-api.io (fichier `oddsapi.key`). Section désactivée.")
            return
        st.caption(
            f"Couverture LoL **limitée** (books : {', '.join(BOOKMAKERS)}) — surtout EM / LCS / "
            "Asia Masters / VCS. Complément de Polymarket. **✅ value actionnable** = edge ≥4 pts "
            "**ET** ligue fiable **ET** data ≥15 g **ET** pas cross-ligue. 🌪️ chaos / ⚠️ x-ligue "
            "= **à ignorer** (le book y a presque toujours raison, cf. backtest EM)."
        )
        try:
            rows = load_oddsapi(days)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"odds-api.io indisponible ({exc}).")
            return
        if not rows:
            st.info("Aucun match LoL coté chez ces books en ce moment.")
            return

        data = []
        for r in rows:
            if not r.get("resolved"):
                data.append({"Signal": "", "Quand (Paris)": r["when"], "Ligue": r["league"],
                             "Match": f"{r['home']} vs {r['away']}", "BO": f"BO{r['bestof']}",
                             "Value": "hors data Elo", "Notre p": None, "Marché": None,
                             "Edge (pts)": None, "Confiance": "—"})
                continue
            if r.get("trust"):
                conf = "✅ fiable"
            elif not r.get("reliable", True):
                conf = "🌪️ chaos"
            elif r.get("xleague"):
                conf = "⚠️ x-ligue"
            else:
                conf = f"⚠️ {min(r['n1'], r['n2'])}g"
            sig = ("✅" if r.get("actionable")
                   else ("🌪️" if not r.get("reliable", True)
                         else ("⚠️x" if r.get("xleague") else "")))
            data.append({
                "Signal": sig, "Quand (Paris)": r["when"], "Ligue": r["league"],
                "Match": f"{r['team1']} vs {r['team2']}", "BO": f"BO{r['bestof']}",
                "Value": (f"{r['value_team']} @{r['best_odd']:.2f}" if r.get("has_value") else "—"),
                "Notre p": round((r.get("our_p") or 0) * 100),
                "Marché": round((r.get("mkt_p") or 0) * 100),
                "Edge (pts)": round(r["edge"] * 100, 1) if r.get("edge") is not None else None,
                "Confiance": conf,
            })
        st.dataframe(
            pd.DataFrame(data), hide_index=True, width="stretch",
            column_config={
                "Notre p": st.column_config.NumberColumn("Notre p", format="%d%%"),
                "Marché": st.column_config.NumberColumn("Marché", format="%d%%"),
            },
        )
        act = [r for r in rows if r.get("actionable")]
        if act:
            for r in act:
                st.success(
                    f"✅ **{r['value_team']} @{r['best_odd']:.2f}** · {r['when']} · {r['league']} "
                    f"— notre {r['our_p']*100:.0f}% vs marché {r['mkt_p']*100:.0f}% "
                    f"(edge **{r['edge']*100:+.1f} pts**). Pose tôt."
                )
        else:
            st.caption(
                "ℹ️ Aucune **value actionnable** aujourd'hui — cohérent avec la discipline : "
                "les matchs cotés sont en EM/Asia (chaos/cross-région) où le book est juste. "
                "Le vrai usage ici = le **live in-map** (page « Série en cours »)."
            )


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

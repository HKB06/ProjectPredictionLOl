"""Front Streamlit — prédicteur de match LoL (LCK) avec prise en compte de la DRAFT.

Saisie : 2 équipes + leurs 5 champions (par rôle). Sortie : probabilités par marché
(vainqueur, first blood/tower/dragon, total kills, durée), via le MatchPredictor.

Lancer :
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src.features.build_features import ROLES
from src.models.predict import MatchPredictor

ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
NONE_OPT = "— (aucun)"

st.set_page_config(page_title="LoL Predictor — Draft", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Entraînement du modèle (une fois)...")
def get_predictor() -> MatchPredictor:
    return MatchPredictor().fit()


def champ_select(side: str, role: str, options: list[str]) -> str | None:
    label = ROLE_LABELS[role]
    val = st.selectbox(label, options, key=f"{side}_{role}", index=0)
    return None if val == NONE_OPT else val


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    mp = get_predictor()
    champ_options = [NONE_OPT] + mp.champions

    st.title("🎯 LoL Predictor — prédiction de match avec draft")
    st.caption(
        "Modèle (LCK 2026, sans fuite de données) : régression logistique régularisée + "
        "priors de draft (winrate champion sur ~5 500 games pro). "
        "Validation held-out (rolling-origin, ~200 matchs) : **AUC 0,74 / ~71 %** au vainqueur."
    )

    with st.sidebar:
        st.header("Paramètres")
        fmt = st.radio("Format", ["1 game (BO1)", "BO3 (gagne 2)", "BO5 (gagne 3)"], index=0)
        wins_needed = {"1 game (BO1)": 1, "BO3 (gagne 2)": 2, "BO5 (gagne 3)": 3}[fmt]
        is_playoffs = st.toggle("Match de playoffs", value=False)
        st.divider()
        st.subheader("Honnêteté du modèle")
        st.markdown(
            "- **Vainqueur / First tower** : signal réel.\n"
            "- **First blood / dragon** : quasi pile/face (≈ 50 %).\n"
            "- **Total kills / durée** : faibles en pré-game (à confirmer en V2 live).\n\n"
            "La draft apporte un gain **modeste mais réel** (+2-3 pts) ; la force "
            "d'équipe (Elo/forme) reste le facteur dominant."
        )

    col_blue, col_mid, col_red = st.columns([5, 1, 5])

    with col_blue:
        st.subheader("🔵 Équipe BLEUE")
        blue_team = st.selectbox("Équipe", mp.teams, key="blue_team", index=0)
        blue_champs = {role: champ_select("blue", role, champ_options) for role in ROLES}

    with col_mid:
        st.markdown("<h2 style='text-align:center;margin-top:2.2rem;'>VS</h2>",
                    unsafe_allow_html=True)

    with col_red:
        st.subheader("🔴 Équipe ROUGE")
        red_idx = 1 if len(mp.teams) > 1 else 0
        red_team = st.selectbox("Équipe", mp.teams, key="red_team", index=red_idx)
        red_champs = {role: champ_select("red", role, champ_options) for role in ROLES}

    st.divider()
    go = st.button("Prédire le match", type="primary", use_container_width=True)

    if not go:
        st.info("Choisis les 2 équipes et (optionnel) les champions par rôle, puis clique sur **Prédire**.")
        return

    if blue_team == red_team:
        st.error("Choisis deux équipes différentes.")
        return

    # Règle LoL : un champion ne peut être pris qu'UNE fois dans toute la partie
    # (les 2 équipes confondues). On bloque la prédiction si doublon.
    picked = [c for c in list(blue_champs.values()) + list(red_champs.values()) if c]
    dupes = sorted({c for c in picked if picked.count(c) > 1})
    if dupes:
        st.error(
            f"Champion(s) en double : **{', '.join(dupes)}**. "
            "Un champion ne peut être choisi qu'une seule fois dans toute la partie "
            "(les 2 équipes confondues). Corrige la draft."
        )
        return

    res = mp.predict_match(blue_team, red_team, blue_champs, red_champs,
                           is_playoffs=int(is_playoffs))
    p_blue = res["winner"]["blue"]

    st.subheader("Vainqueur — 1 game (la map où 🔵 est côté bleu)")
    c1, c2 = st.columns(2)
    c1.metric(f"🔵 {blue_team}", pct(p_blue))
    c2.metric(f"🔴 {red_team}", pct(1 - p_blue))
    st.progress(p_blue, text=f"Probabilité {blue_team} (1 game) : {pct(p_blue)}")

    if wins_needed > 1:
        s = mp.predict_series(blue_team, red_team, blue_champs, red_champs,
                              wins_needed=wins_needed, is_playoffs=int(is_playoffs))
        bo = "BO3" if wins_needed == 2 else "BO5"
        st.subheader(f"Vainqueur de la SÉRIE ({bo})")
        d1, d2 = st.columns(2)
        d1.metric(f"🔵 {blue_team}", pct(s["series_blue"]))
        d2.metric(f"🔴 {red_team}", pct(s["series_red"]))
        st.progress(s["series_blue"], text=f"Probabilité {blue_team} ({bo}) : {pct(s['series_blue'])}")
        st.caption(
            f"Par game (side-neutre) : {blue_team} {pct(s['p_neutral'])} "
            f"[bleu {pct(s['p_on_blue'])} / rouge {pct(s['p_on_red'])}]. "
            f"Série = conversion BO (games indépendantes). La série amplifie le favori : "
            f"un favori à 65 %/game gagne un BO5 ~73 %."
        )

    # Transparence draft (esprit DraftGap) : winrate moyen des compos
    bcw = mp._comp_wr(blue_champs)
    rcw = mp._comp_wr(red_champs)
    st.caption(
        f"Winrate moyen des champions (priors pro) — 🔵 {pct(bcw)} vs 🔴 {pct(rcw)} "
        f"(Δ = {(bcw - rcw) * 100:+.1f} pts). Neutre (50 %) si aucun champion choisi."
    )

    st.subheader("Autres marchés (du point de vue 🔵)")
    m = res["markets"]
    g1, g2, g3 = st.columns(3)
    g1.metric("First blood (bleu)", pct(m.get("First blood", 0.5)))
    g2.metric("First tower (bleu)", pct(m.get("First tower", 0.5)))
    g3.metric("First dragon (bleu)", pct(m.get("First dragon", 0.5)))

    h1, h2 = st.columns(2)
    h1.metric("Total kills (prévu)", f"{m.get('Total kills', float('nan')):.1f}")
    h2.metric("Durée (min, prévue)", f"{m.get('Durée (min)', float('nan')):.1f}")

    st.caption(
        "Rappel : ce sont des probabilités à comparer aux cotes pour chercher de la "
        "valeur — pas une garantie. First blood/dragon ≈ aléatoire."
    )


if __name__ == "__main__":
    main()

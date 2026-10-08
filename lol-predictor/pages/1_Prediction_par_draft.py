"""Page secondaire — prédicteur de match LoL avec prise en compte de la DRAFT.

UI façon DraftGap : score des deux côtés en direct, slots de picks avec icônes
(Data Dragon, CDN officiel Riot), liste centrale des champions classée par
« P(win) de ton côté si tu prends ce champion » (modèle + priors pro).
Clic sur une ligne -> remplit le prochain slot vide du côté actif.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.features.build_features import ROLES
from src.models.predict import MatchPredictor

ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
NONE_OPT = "— (aucun)"
DD_BASE = "https://ddragon.leagueoflegends.com"

st.set_page_config(page_title="Prédiction draft", page_icon="🎯", layout="wide")

from src.ui.auth import require_password  # noqa: E402 - doit suivre set_page_config

require_password()


@st.cache_resource(show_spinner="Entraînement du modèle (une fois)...")
def get_predictor() -> MatchPredictor:
    return MatchPredictor().fit()


# ----------------------------------------------------- icônes Data Dragon (Riot)
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


@st.cache_data(ttl=86_400, show_spinner=False)
def ddragon_icons() -> dict[str, str]:
    """nom normalisé -> URL de l'icône officielle (couvre nom affiché ET id interne).

    Gère les pièges du type Wukong (id MonkeyKing) ou Nunu & Willump (id Nunu).
    """
    import requests
    try:
        ver = requests.get(f"{DD_BASE}/api/versions.json", timeout=10).json()[0]
        data = requests.get(f"{DD_BASE}/cdn/{ver}/data/en_US/champion.json",
                            timeout=10).json()["data"]
    except Exception:
        return {}
    out: dict[str, str] = {}
    for cid, meta in data.items():
        url = f"{DD_BASE}/cdn/{ver}/img/champion/{cid}.png"
        out[_norm(meta["name"])] = url
        out[_norm(cid)] = url
    return out


def icon_url(champ: str | None) -> str:
    if not champ or champ == NONE_OPT:
        return ""
    return ddragon_icons().get(_norm(champ), "")


# ------------------------------------------------- état de draft (session_state)
def picked_champs(side: str) -> dict[str, str | None]:
    """Picks actuels d'un côté, lus depuis le session_state (source de vérité)."""
    out = {}
    for role in ROLES:
        v = st.session_state.get(f"{side}_{role}", NONE_OPT)
        out[role] = None if v == NONE_OPT else v
    return out


def all_picked() -> set[str]:
    return {c for side in ("blue", "red") for c in picked_champs(side).values() if c}


def _reset_draft() -> None:
    for side in ("blue", "red"):
        for role in ROLES:
            st.session_state[f"{side}_{role}"] = NONE_OPT


def _handle_table_click() -> None:
    """Applique le clic sur la liste centrale AVANT d'instancier les widgets.

    Le st.dataframe (key=champ_table) garde sa sélection dans le session_state ;
    on la lit ici pour remplir le prochain slot vide du côté actif, avec un jeton
    anti-répétition (sinon la sélection persistante re-remplirait à chaque rerun).
    """
    ev = st.session_state.get("champ_table")
    rows = getattr(getattr(ev, "selection", None), "rows", None) if ev else None
    if not rows:
        return
    names = st.session_state.get("_table_names", [])
    ridx = rows[0]
    if not (0 <= ridx < len(names)):
        return
    token = (tuple(names), ridx)
    if st.session_state.get("_last_click") == token:
        return
    st.session_state["_last_click"] = token
    champ = names[ridx]
    if champ in all_picked():
        st.toast(f"{champ} est déjà dans la draft.", icon="⚠️")
        return
    side = "blue" if st.session_state.get("fill_side", "🔵 Bleu").startswith("🔵") else "red"
    for role in ROLES:
        key = f"{side}_{role}"
        if st.session_state.get(key, NONE_OPT) == NONE_OPT:
            st.session_state[key] = champ
            st.toast(f"{champ} -> {ROLE_LABELS[role]} ({'bleu' if side == 'blue' else 'rouge'})",
                     icon="✅")
            return
    st.toast("Draft complète de ce côté — libère un slot d'abord.", icon="ℹ️")


# --------------------------------------------------------- suggestions (modèle)
def suggestion_frame(mp: MatchPredictor, blue: str, red: str, side: str,
                     is_playoffs: int, search: str) -> pd.DataFrame:
    """Un rang par champion dispo : winrate pro (prior) + P(win) du côté actif si pické.

    Vectorisé : UNE passe predict_proba sur ~170 variantes de la ligne de features
    (seules blue/red_champ_wr et d_champ_wr changent) -> instantané à l'écran.
    """
    bc, rc = picked_champs("blue"), picked_champs("red")
    taken = all_picked()
    cands = [c for c in mp.champions if c not in taken]
    if search:
        s = _norm(search)
        cands = [c for c in cands if s in _norm(c)]
    if not cands:
        return pd.DataFrame(columns=["icon", "champion", "wr", "p"])

    base = mp._feature_row(blue, red, bc, rc, is_playoffs)
    wr_cand = np.array([mp.champ_idx.asof(c, mp.asof_date) for c in cands])
    cur = [mp.champ_idx.asof(c, mp.asof_date)
           for c in picked_champs(side).values() if c]

    X = pd.concat([base] * len(cands), ignore_index=True)
    new_wr = (sum(cur) + wr_cand) / (len(cur) + 1)
    col = "blue_champ_wr" if side == "blue" else "red_champ_wr"
    X[col] = new_wr
    X["d_champ_wr"] = X["blue_champ_wr"] - X["red_champ_wr"]
    p_blue = mp.bin_models["y_winner"].predict_proba(X[mp.fcols])[:, 1]
    p_side = p_blue if side == "blue" else 1.0 - p_blue

    df = pd.DataFrame({
        "icon": [icon_url(c) for c in cands],
        "champion": cands,
        "wr": wr_cand * 100.0,
        "p": p_side * 100.0,
    })
    return df.sort_values("p", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------- slots
def pick_slot(side: str, role: str, options: list[str]) -> None:
    c_ico, c_sel = st.columns([1, 5])
    val = st.session_state.get(f"{side}_{role}", NONE_OPT)
    with c_ico:
        url = icon_url(val)
        if url:
            st.image(url, width=40)
        else:
            st.markdown(
                "<div style='width:40px;height:40px;border:1px dashed #666;"
                "border-radius:6px;'></div>", unsafe_allow_html=True)
    with c_sel:
        st.selectbox(ROLE_LABELS[role], options, key=f"{side}_{role}",
                     label_visibility="collapsed",
                     help=f"{ROLE_LABELS[role]} — {'bleu' if side == 'blue' else 'rouge'}")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ------------------------------------------------------------------------ page
def main() -> None:
    mp = get_predictor()
    champ_options = [NONE_OPT] + mp.champions
    _handle_table_click()                      # AVANT tout widget de draft

    st.title("🎯 Prédiction de match avec draft")
    st.caption(
        "Modèle (ligues du scope, sans fuite de données) : régression logistique régularisée + "
        "priors de draft (winrate champion). Validation held-out : **AUC ~0,74**. "
        "Icônes : Data Dragon (CDN officiel Riot). ⚠️ Couvre uniquement les **ligues du scope**."
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
            "- **Total kills / durée** : faibles en pré-game.\n\n"
            "La draft apporte un gain **modeste mais réel** (+2-3 pts) ; la force "
            "d'équipe (Elo/forme) reste le facteur dominant."
        )

    # ------------------------------------------------ équipes + score en direct
    t1, tvs, t2 = st.columns([5, 1, 5])
    with t1:
        blue_team = st.selectbox("🔵 Équipe bleue", mp.teams, key="blue_team", index=0)
    with tvs:
        st.markdown("<h3 style='text-align:center;margin-top:1.6rem;'>VS</h3>",
                    unsafe_allow_html=True)
    with t2:
        red_idx = 1 if len(mp.teams) > 1 else 0
        red_team = st.selectbox("🔴 Équipe rouge", mp.teams, key="red_team", index=red_idx)

    same_team = blue_team == red_team
    if same_team:
        st.error("Choisis deux équipes différentes.")

    bc, rc = picked_champs("blue"), picked_champs("red")
    if not same_team:
        live = mp.predict_match(blue_team, red_team, bc, rc, is_playoffs=int(is_playoffs))
        p_blue = live["winner"]["blue"]
        s1, s2 = st.columns(2)
        s1.metric(f"🔵 {blue_team}", pct(p_blue))
        s2.metric(f"🔴 {red_team}", pct(1 - p_blue))
        st.progress(p_blue, text=f"Score en direct (draft partielle prise en compte) — "
                                 f"{blue_team} : {pct(p_blue)}")

    # --------------------------------------------- 3 colonnes façon DraftGap
    col_blue, col_mid, col_red = st.columns([3, 4, 3])

    with col_blue:
        st.subheader("🔵 Picks bleus")
        for role in ROLES:
            pick_slot("blue", role, champ_options)

    with col_red:
        st.subheader("🔴 Picks rouges")
        for role in ROLES:
            pick_slot("red", role, champ_options)

    with col_mid:
        st.subheader("Champions")
        f1, f2 = st.columns([2, 3])
        with f1:
            st.radio("Prochain pick pour", ["🔵 Bleu", "🔴 Rouge"], key="fill_side",
                     horizontal=True)
        with f2:
            search = st.text_input("Recherche", key="champ_search",
                                   placeholder="Nom de champion…")
        side = "blue" if st.session_state.get("fill_side", "🔵 Bleu").startswith("🔵") else "red"

        if same_team:
            st.info("Choisis deux équipes différentes pour voir les suggestions.")
        else:
            sug = suggestion_frame(mp, blue_team, red_team, side,
                                   int(is_playoffs), search or "")
            st.session_state["_table_names"] = sug["champion"].tolist()
            st.dataframe(
                sug,
                key="champ_table",
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                height=430,
                column_config={
                    "icon": st.column_config.ImageColumn("", width="small"),
                    "champion": st.column_config.TextColumn("Champion"),
                    "wr": st.column_config.NumberColumn("WR pro", format="%.1f %%",
                                                        help="Winrate du champion en pro (priors, toutes ligues)"),
                    "p": st.column_config.NumberColumn(
                        f"P(win) {'🔵' if side == 'blue' else '🔴'}", format="%.1f %%",
                        help="Probabilité de victoire de ton côté si tu prends ce champion maintenant"),
                },
            )
            st.caption("Clique sur une ligne pour remplir le **prochain slot vide** du côté choisi. "
                       "Trié par impact modèle (équipes + draft actuelle).")
        st.button("♻️ Réinitialiser la draft", on_click=_reset_draft, width="stretch")

    st.divider()
    go = st.button("Prédire le match (tous marchés)", type="primary", width="stretch")
    if not go:
        st.info("Le score en haut suit la draft en direct. Clique sur **Prédire** pour le "
                "détail complet (série, first blood/tower/dragon, kills, durée).")
        return

    if same_team:
        return

    picked = [c for c in list(bc.values()) + list(rc.values()) if c]
    dupes = sorted({c for c in picked if picked.count(c) > 1})
    if dupes:
        st.error(
            f"Champion(s) en double : **{', '.join(dupes)}**. "
            "Un champion ne peut être choisi qu'une seule fois dans toute la partie. Corrige la draft."
        )
        return

    res = mp.predict_match(blue_team, red_team, bc, rc, is_playoffs=int(is_playoffs))
    p_blue = res["winner"]["blue"]

    st.subheader("Vainqueur — 1 game (la map où 🔵 est côté bleu)")
    c1, c2 = st.columns(2)
    c1.metric(f"🔵 {blue_team}", pct(p_blue))
    c2.metric(f"🔴 {red_team}", pct(1 - p_blue))
    st.progress(p_blue, text=f"Probabilité {blue_team} (1 game) : {pct(p_blue)}")

    if wins_needed > 1:
        s = mp.predict_series(blue_team, red_team, bc, rc,
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
            f"La série amplifie le favori : un favori à 65 %/game gagne un BO5 ~73 %."
        )

    bcw = mp._comp_wr(bc)
    rcw = mp._comp_wr(rc)
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

"""Page secondaire — prédicteur de match LoL avec prise en compte de la DRAFT.

UI minimaliste façon DraftGap : un scoreboard central, deux colonnes de picks,
une liste de champions cliquable.
Workflow : 1) clique un SLOT (rôle) -> il se surligne ; 2) clique un CHAMPION
dans la liste -> il va dans ce slot (remplace l'occupant s'il y en a un).
Icônes : Data Dragon (CDN officiel Riot), rien à héberger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.features.build_features import ROLES
from src.models.predict import MatchPredictor

ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
NONE_OPT = "— (aucun)"          # compat anciennes sessions (ex-selectbox)
BLUE, RED = "#3b82f6", "#ef4444"
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
    """nom normalisé -> URL d'icône (couvre nom affiché ET id interne Riot)."""
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
def get_pick(side: str, role: str) -> str | None:
    v = st.session_state.get(f"{side}_{role}")
    return None if (not v or v == NONE_OPT) else v


def picked_champs(side: str) -> dict[str, str | None]:
    return {role: get_pick(side, role) for role in ROLES}


def all_picked() -> set[str]:
    return {c for side in ("blue", "red") for c in picked_champs(side).values() if c}


def current_target() -> tuple[str, str] | None:
    """Slot cible : celui cliqué par l'utilisateur, sinon le 1er slot libre."""
    t = st.session_state.get("target")
    if t:
        return tuple(t)
    for side in ("blue", "red"):
        for role in ROLES:
            if get_pick(side, role) is None:
                return (side, role)
    return None


def _set_target(side: str, role: str) -> None:
    st.session_state["target"] = (side, role)


def _advance_target(side: str) -> None:
    """Après un pick : slot libre suivant du même côté, sinon l'autre côté."""
    for s in (side, "red" if side == "blue" else "blue"):
        for role in ROLES:
            if get_pick(s, role) is None:
                st.session_state["target"] = (s, role)
                return
    st.session_state.pop("target", None)


def _clear_slot(side: str, role: str) -> None:
    st.session_state[f"{side}_{role}"] = None
    st.session_state["target"] = (side, role)


def _reset_draft() -> None:
    for side in ("blue", "red"):
        for role in ROLES:
            st.session_state[f"{side}_{role}"] = None
    st.session_state.pop("target", None)
    st.session_state.pop("_last_click", None)


def _handle_table_click() -> None:
    """Applique le clic sur la liste AVANT de dessiner la page (jeton anti-répétition)."""
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
        return
    tgt = current_target()
    if tgt is None:
        st.toast("Draft complète — retire un champion (✕) ou efface tout.", icon="ℹ️")
        return
    side, role = tgt
    old = get_pick(side, role)
    st.session_state[f"{side}_{role}"] = champ
    label = f"{ROLE_LABELS[role]} {'bleu' if side == 'blue' else 'rouge'}"
    st.toast(f"{champ} remplace {old} ({label})" if old else f"{champ} → {label}", icon="✅")
    _advance_target(side)


# --------------------------------------------------------- suggestions (modèle)
def suggestion_frame(mp: MatchPredictor, blue: str, red: str,
                     tgt: tuple[str, str] | None,
                     is_playoffs: int, search: str) -> pd.DataFrame:
    """Un rang par champion dispo : P(win) du côté cible s'il est mis dans le slot cible.

    Si le slot cible est occupé, on simule le REMPLACEMENT (l'occupant sort du calcul).
    Vectorisé : une seule passe predict_proba sur ~170 variantes -> instantané.
    """
    side, role = tgt if tgt else ("blue", None)
    bc, rc = picked_champs("blue"), picked_champs("red")
    if role is not None:                       # remplacement : l'occupant sort
        (bc if side == "blue" else rc)[role] = None
    taken = {c for c in list(bc.values()) + list(rc.values()) if c}
    cands = [c for c in mp.champions if c not in taken]
    if search:
        s = _norm(search)
        cands = [c for c in cands if s in _norm(c)]
    if not cands:
        return pd.DataFrame(columns=["icon", "champion", "p"])

    base = mp._feature_row(blue, red, bc, rc, is_playoffs)
    wr_cand = np.array([mp.champ_idx.asof(c, mp.asof_date) for c in cands])
    cur = [mp.champ_idx.asof(c, mp.asof_date)
           for c in (bc if side == "blue" else rc).values() if c]

    X = pd.concat([base] * len(cands), ignore_index=True)
    col = "blue_champ_wr" if side == "blue" else "red_champ_wr"
    X[col] = (sum(cur) + wr_cand) / (len(cur) + 1)
    X["d_champ_wr"] = X["blue_champ_wr"] - X["red_champ_wr"]
    p_blue = mp.bin_models["y_winner"].predict_proba(X[mp.fcols])[:, 1]
    p_side = p_blue if side == "blue" else 1.0 - p_blue

    return (pd.DataFrame({"icon": [icon_url(c) for c in cands],
                          "champion": cands,
                          "p": p_side * 100.0})
            .sort_values("p", ascending=False).reset_index(drop=True))


# ------------------------------------------------------------------- rendu UI
def scoreboard(blue_team: str, red_team: str, p_blue: float) -> None:
    pb, pr = p_blue * 100.0, (1.0 - p_blue) * 100.0
    st.markdown(
        f"""
<div style="border:1px solid rgba(128,128,128,.25);border-radius:14px;
            padding:14px 20px;margin:2px 0 12px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;">
    <div>
      <div style="font-size:.85rem;opacity:.7;">🔵 {blue_team}</div>
      <div style="font-size:2.4rem;font-weight:800;color:{BLUE};line-height:1.1;">{pb:.1f}%</div>
    </div>
    <div style="font-size:1rem;opacity:.45;font-weight:700;align-self:center;">VS</div>
    <div style="text-align:right;">
      <div style="font-size:.85rem;opacity:.7;">{red_team} 🔴</div>
      <div style="font-size:2.4rem;font-weight:800;color:{RED};line-height:1.1;">{pr:.1f}%</div>
    </div>
  </div>
  <div style="height:8px;border-radius:4px;margin-top:10px;
              background:linear-gradient(90deg,{BLUE} {pb:.1f}%,{RED} {pb:.1f}%);"></div>
  <div style="font-size:.75rem;opacity:.55;margin-top:6px;">
    Score en direct — équipes + draft en cours (côté bleu sur cette map).
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def pick_slot(side: str, role: str, active: bool) -> None:
    color = BLUE if side == "blue" else RED
    champ = get_pick(side, role)
    c_ico, c_btn, c_rm = st.columns([1.1, 3.4, 0.8])
    with c_ico:
        url = icon_url(champ)
        if url:
            border = f"3px solid {color}" if active else f"2px solid {color}66"
            st.markdown(
                f"<img src='{url}' width='46' style='border-radius:10px;"
                f"border:{border};display:block;'/>",
                unsafe_allow_html=True)
        else:
            style = f"2px dashed {color}" if active else f"2px dashed {color}44"
            st.markdown(
                f"<div style='width:46px;height:46px;border:{style};"
                f"border-radius:10px;display:flex;align-items:center;justify-content:center;"
                f"font-size:.7rem;opacity:.7;'>{ROLE_LABELS[role][:3]}</div>",
                unsafe_allow_html=True)
    with c_btn:
        label = f"{ROLE_LABELS[role]} · {champ}" if champ else f"{ROLE_LABELS[role]} — libre"
        st.button(label, key=f"sel_{side}_{role}",
                  type="primary" if active else "secondary",
                  on_click=_set_target, args=(side, role), width="stretch",
                  help="Le prochain champion cliqué ira dans ce slot")
    with c_rm:
        if champ:
            st.button("✕", key=f"rm_{side}_{role}", on_click=_clear_slot,
                      args=(side, role), help=f"Retirer {champ}")


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ------------------------------------------------------------------------ page
def main() -> None:
    mp = get_predictor()
    _handle_table_click()                      # avant tout rendu

    st.markdown("### 🎯 Prédiction de match avec draft")

    with st.sidebar:
        st.header("Paramètres")
        fmt = st.radio("Format", ["1 game (BO1)", "BO3 (gagne 2)", "BO5 (gagne 3)"], index=0)
        wins_needed = {"1 game (BO1)": 1, "BO3 (gagne 2)": 2, "BO5 (gagne 3)": 3}[fmt]
        is_playoffs = st.toggle("Match de playoffs", value=False)
        st.divider()
        st.subheader("Honnêteté du modèle")
        st.markdown(
            "- **Vainqueur / First tower** : signal réel (AUC ~0,74).\n"
            "- **First blood / dragon** : quasi pile/face.\n"
            "- **Total kills / durée** : faibles en pré-game.\n\n"
            "La draft pèse **+2-3 pts** ; la force d'équipe (Elo/forme) domine. "
            "Ligues du scope uniquement (`config.yaml`)."
        )

    # ------------------------------------------------------- équipes + score
    t1, t2 = st.columns(2)
    with t1:
        blue_team = st.selectbox("🔵 Équipe bleue", mp.teams, key="blue_team", index=0)
    with t2:
        red_idx = 1 if len(mp.teams) > 1 else 0
        red_team = st.selectbox("🔴 Équipe rouge", mp.teams, key="red_team", index=red_idx)

    same_team = blue_team == red_team
    if same_team:
        st.error("Choisis deux équipes différentes.")
        return

    bc, rc = picked_champs("blue"), picked_champs("red")
    live = mp.predict_match(blue_team, red_team, bc, rc, is_playoffs=int(is_playoffs))
    scoreboard(blue_team, red_team, live["winner"]["blue"])

    # --------------------------------------------- picks + liste cliquable
    tgt = current_target()
    col_blue, col_mid, col_red = st.columns([3, 4.5, 3])

    with col_blue:
        st.markdown(f"<div style='color:{BLUE};font-weight:700;margin-bottom:6px;'>"
                    "PICKS BLEUS</div>", unsafe_allow_html=True)
        for role in ROLES:
            pick_slot("blue", role, active=(tgt == ("blue", role)))

    with col_red:
        st.markdown(f"<div style='color:{RED};font-weight:700;margin-bottom:6px;"
                    "text-align:right;'>PICKS ROUGES</div>", unsafe_allow_html=True)
        for role in ROLES:
            pick_slot("red", role, active=(tgt == ("red", role)))

    with col_mid:
        if tgt:
            side, role = tgt
            color = BLUE if side == "blue" else RED
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:4px 10px;margin-bottom:6px;"
                f"font-weight:600;'>Prochain pick → {ROLE_LABELS[role]} "
                f"{'🔵' if side == 'blue' else '🔴'}</div>", unsafe_allow_html=True)
        search = st.text_input("Recherche", key="champ_search",
                               placeholder="🔎 Chercher un champion…",
                               label_visibility="collapsed")

        sug = suggestion_frame(mp, blue_team, red_team, tgt, int(is_playoffs), search or "")
        st.session_state["_table_names"] = sug["champion"].tolist()
        side_icon = "🔵" if (tgt and tgt[0] == "blue") else "🔴" if tgt else ""
        st.dataframe(
            sug,
            key="champ_table",
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            height=460,
            column_config={
                "icon": st.column_config.ImageColumn("", width="small"),
                "champion": st.column_config.TextColumn("Champion", width="medium"),
                "p": st.column_config.NumberColumn(
                    f"P(win) {side_icon}", format="%.1f %%",
                    help="Probabilité de victoire de ce côté si ce champion prend le slot surligné"),
            },
        )
        st.caption("1) Clique un **slot** à gauche/droite · 2) clique un **champion** ici.")
        st.button("♻️ Tout effacer", on_click=_reset_draft, width="stretch")

    # ------------------------------------------------------------ détail complet
    st.divider()
    if not st.button("Prédire le match (tous marchés)", type="primary", width="stretch"):
        return

    res = mp.predict_match(blue_team, red_team, bc, rc, is_playoffs=int(is_playoffs))
    p_blue = res["winner"]["blue"]

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
            f"La série amplifie le favori."
        )

    bcw, rcw = mp._comp_wr(bc), mp._comp_wr(rc)
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
        "Probabilités à comparer aux cotes pour chercher de la valeur — pas une garantie. "
        "First blood/dragon ≈ aléatoire."
    )


if __name__ == "__main__":
    main()

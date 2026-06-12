"""Page Streamlit — SÉRIE EN COURS : proba conditionnelle BO3/BO5 + edge live entre les maps.

LE créneau validé par le journal (2/2 : KC @2.45, VKS @2.7) : entre deux maps d'un BO,
le marché série reste ouvert et le public/book sur-cote l'équipe qui vient de gagner
(biais de récence). Cette page recalcule NOTRE proba de série en fonction du score
et la compare aux cotes live -> value quand le book s'emballe.

Signal optionnel : penchant draft-only (ratings champions appris sur tout le CSV),
utile quand la draft de la map suivante est connue (AUC ~0.58-0.65 = partiel).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st

from src.ingest.load_oracle import ROOT, load_config
from src.update.elo import RELIABLE_ACC, calibrate, compute_elo, win_prob

st.set_page_config(page_title="LoL — Série en cours", page_icon="🔄", layout="wide")

EDGE_MIN = 0.04
NO_FADE_ODD = 1.20  # règle ferme du journal : jamais fader un favori plus court


@st.cache_data(ttl=1800, show_spinner="Calcul de l'état Elo…")
def elo_state():
    state = compute_elo(load_config())
    return (state["elo"], state["n"], state["league"],
            state["reliability"], state["shrink"], state["global_rel"])


@st.cache_resource(show_spinner="Apprentissage des ratings de champions (draft-only, une fois)…")
def draft_model():
    """Ratings champions (+1 bleu / -1 rouge) appris sur tout le CSV (cf. draft_predict)."""
    from sklearn.linear_model import LogisticRegression
    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["data"]["oracle_csv"], low_memory=False, dtype={"patch": "string"})
    df.columns = [c.strip() for c in df.columns]
    pos = df["position"].str.lower()
    teams, players = df[pos == "team"], df[pos != "team"].copy()
    tb = (teams[teams["side"].str.lower() == "blue"][["gameid", "result"]]
          .rename(columns={"result": "y"}).drop_duplicates("gameid"))
    players["side_l"] = players["side"].str.lower()
    champ = players.groupby(["gameid", "side_l"])["champion"].apply(list).unstack("side_l")
    m = tb.merge(champ, left_on="gameid", right_index=True).dropna(subset=["blue", "red"])
    m = m[m["blue"].apply(lambda x: isinstance(x, list) and len(x) >= 5)
          & m["red"].apply(lambda x: isinstance(x, list) and len(x) >= 5)]
    allc = sorted({c for lst in m["blue"] for c in lst} | {c for lst in m["red"] for c in lst})
    idx = {c: i for i, c in enumerate(allc)}
    X = np.zeros((len(m), len(allc)))
    for i, (b, r) in enumerate(zip(m["blue"], m["red"])):
        for c in b:
            X[i, idx[c]] += 1
        for c in r:
            X[i, idx[c]] -= 1
    clf = LogisticRegression(C=0.3, max_iter=3000).fit(X, m["y"].values)
    return clf, idx, allc, len(m)


def draft_lean(clf, idx, n_champs: int, champs_a: list[str], champs_b: list[str]) -> float:
    """P(A gagne) side-neutre, draft seule (moyenne A-bleu / A-rouge)."""
    def vec(blue, red):
        x = np.zeros((1, n_champs))
        for c in blue:
            if c in idx:
                x[0, idx[c]] += 1
        for c in red:
            if c in idx:
                x[0, idx[c]] -= 1
        return x
    p_blue = clf.predict_proba(vec(champs_a, champs_b))[0, 1]
    p_red = 1 - clf.predict_proba(vec(champs_b, champs_a))[0, 1]
    return float((p_blue + p_red) / 2)


@lru_cache(maxsize=None)
def race(p: float, na: int, nb: int) -> float:
    """P(A gagne la série) si A doit prendre na maps, B nb maps, p = P(A gagne 1 map)."""
    if na <= 0:
        return 1.0
    if nb <= 0:
        return 0.0
    return p * race(p, na - 1, nb) + (1 - p) * race(p, na, nb - 1)


def verdict(edge: float, odd: float, fading_short_fav: bool) -> str:
    if fading_short_fav:
        return "⛔ favori court en face — on ne fade jamais (règle paiN)"
    if edge >= EDGE_MIN:
        return "✅ VALUE — défendable (mise selon fiabilité ligue)"
    if edge > 0:
        return "🟡 léger +EV — couvre à peine la vig"
    return "❌ −EV — pas de pari"


def main() -> None:
    st.title("🔄 Série en cours — proba live entre les maps")
    st.caption(
        "**Le pattern 2/2 du journal** : entre deux maps, le book sur-cote l'équipe qui vient "
        "de gagner (biais de récence). On recalcule notre proba de série **conditionnelle au "
        "score** et on la compare aux cotes live. Rappels : jamais fader un favori < 1.20 ; "
        "ligue 🌪️ = value franche uniquement."
    )

    elo, n, league, reliability, shrink, global_rel = elo_state()
    teams = sorted(elo)

    c = st.columns([3, 3, 1.6, 1.2, 1.2])
    a = c[0].selectbox("Équipe A", teams, index=0)
    b = c[1].selectbox("Équipe B", teams, index=1 if len(teams) > 1 else 0)
    bo = c[2].radio("Format", ["BO3", "BO5"], index=1, horizontal=True)
    wneed = 2 if bo == "BO3" else 3
    wa = c[3].number_input(f"Maps {a.split()[0]}", 0, wneed, 0)
    wb = c[4].number_input(f"Maps {b.split()[0]}", 0, wneed, 0)

    if a == b:
        st.error("Choisis deux équipes différentes.")
        st.stop()
    if wa >= wneed or wb >= wneed:
        st.warning("La série est déjà finie avec ce score 😉. Mets le score AVANT la map à venir.")
        st.stop()

    # --- contexte ligue (fiabilité + calibration) ---
    lg_a, lg_b = league.get(a), league.get(b)
    cand = [c_ for c_ in (lg_a, lg_b) if c_ in reliability]
    code = min(cand, key=lambda c_: reliability[c_]) if cand else None
    rel = reliability.get(code, global_rel)
    shr = shrink.get(code, 1.0)
    xleague = lg_a != lg_b
    reliable = rel >= RELIABLE_ACC

    p_raw = win_prob(elo[a], elo[b])
    p_game = calibrate(p_raw, shr)

    # --- override manuel optionnel (sub, méta, info terrain) ---
    with st.expander("Ajuster notre proba par map (optionnel — sub, roster, info terrain)"):
        ov = st.toggle("Activer l'ajustement manuel", value=False)
        p_adj = st.slider("P(A gagne UNE map) %", 5, 95, int(round(p_game * 100))) / 100
    if ov:
        p_game = p_adj

    p_series_now = race(round(p_game, 4), wneed - int(wa), wneed - int(wb))
    p_series_start = race(round(p_game, 4), wneed, wneed)

    k = st.columns(4)
    k[0].metric(f"P({a}) / map", f"{p_game*100:.0f}%",
                help=f"Elo {elo[a]:.0f} vs {elo[b]:.0f}, calibré ligue {code or '?'}")
    k[1].metric(f"Série avant le match", f"{p_series_start*100:.0f}%")
    k[2].metric(f"Série MAINTENANT ({int(wa)}-{int(wb)})", f"{p_series_now*100:.0f}%",
                delta=f"{(p_series_now - p_series_start)*100:+.0f} pts")
    k[3].metric("Fiabilité ligue", f"{rel*100:.0f}%",
                help="accuracy historique du modèle dans cette ligue")

    warn = []
    if not reliable:
        warn.append(f"🌪️ **{code} chaotique** ({rel*100:.0f}%) : nos probas y valent peu — value franche only.")
    if xleague:
        warn.append(f"⚠️ **Cross-ligue** ({lg_a} vs {lg_b}) : Elo peu comparables (piège KCB/PCIFIC).")
    if min(n[a], n[b]) < 15:
        warn.append(f"⚠️ **Cold-start** : {min(n[a], n[b])} games seulement pour l'une des équipes.")
    for w in warn:
        st.warning(w)

    # --- cotes live série ---
    st.subheader("Cotes live — vainqueur de la SÉRIE")
    cc = st.columns([1.5, 1.5, 2.5])
    odd_a = cc[0].number_input(f"Cote {a}", 1.01, 51.0, 2.00, 0.05, key="odd_a")
    odd_b = cc[1].number_input(f"Cote {b}", 1.01, 51.0, 2.00, 0.05, key="odd_b")
    last_win = cc[2].radio("Qui vient de gagner la dernière map ?", ["— (début de série)", a, b],
                           horizontal=True)

    rows = [(a, p_series_now, odd_a, odd_b), (b, 1 - p_series_now, odd_b, odd_a)]
    for team, p_model, odd, odd_other in rows:
        edge = p_model - 1 / odd
        fading_short = odd_other < NO_FADE_ODD and edge >= EDGE_MIN
        cols = st.columns([2, 1, 1, 1, 3])
        cols[0].markdown(f"**{team}**")
        cols[1].metric("Nous", f"{p_model*100:.0f}%")
        cols[2].metric("Book (1/cote)", f"{(1/odd)*100:.0f}%")
        cols[3].metric("Edge", f"{edge*100:+.1f} pts")
        cols[4].markdown(f"<div style='margin-top:.6rem'>{verdict(edge, odd, fading_short)}</div>",
                         unsafe_allow_html=True)

    # --- le pattern momentum ---
    if last_win in (a, b):
        loser = b if last_win == a else a
        p_loser = (1 - p_series_now) if last_win == a else p_series_now
        odd_loser = odd_b if last_win == a else odd_a
        edge_loser = p_loser - 1 / odd_loser
        if edge_loser >= EDGE_MIN and odd_loser >= 1.7:
            st.success(
                f"🎯 **Pattern KC/VKS détecté** : le book s'emballe sur **{last_win}** (vient de "
                f"gagner) et sous-cote **{loser}** — notre edge sur {loser} : "
                f"**{edge_loser*100:+.1f} pts** @ {odd_loser:.2f}. C'est exactement le fade de "
                "récence qui a payé 2/2. Mise modérée, et **pose vite** (la fenêtre se referme à "
                "la draft suivante)."
            )
        else:
            st.info(
                f"Le book intègre la victoire de **{last_win}** sans s'emballer (pas d'edge ≥4 pts "
                "sur le perdant). Pas de fade ici."
            )

    # --- draft-only (optionnel) ---
    st.divider()
    with st.expander("🧪 Penchant DRAFT de la map suivante (optionnel, signal partiel)"):
        st.caption(
            "Ratings champions appris sur tout le CSV (logistic ±1). Signal **partiel** "
            "(AUC ~0.58-0.65) : à n'utiliser que comme **confirmation** d'un fade, jamais seul. "
            "C'est lui qui avait flairé VKS 64 % en map 3."
        )
        clf, idx, allc, n_games = draft_model()
        opts = [""] + allc
        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**Draft {a}**")
            champs_a = [st.selectbox(f"Pick {i+1} ({a})", opts, key=f"da{i}",
                                     label_visibility="collapsed") for i in range(5)]
        with cb:
            st.markdown(f"**Draft {b}**")
            champs_b = [st.selectbox(f"Pick {i+1} ({b})", opts, key=f"db{i}",
                                     label_visibility="collapsed") for i in range(5)]
        champs_a = [x for x in champs_a if x]
        champs_b = [x for x in champs_b if x]
        if champs_a and champs_b:
            lean = draft_lean(clf, idx, len(allc), champs_a, champs_b)
            d1, d2 = st.columns(2)
            d1.metric(f"Draft → {a}", f"{lean*100:.1f}%")
            d2.metric(f"Draft → {b}", f"{(1-lean)*100:.1f}%")
            agree = (lean >= 0.5) == (p_game >= 0.5)
            st.caption(
                ("✅ La draft **confirme** notre favori Elo." if agree else
                 "⚠️ La draft **contredit** notre favori Elo — c'est le profil des fades gagnants "
                 "(KC, VKS) **si** le book suit l'Elo/momentum.")
                + f" (ratings appris sur {n_games} games)"
            )


if __name__ == "__main__":
    main()

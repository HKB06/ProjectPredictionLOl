"""Blocs d'affichage communs aux pages du nouveau site."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.features.context_factors import FACTORS
from src.v2.external import EXT_FACTORS
from src.v2.extra import EXTRA_FACTORS
from src.v2.player_engine import SIGNALS

ROLE_LABEL = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "ADC", "sup": "Support"}
NEW = "Nouveau · v2 complet"
OLD = "Ancien · calibré par ligue"


def breakeven(p: float) -> float:
    """Cote minimale de rentabilité = 1 / proba."""
    return 1.0 / p if p and p > 0 else float("inf")


def matchup_detail(m: dict, a: str, b: str, bestof: int) -> None:
    """Détail d'une confrontation : toutes les probas, le pourquoi, les joueurs, les profils."""
    t_why, t_players, t_profile, t_models = st.tabs(
        ["Pourquoi ce prono", "Joueurs", "Profil des équipes", "Tous les modèles"])

    with t_why:
        contrib = m["contrib"].head(8)
        top = float(contrib.abs().max()) or 1.0
        why = pd.DataFrame({
            "Signal": contrib.index,
            "Avantage pour": [a if v >= 0 else b for v in contrib],
            "Poids": contrib.abs().to_numpy(),
        })
        st.dataframe(why, hide_index=True, width="stretch", column_config={
            "Poids": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=top)})
        st.caption("Les 8 signaux qui pèsent le plus dans le nouveau modèle, du plus fort au "
                   "plus faible, et l'équipe qu'ils avantagent.")

    with t_players:
        pa, pb = m["player"]
        rows = []
        for role, label in ROLE_LABEL.items():
            na, ea = pa["players"].get(role, ("?", np.nan))
            nb, eb = pb["players"].get(role, ("?", np.nan))
            rows.append({"Rôle": label, a: na, "Elo": ea, b: nb, "Elo ": eb})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Cinq attendu : le dernier cinq aligné, corrigé par l'effectif suivi sur "
                   "Leaguepedia. Elo individuel de chaque joueur ; la force de l'équipe est la "
                   "moyenne des 5. « inconnu » = titulaire parti sans remplaçant annoncé, "
                   "compté au niveau moyen de la ligue.")
        for team, r in zip((a, b), m.get("lineups") or ()):
            if r and r.get("changes"):
                st.info(f"**{team}** — effectif mis à jour : " + " · ".join(r["changes"]))
            elif r and not r.get("known"):
                st.caption(f"{team} : pas d'effectif suivi, dernier cinq aligné conservé.")

    with t_models:
        probs = pd.DataFrame({
            "Modèle": list(m["games"]),
            f"{a} par game %": [round(v * 100, 1) for v in m["games"].values()],
            f"{a} série BO{bestof} %": [round(v * 100, 1) for v in m["series"].values()],
        })
        order = [NEW, OLD] + [k for k in m["games"] if k not in (NEW, OLD)]
        st.dataframe(probs.set_index("Modèle").loc[order].reset_index(),
                     hide_index=True, width="stretch")

    with t_profile:
        (pa, pb), (ca, cb) = m["player"], m["ctx"]
        lines = []
        for key, (bloc, label) in SIGNALS.items():
            k = key[2:]
            lines.append({"Bloc": bloc, "Signal": label, a: round(pa[k], 3), b: round(pb[k], 3)})
        for k, label in FACTORS.items():
            va, vb = ca[k], cb[k]
            lines.append({"Bloc": "Contexte", "Signal": label,
                          a: round(va, 3) if va == va else None,
                          b: round(vb, 3) if vb == vb else None})
        ea, eb = m.get("ext") or ({}, {})
        for k, (bloc, label) in EXT_FACTORS.items():
            va, vb = ea.get(k, np.nan), eb.get(k, np.nan)
            lines.append({"Bloc": bloc, "Signal": label,
                          a: round(va, 3) if va == va else None,
                          b: round(vb, 3) if vb == vb else None})
        xa, xb = m.get("extra") or ({}, {})
        for k, (bloc, label) in EXTRA_FACTORS.items():
            if k in xa or k in xb:
                va, vb = xa.get(k, np.nan), xb.get(k, np.nan)
                lines.append({"Bloc": bloc, "Signal": label,
                              a: round(va, 3) if va == va else None,
                              b: round(vb, 3) if vb == vb else None})
        sa, sb = m.get("soloq") or ({}, {})
        for k, label in (("sq_vol7", "Parties soloQ sur 7 jours (moyenne par joueur)"),
                         ("sq_role14", "Parties soloQ à son poste sur 14 jours"),
                         ("sq_players", "Joueurs suivis en soloQ")):
            va, vb = sa.get(k, np.nan), sb.get(k, np.nan)
            lines.append({"Bloc": "SoloQ", "Signal": label,
                          a: round(va, 1) if va == va else None,
                          b: round(vb, 1) if vb == vb else None})
        h = m.get("h2h") or {}
        for k, label in (("h2h_team", "Face-à-face équipes (> 0 : avantage 1re équipe)"),
                         ("h2h_player", "Face-à-face joueurs, poste pour poste"),
                         ("h2h_lane", "Face-à-face : or à 15 min (milliers)")):
            if k in h:
                lines.append({"Bloc": "Face-à-face", "Signal": label, a: round(h[k], 3), b: None})
        staff = [f"**{t}** : {e['staff'][0]}" for t, e in ((a, ea), (b, eb)) if e.get("staff")]
        if staff:
            st.caption("Head coach (dernier tournoi connu, source Leaguepedia) — " + " · ".join(staff))
        st.dataframe(pd.DataFrame(lines), hide_index=True, width="stretch", height=420)

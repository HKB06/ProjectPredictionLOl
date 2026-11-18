"""Point d'entrée du site : une navigation simple, en haut de page.

  📅 Avant le BO  — les matchs à venir et le prono des deux modèles avant le match.
  🎯 Draft        — on saisit la draft des deux équipes, prono game par game.
  📊 Saison       — tous les matchs joués, leurs stats, et ce que les modèles avaient prédit.
  🧬 Modèles      — le nouveau modèle fait-il mieux que l'ancien ?
  👥 Effectifs    — roster actuel de chaque équipe, mercato, suivi soloQ des joueurs.

Les pages de l'ancien site restent accessibles dans le menu « Ancien site ».

Lancer :
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="LoL Predictor", page_icon="🎮", layout="wide")

from src.ui.auth import require_password  # noqa: E402 - doit suivre set_page_config

require_password()

# Déploiement cloud : expose les secrets Streamlit via os.environ (load_key lit l'env).
try:
    for _k in ("ANTHROPIC_API_KEY", "ODDS_API_KEY"):
        if not os.environ.get(_k) and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # noqa: BLE001 - aucun secret configuré en local
    pass

PAGES = {
    "": [
        st.Page("views/avant_bo.py", title="Avant le BO", icon="📅", default=True),
        st.Page("pages/1_Prediction_par_draft.py", title="Draft", icon="🎯"),
        st.Page("views/saison.py", title="Saison", icon="📊"),
        st.Page("pages/9_Effectifs.py", title="Effectifs", icon="👥"),
        st.Page("views/modeles.py", title="Modèles", icon="🧬"),
    ],
    "Ancien site": [
        st.Page("pages/0_Matchs_a_venir.py", title="Matchs à venir (Elo)", icon="📅"),
        st.Page("pages/2_Bilan_predictions.py", title="Bilan des prédictions", icon="✅"),
        st.Page("pages/3_Serie_en_cours.py", title="Série en cours", icon="🔄"),
        st.Page("pages/4_Journal_paris.py", title="Journal de paris", icon="💰"),
        st.Page("pages/5_Assistant_IA.py", title="Assistant IA", icon="🤖"),
        st.Page("pages/6_Rentabilite_picks.py", title="Rentabilité des picks", icon="💵"),
        st.Page("pages/7_Tickets.py", title="Tickets", icon="🎫"),
        st.Page("pages/8_Plan_de_jeu.py", title="Plan de jeu", icon="🧭"),
    ],
}

st.navigation(PAGES, position="top").run()

"""Verrou d'accès partagé pour l'app déployée.

Pourquoi : l'app publique utilise NOS clés API payantes — en particulier la page
Assistant IA, qui tape Claude Opus. Sans verrou, n'importe quel visiteur tombant
sur l'URL dépense notre crédit. Un mot de passe UNIQUE stocké dans les Secrets
règle le problème sans imposer de comptes : on le donne à qui on veut.

Configuration (Streamlit Cloud → Settings → Secrets) :

    APP_PASSWORD = "le-mot-de-passe-a-partager"

Si aucun mot de passe n'est configuré, l'app reste ouverte : le dev local n'est
jamais bloqué, et l'absence de secret ne casse pas le déploiement.
"""
from __future__ import annotations

import hmac
import os

import streamlit as st

STATE_KEY = "_auth_ok"


def expected_password() -> str | None:
    """APP_PASSWORD : variable d'env d'abord, puis Secrets Streamlit."""
    pwd = os.environ.get("APP_PASSWORD")
    if pwd:
        return pwd.strip()
    try:
        p = st.secrets.get("APP_PASSWORD")
        if p:
            return str(p).strip()
    except Exception:  # noqa: BLE001 - hors Streamlit ou aucun secret configuré
        pass
    return None


def require_password() -> None:
    """Bloque la page tant que le mot de passe n'est pas saisi.

    À appeler EN HAUT de CHAQUE page, juste après `set_page_config`. En multipage
    Streamlit chaque script tourne isolément : un verrou posé uniquement dans
    `app.py` se contournerait en tapant l'URL de la page directement.
    """
    expected = expected_password()
    if not expected or st.session_state.get(STATE_KEY):
        return

    st.title("🔒 Accès protégé")
    st.caption("Cette app tourne sur des clés API payantes. "
               "Demande le mot de passe à son auteur.")
    with st.form("auth_form"):
        typed = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Entrer", type="primary")
    if submitted:
        # compare_digest : évite de fuiter la longueur via le temps de réponse.
        if hmac.compare_digest(typed.encode("utf-8"), expected.encode("utf-8")):
            st.session_state[STATE_KEY] = True
            st.rerun()
        st.error("Mot de passe incorrect.")
    st.stop()

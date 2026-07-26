"""Page Streamlit — Assistant IA (Claude) branché sur nos données.

On donne le contexte d'un match (équipes, draft, infos gol.gg, captures de cotes)
et l'agent Claude interroge NOTRE modèle (Elo calibré, priors champion, forme,
fiabilité ligue) via des outils, puis rend une analyse + une proba argumentée.

Lancer (depuis lol-predictor/) :
    .\\venv\\Scripts\\python.exe -m streamlit run app.py
puis ouvrir la page « Assistant IA » dans la barre latérale.
"""
from __future__ import annotations

import base64
import os

import streamlit as st

from src.assistant.agent import DataContext, load_api_key, load_gemini_key

# Barre de chat multimodale (collage d'images Ctrl+V). Repli sur st.chat_input si absent.
try:
    from st_chat_input_multimodal import multimodal_chat_input
    _HAS_PASTE = True
except Exception:  # noqa: BLE001
    _HAS_PASTE = False

st.set_page_config(page_title="Assistant IA — LoL", page_icon="🤖", layout="wide")

# Déploiement cloud : expose les secrets Streamlit via os.environ (load_api_key lit l'env).
try:
    for _k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ODDS_API_KEY"):
        if not os.environ.get(_k) and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

ALLOWED_IMG = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}
GEMINI_MODELS = {
    "gemini-2.5-flash": "Gemini 2.5 Flash — gratuit, multimodal + rapide (recommandé)",
    "gemini-2.0-flash": "Gemini 2.0 Flash — gratuit, le plus rapide",
    "gemini-2.5-pro": "Gemini 2.5 Pro — gratuit*, le plus fin (quota free tier plus serré)",
}
ANTHROPIC_MODELS = {
    "claude-opus-4-8": "Opus 4.8 — le plus fin · ~$5/$25 par M tokens (payant)",
    "claude-sonnet-4-6": "Sonnet 4.6 — bon rapport vitesse/qualité (payant)",
    "claude-haiku-4-5": "Haiku 4.5 — le plus rapide/économique (payant)",
}
PROVIDERS = {"Gemini (Google · gratuit)": "gemini", "Anthropic (Claude · payant)": "anthropic"}


@st.cache_resource(show_spinner="Chargement du modèle (Elo + priors champion)…")
def get_ctx() -> DataContext:
    return DataContext()


def _compose(team_a, team_b, bestof, league, draft_b, draft_r, notes, question) -> str:
    lines = []
    if team_a and team_b:
        lines.append(f"**Match** : {team_a} vs {team_b} (BO{bestof})")
    elif team_a or team_b:
        lines.append(f"Équipe : {team_a or team_b} (BO{bestof})")
    if league:
        lines.append(f"Ligue / event : {league}")
    if draft_b:
        lines.append(f"Draft bleue : {draft_b}")
    if draft_r:
        lines.append(f"Draft rouge : {draft_r}")
    if notes:
        lines.append(f"Infos en plus : {notes}")
    q = (question or "").strip() or \
        "Analyse ce match et donne ta meilleure proba (game + série) + un verdict pari."
    lines.append("\n" + q)
    return "\n".join(lines)


def _encode_uploads(uploads) -> list[dict]:
    out = []
    for f in uploads or []:
        mt = (f.type or "").lower()
        if mt == "image/jpg":
            mt = "image/jpeg"
        if mt in ALLOWED_IMG:
            out.append({"media_type": mt, "bytes": f.getvalue()})
    return out


def _decode_datauris(files) -> list[dict]:
    """Convertit les images du chat multimodal (data-URL base64) -> {media_type, bytes}."""
    out = []
    for f in files or []:
        data = f.get("data", "") if isinstance(f, dict) else ""
        if not data:
            continue
        if "," in data:
            header, b64 = data.split(",", 1)
            mt = header.split(";")[0].replace("data:", "").strip().lower() or "image/png"
        else:
            b64, mt = data, "image/png"
        if mt == "image/jpg":
            mt = "image/jpeg"
        if mt in ALLOWED_IMG:
            try:
                out.append({"media_type": mt, "bytes": base64.b64decode(b64)})
            except Exception:  # noqa: BLE001
                pass
    return out


def _chat_bar() -> tuple[str, list[dict]] | None:
    """Barre de chat du bas : texte + collage d'images (Ctrl+V). Renvoie (texte, images) ou None."""
    ph = "Suivi — colle un screen (Ctrl+V) : cotes live, draft, gold@15…"
    types = ["png", "jpg", "jpeg", "webp", "gif"]
    if _HAS_PASTE and not st.session_state.get("_mm_broken"):
        try:
            res = multimodal_chat_input(placeholder=ph, accepted_file_types=types,
                                        key="assistant_mm_chat")
            if res and (res.get("text") or res.get("files")):
                return (res.get("text") or "", _decode_datauris(res.get("files")))
            return None
        except Exception:  # noqa: BLE001  composant indispo (ex. API Streamlit retirée) -> repli natif
            st.session_state["_mm_broken"] = True
    # Repli natif : clic + glisser-déposer (le Ctrl+V n'existe pas en st.chat_input natif).
    res = st.chat_input(ph, accept_file="multiple", file_type=types)
    if not res:
        return None
    text = getattr(res, "text", res if isinstance(res, str) else "") or ""
    return (text, _encode_uploads(getattr(res, "files", [])))


def _run_agent(user_text: str, images: list[dict], key: str, model: str,
               provider: str = "gemini") -> None:
    """Affiche le tour user + lance l'agent, journalise les outils, rend la réponse."""
    display = user_text
    if images:
        display += f"\n\n_({len(images)} capture(s) jointe(s))_"
    with st.chat_message("user"):
        st.markdown(display)
    st.session_state.chat.append({"role": "user", "content": display})

    try:
        if provider == "gemini":
            from src.assistant.agent import GeminiAssistant as _Agent
        else:
            from src.assistant.agent import Assistant as _Agent
    except ImportError:
        st.error("Le paquet `anthropic` n'est pas installé.\n\n"
                 "→ `.\\venv\\Scripts\\python.exe -m pip install anthropic`")
        return

    with st.chat_message("assistant"):
        try:
            agent = _Agent(api_key=key, model=model, ctx=get_ctx())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Initialisation impossible : {exc}")
            return

        with st.status("L'agent interroge nos données…", expanded=True) as status:
            def on_event(kind, payload):
                if kind == "tool_call":
                    inp = payload["input"]
                    detail = inp.get("name") or inp.get("champion") or \
                        f"{inp.get('team_a', '')} vs {inp.get('team_b', '')}".strip(" vs")
                    st.write(f"🔧 `{payload['name']}` — {detail}")

            try:
                answer = agent.ask(user_text, images=images,
                                   history=st.session_state.hist, on_event=on_event)
                status.update(label="Analyse terminée ✅", state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label="Erreur", state="error")
                msg = str(exc)
                if "authentication" in msg.lower() or "api_key" in msg.lower() or "401" in msg:
                    st.error("Clé API refusée. Vérifie ta clé API dans la barre latérale.")
                elif "credit" in msg.lower() or "billing" in msg.lower() or "429" in msg:
                    st.error("Quota/crédit insuffisant ou rate-limit sur le compte Anthropic.")
                else:
                    st.error(f"Appel à Claude impossible : {exc}")
                return

        st.markdown(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})
    st.session_state.hist.append({"role": "user", "content": user_text})
    st.session_state.hist.append({"role": "assistant", "content": answer})


def _quick_preview(ctx: DataContext, team_a: str, team_b: str, bestof: int) -> None:
    """Aperçu modèle SANS IA (gratuit) : nos chiffres bruts pour le match."""
    res = ctx.matchup(team_a, team_b, bestof)
    if res.get("error"):
        st.warning(f"{res['error']} (a={res.get('team_a_resolved')}, b={res.get('team_b_resolved')})")
        return
    c = st.columns(4)
    c[0].metric("Favori", res["favorite"])
    c[1].metric("P(game) calibrée", f"{res['p_favorite_game_calibrated']*100:.0f}%",
                help=f"brute {res['p_favorite_game_raw']*100:.0f}%")
    c[2].metric(f"P(série BO{bestof})", f"{res['p_favorite_series']*100:.0f}%")
    c[3].metric("Elo", f"{res['elo_a']:.0f} / {res['elo_b']:.0f}",
                help=f"{res['team_a']} / {res['team_b']}")
    tags = []
    if res["high_confidence"]:
        tags.append("🎯 haute confiance")
    if res["cross_league"]:
        tags.append("⚠️ cross-ligue")
    if not res["favorite_league_reliable"]:
        tags.append("🌪️ ligue chaotique")
    if res["min_games_played"] < 15:
        tags.append(f"⚠️ peu de data ({res['min_games_played']}g)")
    st.caption("  ·  ".join(tags) if tags else "Conditions standard.")
    for note in res.get("notes", []):
        st.caption(f"— {note}")


DRAFT_ROLES = [("top", "Top"), ("jng", "Jungle"), ("mid", "Mid"),
               ("bot", "ADC"), ("sup", "Support")]


def _draft_inputs(champs: list[str]) -> tuple[str, str]:
    """Grille de draft par rôle (selectbox recherchable). Renvoie (draft bleue, draft rouge) en texte."""
    opts = ["—", *champs]
    st.markdown("**🎯 Draft** — pioche par rôle (autocomplétion ; laisse « — » si inconnu)")
    cols = st.columns(2)
    out: dict[str, list[str]] = {"blue": [], "red": []}
    for side, title, col in (("blue", "🔵 Bleue", cols[0]), ("red", "🔴 Rouge", cols[1])):
        with col:
            st.caption(title)
            for rk, rlabel in DRAFT_ROLES:
                champ = st.selectbox(rlabel, opts, index=0, key=f"draft_{side}_{rk}")
                if champ != "—":
                    out[side].append(f"{champ} ({rlabel})")
    return ", ".join(out["blue"]), ", ".join(out["red"])


def main() -> None:
    st.title("🤖 Assistant IA — analyse de match")
    st.caption(
        "Donne le contexte (équipes, draft, infos gol.gg, **captures de cotes**) : l'agent "
        "Claude interroge **nos données** (Elo calibré, priors champion, forme, fiabilité "
        "ligue) et rend une proba argumentée. Il corrige le modèle avec tes infos (roster, "
        "patch, cross-région) — exactement là où l'Elo est aveugle."
    )

    st.session_state.setdefault("chat", [])   # affichage (markdown)
    st.session_state.setdefault("hist", [])   # historique texte pour l'API

    # ----------------------------------------------------------- barre latérale
    with st.sidebar:
        st.subheader("⚙️ Réglages IA")
        prov_label = st.selectbox("Fournisseur", list(PROVIDERS), index=0)
        provider = PROVIDERS[prov_label]

        if provider == "gemini":
            file_key = load_gemini_key()
            typed = st.text_input(
                "Clé API Gemini", type="password",
                value=st.session_state.get("gemini_key", ""),
                placeholder="AIza…" if not file_key else "(détectée via env / gemini.key)",
                help="Gratuite sur aistudio.google.com/apikey. Stockée uniquement pour la "
                     "session. Sinon : variable GEMINI_API_KEY ou fichier `gemini.key` (gitignored).",
            )
            if typed:
                st.session_state["gemini_key"] = typed
            key = st.session_state.get("gemini_key") or file_key
            models = GEMINI_MODELS
        else:
            file_key = load_api_key()
            typed = st.text_input(
                "Clé API Anthropic", type="password",
                value=st.session_state.get("anthropic_key", ""),
                placeholder="sk-ant-…" if not file_key else "(détectée via env / anthropic.key)",
                help="Stockée uniquement pour la session. Sinon : variable ANTHROPIC_API_KEY "
                     "ou fichier `anthropic.key` (déjà gitignored).",
            )
            if typed:
                st.session_state["anthropic_key"] = typed
            key = st.session_state.get("anthropic_key") or file_key
            models = ANTHROPIC_MODELS

        st.caption("✅ Clé détectée" if key else "❌ Pas de clé — ajoute-la ci-dessus.")
        model = st.selectbox("Modèle", list(models), index=0, format_func=lambda m: m)
        st.caption(models[model])

        st.divider()
        if st.button("🗑️ Vider la conversation", width="stretch"):
            st.session_state.chat, st.session_state.hist = [], []
            st.rerun()
        if st.button("🔄 Recharger les données du modèle", width="stretch"):
            get_ctx.clear()
            st.rerun()

    try:
        ctx = get_ctx()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Impossible de charger nos données : {exc}")
        st.stop()
    champs = sorted(ctx.champ.dates)

    # ----------------------------------------------------------- historique chat
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # ----------------------------------------------------------- contexte match
    with st.expander("➕ Nouveau match — contexte structuré", expanded=not st.session_state.chat):
        with st.form("match_form", clear_on_submit=False):
            c = st.columns([2, 2, 1])
            team_a = c[0].text_input("Équipe bleue", placeholder="T1, Karmine Corp…")
            team_b = c[1].text_input("Équipe rouge", placeholder="Gen.G, KCB…")
            bestof = c[2].selectbox("Format", [1, 3, 5], index=1, format_func=lambda b: f"BO{b}")
            league = st.text_input("Ligue / event (optionnel)",
                                   placeholder="LCK, MSI 2026 (utile pour signaler le cross-région)…")
            draft_b, draft_r = _draft_inputs(champs)
            notes = st.text_area("Infos en plus (gol.gg, forme, roster, news, cotes du book…)",
                                 placeholder="Ex : KC joue avec son sub jungler ; cote book T1 @1.02 ; "
                                             "Solary 5 wins d'affilée…", height=80)
            uploads = st.file_uploader("Captures (cotes, draft, stats gol.gg…)",
                                       type=["png", "jpg", "jpeg", "webp", "gif"],
                                       accept_multiple_files=True)
            question = st.text_input("Ta question (optionnel)",
                                     placeholder="Qui parier ? Y a-t-il de la value ?")
            cc = st.columns(2)
            preview = cc[0].form_submit_button("👁️ Aperçu modèle (gratuit, sans IA)", width="stretch")
            submitted = cc[1].form_submit_button("🚀 Analyser avec l'IA", type="primary", width="stretch")

        if preview:
            if team_a and team_b:
                _quick_preview(ctx, team_a, team_b, bestof)
            else:
                st.warning("Renseigne les deux équipes pour l'aperçu.")

    # ----------------------------------------------------------- lancement IA
    pending = None
    if submitted:
        if not key:
            st.error("Ajoute ta clé API Anthropic dans la barre latérale pour lancer l'IA.")
        elif not (team_a or team_b or notes or uploads):
            st.warning("Donne au moins les équipes, une note ou une capture.")
        else:
            pending = (_compose(team_a, team_b, bestof, league, draft_b, draft_r, notes, question),
                       _encode_uploads(uploads))

    chat_in = _chat_bar()
    if chat_in and (chat_in[0].strip() or chat_in[1]):
        if not key:
            st.error("Ajoute ta clé API Anthropic dans la barre latérale.")
        else:
            sig = (chat_in[0], tuple(len(i["bytes"]) for i in chat_in[1]))
            if st.session_state.get("_last_chat_sig") != sig:
                st.session_state["_last_chat_sig"] = sig
                text = chat_in[0].strip() or \
                    "Analyse ce screen (cotes / draft / état de game en live) et donne ton verdict."
                pending = (text, chat_in[1])

    if pending:
        _run_agent(pending[0], pending[1], key, model, provider)


if __name__ == "__main__":
    main()

"""Page Streamlit — TICKETS : ce qu'on doit faire, ce qu'on a validé.

Un tableau de bord minimal pour piloter le projet : on pose un ticket (une idée,
une tâche, un truc à vérifier), on le valide quand c'est fait. Zéro usine à gaz :
un CSV versionnable (data/tickets.csv), deux colonnes d'état, des boutons.

NB Streamlit Cloud : le disque du cloud est ÉPHÉMÈRE — un ticket ajouté en ligne
disparaît au prochain redéploiement. La source de vérité est le fichier du repo :
ajoute/valide en local puis push, ou refais la manip après un redeploy.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.ingest.load_oracle import ROOT

st.set_page_config(page_title="LoL — Tickets", page_icon="🎫", layout="wide")

from src.ui.auth import require_password  # noqa: E402 - doit suivre set_page_config

require_password()

TICKETS_PATH = ROOT / "data" / "tickets.csv"
COLS = ["id", "created", "title", "note", "status", "done_at"]
STATUSES = ["a_faire", "valide"]


def load_tickets() -> pd.DataFrame:
    if not TICKETS_PATH.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(TICKETS_PATH)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    return df[COLS]


def save_tickets(df: pd.DataFrame) -> None:
    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(TICKETS_PATH, index=False)


# ------------------------------------------------------------------ callbacks
# (mutations faites dans des callbacks on_click : pattern sûr avec les reruns)
def _set_status(tid: int, status: str) -> None:
    df = load_tickets()
    m = df["id"] == tid
    df.loc[m, "status"] = status
    df.loc[m, "done_at"] = dt.date.today().isoformat() if status == "valide" else None
    save_tickets(df)


def _delete(tid: int) -> None:
    df = load_tickets()
    save_tickets(df[df["id"] != tid])


def main() -> None:
    st.title("🎫 Tickets — à faire / validé")
    st.caption(
        "Pose ici ce qu'on doit faire ; valide quand c'est fait. "
        "⚠️ En ligne (Streamlit Cloud), le disque est éphémère : un ticket ajouté ici "
        "disparaît au prochain redéploiement — la version durable vit dans le repo "
        "(`data/tickets.csv`), donc ajoute en local + push pour que ça reste."
    )

    df = load_tickets()

    # --- ajout ---
    with st.form("add_ticket", clear_on_submit=True):
        c = st.columns([3, 4, 1])
        title = c[0].text_input("Ticket", placeholder="Vérifier résiliation the-odds-api")
        note = c[1].text_input("Détail (optionnel)", placeholder="contexte, commande, lien…")
        ok = c[2].form_submit_button("➕ Ajouter", type="primary")
    if ok:
        if not title.strip():
            st.error("Le titre du ticket est obligatoire.")
        else:
            nid = int(df["id"].max()) + 1 if len(df) else 1
            row = {"id": nid, "created": dt.date.today().isoformat(),
                   "title": title.strip(), "note": note.strip() or None,
                   "status": "a_faire", "done_at": None}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_tickets(df)
            st.success(f"Ticket #{nid} ajouté.")

    todo = df[df["status"] == "a_faire"]
    done = df[df["status"] == "valide"]

    k = st.columns(3)
    k[0].metric("À faire", len(todo))
    k[1].metric("Validés", len(done))
    k[2].metric("Total", len(df))

    left, right = st.columns(2, gap="large")

    # --- à faire ---
    with left:
        st.subheader(f"📋 À faire ({len(todo)})")
        if todo.empty:
            st.info("Rien en attente. Ajoute un ticket ci-dessus.")
        for r in todo.sort_values("id").itertuples():
            box = st.container(border=True)
            c = box.columns([6, 1.2, 0.8])
            c[0].markdown(f"**{r.title}**")
            if isinstance(r.note, str) and r.note:
                c[0].caption(r.note)
            c[0].caption(f"#{r.id} · créé le {r.created}")
            c[1].button("✅ Valider", key=f"val_{r.id}", width="stretch",
                        on_click=_set_status, args=(int(r.id), "valide"))
            c[2].button("🗑️", key=f"del_{r.id}", width="stretch",
                        help="Supprimer définitivement",
                        on_click=_delete, args=(int(r.id),))

    # --- validés ---
    with right:
        st.subheader(f"✅ Validés ({len(done)})")
        if done.empty:
            st.info("Aucun ticket validé pour l'instant.")
        for r in done.sort_values("done_at", ascending=False).itertuples():
            box = st.container(border=True)
            c = box.columns([6, 1.2, 0.8])
            c[0].markdown(f"~~{r.title}~~")
            if isinstance(r.note, str) and r.note:
                c[0].caption(r.note)
            c[0].caption(f"#{r.id} · validé le {r.done_at}")
            c[1].button("↩️ Rouvrir", key=f"reo_{r.id}", width="stretch",
                        on_click=_set_status, args=(int(r.id), "a_faire"))
            c[2].button("🗑️", key=f"deld_{r.id}", width="stretch",
                        help="Supprimer définitivement",
                        on_click=_delete, args=(int(r.id),))

    # --- édition brute (secours) ---
    with st.expander("✏️ Édition brute (secours)"):
        edited = st.data_editor(
            load_tickets(), width="stretch", hide_index=True, num_rows="dynamic",
            column_config={
                "id": st.column_config.NumberColumn("#", width="small", format="%d"),
                "created": st.column_config.TextColumn("Créé"),
                "title": st.column_config.TextColumn("Ticket", width="large"),
                "note": st.column_config.TextColumn("Détail", width="large"),
                "status": st.column_config.SelectboxColumn("État", options=STATUSES, width="small"),
                "done_at": st.column_config.TextColumn("Validé le"),
            },
        )
        if st.button("💾 Sauvegarder l'édition brute"):
            save_tickets(edited)
            st.success("Tickets sauvegardés.")


if __name__ == "__main__":
    main()

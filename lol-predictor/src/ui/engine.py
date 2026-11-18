"""Moteur partagé par les pages du nouveau site.

Un seul calcul (rejeu des saisons, walk-forward des deux modèles, table de la saison),
fait en tâche de fond par la collecte et gardé sur disque (`src.v2.precompute`). Le site
le relit, le garde en mémoire, et ne le refait lui-même que s'il n'existe pas.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.ingest.load_oracle import ROOT

PARIS = ZoneInfo("Europe/Paris")
JOURS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


def data_version() -> float:
    """Clé de cache : change dès qu'un nouveau moteur est disponible, c'est-à-dire après
    chaque mise à jour des données (CSV Oracle, tables Leaguepedia) ou du code."""
    from src.v2 import precompute
    return precompute.version()


def market_version() -> float:
    """Clé de cache de l'archive des prix Polymarket."""
    files = list((ROOT / "data" / "odds").glob("polymarket_*.csv"))
    return max((p.stat().st_mtime for p in files), default=0.0)


def data_status() -> pd.DataFrame:
    """État de la collecte automatique (`src.update.refresh_all`), pour affichage."""
    from src.update.refresh_all import SOURCES, load_status
    status = load_status()
    rows = []
    for name in SOURCES:
        e = status.get(name)
        if not e:
            rows.append({"Source": name, "État": "jamais lancée", "Dernier succès": None,
                         "Détail": ""})
            continue
        ok = to_paris(e["last_ok"]) if e.get("last_ok") else None
        rows.append({
            "Source": name,
            "État": "ignorée" if e.get("skipped") else "OK" if e.get("ok") else "échec",
            "Dernier succès": f"{ok:%d/%m %H:%M}" if ok else None,
            "Détail": e.get("message", ""),
        })
    return pd.DataFrame(rows)


@st.cache_resource(max_entries=1,
                   show_spinner="Chargement des modèles (plusieurs minutes s'il faut les recalculer)…")
def _load(version: float) -> dict:
    from src.v2 import precompute
    return precompute.get()


def load_engine() -> dict:
    return _load(data_version())


def load_season_board() -> pd.DataFrame | None:
    """Gain du nouveau modèle saison par saison (rien sans les saisons passées)."""
    return load_engine().get("season_board")


@st.cache_data(max_entries=1, show_spinner="Rapprochement avec les prix Polymarket…")
def _market(version: float, mversion: float) -> pd.DataFrame:
    from src.v2.market import attach_market
    eng = load_engine()
    return attach_market(eng["series"], eng["games"], int(eng["df"]["date"].dt.year.max()))


def load_market() -> pd.DataFrame:
    """Les séries de la saison + le prix Polymarket d'avant-match (colonnes `mkt_*`)."""
    return _market(data_version(), market_version())


@st.cache_data(ttl=900, show_spinner="Prix Polymarket des matchs à venir…")
def upcoming_prices(matches: tuple[tuple[str, str, str], ...]) -> dict:
    """(équipe A, équipe B, date) -> marché Polymarket ouvert, ou rien si le match n'est pas coté."""
    from concurrent.futures import ThreadPoolExecutor

    from src.update.polymarket import find_market

    def one(m):
        try:
            return find_market(*m)
        except Exception:  # noqa: BLE001 - un match sans prix ne doit pas casser la page
            return None

    with ThreadPoolExecutor(4) as pool:
        return {m: r for m, r in zip(matches, pool.map(one, matches)) if r}


def refresh_data() -> None:
    """Télécharge le dernier CSV Oracle's Elixir, puis recalcule les modèles."""
    with st.status("Mise à jour des données…", expanded=True) as status:
        try:
            from src.update.download_data import download_latest
            download_latest()
            st.write("✅ Données Oracle's Elixir à jour.")
        except Exception as exc:  # noqa: BLE001 - on garde la data locale
            st.write(f"⚠️ Téléchargement impossible ({exc}). On garde les données locales.")
        from src.v2 import precompute
        st.write("⏳ Modèles : " + precompute.warm() + ".")
        st.cache_data.clear()
        status.update(label="Terminé.", state="complete")


# ------------------------------------------------------------------ dates --
def to_paris(utc_str: str) -> dt.datetime | None:
    """'2026-11-11 10:00:00' (UTC) -> datetime à l'heure de Paris (été/hiver géré)."""
    try:
        d = dt.datetime.strptime(str(utc_str)[:19], "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None
    return d.replace(tzinfo=dt.timezone.utc).astimezone(PARIS)


def day_label(d: dt.date) -> str:
    today = dt.datetime.now(PARIS).date()
    name = f"{JOURS[d.weekday()]} {d:%d/%m}"
    if d == today:
        return f"Aujourd'hui · {name}"
    if d == today + dt.timedelta(days=1):
        return f"Demain · {name}"
    return name


def paris_series(s: pd.Series) -> pd.Series:
    """Colonne de dates UTC (naïves) -> heure de Paris (naïve, pour l'affichage)."""
    return s.dt.tz_localize("UTC").dt.tz_convert("Europe/Paris").dt.tz_localize(None)

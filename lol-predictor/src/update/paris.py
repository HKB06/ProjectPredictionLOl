"""Heure de Paris, été comme hiver (un décalage fixe de +2 h avait une heure de retard l'hiver)."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
UTC = dt.timezone.utc


def to_paris(utc_naive: dt.datetime) -> dt.datetime:
    """Datetime UTC sans fuseau -> heure de Paris, sans fuseau (pour affichage)."""
    return utc_naive.replace(tzinfo=UTC).astimezone(PARIS).replace(tzinfo=None)


def now_paris() -> dt.datetime:
    return dt.datetime.now(PARIS).replace(tzinfo=None)


def midnight_utc(day: dt.date) -> dt.datetime:
    """Minuit à Paris ce jour-là, exprimé en UTC (avec fuseau)."""
    return dt.datetime.combine(day, dt.time(0, 0), tzinfo=PARIS).astimezone(UTC)

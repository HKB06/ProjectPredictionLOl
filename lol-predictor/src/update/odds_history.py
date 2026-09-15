"""Archive des VRAIES cotes de clôture des matchs passés (odds-api.io).

Pourquoi : sans cote réelle, le ROI du passé n'est qu'une projection (« et si on
avait joué à 1.28 ? »). Or odds-api.io expose `/historical/events` et
`/historical/odds`, qui rendent la **cote relevée juste avant le coup d'envoi** —
la closing line. C'est elle qui transforme la projection en mesure.

Contraintes réelles du plan gratuit, toutes vérifiées sur l'API :
  - **100 requêtes/heure**. Une requête par match ⇒ un remplissage complet prend
    plusieurs heures. D'où un cache sur disque et un remplissage *reprenable*.
  - `/historical/events` exige `sport`, `league` et une fenêtre de **31 jours max**.
  - `/historical/odds` exige `eventId` + `bookmakers`.
  - Le marché `ML` cote le vainqueur de **série**, jamais la game isolée.

Cache : `data/odds_history.csv`, une ligne par match, `status` ∈ {pending, ok, none}.
On ne redemande jamais une cote déjà connue.

Usage :
    python -m src.update.odds_history --days 60          # remplit dans la limite du quota
    python -m src.update.odds_history --days 60 --budget 50
    python -m src.update.odds_history --stats            # état du cache, sans requête
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from src.ingest.load_oracle import ROOT
from src.update.oddsapi import BASE, BOOKMAKERS, load_key
from src.update.watchlist import core_tokens

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE_PATH = ROOT / "data" / "odds_history.csv"
QUOTA_PATH = ROOT / "data" / ".odds_quota.json"
PROGRESS_PATH = ROOT / "data" / ".odds_discovery.json"
COLS = ["event_id", "date", "league_slug", "home", "away",
        "odd_home", "odd_away", "books", "status", "fetched_at"]

HOURLY_LIMIT = 100        # plan gratuit odds-api.io (message d'erreur 429 explicite)
SAFETY = 5                # marge : on ne colle jamais au plafond
MAX_SPAN_DAYS = 31        # contrainte /historical/events
TIMEOUT = 40


# ------------------------------------------------------------------ quota local
class Quota:
    """Compte les requêtes de l'heure glissante, persisté pour survivre aux redémarrages.

    Sans ça, relancer le script remettrait le compteur à zéro côté client et on se
    ferait jeter par des 429 en rafale.
    """

    def __init__(self, path: Path = QUOTA_PATH) -> None:
        self.path = path
        self.stamps: list[float] = []
        if path.exists():
            try:
                self.stamps = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                self.stamps = []
        self._prune()

    def _prune(self) -> None:
        cut = time.time() - 3600
        self.stamps = [t for t in self.stamps if t > cut]

    def remaining(self) -> int:
        self._prune()
        return max(0, HOURLY_LIMIT - SAFETY - len(self.stamps))

    def note(self) -> None:
        self.stamps.append(time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stamps))

    def resets_in(self) -> int:
        """Secondes avant qu'une requête se libère."""
        self._prune()
        return int(3600 - (time.time() - min(self.stamps))) if self.stamps else 0


# --------------------------------------------------------------------- API bas niveau
def _get(path: str, quota: Quota, **params) -> tuple[int, object]:
    key = load_key()
    if not key:
        raise RuntimeError("Clé odds-api.io absente (ODDS_API_KEY ou oddsapi.key).")
    r = requests.get(BASE + path, params={**params, "apiKey": key}, timeout=TIMEOUT)
    quota.note()
    try:
        return r.status_code, r.json()
    except Exception:  # noqa: BLE001
        return r.status_code, r.text[:200]


def lol_league_slugs(quota: Quota) -> list[str]:
    st, js = _get("/leagues", quota, sport="esports")
    if st == 429:
        raise RateLimited(str(js)[:120])
    if st != 200 or not isinstance(js, list):
        return []
    return [lg["slug"] for lg in js if "league-of-legends" in str(lg.get("slug", ""))]


def _windows(days: int):
    """Tranches de ≤31 jours (contrainte API), **ancrées sur le calendrier**.

    L'ancrage compte : il rend la clé (ligue, fenêtre) stable d'une exécution à
    l'autre, donc la découverte devient reprenable au lieu de tout refaire.
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days)
    cur = dt.datetime(start.year, start.month, 1)
    while cur < end:
        nxt = min(cur + dt.timedelta(days=MAX_SPAN_DAYS - 1), end)
        yield (cur.strftime("%Y-%m-%dT%H:%M:%SZ"), nxt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        cur = dt.datetime(cur.year + cur.month // 12, cur.month % 12 + 1, 1)


def _load_progress() -> set[str]:
    if not PROGRESS_PATH.exists():
        return set()
    try:
        return set(json.loads(PROGRESS_PATH.read_text()))
    except Exception:  # noqa: BLE001
        return set()


def _save_progress(done: set[str]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(sorted(done)))


def past_events(slug: str, frm: str, to: str, quota: Quota) -> list[dict]:
    st, js = _get("/historical/events", quota, sport="esports", league=slug,
                  **{"from": frm, "to": to}, limit=200)
    if st == 429:
        raise RateLimited(str(js)[:120])
    if st != 200:
        return []
    return (js.get("events") if isinstance(js, dict) else js) or []


class RateLimited(Exception):
    """L'API a répondu 429 : on arrête net plutôt que de brûler le budget à vide."""


# Un marché à 2 issues a une marge : 1/cote_home + 1/cote_away ≈ 1.02 à 1.20.
# Hors de ces bornes, la paire n'est PAS un vainqueur de match (map isolée, marché
# annexe, cotes figées en vrac). C'est notre garde-fou contre les cotes aberrantes.
OVERROUND_MIN, OVERROUND_MAX = 1.00, 1.30


def _plausible(h: float, a: float) -> bool:
    if not (h and a) or h < 1.01 or a < 1.01:
        return False
    return OVERROUND_MIN <= (1.0 / h + 1.0 / a) <= OVERROUND_MAX


def closing_odds(event_id, quota: Quota) -> tuple[float | None, float | None, str]:
    """Meilleure cote ML (vainqueur de série) relevée avant le coup d'envoi.

    On ne prend jamais le max colonne par colonne : un même événement expose
    plusieurs entrées « ML » et mélanger les lignes produit des paires absurdes
    (vu en LCS : 10.22 / 11.80, soit 18 % de probabilité totale). On valide donc
    chaque **paire** via sa marge, puis on fait du line shopping entre paires saines.
    """
    st, js = _get("/historical/odds", quota, eventId=event_id,
                  bookmakers=",".join(BOOKMAKERS))
    if st == 429:
        raise RateLimited(str(js)[:120])
    if st != 200 or not isinstance(js, dict):
        return None, None, ""

    pairs: list[tuple[float, float, str]] = []
    for book, markets in (js.get("bookmakers") or {}).items():
        for m in markets or []:
            if (m.get("name") or "").upper() != "ML":
                continue
            for o in m.get("odds") or []:
                try:
                    h, a = float(o["home"]), float(o["away"])
                except (KeyError, ValueError, TypeError):
                    continue
                if _plausible(h, a):
                    pairs.append((h, a, book))
    if not pairs:
        return None, None, ""
    return (max(p[0] for p in pairs), max(p[1] for p in pairs),
            "+".join(sorted({p[2] for p in pairs})))


# ------------------------------------------------------------------------- cache
def load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(CACHE_PATH)
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLS]


def save_cache(df: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(CACHE_PATH, index=False)


def repair_cache(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remet en file d'attente les cotes invalides (marge aberrante).

    Sert de filet après un correctif d'extraction : les lignes récupérées par une
    version buguée sont réputées fausses et seront simplement redemandées.
    """
    if df.empty:
        return df, 0
    h = pd.to_numeric(df["odd_home"], errors="coerce")
    a = pd.to_numeric(df["odd_away"], errors="coerce")
    sane = pd.Series([_plausible(x, y) for x, y in zip(h.fillna(0), a.fillna(0))],
                     index=df.index)
    bad = (df["status"] == "ok") & ~sane
    n = int(bad.sum())
    if n:
        df.loc[bad, ["odd_home", "odd_away", "books", "fetched_at"]] = pd.NA
        df.loc[bad, "status"] = "pending"
    return df, n


def backfill(days: int = 60, budget: int | None = None) -> dict:
    """Remplit le cache dans la limite du quota. **Reprenable** : relancer continue.

    Deux phases : découverte des matchs (peu de requêtes, une par ligue/fenêtre),
    puis récupération des cotes (une requête par match encore `pending`).
    """
    quota = Quota()
    budget = min(budget or quota.remaining(), quota.remaining())
    cache, repaired = repair_cache(load_cache())
    if repaired:
        save_cache(cache)
    known = set(cache["event_id"].astype(str))
    used = 0

    if budget <= 0:
        return {"added": 0, "fetched": 0, "used": 0,
                "remaining": 0, "resets_in": quota.resets_in(), "cache": len(cache)}

    # --- phase 1 : découverte des matchs.
    # On ne la relance que si la file d'attente est courte : sinon chaque passage
    # gaspillerait la moitié du quota à redécouvrir des matchs déjà connus, alors
    # que le goulot est la récupération des cotes (1 requête par match).
    new_rows: list[dict] = []
    limited = False
    done = _load_progress()
    todo = []                       # couples (ligue, fenêtre) jamais explorés
    if not limited:
        try:
            slugs = lol_league_slugs(quota)
            used += 1
            todo = [(s, f, t) for s in slugs for f, t in _windows(days)
                    if f"{s}|{f[:10]}" not in done]
        except RateLimited:
            limited = True

    # La découverte passe en premier tant qu'elle est incomplète : sans elle, des
    # ligues entières resteraient invisibles. Une fois terminée, tout le quota va
    # aux cotes (le vrai goulot : 1 requête par match).
    discovery_cap = budget if todo else 0
    for slug, frm, to in todo:
        if used >= discovery_cap or quota.remaining() <= 0:
            break
        try:
            evs = past_events(slug, frm, to, quota)
        except RateLimited:
            limited = True
            break
        used += 1
        done.add(f"{slug}|{frm[:10]}")
        for ev in evs:
            eid = str(ev.get("id"))
            if eid in known:
                continue
            known.add(eid)
            new_rows.append({
                "event_id": eid, "date": ev.get("date"), "league_slug": slug,
                "home": ev.get("home"), "away": ev.get("away"),
                "odd_home": pd.NA, "odd_away": pd.NA, "books": pd.NA,
                "status": "pending", "fetched_at": pd.NA,
            })
    _save_progress(done)
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(cache)

    # --- phase 2 : cotes des matchs encore en attente, du plus récent au plus ancien
    pend = cache[cache["status"] == "pending"].sort_values("date", ascending=False)
    fetched = 0
    for i in pend.index:
        if limited:
            break
        if used >= budget or quota.remaining() <= 0:
            break
        try:
            h, a, books = closing_odds(cache.at[i, "event_id"], quota)
        except RateLimited:
            limited = True
            break
        used += 1
        cache.at[i, "odd_home"] = h if h else pd.NA
        cache.at[i, "odd_away"] = a if a else pd.NA
        cache.at[i, "books"] = books or pd.NA
        cache.at[i, "status"] = "ok" if h else "none"
        cache.at[i, "fetched_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        fetched += 1
        if fetched % 10 == 0:
            save_cache(cache)
    save_cache(cache)

    return {"added": len(new_rows), "fetched": fetched, "used": used,
            "repaired": repaired,
            "remaining": quota.remaining(), "resets_in": quota.resets_in(),
            "cache": len(cache), "limited": limited,
            "ok": int((cache["status"] == "ok").sum()),
            "pending": int((cache["status"] == "pending").sum())}


# ---------------------------------------------------------------------- lecture
def lookup(cache: pd.DataFrame, team1: str, team2: str, date: str,
           pick: str, tol_days: int = 1) -> tuple[float, str]:
    """Cote de clôture archivée pour NOTRE pick sur cette série.

    Apparie sur les tokens du nom (les books ajoutent/retirent des sponsors) et
    tolère un jour d'écart (fuseaux horaires).
    """
    if cache.empty:
        return float("nan"), ""
    ok = cache[cache["status"] == "ok"]
    if ok.empty:
        return float("nan"), ""
    d = pd.to_datetime(date, errors="coerce")
    if pd.isna(d):
        return float("nan"), ""
    dates = pd.to_datetime(ok["date"], errors="coerce", utc=True).dt.tz_localize(None)
    tol = pd.Timedelta(days=tol_days)
    near = ok[(dates >= d - tol) & (dates <= d + tol)]

    t1, t2 = core_tokens(team1), core_tokens(team2)
    for r in near.itertuples():
        h, a = core_tokens(str(r.home)), core_tokens(str(r.away))
        if (t1 & h and t2 & a):
            side = "home" if pick == team1 else "away"
        elif (t1 & a and t2 & h):
            side = "away" if pick == team1 else "home"
        else:
            continue
        try:
            h_odd, a_odd = float(r.odd_home), float(r.odd_away)
        except (TypeError, ValueError):
            continue
        if not _plausible(h_odd, a_odd):   # filet si le cache vient d'une version buguée
            continue
        return (h_odd if side == "home" else a_odd), f"closing ({r.books})"
    return float("nan"), ""


# ------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Archive des cotes de clôture passées")
    ap.add_argument("--days", type=int, default=60, help="profondeur à couvrir (def. 60 j)")
    ap.add_argument("--budget", type=int, default=None, help="requêtes max pour ce passage")
    ap.add_argument("--stats", action="store_true", help="état du cache, sans requête")
    args = ap.parse_args()

    if args.stats:
        c = load_cache()
        q = Quota()
        print(f"Cache : {len(c)} matchs  "
              f"({int((c['status'] == 'ok').sum())} avec cote, "
              f"{int((c['status'] == 'none').sum())} sans, "
              f"{int((c['status'] == 'pending').sum())} en attente)")
        print(f"Quota : {q.remaining()} requêtes dispo cette heure")
        return

    res = backfill(days=args.days, budget=args.budget)
    if res.get("limited") and not res["fetched"] and not res["added"]:
        print("Quota horaire epuise cote serveur (100 req/h). "
              "Rien n'a ete perdu : relance plus tard, c'est reprenable.")
        return
    if not res["used"]:
        print(f"Quota epuise. Reessaie dans {res['resets_in'] // 60} min.")
        return
    if res.get("repaired"):
        print(f"{res['repaired']} cotes aberrantes remises en file d'attente.")
    print(f"{res['added']} nouveaux matchs decouverts, {res['fetched']} cotes recuperees "
          f"({res['used']} requetes).")
    print(f"Cache : {res['cache']} matchs, {res.get('ok', 0)} avec cote, "
          f"{res.get('pending', 0)} encore en attente.")
    if res.get("limited"):
        print("!! Quota horaire atteint cote serveur : arret propre, rien n'est perdu.")
    print(f"Quota restant : {res['remaining']}")
    if res.get("pending"):
        print("-> relance la commande dans une heure pour continuer (c'est reprenable).")


if __name__ == "__main__":
    main()

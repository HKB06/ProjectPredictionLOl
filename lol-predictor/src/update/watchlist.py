"""Watchlist pré-match : pour chaque match à venir, calcule NOTRE proba (Elo) à l'avance.

But (cf. SUIVI_PARIS, leçon n°1) : le goulot n'est plus le modèle mais le TIMING de la
mise. On calcule donc le penchant Elo des matchs des prochains jours, AVANT qu'ils
commencent, pour pouvoir poser le pari tôt (quand le marché soft est ouvert) si le book
sur-cote un favori que notre modèle voit plus serré (pattern KC/VKS/Heretics).

Sortie : WATCHLIST.md (racine Projet_Perso) + résumé console.

Usage :
    python -m src.update.watchlist
"""
from __future__ import annotations

import datetime as dt
import unicodedata
from pathlib import Path

from src.ingest.load_oracle import ROOT, load_config
from src.update.elo import (RELIABLE_ACC, calibrate, compute_elo, resolve_league,
                            series_prob, win_prob)

OUT_PATH = ROOT.parent / "WATCHLIST.md"
PARIS_OFFSET = dt.timedelta(hours=2)  # CEST (été). Affichage seulement.
MIN_GAMES_CONF = 15                   # en dessous : cold-start, peu fiable
# Mots "bruit" ignorés pour le matching de noms (Gen.G Esports = Gen.G, etc.)
STOP_WORDS = {"esports", "esport", "gaming", "club", "team", "the", "pro"}


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    tok, cur = [], []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tok.append("".join(cur))
            cur = []
    if cur:
        tok.append("".join(cur))
    return tok


def norm(s: str) -> str:
    """Nom normalisé complet (accents/casse/ponctuation retirés)."""
    return "".join(_tokens(s))


def norm_core(s: str) -> str:
    """Nom normalisé sans mots bruit (esports, gaming, team...) pour le fallback."""
    core = "".join(t for t in _tokens(s) if t not in STOP_WORDS)
    return core or norm(s)


def _fetch_upcoming(days: int) -> list[dict]:
    """lolesports en primaire, Leaguepedia en secours."""
    try:
        from src.update.lolesports import fetch_upcoming
        return fetch_upcoming(days)
    except Exception as exc:  # noqa: BLE001
        print(f"[watchlist] lolesports indisponible ({exc}) -> fallback Leaguepedia")
        from src.update.leaguepedia import fetch_upcoming
        return fetch_upcoming(days)


def _wins_needed(bestof) -> int:
    try:
        bo = int(bestof)
    except (TypeError, ValueError):
        bo = 1
    return {1: 1, 3: 2, 5: 3}.get(bo, (bo + 1) // 2)


def _fmt_paris(datetime_utc: str) -> str:
    try:
        d = dt.datetime.strptime(datetime_utc, "%Y-%m-%d %H:%M:%S") + PARIS_OFFSET
        return d.strftime("%a %d/%m %H:%M")
    except (TypeError, ValueError):
        return str(datetime_utc)


def build_rows(days: int = 7, cfg: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Calcul pur (sans écriture fichier) : renvoie (covered, uncovered).

    Réutilisé par le front Streamlit. `covered` = matchs dont les 2 équipes sont dans
    notre data Elo ; `uncovered` = au moins une équipe non reconnue.
    """
    cfg = cfg or load_config()
    state = compute_elo(cfg)
    elo, n, league = state["elo"], state["n"], state["league"]
    reliability, shrink, global_rel = state["reliability"], state["shrink"], state["global_rel"]
    idx_full = {norm(t): t for t in elo}
    idx_core = {norm_core(t): t for t in elo}  # peut écraser des collisions, acceptable

    def match(name: str) -> str | None:
        return idx_full.get(norm(name)) or idx_core.get(norm_core(name))

    def context(m: dict, a: str, b: str) -> tuple[float, float, str]:
        """Fiabilité + shrink + code ligue du match (pour calibrer la proba/le ⭐)."""
        code = (resolve_league(m.get("overview") or "", reliability)
                or resolve_league(m.get("tournament") or "", reliability))
        if code is None:  # fallback : ligue commune, sinon la plus chaotique des deux
            cand = [c for c in (league.get(a), league.get(b)) if c in reliability]
            code = min(cand, key=lambda c: reliability[c]) if cand else None
        rel = reliability.get(code, global_rel)
        shr = shrink.get(code, 1.0)
        return rel, shr, code or "?"

    matches = _fetch_upcoming(days=days)

    covered: list[dict] = []
    uncovered: list[dict] = []
    for m in matches:
        a = match(m["team1"])
        b = match(m["team2"])
        if not a or not b:
            uncovered.append(m)
            continue
        wn = _wins_needed(m["bestof"])
        rel, shr, code = context(m, a, b)
        p_game_raw = win_prob(elo[a], elo[b])
        p_game = calibrate(p_game_raw, shr)          # aplati selon la fiabilité de la ligue
        p_series = series_prob(p_game, wn)
        conf = min(n[a], n[b]) >= MIN_GAMES_CONF
        reliable = rel >= RELIABLE_ACC
        covered.append({
            "when": _fmt_paris(m["datetime"]),
            "datetime": m["datetime"],
            "league": m.get("overview") or league.get(a) or league.get(b) or "?",
            "team1": a, "team2": b,
            "bestof": m["bestof"],
            "p1": p_series, "p2": 1 - p_series,
            "p1_raw": series_prob(p_game_raw, wn),    # avant calibration (info)
            "elo1": elo[a], "elo2": elo[b],
            "n1": n[a], "n2": n[b],
            "conf": conf,
            "rel": rel, "reliable": reliable, "league_code": code,
            "strong": (p_series >= 0.62 or p_series <= 0.38) and conf and reliable,
            "league_a": league.get(a, "?"), "league_b": league.get(b, "?"),
            "xleague": league.get(a) != league.get(b),  # Elo peu comparable (cf. KCB/PCIFIC)
            "tournament": m.get("tournament") or m.get("overview"),
        })

    covered.sort(key=lambda r: r["datetime"] or "")
    return covered, uncovered


def generate(days: int = 7, cfg: dict | None = None) -> dict:
    covered, uncovered = build_rows(days, cfg)
    _write_markdown(covered, uncovered, days)
    return {"covered": covered, "uncovered": uncovered, "path": OUT_PATH}


def _write_markdown(covered: list[dict], uncovered: list[dict], days: int) -> None:
    now = dt.datetime.now() + PARIS_OFFSET
    lines = [
        "# Watchlist pré-match (auto)",
        "",
        f"> Généré le **{now:%Y-%m-%d %H:%M}** (Paris) · fenêtre **{days} j** · "
        f"{len(covered)} matchs couverts, {len(uncovered)} non couverts.",
        "",
        "**Méthode** : Elo toutes-ligues **K32 + marge de victoire (MOV)**, proba **calibrée par "
        "ligue** (aplatie là où le modèle est chaotique, cf. EM ~56 % d'accuracy). Signal *partiel* "
        "(pas de draft). **Usage** : comparer NOTRE proba à la **cote du book** dès qu'elle ouvre. "
        "Si le book **sur-cote un favori** qu'on voit plus serré → **value** (cf. KC @2.45, VKS @2.7, "
        "Heretics @2.60). ⚠️ **Poser le pari TÔT** (marché soft ouvert) = le vrai enjeu.",
        "",
        "Légende : ⭐ = penchant fort **fiable** (proba ≥62 %, data ≥15 g, **ligue prévisible**). "
        "🌪️ = ligue **chaotique** (EM, LPL, LCS…) → proba peu fiable, **pas de ⭐ même à 70 %**. "
        "⚠️x-ligue = Elo peu comparable (équipes de régions différentes).",
        "",
        "Aide-mémoire value : `edge = notre proba − 1/cote`. On ne fade **jamais** un favori à "
        "cote < ~1.2 (cf. paiN). On vise un **désaccord net sur cote équilibrée**.",
        "",
        "| Quand (Paris) | Ligue | Match | BO | Notre proba | Elo | Fiab. | Cote 1 / 2 | Edge |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in covered:
        fav_strong = "⭐" if r.get("strong") else ""
        chaos = " 🌪️" if not r.get("reliable", True) else ""
        xflag = " ⚠️x-ligue" if r.get("xleague") else ""
        proba = f"**{r['team1']} {r['p1']*100:.0f}%** / {r['team2']} {r['p2']*100:.0f}%"
        elo = f"{r['elo1']:.0f} / {r['elo2']:.0f}"
        fiab = f"{r['rel']*100:.0f}%" + ("" if r["conf"] else f" ⚠️{min(r['n1'], r['n2'])}g")
        lines.append(
            f"| {r['when']} | {r['league']} | {r['team1']} vs {r['team2']} {fav_strong}{chaos}{xflag} | "
            f"BO{r['bestof']} | {proba} | {elo} | {fiab} | _ / _ | _ |"
        )

    if uncovered:
        lines += [
            "",
            f"## Matchs non couverts ({len(uncovered)}) — équipes absentes de notre data",
            "*(noms Leaguepedia ≠ Oracle, ou ligue non collectée. À ignorer ou à mapper plus tard.)*",
            "",
        ]
        seen = set()
        for m in uncovered:
            key = (m["team1"], m["team2"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {_fmt_paris(m['datetime'])} · {m.get('overview', '?')} · {m['team1']} vs {m['team2']}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        res = generate(days=7)
    except Exception as exc:  # noqa: BLE001
        print(f"[watchlist] API matchs indisponible : {exc}")
        return
    cov, unc = res["covered"], res["uncovered"]
    print(f"[watchlist] {len(cov)} matchs couverts, {len(unc)} non couverts -> {res['path']}")
    print("\n  Prochains matchs (Elo K32+MOV, proba calibrée) :")
    for r in cov[:25]:
        flag = "*" if r.get("strong") else ("~" if not r.get("reliable", True) else " ")
        conf = "" if r["conf"] else f" (cold-start {min(r['n1'], r['n2'])}g)"
        print(f"  {flag} {r['when']:14} {r['league']:6} {r['team1']} {r['p1']*100:4.0f}% - "
              f"{r['p2']*100:.0f}% {r['team2']}{conf}")


if __name__ == "__main__":
    main()

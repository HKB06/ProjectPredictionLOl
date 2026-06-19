"""Assistant IA (Claude) branché sur NOS données — cœur agentique.

Idée : un agent Claude (Opus 4.8) à qui on donne le contexte d'un match (draft,
infos gol.gg, screenshots de cotes...) et qui INTERROGE notre modèle via des
*outils* (Elo K32+MOV, proba calibrée par ligue, fiabilité ligue, priors champion,
forme récente) pour rendre une analyse + une proba argumentée, sans inventer de
chiffres.

Deux briques :
  - `DataContext` : charge UNE fois l'état du modèle (Elo, calibration, priors
    champion) et expose des méthodes JSON-sérialisables = les "outils".
  - `Assistant`   : enveloppe l'API Anthropic et fait la boucle d'appels d'outils.

`DataContext` n'importe PAS anthropic : on peut calculer/afficher nos chiffres
même sans clé API. Seul `Assistant` a besoin de la clé.

Usage (CLI rapide, nécessite ANTHROPIC_API_KEY) :
    python -m src.assistant.agent "T1 vs Gen.G en BO5, qui est favori ?"
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config
from src.features.champion_priors import (ChampionWinrateIndex,
                                          load_full_champion_results)
from src.models.high_confidence import CONF_GAME
from src.update.elo import (RELIABLE_ACC, _norm, calibrate, compute_elo,
                            load_games, series_prob, win_prob)

DEFAULT_MODEL = "claude-opus-4-8"   # cf. platform.claude.com (Opus-tier flagship)
MIN_GAMES_CONF = 15                 # data fiable (cohérent avec la watchlist)
MAX_TOOL_STEPS = 10                 # garde-fou anti-boucle d'outils


def load_api_key() -> str | None:
    """ANTHROPIC_API_KEY (env) en priorité, sinon fichier `anthropic.key` (gitignored)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    for p in (ROOT / "anthropic.key", ROOT.parent / "anthropic.key"):
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    return None


# =========================================================================== #
#  DataContext : nos données exposées comme des outils                         #
# =========================================================================== #
class DataContext:
    """Charge l'état du modèle une fois et expose les outils interrogeables."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or load_config()
        state = compute_elo(self.cfg)
        self.elo: dict[str, float] = state["elo"]
        self.n: dict[str, int] = state["n"]
        self.league: dict[str, str] = state["league"]
        self.reliability: dict[str, float] = state["reliability"]
        self.shrink: dict[str, float] = state["shrink"]
        self.global_rel: float = state["global_rel"]

        self.games = load_games(self.cfg)  # une ligne/game (pour la forme récente)
        self._teams_norm = {t: _norm(t) for t in self.elo}

        prior_leagues = self.cfg.get("features", {}).get("champ_prior_leagues")
        self.champ = ChampionWinrateIndex(
            load_full_champion_results(self.cfg, prior_leagues))
        self._champ_by_norm = {}
        for c in self.champ.dates:  # normalisé (apostrophes/espaces) : "ksante" -> "K'Sante"
            self._champ_by_norm.setdefault(_norm(c), c)
        self._champ_date = self.games["date"].max() + pd.Timedelta(seconds=1)
        self._series_res = None   # momentum de série (calcul paresseux)
        self._wp = None           # modèle live win-prob @15 (calcul paresseux)

    # ----------------------------------------------------------- résolution --
    def resolve_team(self, query: str) -> tuple[str | None, list[str]]:
        """Mappe un nom libre ("T1 Academy", "Karmine Corp") vers une équipe connue.

        Renvoie (équipe|None, candidats). Priorité : (1) match exact, (2) recouvrement
        de MOTS (le plus de mots communs gagne), (3) repli sous-chaîne. On classe par
        nb de mots communs puis par nb de games. Ainsi "T1 Academy" → "T1 Esports
        Academy" (2 mots) et non "T1" (1 mot) ; et "T1" seul (exact) → l'équipe LCK.
        C'est ce qui évitait le piège "KT Rolster" vs "KT Rolster Challengers".
        """
        q = _norm(query)
        if not q:
            return None, []
        by_games = sorted(self.elo, key=lambda t: self.n.get(t, 0), reverse=True)

        exact = [t for t in by_games if self._teams_norm[t] == q]
        if exact:
            return exact[0], exact

        qtok = set(_split_tokens(query))
        scored = []
        for t in by_games:
            ttok = set(_split_tokens(t))
            ov = qtok & ttok
            if not ov:
                continue
            subset = 1 if (qtok <= ttok or ttok <= qtok) else 0  # nom inclus dans l'autre
            scored.append((len(ov), subset, self.n.get(t, 0), t))
        if scored:
            scored.sort(reverse=True)
            return scored[0][3], [t for *_, t in scored[:5]]

        # repli : sous-chaîne (noms "collés" sans espace, abréviations partielles)
        contains = [t for t in by_games
                    if q in self._teams_norm[t] or self._teams_norm[t] in q]
        if contains:
            return contains[0], contains[:5]
        return None, []

    def _resolve_champ(self, name: str) -> str | None:
        if not name:
            return None
        q = _norm(name)  # enlève apostrophes/espaces : "Kai'Sa"->"kaisa", "ksante"->"ksante"
        if not q:
            return None
        if q in self._champ_by_norm:
            return self._champ_by_norm[q]
        for nc, orig in self._champ_by_norm.items():
            if q in nc or nc in q:
                return orig
        return None

    # ----------------------------------------------------------- outils -------
    def recent_form(self, team: str, n: int = 10) -> dict:
        g = self.games
        sub = g[(g["blue"] == team) | (g["red"] == team)].tail(n)
        rec, wins = [], 0
        for row in sub.itertuples():
            is_blue = row.blue == team
            won = (row.yb == 1) if is_blue else (row.yb == 0)
            wins += int(won)
            rec.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "opponent": row.red if is_blue else row.blue,
                "side": "blue" if is_blue else "red",
                "result": "W" if won else "L",
            })
        rec.reverse()  # plus récent en premier
        k = len(rec)
        return {"last_n": k, "record": f"{wins}-{k - wins}",
                "winrate": round(wins / k, 3) if k else None, "games": rec}

    def team_info(self, name: str) -> dict:
        t, cands = self.resolve_team(name)
        if not t:
            return {"found": False, "query": name,
                    "hint": "équipe absente de notre data 2026 (nom différent d'Oracle, "
                            "ligue non collectée, ou équipe d'un event tiers)."}
        lg = self.league.get(t, "?")
        rel = self.reliability.get(lg)
        return {
            "found": True,
            "team": t,
            "matched_from": None if _norm(name) == self._teams_norm[t] else name,
            "elo": round(self.elo[t], 1),
            "games_played": int(self.n.get(t, 0)),
            "league": lg,
            "league_reliability": round(rel, 3) if rel is not None else None,
            "league_reliable": bool(rel is not None and rel >= RELIABLE_ACC),
            "calibration_shrink": round(self.shrink.get(lg, 1.0), 2),
            "enough_data": bool(self.n.get(t, 0) >= MIN_GAMES_CONF),
            "recent_form": self.recent_form(t, 10),
            "other_candidates": [c for c in cands if c != t][:4],
        }

    def matchup(self, team_a: str, team_b: str, bestof: int = 3) -> dict:
        a, _ = self.resolve_team(team_a)
        b, _ = self.resolve_team(team_b)
        if not a or not b:
            return {"error": "équipe(s) introuvable(s)",
                    "team_a_resolved": a, "team_b_resolved": b}

        ea = win_prob(self.elo[a], self.elo[b])  # P(a gagne 1 game), side-neutre
        fav, p_fav_raw = (a, ea) if ea >= 0.5 else (b, 1.0 - ea)
        lg_fav = self.league.get(fav, "?")
        shrink_fav = self.shrink.get(lg_fav, 1.0)
        p_fav_cal = max(0.5, min(0.99, calibrate(p_fav_raw, shrink_fav)))

        p_a_cal = p_fav_cal if fav == a else 1.0 - p_fav_cal
        cross = self.league.get(a) != self.league.get(b)
        rel = self.reliability.get(lg_fav, 0.0)
        reliable = rel >= RELIABLE_ACC
        min_g = min(self.n.get(a, 0), self.n.get(b, 0))
        wins_needed = max(1, int(bestof) // 2 + 1)
        p_series_fav = series_prob(p_fav_cal, wins_needed)
        high_conf = bool(reliable and not cross
                         and p_fav_cal >= CONF_GAME and min_g >= MIN_GAMES_CONF)

        notes = []
        if cross:
            notes.append("CROSS-LIGUE : Elo peu comparable entre régions (ex. MSI). "
                         "Proba à prendre avec prudence, croise avec les cotes/roster.")
        if not reliable:
            notes.append(f"LIGUE CHAOTIQUE ({lg_fav} ~{rel*100:.0f}% d'accuracy hist.) : "
                         "le modèle y est sur-confiant, proba déjà aplatie mais peu fiable.")
        if min_g < MIN_GAMES_CONF:
            notes.append(f"PEU DE DATA (min {min_g} games) : Elo encore instable (cold-start).")
        if high_conf:
            notes.append("PICK HAUTE CONFIANCE 🎯 : ligue fiable + favori calibré ≥70%/game "
                         "+ data ≥15g + même ligue (≈83% de bons vainqueurs au backtest).")

        return {
            "team_a": a, "team_b": b, "bestof": int(bestof),
            "favorite": fav,
            "p_favorite_game_raw": round(p_fav_raw, 3),
            "p_favorite_game_calibrated": round(p_fav_cal, 3),
            "p_favorite_series": round(p_series_fav, 3),
            "p_team_a_game": round(p_a_cal, 3),
            "p_team_b_game": round(1.0 - p_a_cal, 3),
            "elo_a": round(self.elo[a], 1), "elo_b": round(self.elo[b], 1),
            "league_a": self.league.get(a), "league_b": self.league.get(b),
            "cross_league": bool(cross),
            "favorite_league_reliability": round(rel, 3),
            "favorite_league_reliable": bool(reliable),
            "min_games_played": int(min_g),
            "high_confidence": high_conf,
            "notes": notes,
            "model": "Elo K32 + marge de victoire (MOV), calibré par ligue. "
                     "AVEUGLE au roster exact, au patch et compare mal entre régions.",
        }

    def champion_winrate(self, champion: str) -> dict:
        name = self._resolve_champ(champion)
        if not name:
            return {"found": False, "champion": champion,
                    "hint": "champion inconnu de la data (orthographe ? ex. 'Wukong', 'Renata Glasc')."}
        wr = self.champ.asof(name, self._champ_date)
        g = self.champ.asof_games(name, self._champ_date)
        return {"found": True, "champion": name, "winrate": round(wr, 3),
                "games": int(g),
                "note": "Winrate pro toutes-ligues sur la saison (lissé). "
                        "PAS pondéré par patch récent : un champion reworké/nerfé "
                        "peut être sur/sous-évalué."}

    def draft_winrate(self, blue_champions: list, red_champions: list) -> dict:
        def side(champs):
            out, wrs = [], []
            for c in champs or []:
                info = self.champion_winrate(c)
                if info.get("found"):
                    out.append({"champion": info["champion"], "winrate": info["winrate"],
                                "games": info["games"]})
                    wrs.append(info["winrate"])
                else:
                    out.append({"champion": c, "winrate": None, "games": 0})
            avg = round(sum(wrs) / len(wrs), 3) if wrs else None
            return out, avg

        b_detail, b_avg = side(blue_champions)
        r_detail, r_avg = side(red_champions)
        diff = round(b_avg - r_avg, 3) if (b_avg is not None and r_avg is not None) else None
        return {
            "blue": {"champions": b_detail, "avg_winrate": b_avg},
            "red": {"champions": r_detail, "avg_winrate": r_avg},
            "blue_minus_red": diff,
            "note": "Signal de draft FAIBLE seul (priors winrate champion). "
                    "Ne dit rien des synergies/contres ni du niveau des joueurs.",
        }

    # ----------------------------------------------------- value / cotes ------
    def value_check(self, prob_team: float, odd_team: float,
                    odd_opponent: float | None = None) -> dict:
        """Compare NOTRE proba à une cote : implicite, dé-vig, edge, verdict discipline."""
        prob_team = float(prob_team)
        odd_team = float(odd_team)
        raw_implied = 1.0 / odd_team
        if odd_opponent:
            inv_o = 1.0 / float(odd_opponent)
            devig = raw_implied / (raw_implied + inv_o)
            vig = round((raw_implied + inv_o - 1) * 100, 1)
        else:
            devig, vig = raw_implied, None
        edge = prob_team - raw_implied            # EV vs cote brute
        if odd_team < 1.20:
            verdict = "⛔ favori court (<1.20) — on ne fade jamais, value rare"
        elif edge > 0.03:
            verdict = "✅ VALUE (+EV) — pari défendable, pose tôt"
        elif edge > 0:
            verdict = "🟡 léger +EV — couvre à peine la vig, prudence"
        else:
            verdict = "❌ -EV — pas de pari"
        return {
            "prob_team": round(prob_team, 3),
            "odd_team": odd_team,
            "implied_raw": round(raw_implied, 3),
            "implied_devig": round(devig, 3),
            "vig_pct": vig,
            "edge_vs_raw": round(edge, 3),
            "verdict": verdict,
            "rule": "value si edge > +3% ; jamais fader un favori < 1.20.",
        }

    # ----------------------------------------------------- série / momentum ---
    def _series_rates(self) -> dict:
        if self._series_res is None:
            from src.models.series_momentum import (analyse, load_games,
                                                    reconstruct_series)
            self._series_res = analyse(reconstruct_series(load_games()))
        return self._series_res

    def _pmap_a(self, a: str, b: str) -> float:
        """Proba calibrée que A gagne UNE map vs B (perspective A)."""
        ea = win_prob(self.elo[a], self.elo[b])
        fav, p_fav_raw = (a, ea) if ea >= 0.5 else (b, 1.0 - ea)
        shrink = self.shrink.get(self.league.get(fav, "?"), 1.0)
        p_fav = max(0.5, min(0.99, calibrate(p_fav_raw, shrink)))
        return p_fav if fav == a else 1.0 - p_fav

    def series_state(self, team_a: str, team_b: str, wins_a: int, wins_b: int,
                     bestof: int = 5, p_map: float | None = None) -> dict:
        """P(A gagne la SÉRIE) selon le score actuel : modèle i.i.d. + taux EMPIRIQUE."""
        from math import comb
        a, _ = self.resolve_team(team_a)
        b, _ = self.resolve_team(team_b)
        if not a or not b:
            return {"error": "équipe(s) introuvable(s)", "a": a, "b": b}
        wins_a, wins_b, bestof = int(wins_a), int(wins_b), int(bestof)
        need = bestof // 2 + 1
        if wins_a >= need:
            return {"result": f"{a} a déjà gagné la série ({wins_a}-{wins_b})."}
        if wins_b >= need:
            return {"result": f"{b} a déjà gagné la série ({wins_b}-{wins_a})."}

        p = float(p_map) if p_map is not None else self._pmap_a(a, b)
        na, nb = need - wins_a, need - wins_b
        q = 1.0 - p
        p_series_a = sum(comb(na - 1 + k, k) * p ** na * q ** k for k in range(nb))

        out = {
            "team_a": a, "team_b": b, "score": f"{wins_a}-{wins_b}", "bestof": bestof,
            "p_map_a": round(p, 3),
            "p_series_a_model": round(p_series_a, 3),
            "p_series_b_model": round(1.0 - p_series_a, 3),
            "model_note": "i.i.d. (maps indépendantes) à partir de notre proba/map calibrée.",
        }
        # taux empirique observé dans la data (leader de l'état courant)
        if wins_a != wins_b:
            leader = a if wins_a > wins_b else b
            res = self._series_rates()
            bo = f"BO{bestof}"
            state = tuple(sorted((wins_a, wins_b), reverse=True))
            recs = res["by_state"].get((bo, state))
            if recs and len(recs) >= 20:
                out["empirical"] = {
                    "leader": leader,
                    "p_series_leader": round(sum(r[0] for r in recs) / len(recs), 3),
                    "p_next_map_leader": round(sum(r[1] for r in recs) / len(recs), 3),
                    "n_series": len(recs),
                    "note": "taux réel observé (toutes ligues). Reflète aussi la force "
                            "du leader, pas que le momentum. À comparer à la cote LIVE.",
                }
        return out

    # ----------------------------------------------------- head-to-head -------
    def head_to_head(self, team_a: str, team_b: str) -> dict:
        a, _ = self.resolve_team(team_a)
        b, _ = self.resolve_team(team_b)
        if not a or not b:
            return {"error": "équipe(s) introuvable(s)", "a": a, "b": b}
        g = self.games
        mask = ((g["blue"] == a) & (g["red"] == b)) | ((g["blue"] == b) & (g["red"] == a))
        sub = g[mask].tail(20)
        wa = wb = 0
        meetings = []
        for row in sub.itertuples():
            winner = row.blue if row.yb == 1 else row.red
            wa += int(winner == a)
            wb += int(winner == b)
            meetings.append({"date": row.date.strftime("%Y-%m-%d"), "winner": winner})
        meetings.reverse()
        return {"team_a": a, "team_b": b, "record_a_b": f"{wa}-{wb}",
                "n_games": len(sub), "recent_meetings": meetings[:8],
                "note": "Saison 2026 uniquement (data du projet)."}

    # ----------------------------------------------------- live win-prob ------
    def _winprob_model(self):
        if self._wp is None:
            try:
                from sklearn.linear_model import LogisticRegression
                proc = ROOT / self.cfg["data"]["processed_dir"] / "team_games.parquet"
                if proc.exists():
                    tg = pd.read_parquet(proc, columns=["golddiffat15", "result"])
                else:  # repli : lecture brute du CSV
                    tg = pd.read_csv(ROOT / self.cfg["data"]["oracle_csv"], low_memory=False,
                                     usecols=["position", "golddiffat15", "result"])
                    tg = tg[tg["position"].str.lower() == "team"]
                tg = tg.dropna(subset=["golddiffat15", "result"])
                X = tg[["golddiffat15"]].astype(float).values
                y = tg["result"].astype(int).values
                clf = LogisticRegression().fit(X, y)
                self._wp = {"coef": float(clf.coef_[0][0]), "intercept": float(clf.intercept_[0])}
            except Exception:  # noqa: BLE001
                self._wp = False
        return self._wp or None

    def live_winprob(self, gold_diff: float, minute: float = 15.0,
                     kill_diff: int | None = None) -> dict:
        """Proba de gagner la game en LIVE depuis l'écart d'or (calibré ~15 min).

        `gold_diff` = or de l'équipe − or adverse (ex. +1500). Modèle logistique
        appris sur golddiff@15 → victoire. Approx hors de ~15 min.
        """
        import math
        model = self._winprob_model()
        if not model:
            return {"error": "modèle live indisponible (data @15 manquante)."}
        gd = float(gold_diff)
        p = 1.0 / (1.0 + math.exp(-(model["intercept"] + model["coef"] * gd)))
        notes = ["Calibré à ~15 min (golddiff@15). Un même écart vaut MOINS tôt "
                 "(<12 min) et PLUS tard. Les kills sans or ni objectifs trompent."]
        if abs(gd) < 800:
            notes.append("Écart d'or faible (<800) → quasi coinflip, peu décisif.")
        return {"p_win": round(p, 3), "gold_diff": gd, "minute": float(minute),
                "kill_diff": kill_diff, "notes": notes}

    # ----------------------------------------------------------- dispatch -----
    def run_tool(self, name: str, args: dict) -> dict:
        try:
            if name == "team_info":
                return self.team_info(args["name"])
            if name == "matchup":
                return self.matchup(args["team_a"], args["team_b"], int(args.get("bestof", 3)))
            if name == "champion_winrate":
                return self.champion_winrate(args["champion"])
            if name == "draft_winrate":
                return self.draft_winrate(args.get("blue_champions", []),
                                          args.get("red_champions", []))
            if name == "value_check":
                return self.value_check(args["prob_team"], args["odd_team"],
                                        args.get("odd_opponent"))
            if name == "series_state":
                return self.series_state(args["team_a"], args["team_b"],
                                         args["wins_a"], args["wins_b"],
                                         int(args.get("bestof", 5)), args.get("p_map"))
            if name == "head_to_head":
                return self.head_to_head(args["team_a"], args["team_b"])
            if name == "live_winprob":
                return self.live_winprob(args["gold_diff"], args.get("minute", 15.0),
                                         args.get("kill_diff"))
        except Exception as exc:  # noqa: BLE001 — on renvoie l'erreur au modèle
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"error": f"outil inconnu: {name}"}


def _split_tokens(s: str) -> list[str]:
    return [w for w in "".join(c if c.isalnum() else " " for c in str(s).lower()).split() if w]


# =========================================================================== #
#  Définition des outils (JSON Schema) + prompt système                        #
# =========================================================================== #
TOOLS = [
    {
        "name": "team_info",
        "description": "Force d'une équipe selon NOTRE modèle : Elo, ligue, fiabilité de "
                       "la ligue, nb de games, et forme récente (10 derniers). À appeler "
                       "pour chaque équipe citée. Ne JAMAIS inventer ces chiffres.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Nom de l'équipe (libre, ex. 'T1', 'KC')."}},
            "required": ["name"],
        },
    },
    {
        "name": "matchup",
        "description": "Proba de victoire d'un match selon notre Elo calibré par ligue : "
                       "proba par game (brute + calibrée), proba de série (BO), drapeaux "
                       "cross-ligue / ligue fiable / data suffisante, et flag 'haute "
                       "confiance' (règle ≥80% mesurée). C'est la source des probas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string"},
                "team_b": {"type": "string"},
                "bestof": {"type": "integer", "description": "1, 3 ou 5 (défaut 3)."},
            },
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "champion_winrate",
        "description": "Winrate pro 'as-of' (saison, toutes ligues) d'un champion + nb de "
                       "games. Pour juger une pioche. Non pondéré par patch.",
        "input_schema": {
            "type": "object",
            "properties": {"champion": {"type": "string"}},
            "required": ["champion"],
        },
    },
    {
        "name": "draft_winrate",
        "description": "Compare deux compositions : winrate moyen des champions de chaque "
                       "side + écart bleu-rouge. Signal de draft faible seul.",
        "input_schema": {
            "type": "object",
            "properties": {
                "blue_champions": {"type": "array", "items": {"type": "string"}},
                "red_champions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["blue_champions", "red_champions"],
        },
    },
    {
        "name": "value_check",
        "description": "Compare une proba (la tienne) à une cote book : proba implicite, "
                       "dé-vig, edge, et verdict selon la discipline (value si edge>+3%, "
                       "jamais fader un favori <1.20). À appeler dès qu'une cote est connue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prob_team": {"type": "number", "description": "Ta proba pour l'équipe (0-1)."},
                "odd_team": {"type": "number", "description": "Cote décimale de l'équipe."},
                "odd_opponent": {"type": "number", "description": "Cote de l'adversaire (pour dé-vig)."},
            },
            "required": ["prob_team", "odd_team"],
        },
    },
    {
        "name": "series_state",
        "description": "P(gagner la SÉRIE) selon le score actuel (ex. mène 2-0) : modèle "
                       "i.i.d. depuis notre proba/map + taux EMPIRIQUE observé (close-out "
                       "réel). À utiliser dès qu'une série est entamée.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string"},
                "team_b": {"type": "string"},
                "wins_a": {"type": "integer", "description": "Maps déjà gagnées par A."},
                "wins_b": {"type": "integer", "description": "Maps déjà gagnées par B."},
                "bestof": {"type": "integer", "description": "3 ou 5 (défaut 5)."},
            },
            "required": ["team_a", "team_b", "wins_a", "wins_b"],
        },
    },
    {
        "name": "head_to_head",
        "description": "Historique des confrontations directes A vs B (saison 2026) : "
                       "bilan + dernières rencontres.",
        "input_schema": {
            "type": "object",
            "properties": {"team_a": {"type": "string"}, "team_b": {"type": "string"}},
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "live_winprob",
        "description": "Proba de gagner la game EN COURS depuis l'écart d'or (or équipe − "
                       "or adverse). Calibré ~15 min. Pour les paris live in-map.",
        "input_schema": {
            "type": "object",
            "properties": {
                "gold_diff": {"type": "number", "description": "Or équipe − or adverse (ex. +1500)."},
                "minute": {"type": "number", "description": "Minute de jeu (contexte)."},
                "kill_diff": {"type": "integer", "description": "Écart de kills (contexte)."},
            },
            "required": ["gold_diff"],
        },
    },
]

SYSTEM_PROMPT = f"""Tu es l'analyste paris LoL esports intégré à l'outil de prédiction de Hugo \
(usage perso, paris). Tu combines NOTRE modèle quantitatif (via des outils) avec les \
infos qualitatives que donne Hugo (draft, stats gol.gg, forme, roster/news, captures \
d'écran de cotes).

NOTRE modèle, ce qu'il sait / ne sait pas :
- Cœur : Elo K32 + marge de victoire (MOV), proba **calibrée par ligue** (aplatie là où \
le modèle est chaotique). Fiabilité = accuracy historique par ligue (seuil fiable ≈ {RELIABLE_ACC:.0%}).
- Priors winrate champion (as-of saison) pour la draft.
- Il est **AVEUGLE** : au roster exact qui joue (remplaçant, bootcamp), au patch, et il \
compare **mal entre régions** (cross-ligue, ex. MSI). C'est là que TES infos qualitatives \
et les captures de cotes corrigent le tir.

OUTILS disponibles (à appeler, ne jamais inventer les chiffres) :
- `team_info` (force/forme d'une équipe) · `matchup` (proba game + série) · `champion_winrate`
  / `draft_winrate` (pioches) · `head_to_head` (confrontations directes).
- `series_state` : dès qu'une SÉRIE est entamée (ex. mène 2-0) → proba de série (modèle + taux réel).
- `live_winprob` : pour un match EN COURS, à partir de l'écart d'or (or équipe − or adverse).
- `value_check` : dès qu'une COTE est connue → implicite, dé-vig, edge, verdict discipline.

RÈGLES :
1. Appelle TOUJOURS les outils pour les chiffres. Pour un match : `matchup` + `team_info` par \
équipe. Série en cours : `series_state`. Game live : `live_winprob`. Cote dispo : `value_check`.
2. Croise le modèle avec le qualitatif (draft, roster, news, capture). En live, méfie-toi des \
kills sans or/objectifs (le score flatte souvent un côté).
3. Sois calibré et honnête. Signale explicitement : cross-ligue, ligue chaotique, ou data \
< {MIN_GAMES_CONF} games → baisse la confiance. Ne survends pas un favori.
4. Discipline paris : on ne fade JAMAIS un favori à cote < 1.20 ; une value n'est retenue \
que si edge = (notre proba − 1/cote) > +3 % ; la vraie value est dans les ligues mineures \
fiables (book plus mou), pas sur les majors/MSI où le book est quasi parfait.
5. Réponse finale STRUCTURÉE et concise (français) :
   - **Proba** : favori, P(game) et P(série) calibrées.
   - **Confiance** : Haute / Moyenne / Faible + pourquoi (fiabilité, data, cross-ligue).
   - **3 facteurs clés** (Elo, forme, draft, roster, info user).
   - **Risques**.
   - **Verdict pari** : value ou pass (si cotes connues), en respectant la discipline.
Réponds en français, en markdown."""


# =========================================================================== #
#  Assistant : boucle agentique Anthropic                                      #
# =========================================================================== #
class Assistant:
    """Agent Claude qui interroge `DataContext` via tool use (+ vision)."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 ctx: DataContext | None = None) -> None:
        import anthropic  # import paresseux : la page marche sans la lib pour le reste
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.ctx = ctx or DataContext()

    def ask(self, user_text: str, images: list[dict] | None = None,
            history: list[dict] | None = None, on_event=None,
            max_tokens: int = 4000) -> str:
        """Pose une question. `images` = [{media_type, bytes}]. `history` = tours texte.

        `on_event(kind, payload)` (optionnel) : kind ∈ {tool_call, tool_result}.
        """
        content: list[dict] = []
        for img in images or []:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": img["media_type"],
                           "data": base64.standard_b64encode(img["bytes"]).decode("utf-8")},
            })
        content.append({"type": "text", "text": user_text})

        messages = list(history or []) + [{"role": "user", "content": content}]

        for _ in range(MAX_TOOL_STEPS):
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text").strip()

            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    if on_event:
                        on_event("tool_call", {"name": block.name, "input": block.input})
                    result = self.ctx.run_tool(block.name, dict(block.input))
                    if on_event:
                        on_event("tool_result", {"name": block.name, "result": result})
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
            messages.append({"role": "user", "content": tool_results})

        return "⚠️ Trop d'étapes d'outils sans réponse finale (boucle interrompue)."


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Présente-toi et explique ce que tu peux analyser."
    key = load_api_key()
    if not key:
        print("Pas de clé : pose ANTHROPIC_API_KEY ou crée le fichier anthropic.key.")
        return

    def log(kind, payload):
        if kind == "tool_call":
            print(f"  [outil] {payload['name']}({payload['input']})")

    agent = Assistant(api_key=key)
    print(agent.ask(question, on_event=log))


if __name__ == "__main__":
    main()

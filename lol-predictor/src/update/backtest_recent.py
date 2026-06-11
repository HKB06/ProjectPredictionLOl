"""Backtest récent : « est-ce qu'on aurait trouvé le vainqueur ? » sur les N derniers jours.

Rejoue l'Elo toutes-ligues en chronologique. Pour CHAQUE game, on prédit le vainqueur
avec l'Elo d'AVANT la game (aucune fuite), côté-neutre (comme la watchlist pré-match),
puis on compare au résultat réel. On agrège par jour, par ligue, et on isole les
favoris ⭐ (≥62 %) — nos « calls confiants » — et les upsets.

⚠️ Ne valide que les games DÉJÀ dans le CSV (MAJ ~quotidienne) : les games du soir
du jour J n'y sont en général pas encore.

Usage :
    python -m src.update.backtest_recent            # 4 derniers jours
    python -m src.update.backtest_recent --days 7
    python -m src.update.backtest_recent --since 2026-06-08
"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config

OUT_PATH = ROOT.parent / "BACKTEST_RECENT.md"


def replay(cfg: dict, cutoff: pd.Timestamp):
    """Utilise le modèle de production (K32 + MOV) en walk-forward, hors cold-start."""
    from src.models.eval_models import production_records
    rec = production_records(cfg)
    rec = rec[rec["nmin"] >= 5]
    max_date = rec["date"].max()
    sub = rec[rec["date"] >= cutoff]
    recs = [{
        "date": row.date, "league": row.league, "blue": row.blue, "red": row.red,
        "fav": row.favori, "p_fav": row.proba_fav, "winner": row.vainqueur,
        "correct": bool(row.correct), "cold": False,
    } for row in sub.itertuples()]
    return recs, max_date


def _acc(rs: list[dict]) -> str:
    if not rs:
        return "—"
    c = sum(r["correct"] for r in rs)
    return f"{c}/{len(rs)} ({c / len(rs) * 100:.0f}%)"


def build_report(recs: list[dict], max_date, cutoff) -> list[str]:
    L: list[str] = []
    L.append("# Backtest récent — « a-t-on trouvé le vainqueur ? »")
    L.append("")
    if not recs:
        L.append(f"_Aucune game dans la data depuis {cutoff:%Y-%m-%d} (data jusqu'au {max_date:%Y-%m-%d})._")
        return L
    n = len(recs)
    c = sum(r["correct"] for r in recs)
    strong = [r for r in recs if r["p_fav"] >= 0.62]
    close = [r for r in recs if r["p_fav"] < 0.62]
    L.append(f"> Data jusqu'au **{max_date:%Y-%m-%d}** · fenêtre depuis **{cutoff:%Y-%m-%d}** · "
             f"**{n} games** (toutes ligues, côté-neutre, sans fuite).")
    L.append("")
    L.append(f"## GLOBAL : **{c}/{n} = {c / n * 100:.1f}%** de vainqueurs trouvés")
    L.append(f"- Favoris **⭐ confiants (≥62 %)** : **{_acc(strong)}** ← nos vrais calls")
    L.append(f"- Matchs **serrés (<62 %)** : {_acc(close)} (≈ pile/face, normal)")
    L.append("")

    by_day = defaultdict(list)
    for r in recs:
        by_day[r["date"].date()].append(r)
    L.append("## Par jour")
    L.append("| Jour | Précision | ⭐ favoris |")
    L.append("|---|---|---|")
    for day in sorted(by_day):
        rs = by_day[day]
        st = [r for r in rs if r["p_fav"] >= 0.62]
        L.append(f"| {day} | {_acc(rs)} | {_acc(st)} |")
    L.append("")

    by_lg = defaultdict(list)
    for r in recs:
        by_lg[r["league"]].append(r)
    L.append("## Par ligue (triées par volume)")
    L.append("| Ligue | Précision | ⭐ favoris |")
    L.append("|---|---|---|")
    for lg in sorted(by_lg, key=lambda k: -len(by_lg[k])):
        rs = by_lg[lg]
        st = [r for r in rs if r["p_fav"] >= 0.62]
        L.append(f"| {lg} | {_acc(rs)} | {_acc(st)} |")
    L.append("")

    wrong_strong = [r for r in strong if not r["correct"]]
    L.append(f"## Upsets — favori ⭐ qui a PERDU ({len(wrong_strong)})")
    if wrong_strong:
        for r in sorted(wrong_strong, key=lambda x: x["date"]):
            L.append(f"- {r['date']:%m-%d} · {r['league']} — **{r['fav']} {r['p_fav']*100:.0f}%** "
                     f"a perdu vs **{r['winner']}**")
    else:
        L.append("- *(aucun : tous nos favoris confiants ont gagné)*")
    L.append("")
    L.append("_Note : côté-neutre (pré-game). Le side bleu (~+4 pts) explique une partie des ratés serrés._")
    return L


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Elo des derniers jours")
    parser.add_argument("--days", type=int, default=4, help="nb de jours en arrière (def. 4)")
    parser.add_argument("--since", type=str, default=None, help="date de départ YYYY-MM-DD")
    args = parser.parse_args()

    cfg = load_config()
    if args.since:
        cutoff = pd.to_datetime(args.since)
    else:
        cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=args.days - 1))

    recs, max_date = replay(cfg, cutoff)
    lines = build_report(recs, max_date, cutoff)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Console (ASCII safe : Windows cp1252 ne gère pas tous les unicode)
    repl = {"⭐": "*", "←": "<-", "≥": ">=", "·": "-", "≈": "~", "’": "'"}
    for ln in lines:
        for k, v in repl.items():
            ln = ln.replace(k, v)
        print(ln.encode("ascii", "replace").decode("ascii"))
    print(f"\n-> ecrit dans {OUT_PATH}")


if __name__ == "__main__":
    main()

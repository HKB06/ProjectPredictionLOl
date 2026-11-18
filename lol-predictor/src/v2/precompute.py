"""Tout ce que le site affiche, calculé UNE fois et gardé sur disque.

Rejouer quatre saisons puis réentraîner les modèles prend d'une à plusieurs minutes
selon la machine. Le site ne doit pas faire attendre pour ça : la collecte automatique
(`src.update.refresh_all`, source `model`) refait le calcul en tâche de fond dès que les
données ou le code changent, et le site ne fait que relire le résultat.

    data/cache/engine.pkl    le moteur (jeu de données, prédictions, états, modèles)
    data/cache/engine.json   avec quoi il a été calculé (signature) et en combien de temps

Règles côté site (`get`) :
  - moteur à jour                       -> on le relit ;
  - moteur en retard, données changées
    il y a moins de `STALE_H` heures    -> on le relit quand même : la collecte est en
                                           train de le refaire ;
  - pas de moteur, ou en retard depuis
    longtemps (pas de collecte en place) -> on le calcule sur place et on l'enregistre.

Usage :
    python -m src.v2.precompute            # recalcule si nécessaire
    python -m src.v2.precompute --force
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import pandas as pd

from src.ingest.load_oracle import ROOT, load_config, season_files

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = ROOT / "data" / "cache"
CACHE = DIR / "engine.pkl"
META = DIR / "engine.json"
FORMAT = 3              # à augmenter si le contenu du moteur change de forme
STALE_H = 1.0           # retard toléré pendant que la collecte refait le calcul
# Code dont dépend le résultat : le modifier doit déclencher un recalcul.
_CODE = ["config.yaml", "src/update/elo.py", "src/features/context_factors.py",
         "src/features/soloq_prep.py", "src/ingest/load_oracle.py", "src/ingest/leaguepedia.py"]


def _inputs(cfg: dict | None = None) -> tuple[list, list]:
    """(fichiers de données, fichiers de code) dont dépend le moteur."""
    cfg = cfg or load_config()
    data = [p for _, p in season_files(cfg)]
    data += sorted((ROOT / "data" / "external" / "leaguepedia").glob("*.csv"))
    code = [ROOT / rel for rel in _CODE] + sorted((ROOT / "src" / "v2").glob("*.py"))
    return [p for p in data if p.exists()], [p for p in code if p.exists()]


def signature(cfg: dict | None = None) -> list:
    """Empreinte des entrées : nom, taille et date de modification de chaque fichier."""
    data, code = _inputs(cfg)
    return [FORMAT] + [[p.relative_to(ROOT).as_posix(), p.stat().st_size, int(p.stat().st_mtime)]
                       for p in data + code]


def _meta() -> dict:
    try:
        return json.loads(META.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_fresh() -> bool:
    return CACHE.exists() and _meta().get("signature") == signature()


def _changed_ago_h() -> float:
    """Heures écoulées depuis la dernière modification d'une entrée (données ou code)."""
    data, code = _inputs()
    newest = max((p.stat().st_mtime for p in data + code), default=0.0)
    return (time.time() - newest) / 3600


def _usable() -> bool:
    return CACHE.exists() and (is_fresh() or _changed_ago_h() < STALE_H)


def version() -> float:
    """Clé de cache du site : change à chaque nouveau moteur disponible."""
    if _usable():
        return CACHE.stat().st_mtime
    data, code = _inputs()
    return max((p.stat().st_mtime for p in data + code), default=0.0)


def compute(cfg: dict | None = None) -> dict:
    """Le calcul complet : rejeu des saisons, walk-forward, table de la saison, modèles."""
    from src.v2.model import (MAIN, SEASON_WARMUP_DAYS, Predictor, build_dataset,
                              comparison_window, first_evaluable, has_history, season_board,
                              season_origin, walk_forward)
    from src.v2.season import build_season

    df, states = build_dataset(cfg)
    try:                                 # notes et résumés des matchs (page « Résumés »)
        from src.v2 import perf
        perf.load(cfg, rebuild=True)
    except Exception as e:               # noqa: BLE001 - l'affichage ne bloque pas le moteur
        print(f"[perf] cache des notes non écrit ({type(e).__name__}: {e})")
    res = walk_forward(df, SEASON_WARMUP_DAYS)
    cmp = comparison_window(res, df)
    games, series = build_season(df, res, cfg)
    board = None
    if has_history(df):
        # Saisons passées : les variantes principales suffisent (pas d'ablation).
        past = walk_forward(df, since=first_evaluable(df), until=season_origin(df),
                            ablation=False)
        keep = ["season", "nmin", "yb", *MAIN]
        board = season_board(pd.concat([past[keep], cmp[keep]], ignore_index=True))
    return {"df": df, "res": res, "cmp": cmp, "states": states,
            "predictor": Predictor(df, states, res), "games": games, "series": series,
            "season_board": board}


def build(cfg: dict | None = None) -> dict:
    """Calcule le moteur et l'enregistre (écriture atomique)."""
    sig = signature(cfg)
    t0 = time.time()
    eng = compute(cfg)
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_name(f"{CACHE.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(eng, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE)
    meta = {"signature": sig, "built": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "seconds": round(time.time() - t0), "games": int(len(eng["df"])),
            "seasons": sorted(int(y) for y in eng["df"]["season"].unique())}
    tmp = META.with_name(f"{META.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tmp.replace(META)
    return eng


def _read() -> dict | None:
    try:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001 - fichier absent, tronqué ou d'une autre version du code
        return None


_mem: dict = {}


def _cached() -> dict | None:
    """Le moteur du disque, gardé en mémoire tant que le fichier ne change pas (une seule
    copie par processus : le site, les picks et les bilans le partagent)."""
    try:
        key = CACHE.stat().st_mtime
    except OSError:
        return None
    if _mem.get("key") != key:
        eng = _read()
        if eng is None:
            return None
        _mem.clear()
        _mem.update(key=key, eng=eng)
    return _mem["eng"]


def peek() -> dict | None:
    """Le dernier moteur calculé, même en retard, SANS jamais lancer de calcul (None s'il
    n'y en a pas). Pour les picks : mieux vaut le modèle d'hier que dix minutes d'attente."""
    return _cached()


def get() -> dict:
    """Le moteur pour le site : relu si possible, calculé sinon (cf. règles en tête)."""
    if _usable():
        eng = _cached()
        if eng is not None:
            return eng
    return build()


def warm(force: bool = False) -> str:
    """Pour la collecte : refait le calcul seulement si les données ou le code ont changé."""
    if not force and is_fresh():
        m = _meta()
        return f"à jour (calculé le {m.get('built', '?')} UTC en {m.get('seconds', '?')} s)"
    eng = build()
    m = _meta()
    return (f"recalculé en {m['seconds']} s · {m['games']} games · saisons "
            f"{', '.join(map(str, m['seasons']))} · {len(eng['series'])} séries cette saison")


def main() -> None:
    ap = argparse.ArgumentParser(description="Précalcule le moteur du site")
    ap.add_argument("--force", action="store_true", help="recalcule même si tout est à jour")
    print(warm(force=ap.parse_args().force))


if __name__ == "__main__":
    main()

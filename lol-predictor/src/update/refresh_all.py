"""UNE commande pour garder toutes les données à jour, faite pour tourner en boucle sur un serveur.

    python -m src.update.refresh_all

À lancer toutes les 30 minutes (cron, systemd ou Planificateur de tâches). Chaque source a
son propre rythme : la commande ne refait que ce qui est dû, donc l'appeler souvent ne
coûte rien et ne martèle aucun site.

  source        rythme   clé            ce que ça met à jour
  oracle        1/jour   -              CSV Oracle's Elixir (toutes les games) + tables.
                                        Une réussite par jour suffit : dès minuit (heure du
                                        serveur), on essaie toutes les 30 min jusqu'à ce que
                                        ça passe (quota Google Drive), puis plus rien
                                        jusqu'au lendemain
  history       1 fois   -              saisons passées (config `data.history_years`) :
                                        CSV Oracle + coachs et lieux Leaguepedia
  leaguepedia   6 h      -              tournois, games, rosters, coachs, comptes soloQ
  rosters       6 h      -              effectif ACTUEL de chaque équipe suivie (mercato,
                                        changements en cours de saison) + journal des
                                        arrivées/départs ; ses joueurs entrent au suivi soloQ
  polymarket    1 h      -              prix d'avant-match de chaque match terminé
  odds          2 h      ODDS_API_KEY   cotes de clôture des bookmakers (odds-api.io)
  model         chaque   -              précalcul des deux modèles pour le site, refait
                passage                 seulement si les données ou le code ont changé
  ladder        168 h    RIOT_API_KEY   comptes des pros trouvés dans les classements
                                        Challenger/GM (12 serveurs), ajoutés au relevé
  sites         chaque   -              comptes actifs des pros, smurfs compris, lus sur
                passage                 TrackingThePros et lolpros.gg (fiche relue 1 fois
                                        par semaine, ~30 min max par passage)
  soloq         12 h     RIOT_API_KEY   relevé quotidien du rang soloQ des pros (2 essais
                                        par jour, le second ne refait rien)
  soloq_hist    chaque   RIOT_API_KEY   historique des parties classées des pros (MATCH-V5),
                passage                 reconstruit puis tenu à jour, ~55 min par passage

Sur le serveur, les sources Riot (file « soloq ») tournent dans leur propre tâche planifiée
(`--only ladder sites soloq soloq_hist`) et la tâche principale les saute
(`--skip ladder sites soloq soloq_hist`) : leurs longs passages ne retardent pas le reste.

Garanties :
  - une source en panne n'arrête pas les autres ; l'erreur est notée et on réessaie
    au passage suivant ;
  - deux lancements ne se marchent jamais dessus (verrou par file) : le relevé soloQ,
    long (~1 h 30), tourne dans sa propre file et ne retarde pas le reste ;
  - `data/refresh_status.json` dit, pour chaque source, quand elle a réussi pour la
    dernière fois. Le site l'affiche.

Options :
    --force            ignore les rythmes (tout refaire maintenant)
    --only a b / --skip a b
    --status           affiche l'état sans rien lancer
Code de sortie : 0 si tout ce qui a tourné a réussi, 1 sinon.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

from src.ingest.load_oracle import ROOT, load_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATUS = ROOT / "data" / "refresh_status.json"
LOCK_STALE_H = 8          # un verrou plus vieux vient d'un lancement tué : on le reprend
ODDS_BUDGET = 40          # requêtes odds-api.io par passage (toutes les 2 h)
DAILY = "jour"            # rythme « une réussite par jour, à partir de minuit »
HIST_MINUTES = 55         # historique soloQ : durée d'un passage (tâche planifiée à part)
SITES_MINUTES = 30        # comptes TrackingThePros / lolpros.gg : durée max d'un passage


class Skip(Exception):
    """La source n'a pas pu tourner pour une raison normale (clé absente…)."""


# ------------------------------------------------------------------ sources
def _year() -> int:
    return dt.datetime.now(dt.timezone.utc).year


def oracle() -> str:
    from src.ingest import build_match_table
    from src.update.download_data import download_latest
    path = download_latest()
    build_match_table.main()
    return f"CSV Oracle {path.stat().st_size / 1e6:.0f} Mo"


def history() -> str:
    """Saisons passées rejouées par le modèle : téléchargées une fois, jamais ensuite.

    Pour chaque saison de `data.history_years` : le CSV Oracle's Elixir, puis les tables
    Leaguepedia (coachs, lieux) qui vont avec.
    """
    from src.ingest import leaguepedia as lp
    from src.update.download_data import download_season
    have, blocked, lp_ko = [], [], []
    for year in sorted(load_config()["data"].get("history_years") or []):
        try:
            download_season(year)
            have.append(str(year))
        except Exception:  # noqa: BLE001 - une saison bloquée n'empêche pas les autres
            blocked.append(str(year))
            continue
        try:
            for name in lp.TABLES:
                lp.load(name, year)              # en cache : aucune requête
        except Exception:  # noqa: BLE001
            lp_ko.append(str(year))
    problems = []
    if blocked:
        problems.append(f"saisons bloquées par le quota Google Drive : {', '.join(blocked)}")
    if lp_ko:
        problems.append(f"Leaguepedia incomplet : {', '.join(lp_ko)}")
    if problems:
        raise RuntimeError(" · ".join(problems)
                           + (f" · déjà présentes : {', '.join(have)}" if have else ""))
    return "saisons présentes : " + ", ".join(have) + " (Oracle + Leaguepedia)"


def leaguepedia() -> str:
    from src.ingest import leaguepedia as lp
    from src.ingest.soloq import build_accounts
    sizes, errors = [], []
    for name in lp.TABLES:
        try:
            sizes.append(f"{name} {len(lp.load(name, _year(), refresh=True))}")
        except Exception as exc:  # noqa: BLE001 - les autres tables restent utiles
            errors.append(f"{name} : {exc}")
    if not errors or not any(e.startswith("players") for e in errors):
        build_accounts(_year())
    if errors:
        raise RuntimeError(" | ".join(errors) + (" · OK : " + ", ".join(sizes) if sizes else ""))
    return ", ".join(sizes)


def rosters() -> str:
    from src.ingest import rosters as ro
    return ro.message(ro.run(_year()))


def polymarket() -> str:
    from src.update.polymarket_history import backfill
    res = backfill(_year())
    msg = f"{res['new']} nouveaux matchs, {res['total']} en archive"
    if res["failed"]:
        msg += f", {res['failed']} à redemander"
    return msg


def odds() -> str:
    from src.update.odds_history import backfill
    from src.update.oddsapi import load_key
    if not load_key():
        raise Skip("clé ODDS_API_KEY absente")
    # La clé (100 requêtes/heure) sert aussi au site : on ne prend qu'une part du quota.
    res = backfill(days=30, budget=ODDS_BUDGET)
    return (f"{res.get('fetched', 0)} cotes récupérées, {res.get('added', 0)} matchs découverts, "
            f"{res.get('pending', '?')} en attente, {res.get('ok', '?')} avec cote "
            f"({res.get('used', 0)} requêtes)")


def soloq() -> str:
    from src.ingest import soloq as sq
    if not sq.load_key():
        raise Skip("clé RIOT_API_KEY absente")
    res = sq.snapshot(_year())
    return (f"{res['new']} comptes relevés, {res['already']} déjà faits aujourd'hui, "
            f"{res['missing']} introuvables")


def ladder() -> str:
    from src.ingest import soloq as sq
    from src.ingest import soloq_ladder as sl
    if not sq.load_key():
        raise Skip("clé RIOT_API_KEY absente")
    res = sl.discover(_year())
    return (f"{res['ladder']} comptes lus, {res['players']} pros reconnus "
            f"({res['found']}), contrôle wiki : {res['check_wrong']} erreurs "
            f"sur {res['check_matched']}")


def sites() -> str:
    from src.ingest import soloq_sites as ss
    return ss.message(ss.run(_year(), minutes=SITES_MINUTES))


def soloq_hist() -> str:
    from src.ingest import soloq as sq
    from src.ingest import soloq_history as sh
    if not sq.load_key():
        raise Skip("clé RIOT_API_KEY absente")
    res = sh.run(_year(), minutes=HIST_MINUTES)
    return sh.message(res) + f" · {res['calls']} requêtes"


def model() -> str:
    """Précalcul du moteur du site : rien à faire tant que les entrées n'ont pas changé."""
    from src.v2 import precompute
    return precompute.warm()


def _low_priority() -> None:
    """La collecte ne doit jamais gêner ce qui tourne à côté sur le serveur."""
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001 - confort, jamais bloquant
        pass


# nom -> (fonction, rythme en heures, file)
SOURCES = {
    "oracle": (oracle, DAILY, "main"),   # quota Drive : une réussite par jour suffit
    "history": (history, 168, "main"),
    "leaguepedia": (leaguepedia, 6, "main"),
    "rosters": (rosters, 6, "main"),      # après leaguepedia : il relit ses tables
    "polymarket": (polymarket, 1, "main"),
    "odds": (odds, 2, "main"),
    "model": (model, 0, "main"),          # après les données : il relit ce qui vient d'arriver
    "ladder": (ladder, 168, "soloq"),     # avant le relevé : il y ajoute ses comptes
    "sites": (sites, 0, "soloq"),         # idem ; ne fait presque rien quand tout est frais
    "soloq": (soloq, 12, "soloq"),            # 2 essais par jour : aucun jour UTC sauté
    "soloq_hist": (soloq_hist, 0, "soloq"),   # avance ~55 min par passage, reprenable
}


# ------------------------------------------------------------- état & verrou
def load_status() -> dict:
    try:
        return json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_status(update: dict) -> None:
    """Relit avant d'écrire : les deux files peuvent écrire en même temps."""
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    status = load_status()
    status.update(update)
    tmp = STATUS.with_name(STATUS.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(status, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def _lock(lane: str) -> bool:
    path = STATUS.with_name(f".refresh_{lane}.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > LOCK_STALE_H * 3600:
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _unlock(lane: str) -> None:
    STATUS.with_name(f".refresh_{lane}.lock").unlink(missing_ok=True)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _age_h(stamp: str | None) -> float:
    if not stamp:
        return float("inf")
    t = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - t).total_seconds() / 3600


def _fresh(prev: dict, every_h) -> bool:
    """La dernière réussite suffit-elle encore ?"""
    if every_h == DAILY:
        stamp = prev.get("last_ok")
        if not stamp:
            return False
        t = dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        return t.astimezone().date() == dt.datetime.now().astimezone().date()
    return _age_h(prev.get("last_ok")) < every_h


def _rhythm(every_h) -> str:
    return "1 fois par jour dès minuit" if every_h == DAILY else f"{every_h} h"


def _log(msg: str) -> None:
    print(f"[{_now()} UTC] {msg}", flush=True)


# -------------------------------------------------------------------- moteur
def run(names: list[str], force: bool = False) -> bool:
    """Lance les sources dues. Renvoie False si l'une d'elles a échoué."""
    ok = True
    for lane in dict.fromkeys(SOURCES[n][2] for n in names):
        if not _lock(lane):
            _log(f"file « {lane} » : un autre lancement est en cours, on passe.")
            continue
        try:
            for name in (n for n in names if SOURCES[n][2] == lane):
                func, every_h, _ = SOURCES[name]
                prev = load_status().get(name, {})
                # Une source en échec, ou ignorée faute de clé, est retentée à chaque passage
                # (et non à son rythme normal) : elle repart dès que le problème est réglé.
                # Rythme quotidien : une réussite dans la journée suffit, même si un essai
                # a échoué depuis.
                if not force and (prev.get("ok") or every_h == DAILY) \
                        and not prev.get("skipped") and _fresh(prev, every_h):
                    continue
                t0 = time.time()
                entry = {**prev, "last_run": _now(), "every_h": every_h, "skipped": False}
                try:
                    entry.update(ok=True, last_ok=_now(), message=func())
                    _log(f"{name:12} OK   {entry['message']}  ({time.time() - t0:.0f}s)")
                except Skip as exc:
                    entry.update(ok=True, skipped=True, message=f"ignoré : {exc}")
                    _log(f"{name:12} --   {entry['message']}")
                except Exception as exc:  # noqa: BLE001 - une source en panne n'arrête pas les autres
                    ok = False
                    entry.update(ok=False, message=f"{type(exc).__name__} : {exc}"[:500])
                    _log(f"{name:12} ÉCHEC {entry['message']}")
                    traceback.print_exc()
                entry["seconds"] = round(time.time() - t0)
                _save_status({name: entry})
        finally:
            _unlock(lane)
    return ok


def print_status() -> None:
    status = load_status()
    for name, (_, every_h, _) in SOURCES.items():
        e = status.get(name)
        if not e:
            print(f"{name:12} jamais lancé")
            continue
        age = _age_h(e.get("last_ok"))
        state = "ignoré" if e.get("skipped") else "OK" if e.get("ok") else "ÉCHEC"
        when = "jamais réussi" if age == float("inf") else f"dernier succès il y a {age:.1f} h"
        print(f"{name:12} {state:6} {when} (rythme {_rhythm(every_h)}) · {e.get('message', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Actualise toutes les données (à planifier)")
    ap.add_argument("--force", action="store_true", help="ignore les rythmes")
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="sources à traiter")
    ap.add_argument("--skip", nargs="*", choices=list(SOURCES), default=[], help="sources à sauter")
    ap.add_argument("--status", action="store_true", help="affiche l'état et sort")
    args = ap.parse_args()
    if args.status:
        print_status()
        return 0
    _low_priority()
    names = [n for n in (args.only or SOURCES) if n not in args.skip]
    return 0 if run(names, force=args.force) else 1


if __name__ == "__main__":
    sys.exit(main())

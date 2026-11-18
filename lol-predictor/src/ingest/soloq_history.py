"""Historique soloQ des pros reconstruit à partir de l'API Riot (MATCH-V5).

Le relevé quotidien (`soloq snapshot`) ne voit que l'état du jour : il faudrait des mois
pour avoir de quoi tester le signal. Riot garde en revanche ~2 ans de parties classées
(vérifié : rien avant ~nov. 2024). On reconstruit donc, pour chaque compte suivi, toutes
ses parties classées solo depuis START, avec la date, le résultat et le champion.

Trois étapes, menées en parallèle sur les routages Riot (americas, asia, europe, sea) dont
les limites sont séparées :
  1. retrouver l'identifiant (PUUID) de chaque compte suivi ;
  2. lister ses parties classées (100 par requête) — mise à jour ensuite une fois par jour ;
  3. télécharger le détail de chaque partie (1 requête par partie, une partie entre pros sert
     à tous les comptes suivis qui y ont joué). D'abord un échantillon réparti dans le temps
     par serveur : il permet de dater TOUTES les parties (les numéros de partie croissent avec
     le temps) avant même d'avoir tout téléchargé ; puis le reste, des plus récentes aux plus
     anciennes.

Reprenable : chaque passage avance pendant `minutes` puis s'arrête proprement ; le suivant
reprend là où il en était. Lancé par `refresh_all` (source `soloq_hist`).

Fichiers (data/external/soloq/history/) :
  accounts.json   compte -> PUUID, routage, date de la dernière liste
  ids.csv         puuid, match_id         (toutes les parties classées listées)
  details.csv     une ligne par (partie, compte suivi) : date, durée, victoire, champion…
  fetched.txt     parties déjà demandées (y compris introuvables)

Usage :
    python -m src.ingest.soloq_history --minutes 10
    python -m src.ingest.soloq_history --status
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import sys
import time
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.ingest import soloq as sq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = sq.DIR / "history"
START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)   # Riot ne garde que ~2 ans
QUEUE = 420                     # classée solo/duo
IDS_EVERY_H = 20                # liste des parties d'un compte : remise à jour 1 fois par jour
CALIB_N = 60                    # parties datées par serveur avant tout le reste
MATCH_ROUTING = {
    "KR": "asia", "JP": "asia",
    "EUW": "europe", "EUNE": "europe", "TR": "europe", "RU": "europe", "ME": "europe",
    "NA": "americas", "BR": "americas", "LAN": "americas", "LAS": "americas",
    "OCE": "sea", "TW": "sea", "VN": "sea", "SEA": "sea", "SG": "sea", "PH": "sea", "TH": "sea",
}
DETAIL_COLS = ["match_id", "ts", "duration", "puuid", "player", "region", "win", "champion",
               "position", "kills", "deaths", "assists", "early_end"]


def _gid(match_id: str) -> int:
    return int(match_id.rsplit("_", 1)[1])


def _platform(match_id: str) -> str:
    return match_id.rsplit("_", 1)[0]


class Store:
    """Fichiers de l'historique, écrits au fil de l'eau (ajout en fin de fichier)."""

    def __init__(self) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        self.acc_path = DIR / "accounts.json"
        self.accounts: dict = json.loads(self.acc_path.read_text(encoding="utf-8")) \
            if self.acc_path.exists() else {}
        self.ids_path, self.det_path, self.fet_path = (DIR / "ids.csv", DIR / "details.csv",
                                                       DIR / "fetched.txt")
        self.fetched = set(self.fet_path.read_text(encoding="utf-8").split()) \
            if self.fet_path.exists() else set()
        self.ids: dict[str, set] = {}               # puuid -> parties
        if self.ids_path.exists():
            for pu, mid in csv.reader(self.ids_path.open(encoding="utf-8")):
                self.ids.setdefault(pu, set()).add(mid)
        # (partie, compte) qui ont déjà leur ligne de détail, et parties déjà redemandées
        self.rows: set[int] = set()
        if self.det_path.exists():
            with self.det_path.open(encoding="utf-8", newline="") as fh:
                rd = csv.reader(fh)
                next(rd, None)
                for row in rd:
                    if len(row) > 3:
                        self.rows.add(hash((row[0], row[3])))
        self.re_path = DIR / "refetched.txt"
        self.refetched = set(self.re_path.read_text(encoding="utf-8").split()) \
            if self.re_path.exists() else set()
        self._re = self.re_path.open("a", encoding="utf-8")
        new = not self.det_path.exists()
        self._det = self.det_path.open("a", encoding="utf-8", newline="")
        self.det = csv.writer(self._det)
        if new:
            self.det.writerow(DETAIL_COLS)
        self._fet = self.fet_path.open("a", encoding="utf-8")
        self._ids = self.ids_path.open("a", encoding="utf-8", newline="")

    def save_accounts(self) -> None:
        tmp = self.acc_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.accounts, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.acc_path)

    def add_ids(self, puuid: str, mids: list[str]) -> None:
        have = self.ids.setdefault(puuid, set())
        w = csv.writer(self._ids)
        for m in mids:
            if m not in have:
                have.add(m)
                w.writerow([puuid, m])
        self._ids.flush()

    def mark(self, mid: str) -> None:
        if mid in self.fetched:                      # partie redemandée : une seule fois
            self.refetched.add(mid)
            self._re.write(mid + "\n")
            return
        self.fetched.add(mid)
        self._fet.write(mid + "\n")

    def missing(self, host: str) -> set[str]:
        """Parties déjà détaillées où il manque la ligne d'un compte suivi : le compte
        n'était pas encore connu quand la partie a été lue (deux pros dans la même partie,
        compte découvert plus tard). Redemandées une fois chacune."""
        out = set()
        for pu, info in self.accounts.items():
            if info["routing"] != host:
                continue
            for m in self.ids.get(pu, ()):
                if m in self.fetched and m not in self.refetched and hash((m, pu)) not in self.rows:
                    out.add(m)
        return out

    def flush(self) -> None:
        for f in (self._det, self._fet, self._ids, self._re):
            f.flush()
        self.save_accounts()

    def close(self) -> None:
        self.flush()
        for f in (self._det, self._fet, self._ids, self._re):
            f.close()


def _accounts(year: int) -> pd.DataFrame:
    acc = sq.tracked_accounts(year)
    acc["rid"] = acc["game_name"] + "#" + acc["tag"].astype(str)
    return acc.drop_duplicates("rid")


def _host_work(host: str, api: sq.Riot, st: Store, acc: pd.DataFrame, puuids: dict,
               aliases: dict, owner: dict):
    """Générateur : UNE requête Riot par pas, toutes sur le même routage `host`."""
    now = time.time()
    # 1. PUUID des comptes dont le compte Riot se cherche sur ce routage
    for r in acc[acc["acc_routing"] == host].itertuples():
        if r.rid in puuids:
            continue
        name, _, tag = aliases.get(r.rid, {}).get("current", r.rid).rpartition("#")
        a = api.get(host, f"/riot/account/v1/accounts/by-riot-id/"
                          f"{quote(name, safe='')}/{quote(tag, safe='')}")
        puuids[r.rid] = a["puuid"] if a else None
        if a:
            owner.setdefault(a["puuid"], (r.player, r.region))
        yield
    # 2. liste des parties classées des comptes dont les parties sont sur ce routage
    for r in acc[acc["match_routing"] == host].itertuples():
        pu = puuids.get(r.rid)
        if not pu:
            continue
        info = st.accounts.setdefault(pu, {"rid": r.rid, "player": r.player,
                                           "region": r.region, "routing": host, "until": 0})
        if now - info["until"] < IDS_EVERY_H * 3600:
            continue
        since = int(max(START.timestamp(), info["until"] - 7200))
        got, start, t0, ok = [], 0, int(time.time()), True
        while True:
            ids = api.get(host, f"/lol/match/v5/matches/by-puuid/{pu}/ids?startTime={since}"
                                f"&queue={QUEUE}&start={start}&count=100")
            yield
            if ids is None:
                ok = False
                break
            got += ids
            if len(ids) < 100:
                break
            start += 100
        st.add_ids(pu, got)
        if ok:
            info["until"] = t0
    # 3. détail des parties : échantillon daté par serveur, puis le reste (récent d'abord)
    mids = {m for pu, info in st.accounts.items() if info["routing"] == host
            for m in st.ids.get(pu, ())} - st.fetched
    redo = st.missing(host)
    if not mids and not redo:
        return
    by_plat: dict[str, list[str]] = {}
    for m in mids:
        by_plat.setdefault(_platform(m), []).append(m)
    order = []
    for plat, ms in by_plat.items():
        ms.sort(key=_gid)
        dated = sum(1 for m in st.fetched if m.startswith(plat + "_"))
        if dated < CALIB_N:
            idx = np.unique(np.linspace(0, len(ms) - 1, min(CALIB_N, len(ms))).astype(int))
            order += [ms[i] for i in idx]
    rest = _recent_first(mids - set(order))
    for mid in order + rest + _recent_first(redo):
        m = api.get(host, f"/lol/match/v5/matches/{mid}")
        yield
        st.mark(mid)
        if not m or "info" not in m:
            continue
        inf = m["info"]
        dur = inf.get("gameDuration", 0)
        for p in inf.get("participants", []):
            o = owner.get(p.get("puuid"))
            if o and hash((mid, p["puuid"])) not in st.rows:
                st.rows.add(hash((mid, p["puuid"])))
                st.det.writerow([mid, inf.get("gameStartTimestamp"), dur, p["puuid"], o[0], o[1],
                                 int(bool(p.get("win"))), p.get("championName"),
                                 p.get("teamPosition"), p.get("kills"), p.get("deaths"),
                                 p.get("assists"),
                                 int(bool(p.get("gameEndedInEarlySurrender")) or dur < 300)])


def _recent_first(mids) -> list[str]:
    """Du plus récent au plus ancien SUR CHAQUE SERVEUR, serveurs en alternance : les numéros
    de partie ne se comparent pas d'un serveur à l'autre (EUW/KR/NA sont dans les milliards,
    BR/TR/JP bien en dessous), un tri global ferait passer ces derniers tout à la fin."""
    by_plat: dict[str, list[str]] = {}
    for m in mids:
        by_plat.setdefault(_platform(m), []).append(m)
    for ms in by_plat.values():
        ms.sort(key=_gid, reverse=True)
    out = []
    for group in itertools.zip_longest(*by_plat.values()):
        out += [m for m in group if m is not None]
    return out


def run(year: int, minutes: float = 55) -> dict:
    """Avance la reconstruction pendant `minutes` puis s'arrête proprement."""
    key = sq.load_key()
    if not key:
        raise sq.KeyError_("Clé Riot absente : définir RIOT_API_KEY ou créer le fichier riot.key.")
    api = sq.Riot(key)
    st = Store()
    puuids, save_puuids = sq._puuid_cache(key)
    aliases = sq.load_aliases()
    acc = _accounts(year)
    acc = acc[acc["region"].isin(MATCH_ROUTING)].copy()
    acc["acc_routing"] = [sq.PLATFORMS[r][1] for r in acc["region"]]
    acc["match_routing"] = acc["region"].map(MATCH_ROUTING)
    # Un compte réattribué (erreur corrigée sur le wiki, joueur mieux identifié) change de
    # propriétaire ici aussi : l'attribution n'est plus figée à la première découverte.
    for r in acc.itertuples():
        info = st.accounts.get(puuids.get(r.rid))
        if info and info["player"] != r.player:
            info["player"] = r.player
    owner = {pu: (i["player"], i["region"]) for pu, i in st.accounts.items()}
    for r in acc.itertuples():                       # comptes déjà connus de cette clé
        if puuids.get(r.rid):
            owner.setdefault(puuids[r.rid], (r.player, r.region))

    hosts = sorted(set(acc["acc_routing"]) | set(acc["match_routing"]))
    gens = {h: _host_work(h, api, st, acc, puuids, aliases, owner) for h in hosts}
    deadline = time.time() + minutes * 60
    calls0, n = api.calls, 0
    try:
        while gens and time.time() < deadline:
            # le routage qui peut repartir le plus tôt (les limites sont par routage)
            h = min(gens, key=lambda x: api.last.get(x, 0.0))
            try:
                next(gens[h])
            except StopIteration:
                del gens[h]
                continue
            n += 1
            if n % 200 == 0:
                st.flush()
                save_puuids()
    finally:
        st.close()
        save_puuids()
    res = status()
    res["calls"] = api.calls - calls0
    res["finished_pass"] = not gens
    return res


def status() -> dict:
    """Où en est la reconstruction (sans appel à Riot)."""
    accp, idp, fp = DIR / "accounts.json", DIR / "ids.csv", DIR / "fetched.txt"
    accounts = json.loads(accp.read_text(encoding="utf-8")) if accp.exists() else {}
    listed = sum(1 for i in accounts.values() if i["until"])
    routing = {pu: i["routing"] for pu, i in accounts.items()}
    todo: dict[str, int] = {}
    total = set()
    redo = 0
    if idp.exists():
        fetched = set(fp.read_text(encoding="utf-8").split()) if fp.exists() else set()
        rp, dp = DIR / "refetched.txt", DIR / "details.csv"
        refetched = set(rp.read_text(encoding="utf-8").split()) if rp.exists() else set()
        rows = set()
        if dp.exists():
            with dp.open(encoding="utf-8", newline="") as fh:
                rd = csv.reader(fh)
                next(rd, None)
                rows = {hash((r[0], r[3])) for r in rd if len(r) > 3}
        seen = set()
        for pu, mid in csv.reader(idp.open(encoding="utf-8")):
            total.add(mid)
            if mid in seen:
                continue
            h = routing.get(pu, "?")
            if mid not in fetched:
                seen.add(mid)
                todo[h] = todo.get(h, 0) + 1
            elif mid not in refetched and hash((mid, pu)) not in rows:
                # détail lu avant que ce compte soit connu : à redemander une fois
                seen.add(mid)
                todo[h] = todo.get(h, 0) + 1
                redo += 1
        done = len(total & fetched)
    else:
        done = 0
    per_hour = 3600 / sq.PAUSE
    eta_h = max(todo.values()) / per_hour if todo else 0.0
    return {"accounts": len(accounts), "listed": listed, "matches": len(total),
            "details": done, "todo": todo, "redo": redo, "eta_h": round(eta_h, 1)}


def message(res: dict) -> str:
    redo = f" dont {res['redo']} à relire (ligne d'un joueur manquante)" if res.get("redo") else ""
    return (f"{res['listed']}/{res['accounts']} comptes listés, {res['matches']} parties, "
            f"{res['details']} détaillées, reste {sum(res['todo'].values())}{redo} "
            f"(≈ {res['eta_h']} h de collecte)")


def load_details() -> pd.DataFrame:
    """Parties détaillées (une ligne par partie et par compte suivi)."""
    p = DIR / "details.csv"
    if not p.exists():
        return pd.DataFrame(columns=DETAIL_COLS)
    df = sq.read_csv(p).drop_duplicates(["match_id", "puuid"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def all_games() -> pd.DataFrame:
    """TOUTES les parties listées, datées : date exacte si détaillée, sinon interpolée à
    partir des parties détaillées du même serveur (les numéros croissent avec le temps)."""
    idp = DIR / "ids.csv"
    if not idp.exists():
        return pd.DataFrame(columns=["puuid", "match_id", "time", "exact"])
    ids = pd.read_csv(idp, names=["puuid", "match_id"]).drop_duplicates()
    ids["platform"] = ids["match_id"].map(_platform)
    ids["gid"] = ids["match_id"].map(_gid)
    det = load_details()
    ref = det.drop_duplicates("match_id")[["match_id", "ts"]]
    ref = ref.assign(platform=ref["match_id"].map(_platform), gid=ref["match_id"].map(_gid))
    out = []
    for plat, g in ids.groupby("platform"):
        r = ref[ref["platform"] == plat].sort_values("gid")
        if len(r) < 2:
            continue
        g = g.copy()
        g["ts"] = np.interp(g["gid"], r["gid"], r["ts"])
        out.append(g)
    if not out:
        return pd.DataFrame(columns=["puuid", "match_id", "time", "exact"])
    df = pd.concat(out, ignore_index=True)
    exact = dict(zip(ref["match_id"], ref["ts"]))
    df["exact"] = df["match_id"].isin(exact)
    df.loc[df["exact"], "ts"] = df.loc[df["exact"], "match_id"].map(exact)
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["puuid", "match_id", "time", "exact"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruction de l'historique soloQ des pros")
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--minutes", type=float, default=55)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(message(status()))
        return
    try:
        res = run(args.year, args.minutes)
    except sq.KeyError_ as exc:
        print(f"[soloq_history] {exc}")
        sys.exit(2)
    print(message(res) + f" · {res['calls']} requêtes")


if __name__ == "__main__":
    main()

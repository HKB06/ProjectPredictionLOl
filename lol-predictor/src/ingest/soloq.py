"""SoloQ des joueurs pros : comptes connus (Leaguepedia) + relevé quotidien (API Riot).

Deux étapes :
  1. `accounts`  : lit la colonne `SoloqueueIds` de Leaguepedia (texte libre du wiki) et en
                   tire une table propre joueur -> (région, pseudo, tag). SANS clé.
  2. `snapshot`  : pour chaque compte, demande à Riot le rang actuel (LP, victoires, défaites)
                   et les champions les plus joués. AVEC clé (`RIOT_API_KEY`).

Pourquoi un relevé quotidien et pas l'historique : Riot ne donne que l'état du moment. La
forme soloQ (LP gagnés, games jouées, champions travaillés) se lit en comparant deux relevés.
Le signal n'existe donc qu'à partir du jour où la collecte tourne — d'où l'intérêt de la
lancer tôt sur un serveur. Rien n'entre dans le modèle tant qu'il n'y a pas assez de recul
pour le mesurer.

Limites connues :
  - seuls les comptes écrits « pseudo#tag » sont interrogeables (les anciens pseudos sans
    tag ne permettent plus de retrouver un compte) ;
  - les comptes du serveur chinois ne sont pas accessibles par l'API Riot ;
  - une clé « Development » expire toutes les 24 h : pour un serveur il faut une clé
    « Personal » (demande gratuite sur developer.riotgames.com, validée sous quelques jours).

Suivi des comptes :
  - un compte retrouvé une fois est suivi par son identifiant permanent (PUUID) : s'il est
    renommé, on le retrouve quand même et on note son nouveau pseudo (`aliases.json`), ce qui
    permet aussi de le retrouver avec une nouvelle clé ;
  - les comptes découverts dans les classements Challenger/GM (`soloq_ladder`) sont relevés
    en plus de ceux du wiki.

Usage :
    python -m src.ingest.soloq accounts            # table des comptes + couverture
    python -m src.ingest.soloq snapshot            # relevé du jour (reprenable)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

from src.ingest import leaguepedia as lp
from src.ingest.load_oracle import ROOT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = ROOT / "data" / "external" / "soloq"
MAX_ACCOUNTS = 3          # comptes interrogés par joueur (les premiers cités sur le wiki)
TOP_CHAMPS = 10
PAUSE = 1.25              # clé Development/Personal : 100 requêtes / 2 min PAR routage
RENAME_CHECK_DAYS = 7     # on revérifie le pseudo actuel d'un compte une fois par semaine
TIMEOUT = 30

# région écrite sur le wiki -> (plateforme Riot, routage régional pour le compte)
PLATFORMS = {
    "KR": ("kr", "asia"), "JP": ("jp1", "asia"),
    "EUW": ("euw1", "europe"), "EUNE": ("eun1", "europe"), "TR": ("tr1", "europe"),
    "RU": ("ru", "europe"), "ME": ("me1", "europe"),
    "NA": ("na1", "americas"), "BR": ("br1", "americas"), "LAN": ("la1", "americas"),
    "LAS": ("la2", "americas"), "OCE": ("oc1", "asia"), "TW": ("tw2", "asia"),
    "VN": ("vn2", "asia"), "SEA": ("sg2", "asia"), "SG": ("sg2", "asia"),
    "PH": ("sg2", "asia"), "TH": ("sg2", "asia"),
}
_ALIAS = {"EU": "EUW", "EUW1": "EUW", "KOREA": "KR", "KR1": "KR", "NA1": "NA", "BR1": "BR",
          "LA1": "LAN", "LA2": "LAS", "OC": "OCE", "OC1": "OCE", "TW2": "TW", "VN2": "VN",
          "TR1": "TR", "JP1": "JP", "EUN": "EUNE", "EUN1": "EUNE"}
_REGION = re.compile(r"'''\s*([A-Za-z0-9]{2,6})\s*:?\s*'''\s*:?")
_SUFFIX = re.compile(r"\(([A-Za-z0-9]{2,6})\)\s*$")


class KeyError_(RuntimeError):
    """Clé Riot absente, invalide ou expirée."""


# ---------------------------------------------------------------- 1. comptes
def _region(code: str) -> str | None:
    code = code.strip().upper()
    code = _ALIAS.get(code, code)
    return code if code in PLATFORMS or code == "CN" else None


def parse_accounts(text) -> list[tuple[str | None, str, str | None]]:
    """Texte wiki -> [(région, pseudo, tag)]. Région None si le wiki ne la précise pas."""
    if not isinstance(text, str):
        return []
    text = html.unescape(html.unescape(text))
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    out, seen = [], set()
    for line in text.split("\n"):
        region = None
        pos = 0
        chunks: list[tuple[str | None, str]] = []
        for m in _REGION.finditer(line):
            if line[pos:m.start()].strip():
                chunks.append((region, line[pos:m.start()]))
            region, pos = _region(m.group(1)), m.end()
        chunks.append((region, line[pos:]))
        for reg, chunk in chunks:
            for raw in chunk.split(","):
                raw = raw.strip().strip("'").strip()
                suffix = _SUFFIX.search(raw)
                r = reg
                if suffix and _region(suffix.group(1)):
                    r, raw = _region(suffix.group(1)), raw[:suffix.start()].strip()
                if not raw:
                    continue
                name, _, tag = raw.partition("#")
                name, tag = name.strip(), tag.strip() or None
                if not name or (r, name.lower(), tag) in seen:
                    continue
                seen.add((r, name.lower(), tag))
                out.append((r, name, tag))
    # une région citée plus haut vaut pour les lignes suivantes qui n'en ont pas
    last = None
    filled = []
    for r, name, tag in out:
        last = r or last
        filled.append((r or last, name, tag))
    return filled


def accounts_table(year: int) -> pd.DataFrame:
    """Une ligne par compte soloQ connu. `usable` = interrogeable par l'API Riot."""
    # en début de saison, les joueurs de la saison précédente restent suivis : presque aucun
    # tournoi de l'année n'existe encore, et un joueur sans équipe peut signer demain
    p = lp.load_season("players", year).drop_duplicates("Page", keep="last")
    # + effectifs actuels (mercato, recrues de ligues mineures, début de saison où peu de
    # tournois de l'année existent encore) : suivis dès l'annonce, sans attendre un match
    cur = DIR.parent / "rosters" / "current.csv"
    if cur.exists():
        c = pd.read_csv(cur, keep_default_na=False, na_values=[""],
                        dtype={"player": str, "id": str})
        c = c.rename(columns={"player": "Page", "id": "ID", "role": "Role"})
        p = pd.concat([p, c[["Page", "ID", "Role", "SoloqueueIds"]]], ignore_index=True)
        p = p.drop_duplicates("Page")
    p = p.dropna(subset=["SoloqueueIds"])
    rows = []
    for page, pid, role, text in zip(p["Page"], p["ID"], p["Role"], p["SoloqueueIds"]):
        for n, (region, name, tag) in enumerate(parse_accounts(text)):
            rows.append({"player": page, "id": pid, "role": role, "n": n, "region": region,
                         "game_name": name, "tag": tag,
                         "usable": bool(tag) and region in PLATFORMS})
    return pd.DataFrame(rows, columns=["player", "id", "role", "n", "region", "game_name", "tag",
                                       "usable"])


def read_csv(path) -> pd.DataFrame:
    """Relecture sûre : sans ça pandas prend la région « NA » (et des tags comme « null »)
    pour des valeurs manquantes."""
    text = {c: str for c in ("player", "id", "game_name", "tag", "region", "current")}
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype=text)


def write_csv(df: pd.DataFrame, path) -> None:
    """Écriture sûre : fichier temporaire puis remplacement. Un arrêt du serveur en pleine
    écriture ne laisse jamais un fichier tronqué."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json(obj, path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, **kw), encoding="utf-8")
    tmp.replace(path)


def carry_over(path_of, year: int):
    """Fichier de l'année ; au changement d'année il repart de celui de l'année d'avant
    (les comptes découverts restent ceux des mêmes joueurs)."""
    p = path_of(year)
    if not p.exists() and path_of(year - 1).exists():
        write_csv(read_csv(path_of(year - 1)), p)
    return p


def accounts_path(year: int):
    return DIR / f"accounts_{year}.csv"


def build_accounts(year: int) -> pd.DataFrame:
    df = accounts_table(year)
    write_csv(df, accounts_path(year))
    return df


# ---------------------------------------------------------------- 2. relevé
def load_key() -> str | None:
    """RIOT_API_KEY : variable d'environnement, puis fichier `riot.key` (gitignoré)."""
    key = os.environ.get("RIOT_API_KEY")
    if key and key.strip():
        return key.strip()
    for p in (ROOT / "riot.key", ROOT.parent / "riot.key"):
        if p.exists() and p.read_text(encoding="utf-8").strip():
            return p.read_text(encoding="utf-8").strip()
    return None


class Riot:
    def __init__(self, key: str) -> None:
        self.s = requests.Session()
        self.s.headers["X-Riot-Token"] = key
        self.calls = 0
        self.last: dict[str, float] = {}

    def _wait(self, host: str) -> None:
        """Les limites de Riot s'appliquent par routage (kr, euw1, asia, europe…) : on
        espace les appels d'un même routage et on alterne librement entre routages."""
        wait = PAUSE - (time.monotonic() - self.last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self.last[host] = time.monotonic()

    def get(self, host: str, path: str):
        """JSON, ou None si le compte n'existe pas (404). Respecte les limites de débit."""
        for _ in range(6):
            self._wait(host)
            self.calls += 1
            r = self.s.get(f"https://{host}.api.riotgames.com{path}", timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise KeyError_("Clé Riot refusée (invalide ou expirée : une clé Development "
                                "ne dure que 24 h).")
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 10)) + 1)
                continue
            if r.status_code >= 500:
                time.sleep(5)
                continue
            raise RuntimeError(f"Riot {r.status_code} sur {path.split('?')[0]}")
        raise RuntimeError("Riot : trop d'essais (limite de débit ou panne).")


def snapshots_path(year: int):
    return DIR / f"snapshots_{year}.csv"


def _puuid_cache(key: str) -> tuple[dict, callable]:
    """Les PUUID sont chiffrés PAR CLÉ : le cache est lié à l'empreinte de la clé."""
    path = DIR / f".puuids_{hashlib.sha256(key.encode()).hexdigest()[:10]}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save() -> None:
        write_json(cache, path)
    return cache, save


def aliases_path():
    return DIR / "aliases.json"


def load_aliases() -> dict:
    """pseudo#tag d'origine -> {"current": pseudo#tag actuel, "checked": date}.
    Indépendant de la clé : c'est ce qui permet de retrouver un compte renommé."""
    p = aliases_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def discovered_path(year: int):
    return DIR / f"discovered_{year}.csv"


def tracked_accounts(year: int) -> pd.DataFrame:
    """Comptes à relever : ceux du wiki (les MAX_ACCOUNTS premiers par joueur), ceux
    découverts dans les classements, et ceux de TrackingThePros / lolpros.gg (comptes
    actifs, smurfs compris) — ces deux dernières sources en confiance « fort » ou « moyen »."""
    acc = build_accounts(year)
    # les MAX_ACCOUNTS premiers comptes INTERROGEABLES : compter aussi les comptes sans tag
    # ou chinois écartait des comptes utilisables (Bin : 1 suivi sur 4)
    acc = acc[acc["usable"].astype(bool)].copy()
    acc = acc[acc.groupby("player").cumcount() < MAX_ACCOUNTS].copy()
    acc["source"] = "wiki"
    for path, source in ((carry_over(discovered_path, year), "ladder"),
                         (carry_over(lambda y: DIR / f"sites_{y}.csv", year), None)):
        if not path.exists():
            continue
        d = read_csv(path)
        d = d[d["confidence"].isin(["fort", "moyen"])]
        known = set(zip(acc["game_name"].astype(str).str.lower(), acc["tag"].astype(str).str.lower()))
        d = d[[(str(g).lower(), str(t).lower()) not in known
               for g, t in zip(d["game_name"], d["tag"])]]
        d = d.assign(n=-1, usable=True)
        if source:
            d["source"] = source
        acc = pd.concat([acc, d[acc.columns]], ignore_index=True)
    return acc


def snapshot(year: int, limit: int | None = None) -> dict:
    """Relevé du jour pour tous les comptes interrogeables. Reprenable dans la journée."""
    key = load_key()
    if not key:
        raise KeyError_("Clé Riot absente : définir RIOT_API_KEY ou créer le fichier riot.key.")
    acc = tracked_accounts(year)
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    path = snapshots_path(year)
    old = read_csv(path) if path.exists() else pd.DataFrame()
    done = set()
    if len(old):
        t = old[old["date"] == today]
        done = set(zip(t["player"], t["region"], t["game_name"], t["tag"]))
    todo = [r for r in acc.itertuples()
            if (r.player, r.region, r.game_name, r.tag) not in done]
    if limit:
        todo = todo[:limit]

    api = Riot(key)
    puuids, save_puuids = _puuid_cache(key)
    aliases = load_aliases()
    rows, missing, renamed = [], 0, 0
    # comptes déjà relevés aujourd'hui (passage précédent) : un compte listé sous deux
    # joueurs ne doit pas être relevé une seconde fois sous l'autre nom
    seen = {puuids.get(f"{g}#{t}") for _p, _r, g, t in done} - {None}
    written = 0

    def flush() -> None:
        """Ajoute en fin de fichier les lignes pas encore écrites (jamais de réécriture
        complète : le fichier des relevés ne peut pas être perdu par un arrêt brutal)."""
        nonlocal written
        new = rows[written:]
        if new:
            df = pd.DataFrame(new)
            if len(old):
                df = df.reindex(columns=list(old.columns))
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, mode="a", index=False, header=not path.exists())
            written = len(rows)
        save_puuids()
        write_json(aliases, aliases_path(), ensure_ascii=False, indent=0)

    try:
        for i, r in enumerate(todo, 1):
            platform, routing = PLATFORMS[r.region]
            rid = f"{r.game_name}#{r.tag}"
            if rid not in puuids:
                # nouvelle clé : on cherche sous le DERNIER pseudo connu (compte renommé)
                name, _, tag = aliases.get(rid, {}).get("current", rid).rpartition("#")
                a = api.get(routing, f"/riot/account/v1/accounts/by-riot-id/"
                                     f"{quote(name, safe='')}/{quote(tag, safe='')}")
                puuids[rid] = a["puuid"] if a else None
                if a:                                     # pseudo actuel connu à l'instant
                    aliases[rid] = {"current": f"{name}#{tag}",
                                    "checked": dt.date.today().isoformat()}
            puuid = puuids[rid]
            if not puuid:
                missing += 1                              # renommé avant qu'on le connaisse
                continue
            if puuid in seen:                             # même compte listé deux fois
                continue
            seen.add(puuid)
            al = aliases.setdefault(rid, {"current": rid, "checked": ""})
            stale = (dt.date.today() - dt.timedelta(days=RENAME_CHECK_DAYS)).isoformat()
            if al["checked"] < stale:
                a = api.get(routing, f"/riot/account/v1/accounts/by-puuid/{puuid}")
                if a and a.get("gameName"):
                    now = f"{a['gameName']}#{a.get('tagLine', '')}"
                    renamed += now.lower() != al["current"].lower()
                    al["current"] = now
                al["checked"] = dt.date.today().isoformat()
            entries = api.get(platform, f"/lol/league/v4/entries/by-puuid/{puuid}") or []
            solo = next((e for e in entries if e.get("queueType") == "RANKED_SOLO_5x5"), {})
            champs = api.get(platform, f"/lol/champion-mastery/v4/champion-masteries/by-puuid/"
                                       f"{puuid}/top?count={TOP_CHAMPS}") or []
            rows.append({
                "date": today, "player": r.player, "id": r.id, "role": r.role,
                "region": r.region, "game_name": r.game_name, "tag": r.tag,
                "current": al["current"], "source": r.source,
                "tier": solo.get("tier"), "rank": solo.get("rank"),
                "lp": solo.get("leaguePoints"), "wins": solo.get("wins"),
                "losses": solo.get("losses"),
                # [champion, points de maîtrise, dernière partie (ms)] : l'écart entre deux
                # relevés dit quels champions ont été joués entre-temps
                "champions": json.dumps([[c.get("championId"), c.get("championPoints"),
                                          c.get("lastPlayTime")] for c in champs]),
            })
            if i % 50 == 0:
                flush()
                print(f"  {i}/{len(todo)} comptes…", flush=True)
    finally:
        flush()
    return {"new": len(rows), "missing": missing, "renamed": renamed, "calls": api.calls,
            "already": len(done), "accounts": len(acc)}


def main() -> None:
    ap = argparse.ArgumentParser(description="SoloQ des joueurs pros")
    ap.add_argument("action", choices=["accounts", "snapshot"])
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--limit", type=int, default=None, help="nombre max de comptes à relever")
    args = ap.parse_args()
    if args.action == "accounts":
        df = build_accounts(args.year)
        players = lp.load("players", args.year)
        ok = df[df["usable"]]
        print(f"{len(df)} comptes pour {df['player'].nunique()} joueurs "
              f"(sur {len(players)} joueurs inscrits en {args.year}).")
        print(f"Interrogeables (pseudo#tag + région connue) : {len(ok)} comptes, "
              f"{ok['player'].nunique()} joueurs.")
        print("Par région :", ok["region"].value_counts().to_dict())
        print("Non interrogeables :", {
            "sans tag": int(df["tag"].isna().sum()),
            "région inconnue": int((df["tag"].notna() & df["region"].isna()).sum()),
            "serveur chinois": int((df["region"] == "CN").sum())})
        return
    try:
        res = snapshot(args.year, args.limit)
    except KeyError_ as exc:
        print(f"[soloq] {exc}")
        sys.exit(2)
    print(f"{res['new']} comptes relevés ({res['already']} déjà faits aujourd'hui, "
          f"{res['missing']} introuvables, {res['renamed']} renommés) · "
          f"{res['calls']} requêtes · "
          f"{res['accounts']} comptes suivis.")


if __name__ == "__main__":
    main()

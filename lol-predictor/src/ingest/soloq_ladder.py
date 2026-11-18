"""Découverte des comptes soloQ des pros dans les classements Challenger / Grandmaster.

Le wiki (Leaguepedia) ne connaît qu'une partie des comptes, souvent avec d'anciens pseudos.
Or les pros jouent presque tous tout en haut du classement, et beaucoup affichent leur équipe
dans leur pseudo (« T1 Faker », « HLE Zeka »). On lit donc le haut du classement de chaque
serveur, on récupère le pseudo de chaque compte, et on le compare aux pros connus.

Confiance d'une correspondance :
  fort    pseudo = abréviation d'équipe + nom du pro (« HLE Zeka », « T1Faker »)
  moyen   pseudo = exactement le nom du pro (« Ruler »)
  faible  le nom du pro n'est qu'un mot du pseudo (« faker fan ») : noté, jamais relevé
  ambigu  plusieurs pros possibles au même niveau : noté, jamais relevé
Un compte trouvé sur un serveur qui n'est pas celui de la ligue du pro (bootcamp, etc.) perd
un niveau. Les comptes « fort » et « moyen » sont ajoutés au relevé quotidien (`soloq`).

Contrôle : les comptes déjà connus par le wiki qui apparaissent dans le classement servent de
vérité terrain (un pseudo attribué au mauvais pro = faux positif).

Usage :
    python -m src.ingest.soloq_ladder                       # 12 serveurs, Challenger + GM
    python -m src.ingest.soloq_ladder --regions KR --tiers challenger
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import unicodedata

import pandas as pd

from src.ingest import leaguepedia as lp
from src.ingest import soloq as sq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Serveurs des 6 ligues qualificatives pour les Mondiaux ET de leurs ligues secondaires :
#   LCK -> KR (LPL : pas d'API pour le serveur chinois, les joueurs en bootcamp jouent en KR)
#   LEC + ligues régionales -> EUW, EUNE (Balkans, Grèce, Pologne, Tchéquie), TR (TCL),
#                              ME (Arabian League)
#   LCS + NACL / Liga Regional Norte -> NA, LAN · CBLOL + CD / Liga Regional Sur -> BR, LAS
#   LCP + PCS / VCS / LJL -> TW, VN, JP
REGIONS = ["KR", "EUW", "EUNE", "TR", "ME", "NA", "LAN", "BR", "LAS", "TW", "VN", "JP"]
TIERS = {"challenger": "challengerleagues", "grandmaster": "grandmasterleagues",
         "master": "masterleagues"}
DEFAULT_TIERS = ["challenger", "grandmaster"]
NAME_TTL_DAYS = 14        # pseudo d'un compte du classement : relu au plus toutes les 2 semaines
MIN_ID = 3                # noms de pros trop courts (« Bo », « Ra ») : trop de faux positifs

# région d'équipe (Leaguepedia) -> serveurs où l'on s'attend à trouver ses joueurs
HOME = {
    "Korea": {"KR"}, "China": {"KR"},              # les joueurs LPL en bootcamp jouent en KR
    "EMEA": {"EUW", "EUNE", "TR", "ME"}, "Europe": {"EUW", "EUNE", "TR"}, "Turkey": {"TR"},
    "MENA": {"ME", "EUW", "EUNE"}, "Middle East": {"ME", "EUW", "EUNE"},
    "CIS": {"EUW", "EUNE", "RU"},
    # depuis 2026, la Liga Regional Norte dépend de la LCS et la Liga Regional Sur du CBLOL :
    # le wiki classe leurs équipes « North America » et « Brazil »
    "North America": {"NA", "LAN"}, "Brazil": {"BR", "LAS"}, "Latin America": {"LAN", "LAS"},
    "LAN": {"LAN"}, "LAS": {"LAS"}, "Americas": {"NA", "BR", "LAN", "LAS"},
    "Asia Pacific": {"VN", "TW", "SEA", "JP", "OCE"}, "Vietnam": {"VN"}, "SEA": {"SEA"},
    "PCS": {"TW", "SEA"}, "LMS": {"TW"}, "Japan": {"JP"}, "Oceania": {"OCE"},
}
GENERIC = {"esports", "esport", "gaming", "team", "club", "academy", "challengers", "youth",
           "the", "of", "and", "e", "sports", "global", "rookies", "scholars"}
LEVELS = ["fort", "moyen", "faible"]


def norm(s) -> str:
    """Minuscules, sans accents, lettres et chiffres seulement (garde le coréen, etc.)."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return "".join(c for c in s if c.isalnum())


def tokens(s) -> list[str]:
    return [t for t in (norm(w) for w in re.split(r"[\W_]+", str(s))) if t]


# ------------------------------------------------------------------- les pros
def _teams(year: int) -> pd.DataFrame:
    """Abréviation et région de chaque équipe de l'année (mise en cache)."""
    path = sq.DIR / f"teams_{year}.csv"
    if path.exists() and (dt.datetime.now().timestamp() - path.stat().st_mtime) < 7 * 86400:
        return sq.read_csv(path)
    # en début de saison, la précédente aussi : presque aucune équipe n'a encore joué
    df = lp.cargo("TournamentPlayers=TP,Tournaments=T,Teams=Te",
                  "Te.OverviewPage=Team,Te.Short=Short,Te.Region=Region",
                  where=f"T.Year IN ({', '.join(str(y) for y in lp.season_years(year))})",
                  join_on="TP.OverviewPage=T.OverviewPage,TP.Team=Te.OverviewPage",
                  group_by="Te.OverviewPage")
    if "Team" not in df.columns:                  # réponse vide
        if path.exists():
            return sq.read_csv(path)
        return pd.DataFrame(columns=["Team", "Short", "Region"])
    df = df.dropna(subset=["Team"])
    sq.write_csv(df, path)
    return df


def team_tags(team: str, short) -> set[str]:
    """Façons courantes d'écrire l'équipe dans un pseudo : HLE, Hanwha, T1, GEN…"""
    tags = set()
    if isinstance(short, str) and short.strip():
        tags.add(norm(short.split(".")[0]))
        tags.add(norm(short))
    words = tokens(team)
    if words:
        tags.add(words[0])
        tags.add("".join(words))
        sig = [w for w in words if w not in GENERIC]
        if len(sig) >= 2:
            tags.add("".join(w[0] for w in sig))
        if len(words) >= 2:
            tags.add("".join(w[0] for w in words))
    return {t for t in tags if len(t) >= 2}


def pros(year: int) -> pd.DataFrame:
    """Un pro par ligne : page, nom (ID), équipe la plus récente, abréviations, serveurs."""
    # saison en cours + précédente (voir `_teams`), et l'équipe ACTUELLE du joueur quand le
    # suivi des effectifs la connaît (mercato : l'ancien maillot ne vaut plus)
    players = lp.load_season("players", year).drop_duplicates("Page", keep="last")
    rosters = lp.load_season("rosters", year)
    tourn = lp.load_season("tournaments", year)[["OverviewPage", "DateStart"]]
    teams = _teams(year)
    r = rosters.merge(tourn, on="OverviewPage", how="left").sort_values("DateStart")
    last = r.drop_duplicates("Player", keep="last")[["Player", "Team"]]
    cur_path = sq.DIR.parent / "rosters" / "current.csv"
    if cur_path.exists():
        cur = pd.read_csv(cur_path, keep_default_na=False, na_values=[""],
                          dtype={"player": str, "team": str})
        cur = cur.rename(columns={"player": "Player", "team": "Team"})[["Player", "Team"]]
        last = pd.concat([last, cur], ignore_index=True).drop_duplicates("Player", keep="last")
    df = players.merge(last, left_on="Page", right_on="Player", how="left")
    df = df.merge(teams.rename(columns={"Region": "TeamRegion"}), on="Team", how="left")
    df["key"] = df["ID"].map(norm)
    df = df[df["key"].str.len() >= MIN_ID]
    df["tags"] = [team_tags(t, s) if isinstance(t, str) else set()
                  for t, s in zip(df["Team"], df["Short"])]
    df["home"] = [HOME.get(tr if isinstance(tr, str) else res, set()) | HOME.get(res, set())
                  for tr, res in zip(df["TeamRegion"], df["Residency"])]
    return df.reset_index(drop=True)


def match(game_name: str, region: str, index: dict[str, list[dict]]) -> tuple[dict | None, str]:
    """(pro, confiance) pour un pseudo du classement, ou (None, '') si aucun pro."""
    full, toks = norm(game_name), tokens(game_name)
    found: dict[str, tuple[dict, int]] = {}
    for key in set(toks) | {full}:
        for p in index.get(key, []):
            others = [t for t in toks if t != key]
            if any(t in p["tags"] for t in others) or \
                    any(full == tag + key or full == key + tag for tag in p["tags"]):
                lvl = 0
            elif full == key:
                lvl = 1
            else:
                lvl = 2
            if region not in p["home"]:
                lvl += 1
            if lvl < len(LEVELS):
                prev = found.get(p["Page"])
                if prev is None or lvl < prev[1]:
                    found[p["Page"]] = (p, lvl)
    # un pseudo qui se termine par l'abréviation d'équipe (« Faker T1 ») est déjà couvert
    if not found:
        # pseudo collé « hlezeka » : on cherche une abréviation en préfixe
        for key, ps in index.items():
            if len(key) >= 4 and full.endswith(key) and full != key:
                for p in ps:
                    if full[: -len(key)] in p["tags"]:
                        lvl = 0 if region in p["home"] else 1
                        found[p["Page"]] = (p, lvl)
    if not found:
        return None, ""
    best = min(lvl for _, lvl in found.values())
    top = [p for p, lvl in found.values() if lvl == best]
    if len(top) > 1:
        return None, "ambigu"
    return top[0], LEVELS[best]


# --------------------------------------------------------------- classements
def _names_cache(key: str):
    path = sq.DIR / f".ladder_names_{hashlib.sha256(key.encode()).hexdigest()[:10]}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save() -> None:
        sq.write_json(cache, path, ensure_ascii=False)
    return cache, save


def discover(year: int, regions: list[str] = REGIONS, tiers: list[str] = DEFAULT_TIERS) -> dict:
    key = sq.load_key()
    if not key:
        raise sq.KeyError_("Clé Riot absente : définir RIOT_API_KEY ou créer le fichier riot.key.")
    api = sq.Riot(key)
    today = dt.date.today().isoformat()

    # 1. le haut du classement de chaque serveur
    ladder = []
    for reg in regions:
        platform, routing = sq.PLATFORMS[reg]
        for tier in tiers:
            data = api.get(platform, f"/lol/league/v4/{TIERS[tier]}/by-queue/RANKED_SOLO_5x5") or {}
            for e in data.get("entries", []):
                if e.get("puuid"):
                    ladder.append({"region": reg, "routing": routing, "tier": tier,
                                   "puuid": e["puuid"], "lp": e.get("leaguePoints"),
                                   "wins": e.get("wins"), "losses": e.get("losses")})
    lad = pd.DataFrame(ladder, columns=["region", "routing", "tier", "puuid", "lp", "wins",
                                        "losses"])
    print(f"  classements : {len(lad)} comptes ({lad.groupby('region').size().to_dict()})",
          flush=True)
    if lad.empty:
        # remise à zéro de début de saison : on garde les comptes déjà découverts
        return {"ladder": 0, "calls": api.calls, "found": {}, "players": 0,
                "wiki_in_ladder": 0, "check_matched": 0, "check_wrong": 0}

    # 2. les comptes déjà connus par le wiki : leur pseudo est connu, pas besoin d'appel
    known = {}                                   # puuid -> (joueur du wiki, pseudo#tag)
    puuids, _ = sq._puuid_cache(key)
    acc = sq.read_csv(sq.accounts_path(year)) if sq.accounts_path(year).exists() \
        else sq.build_accounts(year)
    by_rid = {f"{g}#{t}": p for g, t, p in zip(acc["game_name"], acc["tag"], acc["player"])}
    for rid, pu in puuids.items():
        if pu and rid in by_rid:
            known[pu] = (by_rid[rid], rid)

    # 3. pseudo de chaque compte inconnu (1 requête par compte, mise en cache), en alternant
    #    les routages pour aller plus vite
    names, save_names = _names_cache(key)
    stale = (dt.date.today() - dt.timedelta(days=NAME_TTL_DAYS)).isoformat()
    todo = lad[[p not in known and names.get(p, ["", ""])[1] < stale for p in lad["puuid"]]]
    todo = todo.assign(_k=todo.groupby("routing").cumcount()).sort_values("_k")
    print(f"  pseudos à demander : {len(todo)} (≈ {len(todo) / max(todo['routing'].nunique(), 1) * sq.PAUSE / 60:.0f} min)",
          flush=True)
    try:
        for i, (pu, routing) in enumerate(zip(todo["puuid"], todo["routing"]), 1):
            a = api.get(routing, f"/riot/account/v1/accounts/by-puuid/{pu}")
            names[pu] = [f"{a['gameName']}#{a.get('tagLine', '')}" if a and a.get("gameName")
                         else "", today]
            if i % 200 == 0:
                save_names()
                print(f"    {i}/{len(todo)}", flush=True)
    finally:
        save_names()

    # 4. correspondance pseudo -> pro
    pr = pros(year)
    index: dict[str, list[dict]] = {}
    for p in pr.to_dict("records"):
        index.setdefault(p["key"], []).append(p)
    rows, checks = [], []
    for e in lad.itertuples():
        rid = known[e.puuid][1] if e.puuid in known else names.get(e.puuid, [""])[0]
        if not rid:
            continue
        game_name, _, tag = rid.rpartition("#")
        p, conf = match(game_name, e.region, index)
        if e.puuid in known:                     # vérité terrain : compte connu du wiki
            checks.append({"truth": known[e.puuid][0], "found": p["Page"] if p else None,
                           "confidence": conf})
            continue
        if not conf:
            continue
        rows.append({"player": p["Page"] if p else None, "id": p["ID"] if p else None,
                     "role": p["Role"] if p else None, "region": e.region,
                     "game_name": game_name, "tag": tag, "confidence": conf,
                     "tier": e.tier, "lp": e.lp, "team": p["Team"] if p else None,
                     "first_seen": today, "last_seen": today})

    # 5. fusion avec les découvertes précédentes (un compte reste connu même s'il descend)
    new = pd.DataFrame(rows)
    path = sq.carry_over(sq.discovered_path, year)
    if path.exists() and len(new):
        old = sq.read_csv(path)

        def keys(df: pd.DataFrame) -> list:
            return list(zip(df["region"], df["game_name"].astype(str).str.lower(),
                            df["tag"].astype(str).str.lower()))
        first = dict(zip(keys(old), old["first_seen"]))
        new["first_seen"] = [first.get(k, today) for k in keys(new)]
        # un compte déjà découvert reste connu même s'il n'est plus dans le haut du
        # classement cette semaine (smurf redescendu, classement vide un jour de panne)
        found = set(keys(new))
        keep = old[[k not in found for k in keys(old)]]
        new = pd.concat([keep, new], ignore_index=True)
    elif path.exists():
        new = sq.read_csv(path)
    if len(new):
        sq.write_csv(new, path)

    ch = pd.DataFrame(checks)
    usable = ch[ch["confidence"].isin(["fort", "moyen"])] if len(ch) else ch
    cur = new[new["region"].isin(regions)] if len(new) else new
    res = {
        "ladder": len(lad), "calls": api.calls,
        "found": cur["confidence"].value_counts().to_dict() if len(cur) else {},
        "players": int(cur[cur["confidence"].isin(["fort", "moyen"])]["player"].nunique())
        if len(cur) else 0,
        "wiki_in_ladder": len(ch),
        "check_matched": len(usable),
        "check_wrong": int((usable["found"] != usable["truth"]).sum()) if len(usable) else 0,
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Comptes soloQ des pros trouvés dans les classements")
    ap.add_argument("--year", type=int, default=dt.date.today().year)
    ap.add_argument("--regions", nargs="*", default=REGIONS, choices=list(sq.PLATFORMS))
    ap.add_argument("--tiers", nargs="*", default=DEFAULT_TIERS, choices=list(TIERS))
    args = ap.parse_args()
    try:
        res = discover(args.year, args.regions, args.tiers)
    except sq.KeyError_ as exc:
        print(f"[ladder] {exc}")
        sys.exit(2)
    print(f"{res['ladder']} comptes lus dans les classements · {res['calls']} requêtes")
    print(f"Correspondances : {res['found']} → {res['players']} pros suivis en plus")
    print(f"Contrôle sur les {res['wiki_in_ladder']} comptes du wiki présents dans le classement : "
          f"{res['check_matched']} reconnus par leur pseudo, dont {res['check_wrong']} attribués "
          f"au mauvais pro")


if __name__ == "__main__":
    main()

"""Préparation soloQ des pros avant chaque match, à la lumière de la méta de LEUR ligue.

Idée : on ne voit pas les scrims, mais la soloQ est publique. Avant un match, un joueur
qui enchaîne les parties sur les champions que sa ligue joue en ce moment se prépare ; un
joueur qui joue peu, ou hors méta, peut-être moins (ou il scrime beaucoup : c'est le test qui
dira dans quel sens ça joue, rien n'est supposé d'avance).

Trois briques, toutes calculées SANS regarder l'avenir (seulement ce qui précède le match) :

1. Méta par ligue et par poste (`meta_sets`) : les champions pris dans la ligue à ce poste
   sur les `META_DAYS` derniers jours. Les ligues ne jouent pas la même méta sur un même
   patch (en 2026, seuls 6 à 8 des 20 champions les plus présents sont communs aux six
   grandes ligues), d'où une méta par ligue. Trop peu de matchs récents dans la ligue
   (début de saison) : on complète avec toutes les ligues pariables.

2. Lien joueur soloQ (page Leaguepedia) -> joueur des matchs (nom Oracle) : même nom ; en
   cas d'homonymes, même équipe d'après les effectifs Leaguepedia.

3. Indicateurs par joueur et par match (`features`) :
     vol_7          parties classées jouées les 7 jours d'avant (tous ses comptes suivis)
     vol_ratio      vol_7 rapporté à son rythme habituel (60 jours d'avant)
     role_14        parties à SON poste sur 14 jours (détaillées)
     meta_share_14  part de ces parties sur des champions de la méta de sa ligue
     meta_champs_14 nombre de champions méta différents travaillés
     meta_wr_14     victoires sur ces champions
     pick_practiced le champion joué dans CE match avait-il été joué en soloQ (14 jours) ?
                    -> sert au contrôle de cohérence, pas à la prédiction (connu après draft)
   `covered` : le joueur a au moins une partie soloQ suivie dans les 60 jours d'avant.

Usage :
    python -m src.features.soloq_prep            # contrôle de cohérence + aperçu
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.ingest import leaguepedia as lp
from src.ingest import soloq as sq
from src.ingest import soloq_history as sh
from src.ingest import soloq_ladder as sl
from src.ingest.load_oracle import load_config, read_seasons
from src.update.bet_universe import universe

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

META_DAYS = 21            # fenêtre de la méta d'une ligue
META_MIN_PICKS = 2        # un champion est « méta » s'il a été pris au moins 2 fois au poste
META_MIN_GAMES = 20       # en dessous (début de saison), on complète avec les autres ligues
WINDOWS = (3, 7, 14)
BASE_DAYS = 60
ROLE_SOLOQ = {"top": "TOP", "jng": "JUNGLE", "mid": "MIDDLE", "bot": "BOTTOM", "sup": "UTILITY"}
_COLS = {"gameid", "date", "league", "patch", "teamname", "playername", "position", "champion"}
# Riot écrit « LeeSin », « Kaisa », « MonkeyKing » ; Oracle « Lee Sin », « Kai'Sa », « Wukong ».
_CHAMP_ALIAS = {"monkeyking": "wukong", "renata": "renataglasc", "nunu": "nunuwillump"}


def champ_key(name) -> str:
    """Clé commune aux deux écritures d'un champion (sans espace, apostrophe ni casse)."""
    k = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return _CHAMP_ALIAS.get(k, k)


# ------------------------------------------------------------------- matchs pros
def pro_games(cfg: dict | None = None) -> pd.DataFrame:
    """Une ligne par joueur et par game pro (saisons passées comprises)."""
    cfg = cfg or load_config()
    df = read_seasons(cfg, _COLS, dtype={"patch": "string"}, history=True)
    df = df[df["position"].isin(ROLE_SOLOQ)].dropna(subset=["playername", "champion"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df.dropna(subset=["date"]).reset_index(drop=True)


# --------------------------------------------------------------------- 1. méta
def meta_sets(pg: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """{(ligue, poste, date du match): {champion: part des picks}} pour chaque clé demandée.

    Fenêtre [date - META_DAYS, date) ; ligue trop calme -> picks de toutes les ligues de
    `pg` sur la même fenêtre (pg est déjà restreint aux ligues voulues)."""
    pg = pg.sort_values("date")
    t_all = pg["date"].dt.tz_convert(None).to_numpy()
    out = {}
    by_league = {lg: g for lg, g in pg.groupby("league")}
    for (lg, day), grp in keys.groupby(["league", "day"]):
        t = pd.Timestamp(day, tz="UTC")
        lo = t - pd.Timedelta(days=META_DAYS)
        g = by_league.get(lg)
        win = g[(g["date"] >= lo) & (g["date"] < t)] if g is not None else pg.iloc[0:0]
        if win["gameid"].nunique() < META_MIN_GAMES:
            i0 = np.searchsorted(t_all, lo.tz_convert(None).to_datetime64())
            i1 = np.searchsorted(t_all, t.tz_convert(None).to_datetime64())
            win = pg.iloc[i0:i1]
        for role, w in win.groupby("position"):
            c = w["champion"].value_counts()
            c = c[c >= META_MIN_PICKS]
            out[(lg, role, day)] = (c / c.sum()).to_dict() if len(c) else {}
    return out


# ------------------------------------------------------------- 2. lien joueurs
HOME_SHARE = 0.10         # un serveur est « celui de la ligue » s'il porte ≥ 10 % de ses comptes


def link_players(pg: pd.DataFrame, pages: set[str],
                 servers: dict[str, set] | None = None) -> dict[tuple[str, str], str]:
    """(nom Oracle, équipe Oracle) -> page Leaguepedia, pour les joueurs suivis en soloQ.

    Un nom ne suffit pas : « Ice » (Team Heretics) n'est pas « Ice (Gerardo Arroyo) » des
    ligues mexicaines. Un lien est retenu si la page a porté ce maillot (effectifs
    Leaguepedia) ; à défaut, pour un candidat unique, si l'un de ses comptes soloQ
    (`servers` : page -> serveurs) est sur un serveur où jouent les joueurs de cette ligue
    (appris sur les liens vérifiés par le maillot)."""
    ids, teams = {}, {}
    import datetime as dt
    for year in range(2025, dt.date.today().year + 1):
        try:
            p = lp.load("players", year)
            r = lp.load("rosters", year)
        except Exception:  # noqa: BLE001 - saison absente : on fait avec l'autre
            continue
        for page, pid in zip(p["Page"], p["ID"]):
            if page in pages and isinstance(pid, str):
                ids[page] = pid
        for page, team in zip(r["Player"], r["Team"]):
            if page in pages and isinstance(team, str):
                teams.setdefault(page, set()).add(team)
    for page in pages:                                   # page sans fiche : son nom fait foi
        ids.setdefault(page, page.split(" (")[0])
    by_key: dict[str, list[str]] = {}
    for page, pid in ids.items():
        by_key.setdefault(sl.norm(pid), []).append(page)

    def same_team(page: str, team: str) -> bool:
        tags = sl.team_tags(team, None)
        return any(sl.norm(t) == sl.norm(team) or tags & sl.team_tags(t, None)
                   for t in teams.get(page, ()))

    servers = servers or {}
    pairs = pg.groupby(["playername", "teamname"])["league"].agg(set)
    out, pending = {}, []
    for (name, team), leagues in pairs.items():
        cands = by_key.get(sl.norm(name), [])
        checked = [c for c in cands if same_team(c, team)]
        if len(checked) == 1:
            out[(name, team)] = checked[0]
        elif not checked and len(cands) == 1:
            pending.append((name, team, leagues, cands[0]))
    # serveurs de chaque ligue, d'après les joueurs dont le lien est vérifié par le maillot
    seen: dict[str, dict] = {}
    for (name, team), page in out.items():
        for lg in pairs[(name, team)]:
            for srv in servers.get(page, ()):
                seen.setdefault(lg, {}).setdefault(srv, 0)
                seen[lg][srv] += 1
    home = {lg: {srv for srv, n in c.items() if n >= HOME_SHARE * sum(c.values())}
            for lg, c in seen.items()}
    for name, team, leagues, page in pending:
        local = set().union(*(home.get(lg, set()) for lg in leagues))
        if servers.get(page, set()) & local:
            out[(name, team)] = page
    return out


# ------------------------------------------------------------ 3. soloQ par joueur
def _impostors(vol: pd.DataFrame, acc: dict) -> set[str]:
    """Comptes attribués à tort à un pro : deux comptes du « même » joueur dans la MÊME partie,
    c'est impossible (ex. un compte « Faker#… » du classement, croisé face à Hide on bush).
    On garde le compte donné par le wiki et on écarte l'autre ; sans compte wiki, les deux."""
    dup = vol[vol.duplicated(["player", "match_id"], keep=False)]
    if dup.empty:
        return set()
    wiki: set[str] = set()
    for path in sq.DIR.glob("accounts_*.csv"):
        a = sq.read_csv(path)
        a = a[a["usable"].astype(str) == "True"]
        wiki |= set((a["game_name"].astype(str) + "#" + a["tag"].astype(str)).str.lower())
    bad: set[str] = set()
    for _, g in dup.groupby("player"):
        pus = set(g["puuid"])
        trusted = {pu for pu in pus if str(acc[pu]["rid"]).lower() in wiki}
        bad |= (pus - trusted) if trusted else pus
    return bad


def _valid_from(vol: pd.DataFrame, det: pd.DataFrame, owner: dict) -> dict:
    """Date à partir de laquelle l'historique soloQ d'un joueur est COMPLET.

    Riot ne liste que les ~1 000 dernières parties d'un compte (toutes files confondues,
    vérifié le 18/11/2026 : ni fenêtre de dates ni pagination ne remontent plus loin). Pour
    un compte très actif, la liste commence donc en cours de saison ; ce qui précède est
    perdu, sauf les parties jouées avec un autre compte suivi (détail lu par ce biais).
    Un compte est « tronqué » s'il a une partie connue antérieure au début de sa liste ;
    le joueur n'est fiable qu'après le début de liste le plus tardif de ses comptes tronqués.
    Les joueurs absents du dictionnaire ont un historique complet."""
    first_list = vol.groupby("puuid")["time"].min()
    first_det = det.groupby("puuid")["time"].min()
    out: dict = {}
    for pu, t0 in first_list.items():
        td = first_det.get(pu)
        if td is not None and td < t0 - pd.Timedelta(days=1):
            p = owner[pu]
            out[p] = max(out.get(p, t0), t0)
    return {p: np.datetime64(t.tz_convert(None).to_datetime64()) for p, t in out.items()}


_raw_cache: dict = {}


def _raw_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(parties listées, détails), relus seulement quand les fichiers changent."""
    key = tuple((p.name, p.stat().st_mtime) for p in (sh.DIR / "ids.csv", sh.DIR / "details.csv")
                if p.exists())
    if _raw_cache.get("key") != key:
        _raw_cache.update(key=key, vol=sh.all_games(), det=sh.load_details())
    return _raw_cache["vol"].copy(), _raw_cache["det"]


def soloq_tables(champions: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(volume : parties datées ; détail : champion/poste/victoire), une ligne par partie et
    par joueur (page Leaguepedia).

    Seules les parties à la date EXACTE (détail récupéré) comptent : une date estimée peut
    se tromper de plusieurs heures et faire passer une partie jouée après le match pour une
    partie d'avant. `attrs["exact"]` = part des parties listées qui ont leur date exacte ;
    tant qu'elle est loin de 100 %, l'historique est incomplet et les volumes sous-estimés.
    `champions` : noms Oracle, pour écrire les champions soloQ de la même façon."""
    acc = json.loads((sh.DIR / "accounts.json").read_text(encoding="utf-8"))
    owner = {pu: v["player"] for pu, v in acc.items()}
    vol, det = _raw_tables()
    vol["player"] = vol["puuid"].map(owner)
    vol = vol.dropna(subset=["player"])
    bad = _impostors(vol, acc)
    vol = vol[~vol["puuid"].isin(bad)].drop_duplicates(["player", "match_id"])
    exact = float(vol["exact"].mean()) if len(vol) else 0.0
    vol = vol[vol["exact"]]
    det = det[~det["puuid"].isin(bad)]
    valid_from = _valid_from(vol, det, owner)
    det = det[det["early_end"] == 0].copy()
    # propriétaire ACTUEL du compte (la colonne écrite à l'époque peut être périmée)
    det["player"] = det["puuid"].map(owner).fillna(det["player"])
    det = det.drop_duplicates(["player", "match_id"])
    if champions:
        names = {champ_key(c): c for c in champions}
        det["champion"] = det["champion"].map(lambda c: names.get(champ_key(c), c))
    vol, det = vol[["player", "time"]], det[["player", "time", "champion", "position", "win"]]
    vol.attrs["exact"] = exact
    vol.attrs["impostors"] = len(bad)
    vol.attrs["valid_from"] = valid_from
    servers: dict[str, set] = {}
    for v in acc.values():
        servers.setdefault(v["player"], set()).add(v["region"])
    vol.attrs["servers"] = servers
    return vol, det


def _windows(times: np.ndarray, t: np.datetime64, days: int, end_days: int = 0) -> tuple[int, int]:
    lo = np.searchsorted(times, t - np.timedelta64(days, "D"))
    hi = np.searchsorted(times, t - np.timedelta64(end_days, "D"))
    return lo, hi


def features(pg: pd.DataFrame, link: dict, vol: pd.DataFrame, det: pd.DataFrame,
             meta: dict, valid_from: dict | None = None) -> pd.DataFrame:
    """Indicateurs de préparation pour chaque ligne de `pg` (joueur x game pro).

    `valid_from` (voir `_valid_from`) : un joueur dont la fenêtre d'avant-match déborde sur
    la partie tronquée de son historique n'est PAS couvert (`complete` = False) — ses volumes
    seraient sous-estimés, et d'autant plus qu'il joue beaucoup."""
    valid_from = valid_from or {}
    # Tableaux numpy par joueur, préparés une fois : la boucle tourne sur des dizaines de
    # milliers de lignes, et pandas recopiait à chaque découpe les annexes de la table
    # (`attrs` : serveurs et dates de validité de tous les joueurs).
    vol = pd.DataFrame({"player": vol["player"].to_numpy(),
                        "time": vol["time"].dt.tz_convert(None).to_numpy()})
    vol_t = {p: np.sort(g.to_numpy()) for p, g in vol.groupby("player")["time"]}
    det = det.sort_values("time")
    det = pd.DataFrame({"player": det["player"].to_numpy(),
                        "time": det["time"].dt.tz_convert(None).to_numpy(),
                        "champion": det["champion"].to_numpy(dtype=object),
                        "position": det["position"].to_numpy(dtype=object),
                        "win": det["win"].to_numpy(dtype=float)})
    det_g = {p: (g["time"].to_numpy(), g["champion"].to_numpy(dtype=object),
                 g["position"].to_numpy(dtype=object), g["win"].to_numpy())
             for p, g in det.groupby("player")}
    rows = []
    for r in pg.itertuples(index=False):
        page = link.get((r.playername, r.teamname))
        t = np.datetime64(r.date.tz_convert(None).to_datetime64())
        rec = {"gameid": r.gameid, "playername": r.playername, "page": page}
        vt = vol_t.get(page)
        if page is None or vt is None:
            rows.append({**rec, "covered": False})
            continue
        lo60, hi = _windows(vt, t, BASE_DAYS)
        vf = valid_from.get(page)
        rec["complete"] = bool(vf is None or t - np.timedelta64(BASE_DAYS + 7, "D") >= vf)
        rec["covered"] = bool(hi > lo60) and rec["complete"]
        for d in WINDOWS:
            lo, _ = _windows(vt, t, d)
            rec[f"vol_{d}"] = int(hi - lo)
        lo_b, hi_b = _windows(vt, t, BASE_DAYS + 7, 7)
        base_week = (hi_b - lo_b) / BASE_DAYS * 7
        rec["vol_ratio"] = (rec["vol_7"] + 1) / (base_week + 1)
        g = det_g.get(page)
        m = meta.get((r.league, r.position, r.date.normalize().tz_convert(None).date()), {})
        if g is not None:
            gt, gchamp, gpos, gwin = g
            lo, hi = _windows(gt, t, 14)
            champs, wins = gchamp[lo:hi], gwin[lo:hi]
            at_role = gpos[lo:hi] == ROLE_SOLOQ[r.position]
            rchamps, rwins = champs[at_role], wins[at_role]
            in_meta = np.array([c in m for c in rchamps], dtype=bool)
            rec["det_14"] = int(hi - lo)
            rec["role_14"] = int(at_role.sum())
            rec["meta_share_14"] = float(in_meta.mean()) if len(rchamps) else np.nan
            rec["meta_champs_14"] = len(set(rchamps[in_meta]))
            rec["meta_wr_14"] = float(rwins[in_meta].mean()) if in_meta.any() else np.nan
            rec["pick_practiced"] = bool((champs == r.champion).any())
            rec["pick_in_meta"] = r.champion in m
            # hasard : chance qu'un pick tiré dans la méta (selon sa fréquence) soit un
            # champion qu'il a joué en soloQ -> référence du contrôle de cohérence
            played = set(champs)
            rec["pick_chance"] = float(sum(s for c, s in m.items() if c in played))
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ rapport
def build(since: str = "2026-05-01") -> pd.DataFrame:
    """Tout bout à bout, pour les matchs des ligues pariables depuis `since`."""
    cfg = load_config()
    leagues, events = universe(cfg)
    pg_all = pro_games(cfg)
    pg_all = pg_all[pg_all["league"].isin(leagues | events)]
    pg = pg_all[pg_all["date"] >= pd.Timestamp(since, tz="UTC")].copy()
    pg["day"] = pg["date"].dt.tz_convert(None).dt.normalize().dt.date
    meta = meta_sets(pg_all, pg[["league", "day"]].drop_duplicates())
    vol, det = soloq_tables(set(pg_all["champion"].unique()))
    link = link_players(pg, set(vol["player"].unique()), vol.attrs["servers"])
    f = features(pg.drop(columns="day"), link, vol, det, meta, vol.attrs.get("valid_from"))
    keep = ["gameid", "date", "league", "patch", "teamname", "playername", "position", "champion"]
    out = pg[keep].reset_index(drop=True).join(f.drop(columns=["gameid", "playername"]))
    out.attrs["exact"] = vol.attrs["exact"]
    out.attrs["impostors"] = vol.attrs["impostors"]
    out.attrs["truncated"] = len(vol.attrs.get("valid_from") or {})
    return out


def main() -> None:
    f = build()
    pd.set_option("display.width", 200)
    n = len(f)
    linked = f["page"].notna()
    cov = f["covered"].fillna(False).astype(bool)
    print(f"{n} lignes joueur x game (ligues pariables, depuis mai 2026)")
    print(f"  reliées à un joueur suivi : {linked.mean():.0%} · soloQ dans les 60 j : {cov.mean():.0%}")
    print("  couverture par ligue :", (f.assign(c=cov).groupby("league")["c"].mean() * 100)
          .round(0).sort_values(ascending=False).to_dict())

    d = f[cov & (f["det_14"].fillna(0) >= 5)]
    print(f"\n=== CONTRÔLE DE COHÉRENCE ({len(d)} lignes avec ≥ 5 parties détaillées sur 14 j) ===")
    print(f"  champion joué en match déjà joué en soloQ les 14 jours d'avant : "
          f"{d['pick_practiced'].mean():.0%}")
    print(f"  champion joué en match qui est dans la méta de la ligue : {d['pick_in_meta'].mean():.0%}")
    print("  par ligue :", (d.groupby("league")["pick_practiced"].mean() * 100).round(0)
          .sort_values(ascending=False).to_dict())
    print("\n=== INDICATEURS (médianes, joueurs couverts) ===")
    cols = ["vol_3", "vol_7", "vol_14", "vol_ratio", "role_14", "meta_share_14", "meta_champs_14",
            "meta_wr_14"]
    print(f[cov][cols].describe().loc[["count", "50%", "mean"]].round(2).to_string())


if __name__ == "__main__":
    main()

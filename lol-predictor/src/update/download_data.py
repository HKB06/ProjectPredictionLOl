"""Télécharge le CSV Oracle's Elixir 2026 le plus récent depuis Google Drive.

Le fichier est mis à jour quotidiennement par Oracle's Elixir ; son ID Drive est
stable (config.yaml -> data.oracle_drive_id).

⚠️ Ce fichier est TRÈS téléchargé : Google impose un quota partagé ("Too many users...").
On gère ça proprement :
  - on écrit dans un .tmp puis on ne remplace la data que si le contenu est un vrai CSV ;
  - on essaie gdown, puis un fallback requests ;
  - si tout échoue (quota), on lève une erreur SANS abîmer la data locale (daily.py continue).

Usage :
    python -m src.update.download_data
"""
from __future__ import annotations

from pathlib import Path

from src.ingest.load_oracle import ROOT, load_config

# Fallback si absent de config.yaml (ID du 2026_LoL_esports_match_data_from_OraclesElixir.csv)
OE_2026_FILE_ID = "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"
MIN_BYTES = 1_000_000  # un vrai CSV 2026 fait des dizaines de Mo

# Dossier public d'Oracle's Elixir : un CSV par saison (ID Drive stable de chaque fichier).
OE_FOLDER_ID = "1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"
OE_FILE_IDS = {
    2014: "12syQsRH2QnKrQZTQQ6G5zyVeTG2pAYvu", 2015: "1qyckLuw0-hJM8XqFhlV9l1xAbr3H78T_",
    2016: "1muyfpaIqk8_0BFkgLCWXDGNgWSXoPBwG", 2017: "11fx3nNjSYB0X8vKxLAbYOrS2Bu6avm9A",
    2018: "1GsNetJQOMx0QJ6_FN8M1kwGvU_GPPcPZ", 2019: "11eKtScnZcpfZcD3w3UrD7nnpfLHvj9_t",
    2020: "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d", 2021: "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy", 2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN", 2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: OE_2026_FILE_ID,
}


def _looks_like_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.readline().lower()
    except OSError:
        return False
    return "gameid" in head  # en-tête Oracle's Elixir


def _try_gdown(file_id: str, tmp: Path) -> bool:
    try:
        import gdown
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(tmp), quiet=False)
        return tmp.exists()
    except Exception as exc:  # noqa: BLE001
        # gdown écrit dix lignes par refus : on garde la raison, le journal tourne en boucle
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        reason = next((ln for ln in lines if "Too many users" in ln), lines[0] if lines else "?")
        print(f"[download] gdown KO : {reason}")
        return False


def _try_requests(file_id: str, tmp: Path) -> bool:
    """Fallback : endpoint usercontent (gère souvent les gros fichiers sans gdown)."""
    try:
        import requests
        url = "https://drive.usercontent.google.com/download"
        sess = requests.Session()
        r = sess.get(url, params={"id": file_id, "export": "download", "confirm": "t"},
                     stream=True, timeout=120)
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            print("[download] requests KO : page HTML (quota/confirmation).")
            return False
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        return tmp.exists()
    except Exception as exc:  # noqa: BLE001
        print(f"[download] requests KO : {exc}")
        return False


def _google_key() -> str | None:
    """GOOGLE_API_KEY : variable d'environnement, puis fichier `google.key` (gitignoré)."""
    import os
    key = os.environ.get("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()
    for p in (ROOT / "google.key", ROOT.parent / "google.key"):
        if p.exists() and p.read_text(encoding="utf-8").strip():
            return p.read_text(encoding="utf-8").strip()
    return None


def _try_api(file_id: str, tmp: Path) -> bool:
    """Secours quand le lien public est saturé (« Too many users… ») : l'API Google Drive,
    avec une clé gratuite, a son propre quota. Sans clé, cette voie est simplement ignorée."""
    key = _google_key()
    if not key:
        return False
    try:
        import requests
        r = requests.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                         params={"alt": "media", "key": key}, stream=True, timeout=120)
        if r.status_code != 200:
            print(f"[download] API Drive KO : HTTP {r.status_code}")
            return False
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        return tmp.exists()
    except Exception as exc:  # noqa: BLE001
        print(f"[download] API Drive KO : {exc}")
        return False


def season_path(year: int, cfg: dict | None = None) -> Path:
    """Emplacement local du CSV d'une saison (même dossier que la saison en cours)."""
    cfg = cfg or load_config()
    return (ROOT / cfg["data"]["oracle_csv"]).with_name(
        f"{year}_LoL_esports_match_data_from_OraclesElixir.csv")


def download_season(year: int, cfg: dict | None = None, force: bool = False) -> Path:
    """Télécharge le CSV d'une saison passée. Une saison terminée ne change plus : on ne
    la retélécharge pas si elle est déjà là (sauf `force`)."""
    if year not in OE_FILE_IDS:
        raise ValueError(f"Saison {year} absente du dossier Oracle's Elixir ({min(OE_FILE_IDS)}"
                         f"-{max(OE_FILE_IDS)}).")
    out = season_path(year, cfg)
    if _looks_like_csv(out) and not force:
        return out
    return _download(OE_FILE_IDS[year], out)


def download_latest(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = ROOT / cfg["data"]["oracle_csv"]
    file_id = cfg["data"].get("oracle_drive_id") or OE_2026_FILE_ID
    return _download(file_id, out)


def _download(file_id: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")

    print(f"[download] fichier {file_id} -> {out}")
    ok = False
    for attempt in (_try_gdown, _try_requests, _try_api):
        if tmp.exists():
            tmp.unlink()                     # reste d'un essai précédent (page HTML d'erreur)
        if attempt(file_id, tmp) and _looks_like_csv(tmp):
            ok = True
            break

    if ok and _looks_like_csv(tmp):
        tmp.replace(out)
        print(f"[download] OK ({out.stat().st_size / 1e6:.1f} Mo)")
        return out

    if tmp.exists():
        tmp.unlink()
    raise RuntimeError(
        "Téléchargement impossible (quota Drive partagé ?). Data locale conservée. "
        f"Réessaie plus tard, ou télécharge à la main : https://drive.google.com/uc?id={file_id}"
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Télécharge les CSV Oracle's Elixir")
    ap.add_argument("--seasons", type=int, nargs="*", metavar="ANNÉE",
                    help="saisons passées à récupérer (ex. --seasons 2024 2025) ; "
                         "sans cette option : la saison en cours")
    ap.add_argument("--force", action="store_true", help="retélécharge même si déjà présent")
    args = ap.parse_args()
    if not args.seasons:
        download_latest()
        return
    for year in args.seasons:
        try:
            path = download_season(year, force=args.force)
            print(f"[download] saison {year} : {path.name} ({path.stat().st_size / 1e6:.1f} Mo)")
        except Exception as exc:  # noqa: BLE001 - une saison bloquée n'empêche pas les autres
            print(f"[download] saison {year} : ÉCHEC ({exc})")


if __name__ == "__main__":
    main()

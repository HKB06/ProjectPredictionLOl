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
        print(f"[download] gdown KO : {exc}")
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


def download_latest(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    out = ROOT / cfg["data"]["oracle_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    file_id = cfg["data"].get("oracle_drive_id") or OE_2026_FILE_ID
    tmp = out.with_name(out.name + ".tmp")

    print(f"[download] fichier {file_id} -> {out}")
    ok = _try_gdown(file_id, tmp) or _try_requests(file_id, tmp)

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
    download_latest()


if __name__ == "__main__":
    main()

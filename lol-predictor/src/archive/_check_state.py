import os
from pathlib import Path
import pandas as pd

raw = Path(r"data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv")
proc = Path(r"data/processed")

d = pd.read_csv(raw, usecols=["date", "league"], low_memory=False)
d["date"] = pd.to_datetime(d["date"], errors="coerce")
lck = d[d.league == "LCK"]
print(f"RAW CSV       : {len(d)} lignes | -> {d['date'].max().date()} | LCK -> {lck['date'].max().date()}")

for name in ["matches.parquet", "team_games.parquet", "features.parquet"]:
    p = proc / name
    if not p.exists():
        print(f"{name:20}: ABSENT")
        continue
    mtime = pd.Timestamp(os.path.getmtime(p), unit="s", tz="UTC").tz_convert("Europe/Paris")
    try:
        df = pd.read_parquet(p)
        dcol = "date" if "date" in df.columns else None
        maxd = pd.to_datetime(df[dcol]).max().date() if dcol else "?"
        print(f"{name:20}: {len(df):>6} l. | derniere date {maxd} | ecrit {mtime:%Y-%m-%d %H:%M}")
    except Exception as e:
        print(f"{name:20}: ERREUR lecture ({e})")

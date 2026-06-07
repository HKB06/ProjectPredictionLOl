# ProjectPredictionLOl — Prédiction LoL esports (LCK / LEC) + paris

Pipeline de prédiction (données Oracle's Elixir → modèles winner / kills / valeur + front Streamlit).
Sauvegarde pour reprendre le travail sur un autre PC.

## Structure
- `lol-predictor/` — le pipeline (`src/`, `app.py` Streamlit, `data/`, `config.yaml`, `requirements.txt`).
- `POINT_ETAPE.md` — **journal scientifique** (toutes les hypothèses testées + résultats). **À lire en premier.**
- `SUIVI_PARIS.md` — forward-test des paris (draft-only vs cotes book).
- `SUIVI_PREDICTIONS.md` — suivi des prédictions vs résultats réels.
- `cursor_projet_de_master_big_data_ia.md` — export complet de la conversation Cursor (tout le contexte).

## Reprendre sur le portable
```powershell
git clone https://github.com/HKB06/ProjectPredictionLOl.git
cd ProjectPredictionLOl/lol-predictor
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

## Scripts clés (depuis `lol-predictor`)
```powershell
.\venv\Scripts\python.exe -m src.models.winprob_stages   # draft vs exécution (gold @10/@15)
.\venv\Scripts\python.exe -m src.models.draft_predict     # penchant draft-only d'un matchup
.\venv\Scripts\python.exe -m src.models.value_backtest    # backtest valeur vs cotes
```

## Reprendre une session avec l'assistant (IA)
L'assistant **ne garde PAS la mémoire** entre deux sessions (rien n'est automatique). Pour qu'il
retrouve tout le contexte demain sur le portable :
1. Ouvre le dossier `ProjectPredictionLOl` dans Cursor.
2. Demande-lui de lire d'abord ces deux fichiers (ils contiennent TOUT l'historique) :
   - `POINT_ETAPE.md` (journal scientifique : hypothèses testées + résultats)
   - `cursor_projet_de_master_big_data_ia.md` (export complet de notre conversation)
   Exemple de message : *« Lis POINT_ETAPE.md et cursor_projet_de_master_big_data_ia.md, puis fais-moi
   un point sur où on en est avant de continuer. »*

## Sauvegarder à la fin d'une session
```powershell
git add -A
git commit -m "décris ce que tu as fait"
git push
```

## Notes
- **Données incluses** : `lol-predictor/data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv` (~42 Mo).
  Mise à jour : https://oracleselixir.com/tools/downloads
- **Exclus** (regénérables) : `venv/`, `node_modules/`, `.next/`.
- **Front Next.js d'inspiration** (non inclus) — re-cloner si besoin :
  `git clone https://github.com/Flames1217/LOL-DeepWinPredictor.git`
- **Verdict actuel** : meilleur modèle pré-game pariable = Elo + forme + side + draft (~0.74 AUC) ;
  +état @15 ≈ 0.823 (non pariable en live) ; edge réaliste = mispricing (fader un favori survalué) + régionales molles.

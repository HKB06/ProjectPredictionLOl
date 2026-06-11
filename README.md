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

## Lancer l'app en 1 clic
Double-clique sur **`Lancer_LoL.bat`** (racine du projet) : il (1) tente de rafraîchir la data
Oracle's Elixir depuis le Drive, puis (2) ouvre l'app Streamlit dans le navigateur.

L'app a 2 pages :
- **📅 Matchs à venir (accueil)** — calendrier des prochains jours (API lolesports) avec NOTRE proba Elo
  (toutes équipes), filtres par ligue, bouton **🔄 Actualiser**, et un **calculateur de value** (saisis les
  cotes → edge + verdict). Les matchs sont **récupérés en direct à chaque ouverture**.
- **🎯 Prédiction par draft** — saisie 2 équipes + 10 champions → probas par marché (ligues du scope).

## Scripts clés (depuis `lol-predictor`)
```powershell
.\venv\Scripts\python.exe -m src.models.winprob_stages   # draft vs exécution (gold @10/@15)
.\venv\Scripts\python.exe -m src.models.draft_predict     # penchant draft-only d'un matchup
.\venv\Scripts\python.exe -m src.models.value_backtest    # backtest valeur vs cotes
```

## Automatisation quotidienne (data fraîche + watchlist pré-match)
Objectif (cf. `SUIVI_PARIS.md`, leçon n°1) : le modèle voit juste, le **goulot c'est le timing de la mise**.
La watchlist calcule NOTRE proba (Elo toutes-ligues) sur les matchs des prochains jours, **à l'avance**,
pour repérer tôt un favori que le book va peut-être sur-coter (pattern KC / VKS / Heretics).

```powershell
# tout faire en 1 commande (data Drive -> tables -> WATCHLIST.md) :
.\venv\Scripts\python.exe -m src.update.daily

# options : fenêtre 5 jours, sans retélécharger la data :
.\venv\Scripts\python.exe -m src.update.daily --days 5 --no-download

# briques individuelles :
.\venv\Scripts\python.exe -m src.update.download_data   # MAJ CSV 2026 depuis Google Drive
.\venv\Scripts\python.exe -m src.update.leaguepedia     # liste les matchs à venir
.\venv\Scripts\python.exe -m src.update.watchlist       # génère WATCHLIST.md
```
Sortie : **`WATCHLIST.md`** (racine) — table des matchs à venir avec notre proba + colonnes `Cote`/`Edge` à remplir.

**Planifier 1×/jour (Windows Task Scheduler)** — lance la tâche tous les jours à 9h :
```powershell
$py  = "d:\Downloads\MemoireM2\Projet_Perso\lol-predictor\venv\Scripts\python.exe"
$cwd = "d:\Downloads\MemoireM2\Projet_Perso\lol-predictor"
schtasks /create /tn "LoL daily watchlist" /tr "cmd /c cd /d $cwd && $py -m src.update.daily" /sc daily /st 09:00 /f
```

**Caveats**
- Le CSV OE est très téléchargé → quota Drive partagé fréquent ("Too many users..."). Le script **ne casse rien**
  (garde la data locale et continue) ; relance plus tard ou récupère via `git pull` / le navigateur.
- L'API Leaguepedia limite les requêtes anonymes : **1 appel/jour passe sans souci** (en test rapproché ça peut throttle).
- Watchlist = **Elo-only** (signal partiel, sans draft) → un repère pré-match, pas une reco finale.

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

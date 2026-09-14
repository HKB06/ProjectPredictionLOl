# ProjectPredictionLOl — Prédiction LoL esports + détection de valeur

Pipeline complet : données **Oracle's Elixir** → Elo calibré par ligue → probabilités
pré-match → comparaison aux **cotes du book** → mesure du **ROI réel**.

> ⚠️ Projet **perso**, distinct du mémoire M2 (qui porte sur l'aide à la décision pour
> coachs/analystes). Les livrables école sont dans `../Livrables_M2/`.

## L'essentiel en 3 chiffres

| | |
|---|---|
| **Data** | 3 903 matchs · 14/01 → 13/09/2026 · 12 ligues (LCK, LEC, LCS, LPL, LFL, PRM, TCL, LJL, LAS, LCP, EBL, HLL) |
| **Picks 🎯 haute confiance** | **81,0 %** de vainqueurs trouvés (458 picks / 60 j, walk-forward sans fuite) |
| **Cote minimale de rentabilité** | **1,23** — en dessous, on perd **même en ayant raison** |

La leçon centrale du projet : **l'accuracy ne suffit pas**. Un favori à 85 % joué à cote
1,10 est perdant sur la durée. Tout l'outillage sert à trouver les spots où la cote paie
plus que le risque.

## Structure

- `lol-predictor/` — le pipeline : `src/`, `app.py` (Streamlit), `data/`, `config.yaml`.
- `POINT_ETAPE.md` — **journal scientifique** (hypothèses testées + résultats). **À lire en premier.**
- `SUIVI_PARIS.md` — forward-test des paris et leçons d'exécution (marchés qui ferment, timing).
- `SUIVI_PREDICTIONS.md` — suivi des prédictions vs résultats réels.
- `HIGH_CONFIDENCE.md`, `WATCHLIST.md`, `BACKTEST_RECENT.md` — sorties générées.
- `Data Oracle LOL/` — historique complet 2014→2026 (local, **non versionné** : ~810 Mo).

## Lancer l'app

Double-clique sur **`Lancer_LoL.bat`** : rafraîchit la data depuis le Drive, puis ouvre
Streamlit. Ou manuellement, depuis `lol-predictor/` :

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

### Les 7 pages

| Page | Ce qu'elle fait |
|---|---|
| 📅 **Matchs à venir** (accueil) | Calendrier live (API lolesports) + notre proba Elo, picks 🎯, **cote mini** par match, calculateur de value |
| 🎯 **Prédiction par draft** | 2 équipes + 10 champions → probas par marché (vainqueur, kills, tours, dragons, durée) |
| 📊 **Bilan prédictions** | Backtest walk-forward : réussite par jour, par ligue, upsets |
| 🔴 **Série en cours** | Suivi live d'une série (score par map + cotes live via odds-api.io) |
| 💰 **Journal de paris** | Paris réellement posés : P/L, ROI, winrate, **CLV** |
| 🤖 **Assistant IA** | Questions en langage naturel sur la data du projet |
| 💵 **Rentabilité des picks** | **Les picks 🎯 rapportent-ils ?** Simulation ROI à mise fixe + journal auto-réglé |

## Le modèle

**Elo toutes-ligues, K=32 + marge de victoire (MOV)**, avec **calibration par ligue** :
la proba est aplatie là où le modèle est historiquement mauvais (ligues chaotiques), et
conservée là où il est fiable. Résultat : probabilités **nativement bien calibrées**
(sans couche de recalibration).

Un **pick 🎯** = ligue fiable **+** favori ≥ 70 %/game **+** data ≥ 15 games/équipe
**+** pas de cross-ligue. Le levier n'est pas le modèle mais la **sélectivité** :
prédire *tout* plafonne à ~65 %, être sélectif monte à ~81 %.

### Rentabilité par ligue (60 derniers jours)

| À jouer | Réussite | Cote mini | | À éviter | Réussite | Cote mini |
|---|---|---|---|---|---|---|
| LES | 94,6 % | **1,06** | | PRM | 65,6 % | 1,52 |
| NLC | 92,9 % | **1,08** | | LAS | 66,7 % | 1,50 |
| ROL | 87,0 % | **1,15** | | LFL | 69,4 % | 1,44 |
| HM | 85,7 % | **1,17** | | TCL | 76,9 % | 1,30 |

LFL et PRM passent le filtre « ligue fiable » mais exigent des cotes que le book ne
donne jamais → à sortir de la sélection.

## Commandes clés (depuis `lol-predictor/`)

```powershell
# --- Quotidien : data fraîche -> tables -> watchlist pré-match ---
.\venv\Scripts\python.exe -m src.update.daily
.\venv\Scripts\python.exe -m src.update.daily --days 5 --no-download

# --- Mise à jour manuelle de la data ---
.\venv\Scripts\python.exe -m src.update.download_data      # CSV 2026 depuis Google Drive
.\venv\Scripts\python.exe -m src.ingest.build_match_table  # -> matches / team_games
.\venv\Scripts\python.exe -m src.features.build_features   # -> features
.\venv\Scripts\python.exe -m src.ingest.load_oracle        # sanity-check

# --- Sélection & valeur ---
.\venv\Scripts\python.exe -m src.update.watchlist          # -> WATCHLIST.md
.\venv\Scripts\python.exe -m src.models.high_confidence    # -> HIGH_CONFIDENCE.md
.\venv\Scripts\python.exe -m src.update.oddsapi            # cotes book + edge
.\venv\Scripts\python.exe -m src.models.picks_roi          # ROI des picks (mise fixe)
.\venv\Scripts\python.exe -m src.models.picks_roi --capture

# --- Évaluation ---
.\venv\Scripts\python.exe -m src.update.backtest_recent --days 7
.\venv\Scripts\python.exe -m src.models.eval_models        # comparatif de variantes
.\venv\Scripts\python.exe -m src.models.audit_calibration  # audit reproductible -> reports/
```

### Planifier 1×/jour (Windows Task Scheduler)

```powershell
$py  = "d:\Downloads\MemoireM2\Projet_Perso\lol-predictor\venv\Scripts\python.exe"
$cwd = "d:\Downloads\MemoireM2\Projet_Perso\lol-predictor"
schtasks /create /tn "LoL daily watchlist" /tr "cmd /c cd /d $cwd && $py -m src.update.daily" /sc daily /st 09:00 /f
```

## Discipline de pari (ce que le projet a appris à ses dépens)

1. **Marché vainqueur uniquement.** Les props (total maps, kills) ont été le principal
   poste de pertes.
2. **Porte de valeur obligatoire** : ne parier que si `cote_book ≥ max(1.30 ; cote_mini × 1.10)`.
   Sinon → **skip**. La plupart des jours = 0 pari, et c'est normal.
3. **Mise fixe**, jamais de mise variable « au feeling ».
4. **Ne jamais fader un favori court** (cote < 1,2) sur une ligue chaotique : le book y
   est sharp, un gros « edge » affiché est une erreur du modèle, pas une valeur.
5. **Le goulot n'est pas la prédiction mais l'exécution** : marchés qui ferment à la fin
   de la draft, matchs lancés en avance, absence de marché (cf. `SUIVI_PARIS.md`).
6. **Tout logger** : sans journal, il n'y a pas de preuve. Viser 25-30 paris avant de
   conclure quoi que ce soit.

## Configuration

`config.yaml` — ligues du scope, année, marchés, fenêtres de forme, seed.
Clé odds-api.io : variable `ODDS_API_KEY` ou fichier `oddsapi.key` (gitignored).

## Notes

- **Data versionnée** : `lol-predictor/data/raw/2026_...csv` (~67 Mo). GitHub avertit
  au-delà de 50 Mo (la limite bloquante est à 100 Mo). Le fichier étant re-téléchargeable
  via `download_data.py`, on pourra à terme ne versionner que les Parquet.
- **Exclus du dépôt** : `venv/`, `Data Oracle LOL/`, `reports/`, secrets (`*.key`, `.env`).
- **Front Next.js d'inspiration** (non inclus) :
  `git clone https://github.com/Flames1217/LOL-DeepWinPredictor.git`
- **Jeu responsable** : ce projet est un outil d'analyse, pas une promesse de gain.
  Aide et information : [joueurs-info-service.fr](https://www.joueurs-info-service.fr) — 09 74 75 13 13.

# LoL Predictor — pipeline technique

Prédiction multi-marchés sur les matchs LoL esports à partir de données pro
historiques, **sans fuite de données** et avec **probabilités calibrées**.

> Projet PERSO, séparé du mémoire école. Vue d'ensemble et discipline de pari :
> voir le [README racine](../README.md).

## Objectif

Pour un match à venir (équipes + draft), sortir des **probabilités calibrées** par
marché : vainqueur, total kills, première tour, premier dragon, first blood, durée.
But final : comparer ces probas aux **cotes** pour détecter de la **valeur**.

## Données

- **Backbone** : CSV Oracle's Elixir (par-game : GD@15, picks/bans, first objectives).
- **Périmètre actuel** (`config.yaml`) : 12 ligues, année 2026 → **3 903 matchs**
  (14/01 → 13/09/2026), baseline côté bleu **53,9 %**.
- **Calendrier des matchs à venir** : API lolesports (fallback Leaguepedia + saisie manuelle).
- **Cotes** : odds-api.io (books mous) et Polymarket.

### Récupérer les données

Automatique (recommandé) :

```powershell
.\venv\Scripts\python.exe -m src.update.download_data
```

Manuel : compte gratuit sur <https://oracleselixir.com> (Tools → Downloads), télécharger
`2026_LoL_esports_match_data_from_OraclesElixir.csv`, le placer dans `data/raw/`.

## Règles d'or (anti-pièges)

1. **Anti-fuite** : toute feature historique n'utilise que les games **antérieures** au
   match prédit. Jamais d'agrégats « à aujourd'hui » pour l'entraînement.
2. **Baseline réelle = 53,9 %** (toujours côté bleu), pas 50 %.
3. **Calibration par ligue** : la proba est aplatie là où le modèle est historiquement
   mauvais. Toute recalibration future devra être ajustée **par fenêtre**, jamais sur
   l'ensemble (sinon la fuite revient par la couche de calibration).
4. **Pas de wrapper de calibration sur le winner** : mesuré, `CalibratedClassifierCV`
   dégrade l'AUC (0,59 vs 0,74) sur ce volume. La régression logistique régularisée est
   déjà bien calibrée.

## Installation

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Pipeline de données

```powershell
.\venv\Scripts\python.exe -m src.update.download_data      # CSV brut
.\venv\Scripts\python.exe -m src.ingest.build_match_table  # -> matches.parquet, team_games.parquet
.\venv\Scripts\python.exe -m src.features.build_features   # -> features.parquet (38 features)
.\venv\Scripts\python.exe -m src.ingest.load_oracle        # sanity-check (games, période, WR bleu)
```

Ou tout en une commande : `python -m src.update.daily`.

## Lancer le front

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

## Structure

```
lol-predictor/
├── app.py                      # accueil : matchs à venir + proba Elo + value
├── pages/                      # 1 draft · 2 bilan · 3 série live · 4 journal
│                               # 5 assistant IA · 6 rentabilité des picks
├── data/{raw,interim,processed} # CSV brut -> Parquet
├── src/
│   ├── ingest/                 # load_oracle, build_match_table
│   ├── features/               # build_features, champion_priors
│   ├── models/                 # predict, eval_models, high_confidence,
│   │                           # audit_calibration, picks_roi
│   └── update/                 # elo, watchlist, lolesports, oddsapi,
│                               # polymarket, backtest_recent, daily, download_data
├── models/                     # modèles + calibrateurs
├── reports/                    # sorties d'audit (non versionnées)
└── config.yaml                 # ligues, année, marchés, fenêtres, seed
```

## Modèles

| Brique | Rôle |
|---|---|
| `src/update/elo.py` | Elo K32 + MOV, fiabilité et shrink **par ligue** |
| `src/models/predict.py` | `MatchPredictor` : tous les marchés + proba de série |
| `src/models/eval_models.py` | Replay walk-forward, Brier / LogLoss / ECE, comparatif de variantes |
| `src/models/high_confidence.py` | La règle de **sélectivité** (picks 🎯) |
| `src/models/audit_calibration.py` | Audit reproductible : Murphy, ECE, ROC, IC bootstrap |
| `src/models/picks_roi.py` | Rentabilité : ROI à mise fixe, cote minimale |

## Évaluation

```powershell
.\venv\Scripts\python.exe -m src.models.eval_models         # comparatif de variantes Elo
.\venv\Scripts\python.exe -m src.models.audit_calibration   # audit complet -> reports/
.\venv\Scripts\python.exe -m src.update.backtest_recent --days 7
.\venv\Scripts\python.exe -m src.models.picks_roi --days 60
```

**Verdict actuel** : le modèle **sélectionne bien mais gradue mal** — il identifie le
vainqueur correctement (81 % sur les picks sélectifs) mais manque de résolution pour
s'éloigner de la base rate. Le gain attendu vient de la **draft / du lineup confirmé**,
pas d'un réglage d'Elo. Détail et intervalles de confiance : `src/models/audit_calibration.py`.

## Configuration

`config.yaml` : ligues du scope, année, `date_min`, marchés, fenêtres de forme, seed.
Clé odds-api.io : variable d'environnement `ODDS_API_KEY` ou fichier `oddsapi.key`
(gitignored). Sur Streamlit Cloud, passer par `st.secrets`.

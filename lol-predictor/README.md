# LoL Predictor (projet perso / paris)

Prédiction multi-marchés sur les matchs **LCK 2026** (puis LEC, LPL...) à partir de
données pro historiques, **sans fuite de données** et avec **calibration des probabilités**.

> Projet PERSO, séparé du mémoire école (qui porte sur l'aide à la draft pour coachs).

## Objectif
Pour un match à venir (draft + équipes), sortir des **probabilités calibrées** par marché :
vainqueur, total kills, première tour, premier dragon, first blood, durée de partie.
But final : comparer ces probas aux **cotes** pour détecter de la **valeur**.

## Données
- **Backbone** : CSV Oracle's Elixir (par-game, propre, contient GD@15, picks/bans, first objectives).
- **Enrichissement / live (plus tard)** : Gol.gg premium.

### Récupérer les données
1. Créer un compte gratuit sur https://oracleselixir.com (Tools -> Downloads).
2. Télécharger `2026_LoL_esports_match_data_from_OraclesElixir.csv`.
3. Le placer dans `data/raw/`.

## Règles d'or (anti-pièges)
1. **Anti-fuite** : toute feature historique est calculée uniquement avec les games
   ANTÉRIEURES au match prédit. Jamais les agrégats "à aujourd'hui" de Gol.gg pour l'entraînement.
2. **Petit dataset** (~329 games LCK 2026) : features LEAN, modèles simples, lissage.
3. **Baseline réelle = 56,8%** (toujours côté bleu), pas 50%.

## Installation
```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Utilisation
```powershell
# Charger + résumer les données (sanity-check vs Gol.gg : 329 games, 56.8% bleu, 31:55)
.\venv\Scripts\python.exe -m src.ingest.load_oracle
```

## Structure
```
lol-predictor/
├── data/{raw,interim,processed}/   # CSV brut -> features Parquet
├── src/{ingest,features,models,eval}/
├── models/                         # modèles + calibrateurs
├── reports/                        # backtests + figures
└── config.yaml                     # ligues, année, marchés, fenêtres, seed
```

## Roadmap
- P0 Scaffolding (fait)
- P1 Ingestion + exploration Oracle (LCK 2026)
- P2 Nettoyage -> table 1 ligne/game
- P3 Feature engineering (Elo, forme lissée, H2H, draft, GD@15) sans fuite
- P4 Modèles par marché + calibration + backtest temporel
- P5+ Front, V2 live (at10/at15), cotes/valeur

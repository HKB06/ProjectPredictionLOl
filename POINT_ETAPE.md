# Point d'étape — Projet perso (LOL DeepWin / prédiction LCK-LEC)

_Dernière mise à jour : 4 juin 2026 (session du soir)_

## Où on en est

### Ce qui est fait
- **Git + Python 3.11** installés sur la machine.
- Repo **LOL-DeepWinPredictor** cloné dans `d:\Downloads\MemoireM2\Projet_Perso\LOL-DeepWinPredictor`.
- **Front Next.js installé et lancé** → visible sur **http://localhost:3000** (`npm run dev` dans le dossier `frontend`).
  - Pages : `/` (assistant de draft), `/champions`, `/players`, `/schedule`.
  - Marche : design, navigation, sélecteur de champions (données de démo via `mock-data.ts`).
  - Ne marche pas : icônes (servies par le backend), stats live, bouton "Prédire".

### Analyse critique du projet (важно pour le mémoire)
Le modèle "BiLSTM-Attention" est en grande partie du marketing :
1. **L'Attention est du code mort** (définie mais jamais appelée dans le `forward`).
2. **Faux temporel** : entrée reshapée en séquence de longueur 1 → c'est un MLP déguisé.
3. **IDs en chiffres bruts** (`championId`, `teamId`) → features quasi inutiles, pas d'embeddings.
4. **Split aléatoire = fuite de données** (le repo l'avoue dans son endpoint diagnostic).
5. **Pas de prédiction de kills** : il sort UNIQUEMENT le vainqueur (1 proba sigmoïde).
6. La prédiction réelle = `0.55 × sortie_BiLSTM + 0.45 × heuristique_force_équipe` (bornée 5-95%).
   → C'est surtout l'heuristique qui porte la performance.

### Ce qui manque pour le faire tourner "pour de vrai"
- `BILSTM_Att.pt` (poids entraînés) — **absent** du repo et des releases.
- Le dataset d'entraînement — **non fourni** (données LPL privées).
- Données live (stats équipes, rosters, winrates) — via son scraper LPL/OP.GG (Chine).

### Conclusion sur la "justesse" (chiffres vérifiés, LoL PC uniquement)
ATTENTION : ne pas mélanger les jeux. Les chiffres style "88% in-game" (MSC 2025) = Mobile Legends ;
"91% DraftNet" = Dota 2. NON transférables à LoL.

Tableau LoL seulement (sources vérifiées) :
- Pro, draft seul : ~55-70% (LDANet 70% max ; LoLAnalyzer 55,6%)
- Pro, pré-game + stats historiques joueurs : AUC 0,97 MAIS suspect de fuite de données (IEEE CoG 2021)
- SoloQ, pré-game (maîtrise champion) : ~62-75% (Do et al. 75,1% ; MDPI 62%)
- SoloQ, pré-game + early in-game : ~76,8% (MDPI)
- In-game early (10-15 min) : ~67-75% (MDPI 74,6% ; Jadowski 67%)
- In-game mid/late : ~81-95% (LightGBM 81,6% ; Lin 95% = overfit reconnu)

Pièges (matière mémoire) :
- AUC ≠ accuracy ; AUC 0,97 pré-game pro = drapeau rouge (winrate historique calculé sur la même période = fuite).
- 95% in-game = overfit (features de fin de partie corrélées au résultat).

Cible honnête pour CE projet (LoL pro LCK/LEC) :
- Draft seul : ~60-65% (plafond optimiste 70%, suppose équipes de niveau égal).
- Pré-game + ratings équipe + forme + historique joueurs (sans fuite) : ~65-72%.
- + données live à 10-15 min (golddiffat10 dispo dans Oracle's Elixir) : ~75-81%.

## PÉRIMÈTRE PROJET PERSO (paris) — défini le 5 juin

Projet PERSO uniquement (paris), séparé du mémoire école (draft coachs).
But réel = probabilités CALIBRÉES par marché, à comparer aux cotes pour trouver de la VALEUR (pas l'accuracy brute).

### V1 — Pré-game (LCK + LEC), données Oracle's Elixir
Entrées : 2 équipes + rosters, draft (5 picks/équipe), bans (5/équipe), patch, side,
historique passé only (Elo/Glicko, forme récente, H2H, winrate/KDA joueurs sur champion).

Marchés (1 sortie calibrée chacun) + prédictibilité pré-game honnête :
- Vainqueur (binaire, result) : ~65-72% — le plus fiable
- Total kills (régression + O/U) : MAE ~5-6 ; O/U ~55-60%
- Première tourelle (firsttower) : ~57-62%
- First blood (firstblood) : ~52-56% (quasi pile/face)
- Premier dragon (firstdragon) : ~55-60%
- Durée partie (gamelength, O/U) : ~56-60%
- Total tours / drakes (régression) : modéré
- Premier baron (firstbaron) : faible (event tardif)
→ Signal fort = vainqueur + total kills. Faible = first blood + premier baron.

Modèle : 1 LightGBM par marché + calibration isotonic. Option : NN à embeddings champions (comparaison).
Validation : split TEMPOREL, zéro fuite. Métriques : AUC/Brier (classif), MAE/RMSE (régression).

### V2 — Live (10-15 min)
Ajoute snapshots Oracle's Elixir (golddiffat10/15, kills/drakes/tours à 10-15 min) → ~75-81%. Modèle séparé.

### Front
Garder le front cloné. Adapter : saisie picks + bans des 2 équipes ; panneau résultats = toutes les probas par marché.

### Data
Oracle's Elixir = entraînement (résultats + first X + stats). PAS de cotes dedans →
source de cotes séparée (Gol.gg / sites paris) à ajouter plus tard pour la détection de valeur.

### Décisions data/features (6 juin) — l'utilisateur a Gol.gg PREMIUM (9$/mois)
Sources Gol.gg premium dispo : Team comparison, Champions betting stats, Meta Infographic,
Champion Synergy, Game Search Engine, Team Graph, Player Graph.
Axes Team Graph : Winrate, Game Duration, Firstblood%, First Dragon%, First Rift Herald%,
First Tower%, GD@15, Dragon%, Nashor%, CKPM, CSM, DPM, GPM, WPM, WCPM, VSPM, PPG.

RÈGLES OBJECTIVES (3 nuances critiques) :
1. FUITE : les agrégats Gol.gg sont "à aujourd'hui" → INTERDIT pour entraîner.
   - Entraînement/backtest = données PAR-GAME Oracle's Elixir, features recalculées "à la date du match".
   - Gol.gg agrégé = inférence sur match à venir + cross-check + features champions/synergie.
2. PETIT DATASET (~400-500 games LCK 2026) : plus de features ≠ mieux. Commencer LEAN,
   ajouter une feature seulement si elle améliore le backtest.
3. VARIANCE forme récente / H2H : lisser (shrinkage vers moyenne ligue). Préférer last 10 à last 5.

SET DE FEATURES V1 (lean, fort signal) : winrate (overall+last10 lissé+blue/red), GD@15,
avg kill diff, first tower%, first dragon%, total kills avg/CKPM, game duration, H2H lissé.
+ deltas (A-B). Plus tard : DPM/GPM, synergie champions, stats joueurs.
ÉCARTÉ V1 : CSM/WPM/WCPM/VSPM (faible signal vainqueur), Meta Infographic (viz).

VERDICT MARCHÉS : Vainqueur ✅, Total kills ✅, Game time 🟡, First tower/dragon 🟡,
First blood/baron ❌ (quasi aléatoire — tes screens : FB 56%/53%).
Outils premium : Game Search Engine = extraction ; Team comparison = inférence live.

## Décision prise
- **Ne PAS** ressusciter / réentraîner son BiLSTM (architecture bancale, infra Spark, données LPL).
- **Garder le front** comme inspiration UI.
- **Construire notre propre pipeline** : données **Oracle's Elixir** (CSV téléchargeables) + modèle simple (**LightGBM/XGBoost**) pour **winner + kills**, objectif honnête (~64-67% winner pré-game).

## Avancement
- [FAIT] P0 Scaffolding : dossier propre `Projet_Perso/lol-predictor/` créé (config.yaml LCK/2026,
  requirements.txt installé dans venv, README, structure src/, script src/ingest/load_oracle.py).
- Décisions actées : source = Oracle's Elixir CSV (assez riche : GD@15, picks/bans, first objectives) ;
  stockage = CSV -> Parquet (PAS MongoDB, overkill pour du tabulaire) ; PAS de copier-coller manuel.
- Baseline réelle à battre = 56,8% (toujours bleu), pas 50%. Avg game time LCK 2026 = 31:55. 329 games.

## RÉSULTATS V1 BACKTEST (6 juin) — LCK 2026, test = 62 matchs (15-31 mai)
Pipeline complet construit : load_oracle -> build_match_table -> build_features (sans fuite) -> train_backtest.
Contrôle anti-fuite OK : corr(d_elo, y_winner) = +0.374 (réaliste, pas truqué).

Marché VAINQUEUR :
- Baseline (toujours majorité) 64,5% ; Elo-sign 75,8% (gonflé par petit échantillon) ;
- LogReg : acc 69,4%, AUC 0,717, Brier 0,210  <- MEILLEUR modèle V1
- LGBM : acc 53,2%, AUC 0,52 -> OVERFIT (confirme : petit dataset => modèle simple gagne)
First tower : LogReg AUC 0,704 (signal modeste réel).
First blood / first dragon : AUC ~0,50 -> quasi aléatoire (comme prévu).
Total kills : MAE 7,16 vs baseline 7,27 (le modèle n'apporte presque rien pré-game).
Durée : MAE 4,13 > baseline 3,71 (mieux vaut la moyenne) -> pré-game faible.

CONCLUSIONS :
- Le winner marche (~69%/AUC 0,72), cohérent avec la littérature pro. LogReg > LGBM sur 250 games.
- Kills/durée : faibles en pré-game -> nécessitent la V2 LIVE (golddiffat15, kills à 10-15 min).
- Pour faire monter : plus de data (LEC/LPL/saisons passées) réduira la variance et débridera LGBM.
- Rapports CSV dans lol-predictor/reports/.

## SESSION 6 juin (nuit) — DRAFT + FRONT STREAMLIT

### Features de DRAFT ajoutées (winrate champion, esprit DraftGap composante 1)
- Nouveau `src/features/champion_priors.py` : winrate champion "as-of" (sans fuite),
  calculé sur le POOL PRO COMPLET du CSV (5 548 games, 171 champions, toutes ligues).
  Anti-fuite = bisect strict sur la date (le match courant est exclu).
- Intégré dans `build_features.py` : `blue_champ_wr`, `red_champ_wr`, `d_champ_wr`
  (moyenne du winrate des 5 picks par équipe). corr(d_champ_wr, y_winner) = +0.155.

### Expérience source des priors (répond à la crainte "ligues mineures faussent")
`src/models/exp_champ_leagues.py` — marché vainqueur, LCK held-out, 2 protocoles :
| Source priors            | Pool | Split AUC | Roll AUC | Roll acc% |
|--------------------------|------|-----------|----------|-----------|
| Aucune draft             | 0    | 0.717     | 0.726    | 67.9      |
| **Toutes ligues**        | 5548 | **0.772** | **0.734**| **70.3**  |
| Majeures (LPL/LCK/LEC/LCS)| 1146| 0.713     | 0.723    | 68.9      |
| LCK seule                | 329  | 0.702     | 0.711    | 68.9      |
-> CONTRE-INTUITIF : "toutes ligues" gagne (le winrate champion est auto-normalisé,
   le volume réduit la variance plus que les ligues faibles ne biaisent). Config =
   `champ_prior_leagues: null`.
-> HONNÊTE : le gain robuste (rolling, ~200 matchs) est MODESTE : +2.4 pts acc.
   Le split unique (74.2% / 0.772) était optimiste (fenêtre de 62 favorable).
   La force d'équipe (Elo corr +0.374) reste dominante devant la draft (+0.155).

### Inférence + Front
- `src/models/predict.py` : `MatchPredictor` (entraîne tous les marchés + prédit
  un match isolé : 2 équipes + 10 champions -> probas par marché). Réutilise l'état
  final (Elo/forme/H2H) via `build_features(..., return_state=True)`.
- `app.py` : front **Streamlit** (saisie équipes + draft par rôle, panneau probas).
  Lancer : `.\venv\Scripts\python.exe -m streamlit run app.py` -> http://localhost:8501

## SESSION 6 juin (matin) — DÉCOUVERTE CALIBRATION + BO5 + sous-confiance

### Le wrapper de calibration DÉTRUISAIT le modèle (résultat majeur)
Diagnostic via re-prédiction du match HLE vs BRION (déjà dans la data) puis DK vs BRION
(match du jour, hors data). Le front sortait des probas trop molles (DK ~52% side-neutre)
alors que l'Elo (1539 vs 1429 = ~65%) et le marché (~65-77%) disaient DK gros favori.
Chaîne de diagnostic (3 hypothèses fausses corrigées par la data) :
- PAS un manque de data (saison finie le 31 mai = tout est là).
- PAS un sur-poids du side : `src/models/side_check.py` -> side modèle +6.9% ≈ réel +6.8% (x1.0).
- LE vrai coupable : `src/models/tune_winner.py` (rolling-origin, 212 matchs) montre que
  CalibratedClassifierCV(sigmoid, cv=3) sur ~250 games **dégrade tout** :
  | Config | AUC | acc% | Brier |
  |---|---|---|---|
  | C=0.5 + sigmoid (ancien) | 0.593 | 56.1 | 0.233 |
  | C=0.1 BRUT (nouveau) | 0.742 | 70.3 | 0.199 |
  -> La régression logistique régularisée est DÉJÀ bien calibrée ; le wrapper ajoutait
     du bruit. CHANGÉ : `predict.py` + `train_backtest.py` = LogReg C=0.1 SANS calibration.

### BO5/BO3 ajouté + side neutralisé proprement
- `bo_series_prob` + `MatchPredictor.predict_series` (app : sélecteur Format BO1/BO3/BO5).
- Neutralisation du side EXACTE en log-odds (moyenne des 2 sides) -> proba "série" pure
  force+draft, sans hypothèse de side. DK vs BRION : 69.2%/game, 82.6% BO5 (≈ marché 77%).

### Chiffres honnêtes après correction
- Rolling-origin (~200 matchs, robuste) : AUC 0.742, acc ~71%, Brier 0.199.
- Split unique (62) : acc 77.4%, AUC 0.777, Brier 0.180.
- Scripts diagnostic ajoutés : blind_check, side_check, reg_check, tune_winner.

### Rappel "ça rapporte ?" (important)
On MATCHE désormais le marché sur les matchs clairs -> bon signe de fiabilité, mais
"matcher le marché" = AUCUN edge pour parier. La value reste à prouver via un BACKTEST
DE VALEUR vs cotes réelles (prochain gros jalon). Pas de value sur DK 1.2 (piège évité).

## SESSION 6-7 juin (nuit) — BACKTEST DE VALEUR vs COTES (jalon majeur)

### Infra construite
- `data/odds/lck_2026_odds.csv` : cotes série BO3/BO5 LCK 2026 (32 séries jan->mi-fév
  saisies, pages mars->mai à venir). Format : date, team1, team2, score, odd1, odd2.
- `src/models/value_backtest.py` : moteur walk-forward SANS FUITE.
  - Pour chaque série au jour D : modèle vainqueur réentraîné sur games < D ;
    features = état (Elo/forme/H2H) au début de D (snapshot) ; SANS draft (cotes
    posées avant les drafts) ; side neutralisé -> proba game -> proba série (BO3/BO5).
  - Anti cold-start : ne parie que si les 2 équipes ont >= min-games games passées.
  - Garde-fou pro : ignore les "edges" > 30% (EV implausible = erreur modèle).
  - Sorties : EV par camp, paris simulés (flat + Kelly), ROI, hit rate.
- Mapping noms cotes->Oracle OK (FearX=BNK FEARX, KRX=Kiwoom DRX,
  OKSavingsBank/Hanjin Brion=HANJIN BRION, Hanwha Life=...Esports).

### RÉSULTAT 1er backtest (jan->fév = DÉBUT de saison)
- 32 séries -> 12 évaluables, 20 skip cold-start (warmup 5).
- Sans garde-fou : 11 paris, 27% gagnés, ROI flat **-18,3%**.
- Avec garde-fou (edge<=30%) : 3 paris, ROI flat **-10%**.
- => ON PERD. MAIS diagnostic en or :

### DIAGNOSTIC (très important)
- Les "edges" revendiqués sont ABSURDES (+100% à +294%) = signature d'un modèle qui
  se trompe lourdement, PAS de la value. Un vrai edge = 2-10%.
- Cause = COLD-START : Elo part à 1500 pour tous (pas de data 2025). Le modèle
  sur-évalue des équipes que le marché sait faibles (ex. Kiwoom DRX/KRX, price à
  5.5-12.0 et qui perd, mais que notre Elo met favori sur 2-3 résultats précoces).
- Le marché a des PRIORS bien meilleurs que nous en début de saison (rosters, 2025).
- CONCLUSION : toute la fenêtre jan-fév est inexploitable pour nous. Le test équitable
  = saison warmée (mars-mai) ET/OU amorçage Elo avec 2025.

### RÉSULTAT 2 : SAISON COMPLÈTE (130 séries jan->mai) = VERDICT DÉFINITIF
Cotes complètes saisies (130 séries, toute la saison LCK 2026). Backtest walk-forward :
- 110 évaluables, 34 paris, 50% gagnés, ROI flat **-9,3%** (Kelly -12,3%).
- CALIBRATION (métrique robuste, sur 110 séries, indépendante du seuil de pari) :
  | Warmup | Nous acc | Marché acc | Brier nous | Brier marché |
  |--------|----------|------------|------------|--------------|
  | 5      | 68%      | 76%        | 0.196      | 0.169        |
  | 15     | 70%      | 76%        | 0.188      | 0.172        |
  | 20 (avr-mai) | 74% | 80%        | 0.172      | 0.149        |

VERDICT : **on NE bat PAS le moneyline LCK.** Le marché est ~6 pts plus précis ET mieux
calibré, et l'écart PERSISTE quand on est chaud (donc PAS du cold-start = vraie supériorité
marché : rosters, scrims, news, argent sharp). Notre modèle est bon (74% chaud, conforme
littérature) mais le marché est meilleur. -9% ROI = on matche le marché et on paie la vig.

IMPLICATIONS STRATÉGIQUES :
- PARIS : abandonner le winner moneyline LCK (efficient). Pistes de vrai edge =
  (a) marchés SECONDAIRES (handicap de map, total maps, kills/durée O/U, first tower) que
  les books pricent plus mollement -> besoin de CES cotes ; (b) ligues moins efficientes ;
  (c) LIVE (V2, gold@10/15 -> 75-81%, lignes plus lentes). Le moneyline pré-game = mur.
- MÉMOIRE : résultat EN OR. Backtest temporel sans fuite démontrant "matcher mais ne pas
  battre un marché efficient", avec comparaison de calibration (Brier/accuracy), analyse
  cold-start et sensibilité. C'est exactement la "prise de hauteur" demandée (efficience de
  marché, calibration, protocole sans fuite, cold-start des ratings).

### Prochaines actions actées
1. **DATA 2025** (oracleselixir, CSV) -> amorcer Elo/forme = fix cold-start (rapproche du
   marché en début de saison, mais NE crée PAS d'edge : l'écart persiste à chaud).
2. Si on poursuit les paris : obtenir des cotes de MARCHÉS SECONDAIRES (pas juste moneyline).
3. Élargir multi-ligues (LEC/LPL/LCS) pour estimer les coefficients plus robustement.
4. Exploiter le résultat pour le MÉMOIRE (chapitre efficience de marché / calibration).

## Diagnostic "ingénieur data" (7 juin) — où sont les vrais leviers
Hiérarchie des problèmes (du code réel + 3 tests aveugle DK-BRION + backtest valeur) :
- #1 Mesurer la VALEUR (fait : infra OK). Battre le book != accuracy, c'est CLV/ROI sur
  marchés mal price-és + probas bien calibrées. Le winner LCK est efficient (on le matche).
- #2 Dataset minuscule (~250 games) = cause racine (overfit LGBM, kills/durée faibles,
  cold-start). Fix : 2025 + multi-ligues + contexte ligue.
- #3 Mauvaises features par marché : kills/durée/dragon ratent les STOMPS faute de
  |d_elo| (ampleur du mismatch) + tempo absolu (CKPM/len combinés). Winner = trop de
  features redondantes (raw+delta) pour 250 lignes -> passer en delta-only.
- #4 Draft superficielle (winrate global moyen) ET signal faible en pro (corr 0.155 vs
  0.374 Elo ; pointait vers BRION les 3 games perdues). A creuser par rôle + patch, mais
  pas la priorité "argent" (plutôt mémoire).
- #5 Elo simpliste (K fixe, pas de carry-over/roster) -> Glicko-2 + seed 2025.
- #6 Calibration jamais vérifiée formellement (reliability/ECE).

## SESSION 7 juin (nuit) — MARCHÉS SECONDAIRES + EDGE DE STYLE

### Audit data (réponse "a-t-on les first X ?") : OUI, 100% rempli LCK
firstblood, firsttower, firstdragon, firstherald, firstbaron, firsttothreetowers,
firstmidtower -> tous à 100%. + totaux (towers/dragons/barons/inhibitors).
+ SNAPSHOTS LIVE à 100% : golddiffat10/15, killsat10/15/20/25, gold/xp/cs at10/15.
=> le modèle LIVE V2 est constructible MAINTENANT (data déjà là). PAS dispo : "first to
5/10/15 kills" comme course chronométrée (pas de timeline kill par kill).

### Pouvoir prédictif par marché (src/models/secondary_markets.py, test 62 games)
| Marché binaire | AUC | acc% | base% |
|----------------|-----|------|-------|
| winner         |0.78 | 77   | 64    |
| first_herald   |0.75 | 61   | 58    |
| first_baron    |0.67 | 61   | 50    |
| first_tower    |0.63 | 60   | 56    |
| first_blood    |0.61 | 65   | 60    |
| first_dragon   |0.47 | 56   | 56  (bruit) |
Numériques O/U : on bat à peine la moyenne (total_kills +5%, towers +2%), reste négatif.
PIÈGES : (1) marges book secondaires ~8% (vs ~3% moneyline) -> il faut +8% d'edge ;
(2) ce qu'on prédit (herald/baron/tower) est CORRÉLÉ au winner -> book efficient là aussi.

### EDGE DE STYLE (src/models/style_edge.py) — le finding intéressant
Test stabilité (split-half) + indépendance (corr winrate) de chaque trait :
| Trait      | stable? | indép. force? | verdict |
|------------|---------|---------------|---------|
| total_kills| 0.60 ✅ | 0.12 ✅       | TRAIT DE STYLE réel |
| ckpm (tempo)| 0.63 ✅| 0.11 ✅       | TRAIT DE STYLE réel |
| first_blood| -0.32 ❌| 0.70 ❌       | bruit + = la force |
| first_herald| 0.08 ❌| 0.70 ❌       | = la force |
| duration   | 0.11 ❌ | -0.03         | peu stable |
=> Le "kill volume / tempo" est STABLE et INDÉPENDANT de la force. Ex concret : Nongshim
(34% WR mais 32.5 kills, ckpm 1.04 = faible mais agressive) vs KT/DK (55% WR, ~28 kills,
ckpm 0.88 = lents). L'intuition "équipe faible agressive" est VRAIE — mais via TOTAL KILLS,
pas via first blood (qui est du bruit corrélé à la force).

### Modèle tempo total kills (src/models/kills_tempo.py)
- MAE : tempo ne bat la moyenne que de ~5% (≈ LGBM) -> prédiction du total exact modeste.
- Directionnel vs ligne PARESSEUSE (moyenne glissante) : 69% (tous) / 79% (écarts nets)
  du bon côté -> BIEN au-dessus des ~54% requis.
- MAIS : valable seulement SI le book poste des lignes kills proches de la moyenne. S'il
  ajuste au matchup (il voit le ckpm public), l'edge fond. NON CONFIRMÉ sans lignes réelles.

### CONCLUSION paris (honnête)
- Moneyline LCK : imbattable (prouvé). Favoris = break-even (marché efficient).
- Secondaires "objectifs" : signal modeste + 8% marge + corrélé winner -> pas d'edge clair.
- SEUL LEAD pré-game : TOTAL KILLS O/U via tempo (style réel) -> à confirmer avec lignes O/U.
- Vraie piste edge : LIVE (at10/15, data prête) + ligues moins efficientes.

## SESSION 7 juin (matin) — TEST SUR VRAIES LIGNES O/U KILLS (KT vs DK)

Lignes réelles relevées (book, map 1) -> src/models/match_kills_value.py + team_kills_backtest.py
Modèle attaque/défense par équipe : kills_A = (kills_pour_A + kills_contre_B) / 2, loi NB.

RÉSULTAT (le test décisif du lead "kills O/U") :
| | Notre modèle | Book (ligne dé-viggée) | Verdict |
|--|--|--|--|
| DK kills | 14.1 | 14.2 | book SHARP (collé) -> EV ~0% |
| KT kills | 13.4 | 14.3 | écart ~1 kill -> faux "edge UNDER +10%" |
| Combiné | 27.5 | 28.5 | écart ~1 |

Backtest erreur du modèle kills/équipe (606 préds, no leak) :
- MAE 5.72 kills (baseline moyenne 5.91 -> +3.3% SEULEMENT) ; std réel 7.07.
- Prédictions à ±1 kill : 10% ; à ±3 kills : 30%.
=> Un écart de 1 kill avec le book est DANS LE BRUIT. Le "+10% EV" sur KT est un mirage.

### VERDICT FINAL — LEAD KILLS O/U MORT (LCK)
La ligne kills du book N'EST PAS paresseuse : elle est collée à notre tempo (DK 14.1 vs 14.2).
Double peine : (1) kills par équipe trop bruités (std 7) pour prédire à la précision requise ;
(2) le book intègre déjà le tempo (ckpm public). Aucun edge pré-game exploitable sur kills.
Récap efficience LCK testée : moneyline ✗, favoris ✗, objectifs ✗, kills O/U ✗. Book sharp partout.

### SEULES PISTES RESTANTES (réalistes)
- LIVE in-play (at10/at15 -> winner/kills) : data 100% prête, incertitude effondrée, lignes plus molles.
- Ligues mineures (lignes moins efficientes) : besoin data + cotes.
- MÉMOIRE : démonstration rigoureuse d'efficience multi-marchés = excellent matériau.

## SESSION 7 juin (matin) — PRÉVISIBILITÉ PAR LIGUE (src/models/league_predictability.py)

Question : "ligues mineures = plus 50-50, plus dures à prédire ?" -> MESURÉ (Elo WF 2026).
AUC vainqueur (Elo seul, plancher ; +0.06 avec modèle complet) :
- PLUS prévisibles que LCK : LAS .79, EBL .78, LJL .77, TCL .76, HLL .76, PRM .75, LFL .73
- ~LCK : LCK .716, LCP .72, LCS .70
- MOINS (chaos, intuition user juste) : LEC .67, LCKC .64, NACL .62, CBLOL .61, LPL .60, CD .57

CONCLUSION : l'intuition "mineure = chaos" est VRAIE pour les ligues ACADÉMIE (LCKC, NACL)
mais FAUSSE pour les régionales pro établies (LFL/PRM/TCL/LJL/HLL = 2-3 orgs dominent ->
PLUS prévisibles que LCK). LPL (majeure) = la moins prévisible (chaos).
=> SWEET SPOT EDGE = régionale prévisible (modèle précis) + book paresseux (peu surveillée)
   = LFL / PRM / TCL / LJL / HLL / EBL. NB : prévisibilité != edge, besoin des COTES pour prouver.

### PLAFOND DE SIGNAL KILLS (src/models/kills_ceiling.py + kills_by_patch.py)
Pourquoi le total kills est ininjouable, STRUCTURELLEMENT (pas juste "book sharp") :
- Meilleur modèle possible (LGBM toutes features) : R²=0.11, corr=0.38 (MAE 6.72 vs 7.27).
- ICC : l'identité des équipes explique 5% de la variance -> ~89% = BRUIT irréductible.
- Patch : swing moy 4.2 kills < bruit intra-patch 8.0 -> effet méta noyé (pas d'edge patch-lag).
=> Contraste clé : le VAINQUEUR est prévisible (AUC .72-.79) -> exploitable si book mou.
   Les KILLS ne sont PAS prévisibles (89% bruit) -> aucun edge, même book mou (rien à prédire).
   Donc cibler le VAINQUEUR en régionale, PAS les kills.

### DRAFT SEULE -> VAINQUEUR (src/models/draft_model.py) — test de la thèse "draft"
Modèle ratings de champions appris (logistic, X=+1 bleu/-1 rouge) sur 4533 games pro, test 65 LCK :
- Draft seule (moyenne winrate) : AUC 0.580 | acc 60% | Brier 0.236
- Draft seule (ratings appris)   : AUC 0.572 | acc 60% (pas mieux : pro data trop sparse)
- Baseline "toujours bleu" : acc 64.6% (!) -> la draft seule fait MOINS bien que le side.
- Modèle complet (Elo+forme+side+draft) : AUC ~0.74.
=> La draft SEULE = signal FAIBLE (0.58, à peine > coin-flip). L'exécution+forme dominent en pro.
   Anecdote : sur KT vs DK (7 juin) la draft seule donnait KT 44.6% -> penchait DK (correct).
   Limite : modèle additif, SANS synergies (Ashe+Ori) ni matchups de lane -> vrai signal un peu
   plus haut mais nécessite data SOLO-QUEUE (Lolalytics) pour des matchups fiables (pro trop rare).
   Top ratings : Lux/A.Sol/Morgana/Nasus/Vayne ; flop : Vladimir/Taric/Kindred/Viego/Yuumi.

### HYPOTHÈSE "draft décide entre égaux (top ligues)" (src/models/draft_by_tier.py) -> REJETÉE
Draft-only AUC par tier (ratings appris, test 20% recent/ligue) :
- LCK 0.547 (n65), LPL 0.540 (n87) -> gros échantillons = signal le PLUS FAIBLE
- LEC 0.654 (n47), LCS 0.648 (n28) -> plus haut MAIS petits échantillons (bruit)
- TOP3 0.573 vs AUTRES (régionales) 0.621 -> draft prédit MOINS en top3.
INTERPRÉTATION : en régionale l'AUC draft est gonflée (équipes faibles draftent mal ->
draft = proxy du skill). En LCK/LPL (vrais égaux) tout le monde drafte bien -> draft ~0.54
(quasi hasard) -> c'est l'EXÉCUTION qui décide, pas la draft. Hypothèse user infirmée.
Caveat : modèle additif (sans synergies/matchups) ; vrai signal peut-être un peu + haut en
top ligue mais nécessite data solo-queue. Excellent matériau mémoire (test d'hypothèse propre).

### DRAFT vs EXÉCUTION — win-prob par ÉTAPES (src/models/winprob_stages.py) — CHIFFRE LA THÈSE
Modèle emboîté, split temporel (train 8180 / test 2046 lignes-équipe ; draft appris sur 4090 games) :
- Draft seule (proba ratings champions)   : AUC 0.605
- Gold/kills @10 seul                       : AUC 0.750
- Gold/kills @15 seul                       : AUC 0.819
- Draft + @10 + @15 (empilé)                : AUC 0.823  <- MEILLEUR (empiler aide)
- @15 seul 0.819 -> @15 + draft 0.826       : +0.007 seulement (effet "scaling" réel mais minuscule)
Poids standardisés (modèle complet) : golddiffat15 +1.33 >> draft_prob +0.45 (~3x moins) > xpdiff ~0.4 ;
killdiff ~0 voire négatif une fois gold/xp connus.
=> VERDICT : (1) combiner draft+état = optimal (user a raison sur l'empilement) ; MAIS
   (2) l'EXÉCUTION écrase la draft (gold@15 seul ≈ modèle complet ; coef ~3x) ;
   (3) "la draft décide au-delà du lead" = vrai mais marginal (+0.007 AUC) ;
   (4) les KILLS bruts ne comptent pas, seul le LEAD or/xp compte (explique "bon early sans lead = rien").
   Pari : le gros prédicteur (gold@15) n'est PAS pré-game -> non pariable retail ; la draft pré-game
   reste faible (0.605) -> l'edge pré-game = MISPRICING du book (fader le momentum), pas out-predict.

### Live : ÉCARTÉ (insight user) — book suspend le marché à chaque action + retard stream
=> live "réagir à l'événement" non exploitable en retail (réservé syndicats à flux officiel).

## 2026-06-12 — UPGRADE MODÈLE (mesuré, pas deviné) + page Bilan Streamlit
Banc d'essai walk-forward `src/models/eval_models.py` (5734 games, burn-in ≥5 g/équipe ;
métriques accuracy / Brier / log-loss / ECE ; global + par ligue). Comparatif de 12 variantes.
**RETENU : Elo K32 + marge de victoire (MOV via écart de kills, multiplicateur log1p).**
- Accuracy saison : 64.0 % (K24 baseline) → **65.0 %** ; récent 45 j : 64.1 % → **65.7 %**.
- Brier 0.2226 → 0.2191 · log-loss 0.636 → 0.628 · ECE 0.044 → 0.026 (tout s'améliore).
- Forme/momentum (winrate L10) : ~0 sur l'accuracy → **écartée**. Side bleu : +0.7 pt mais
  **inutilisable pré-game** (sides inconnus avant la draft) ; gardé comme borne haute info.
**Calibration PAR LIGUE (la vraie trouvaille)** : la prévisibilité varie énormément.
- Fiables (acc ≥ 0.62) : LJL/LAS/TCL/HLL ~0.71-0.73, AL/PRM/EWC ~0.67, LEC 0.66, LCK 0.64, LFL 0.64.
- **CHAOTIQUES** : EM 0.56, LCS 0.57, CD 0.57, LPL/CBLOL 0.58.
- En **EM** le modèle est **sur-confiant** : un « 60-80 % » brut ne gagne que ~49-59 % (écart +15 pts).
  En **LCK/LEC** c'est l'inverse (sous-confiant). → `compute_elo` apprend `reliability[ligue]`
  (accuracy hist.) + `shrink[ligue]` (aplatit la proba vers 50 % là où on se trompe).
- **Flag ⭐ bridé** : étoile seulement si proba ≥62 % **ET** data ≥15 g **ET** ligue fiable.
  L'EM ne reçoit plus d'étoile (validé en live : Misa 71 % EM → 🌪️, a perdu 0-2 vs UCAM le 11).
Page Streamlit **Bilan** (`pages/2_Bilan_predictions.py`) : prévisions vs résultats en walk-forward
(bons/faux par jour & par ligue, table de calibration, détail filtrable « erreurs seulement »).
**Signal exploitable repéré** : les « risers » que l'Elo capte en retard (UCAM, KCB, Ruddy)
battent les favoris en EM → angle = parier SUR eux en outsider avant que book/Elo rattrapent.
**Limites** : MOV aide globalement mais en EM les stomps sont du bruit (stomp puis défaite) ;
cold-start non seedé (pas de data 2025) ; pools isolés (Elo EM gonflé → prudence cross-ligue).

## SESSION 12-13 juin (nuit) — POLYMARKET + RÈGLE 75 % + FRONT COMPLET (A/B/C)

### Pont Polymarket (le « pariable » devient mesurable)
- `src/update/polymarket.py` : API Gamma publique (sans clé), prix en cents = probas ≈ sans
  marge. `public-search?q=<équipe>` retrouve l'événement, marché `sportsMarketType=moneyline`
  = vainqueur de série. Croise notre watchlist -> `WATCHLIST_PARIABLE.md` (edge auto).
- Constat clé : Polymarket ne cote que majors + events -> nos pépites (LJL/TCL...) n'y sont
  pas. Le 13/06 : 4 matchs cotés, 1 seule value propre (Gen.G +4,3 pts, ligue fiable) ; les
  « gros edges » (TL +14,7, Keyd +8,6) = chaos/x-ligue -> correctement refusés.

### RÈGLE 75 % — la sélectivité MESURÉE (4 766 games walk-forward, hors cold-start)
- Tous matchs : 65 %. **Fiable + p>=0.65 : 81,1 % (n=1 174, ~56 picks/sem)** ;
  fiable + p>=0.70 : 83,4 % ; + data>=15 g + p>=0.75 : 84,4 %.
- Ligues chaotiques avec le MÊME filtre p>=0.65 : 64,4 % (EM : 52,7 % !) -> la sélection de
  ligue est la moitié du winrate. -> flag **🎯 pick** = fiable + >=15 g + p>=0.65 + pas x-ligue.

### Front industrialisé (A/B/C dans la même nuit)
- **A — Accueil** : signal 🎯 + filtre dédié + colonnes Polymarket auto (Value PM / lien,
  cache 15 min). `daily.py` = 4 étapes (data -> tables -> watchlist -> Polymarket) ;
  `Lancer_LoL.bat` lance désormais le daily complet.
- **B — `pages/3_Serie_en_cours.py`** : proba de série CONDITIONNELLE au score (race DP),
  cotes live -> edge par camp, détection auto du pattern KC/VKS (book s'emballe sur le
  gagnant de la map), draft-only optionnel (ratings champions, cache), garde-fous intégrés
  (jamais fader <1.20, chaos, x-ligue, cold-start).
- **C — `pages/4_Journal_paris.py`** + `data/bets.csv` : log de chaque pari (cote, mise,
  notre proba, clôture) -> P/L, ROI, winrate, **CLV**, courbe bankroll, ventilation par
  pattern. LA preuve finale = cette page (accuracy != profit).

### Picks week-end (règle 🎯) : Rising Gaming 80 % & FENNEL 69 % (LJL, books esports type
GG.bet) · Gen.G 94 % @~1,12 sur Polymarket (edge +4,3) · Saigon Dino 79 % (VCS, mise réduite).
À ignorer malgré les « edges » : TL/C9 (LCS), Keyd/LOS (x-région), BLG/WE (LPL), tout l'EM.

## Prochaine étape (à reprendre demain)
1. **PRIORITÉ : anciennes cotes régionales** (l'utilisateur peut en fournir : LJL/TCL/LFL/PRM,
   format date,team1,team2,score,odd1,odd2) -> backtest valeur sur régionales = savoir si les
   favoris 🎯 sont sous-cotés (la règle 81 % est-elle +ROI ?). Dernier maillon manquant.
2. **Seed Elo avec data 2025** : moins d'erreurs en début de split + plus de matchs éligibles
   🎯 (le burn-in en disqualifie). Gain attendu réel mais borné.
3. **Side bleu dans la page Série** (sides connus entre les maps -> +0,7 pt mesuré, gratuit).
4. **Journal : 20-30 paris loggés** (réels ou paper) -> ROI/CLV. Discipline > modèle.
5. (Optionnel) The Odds API (cotes Winamax/Betclic -> LFL auto) ; Glicko-2/K-adaptatif au banc
   d'essai (plafond proche, ne pas sur-investir) ; roster-changes = angle mort connu de l'Elo.
6. PORTES FERMÉES (ne pas rouvrir) : moneyline LCK, kills O/U, secondaires, live in-game,
   draft-only pré-game (marché fermé à la draft).
7. MÉMOIRE : tout ce qui précède = matériau (efficience, calibration, sélectivité, CLV).

## Rappels projet
- **2 projets séparés** :
  - **Mémoire école** = aide à la **draft** pour coachs LCK (pas de paris). Livrables dans `d:\Downloads\MemoireM2\Livrables_M2`.
  - **Projet perso** = prédiction winner/kills/cotes LCK-LEC (ce dossier `Projet_Perso`).
- Mémoire à rendre le **10 septembre**. Prof veut : rédiger vite + "prise de hauteur" théorique (apprentissage supervisé, multi-tâches, protocole temporel sans fuite, calibration, biais/dérive patchs).
- L'analyse critique ci-dessus = matière directe pour la "prise de hauteur".

## Note technique
- Serveur de dev front lancé en arrière-plan (`npm run dev`). Pour le relancer demain :
  `cd d:\Downloads\MemoireM2\Projet_Perso\LOL-DeepWinPredictor\frontend ; npm run dev`

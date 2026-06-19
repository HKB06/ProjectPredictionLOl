# LolPredictor — Ligne directrice & grand point projet

> **Boussole du projet**
> Le projet vaut par **une seule chaîne courte et honnête** :
> **Données récentes → Elo calibré par ligue → filtre « haute confiance » + écart aux cotes → Assistant IA qui tranche.**
> Tout ce qui ne nourrit pas cette chaîne est de la R&D déjà terminée : on l'archive pour arrêter de se perdre.

## Bilan chiffré (51 fichiers Python au total)

| Catégorie | Nombre | Décision |
|---|---|---|
| Fichiers cœur | 23 | **Garder** (branchés au produit) |
| Outils d'analyse | 2 | **Garder** (lancés ponctuellement) |
| R&D terminée | 21 | **Archiver** (conclusion déjà intégrée) |
| Scratch / debug | 5 | **Supprimer** |

---

## 1. Le cœur qui a de la valeur — à GARDER

Tout ce qui est branché au produit (app, pages, agent, refresh quotidien). C'est l'actif réel.

| Sous-système | Fichiers clés | Pourquoi c'est précieux |
|---|---|---|
| **Données** | `load_oracle.py`, `build_match_table.py`, `build_features.py`, `champion_priors.py` | Ingestion Oracle's Elixir → tables propres + features calculées *as-of* (anti-fuite). Base de tout. |
| **Moteur Elo** | `update/elo.py` | Cœur prédictif : Elo K32+MOV calibré par ligue + score de fiabilité. C'est CE qui marche vraiment. |
| **Prédiction draft** | `models/predict.py` | MatchPredictor (page 1) : proba game/série à partir des 2 équipes + draft (priors champion). |
| **Backtest production** | `models/eval_models.py` | Mesure la VRAIE accuracy historique (page 2 « Bilan »). Garde-fou d'honnêteté. |
| **Sélectivité & séries** | `high_confidence.py`, `series_momentum.py` | Règle « haute confiance » (seuil + ligues fiables) + dynamique de séries. C'est là qu'est l'argent. |
| **Cotes & watchlist** | `watchlist.py`, `oddsapi.py`, `polymarket.py`, `lolesports.py`, `leaguepedia.py`, `download_data.py`, `daily.py` | Détection de value vs bookmakers, calendrier des matchs, refresh quotidien automatique. |
| **Assistant IA** | `assistant/agent.py` | Agrège Elo, forme, draft, cotes et live en 1 verdict argumenté. La pièce maîtresse récente. |
| **Interface** | `app.py` + `pages/1–5` | Streamlit : prédiction draft, bilan, série live, journal de paris, Assistant IA. |

## 2. Outils d'analyse — à GARDER (lancés ponctuellement)

| Fichier | Rôle |
|---|---|
| `update/backtest_recent.py` | Recalcule l'accuracy récente → `BACKTEST_RECENT.md`. Contrôle périodique. |
| `models/tune_winner.py` | Réglage hyperparamètres (rolling-origin). Justifie `C=0.1` / pas de recalibration. Référence reproductible. |

---

## 3. À ARCHIVER — R&D terminée (conclusion déjà intégrée)

Pas du code « mauvais » : du code qui a fait son travail. La décision qu'il a produite vit déjà dans le cœur.
**Suggestion : les déplacer dans un dossier `archive/`** plutôt que les jeter (traçabilité).

| Catégorie | Fichiers | Pourquoi on archive |
|---|---|---|
| Modèles draft séparés | `draft_model.py`, `draft_by_tier.py`, `draft_predict.py`, `eval_draft.py` | Remplacés par les priors champion intégrés dans `predict.py`. |
| Marchés secondaires (kills/totaux/style) | `kills_ceiling.py`, `kills_tempo.py`, `kills_by_patch.py`, `match_kills_value.py`, `team_kills_backtest.py`, `secondary_markets.py`, `style_edge.py` | Piste explorée à fond, pas d'edge stable → jamais branchée au produit. |
| Explorations value / edge | `value_backtest.py`, `quick_edge.py`, `train_backtest.py` | Remplacées par la watchlist + `eval_models` + l'outil `value_check` de l'agent. |
| Contrôles ponctuels | `blind_check.py`, `side_check.py`, `reg_check.py` | Sanity-checks one-shot, déjà validés. |
| Études déjà intégrées | `exp_champ_leagues.py`, `league_predictability.py`, `winprob_stages.py`, `fav_strategy.py` | Conclusions absorbées dans `config.yaml`, la fiabilité Elo et l'outil live de l'agent. |

## 4. À SUPPRIMER — scratch & doublons

| Fichier | Raison |
|---|---|
| `lol-predictor/_check_state.py` | Script de debug non suivi par git. |
| `lol-predictor/tmp_selectivity2.py` | Brouillon temporaire. |
| `lol-predictor/tmp_value_em.py` | Brouillon temporaire. |
| `src/models/_inspect_series.py` | Inspection de debug (préfixe `_`). |
| `src/ingest/inspect_game.py` | Inspection de debug. |

---

## 5. Documentation & sorties

| Fichier | Décision |
|---|---|
| `POINT_ETAPE.md` | Garder — journal du projet, à tenir à jour. |
| `SUIVI_PARIS.md` | Garder — bankroll / ROI réel des paris. |
| `README.md` (×2) | Garder — fusionner racine + `lol-predictor` en un seul. |
| `WATCHLIST*.md`, `HIGH_CONFIDENCE.md`, `BACKTEST_RECENT.md` | Sorties auto-générées — garder, idéalement `.gitignore`. |
| `SUIVI_PREDICTIONS.md` | Doublon → fusionner dans `SUIVI_PARIS.md`. |
| `cursor_projet_de_master_big_data_ia.md` | Hors-sujet (projet Master) → sortir de ce repo. |

---

## 6. Prochaines valeurs à construire

1. **Données joueurs / rosters** — le SEUL vrai levier d'accuracy restant. Qui joue (titulaire, sub, nouveau roster) compte plus que +4 ans d'historique ancien.
2. **Suivi résultats automatique** — comparer prédiction vs réalité + *closing line value*, pour PROUVER l'edge au lieu de le supposer.
3. **Discipline bankroll** — mise Kelly fractionnée, plafonnée aux paris « haute confiance » ou à value positive. Le modèle ne sert à rien sans gestion de mise.

## Déjà tranché — ne pas y revenir

**Ne PAS réinjecter d'historique trop ancien (2022–2023)** pour « avoir plus de données » : meta, patchs et rosters ont trop changé, ça dégrade les probas actuelles au lieu de les améliorer. Le levier, c'est la **qualité** (joueurs), pas la quantité d'années.

# Picks HAUTE CONFIANCE — viser ≥80 % de vrais vainqueurs

> Backtest walk-forward **sans fuite** · 4924 games notées · 2026-01-15 -> 2026-06-21

## La vérité en 1 phrase
Prédire **tout** = ~65 % (un coinflip reste un coinflip). Pour **≥80 % réel** il faut être **sélectif** : (1) **ligues fiables** uniquement, (2) **confiance élevée**.

## Filtre 1 — TOUTES ligues (seuil de confiance seul)
| seuil | matchs | couv% | acc_reel% |
|---|---|---|---|
| >=50% | 4924 | 100.0 | 64.7 |
| >=55% | 3678 | 74.7 | 68.7 |
| >=60% | 2598 | 52.8 | 72.2 |
| >=65% | 1689 | 34.3 | 76.4 |
| >=70% | 1035 | 21.0 | 79.5 |
| >=75% | 593 | 12.0 | 82.0 |
| >=80% | 312 | 6.3 | 82.7 |
| >=85% | 126 | 2.6 | 83.3 |

## Filtre 2 — LIGUES FIABLES seulement (on exclut ['Asia Master', 'CBLOL', 'CD', 'EM', 'FST', 'HC', 'LCKC', 'LCS', 'LPL', 'NACL', 'PCS'])
| seuil | matchs | couv% | acc_reel% |
|---|---|---|---|
| >=50% | 3221 | 100.0 | 68.3 |
| >=55% | 2420 | 75.1 | 73.0 |
| >=60% | 1734 | 53.8 | 76.5 |
| >=65% | 1164 | 36.1 | 81.3 |
| >=70% | 743 | 23.1 | 83.7 |
| >=75% | 436 | 13.5 | 85.6 |
| >=80% | 241 | 7.5 | 87.1 |
| >=85% | 100 | 3.1 | 88.0 |

➡️ **Recette ≥80 %** : ligues fiables + confiance **≥65%/game** → **81.3%** réel sur **1164 matchs** (36% des matchs fiables).

## Par ligue fiable (confiance ≥70%/game)
| ligue | matchs_HC | acc_HC% |
|---|---|---|
| EBL | 18 | 100.0 |
| HM | 30 | 100.0 |
| LIT | 15 | 100.0 |
| LRN | 13 | 92.3 |
| ROL | 31 | 90.3 |
| LJL | 58 | 89.7 |
| TCL | 27 | 88.9 |
| LES | 26 | 88.5 |
| LFL | 23 | 87.0 |
| LCK | 87 | 85.1 |
| HLL | 31 | 83.9 |
| PRM | 56 | 82.1 |
| LAS | 93 | 81.7 |
| RL | 15 | 80.0 |
| EWC | 56 | 76.8 |
| LCP | 34 | 76.5 |
| LEC | 33 | 72.7 |
| NLC | 18 | 72.2 |
| AL | 35 | 71.4 |
| VCS | 25 | 68.0 |

**Règle d'or** : un pick 🎯 = ligue fiable **+** favori ≥70 %/game **+** data ≥15 g **+** pas de cross-ligue. En BO3/BO5, la série amplifie encore l'avantage. ⚠️ Haute *accuracy* ≠ profit auto : à cote courte le gain est faible — la value vient des ligues **mineures fiables** (LJL, LAS, PRM, TCL...) où le book est plus mou.

# Picks HAUTE CONFIANCE — viser ≥80 % de vrais vainqueurs

> Backtest walk-forward **sans fuite** · 5357 games notées · 2026-01-15 -> 2026-07-22

## La vérité en 1 phrase
Prédire **tout** = ~65 % (un coinflip reste un coinflip). Pour **≥80 % réel** il faut être **sélectif** : (1) **ligues fiables** uniquement, (2) **confiance élevée**.

## Filtre 1 — TOUTES ligues (seuil de confiance seul)
| seuil | matchs | couv% | acc_reel% |
|---|---|---|---|
| >=50% | 5357 | 100.0 | 64.7 |
| >=55% | 4036 | 75.3 | 68.5 |
| >=60% | 2876 | 53.7 | 71.9 |
| >=65% | 1893 | 35.3 | 75.9 |
| >=70% | 1172 | 21.9 | 79.0 |
| >=75% | 685 | 12.8 | 80.6 |
| >=80% | 369 | 6.9 | 81.3 |
| >=85% | 154 | 2.9 | 81.8 |

## Filtre 2 — LIGUES FIABLES seulement (on exclut ['Asia Master', 'CBLOL', 'CD', 'EM', 'FST', 'HC', 'KeSPA Cup', 'LCKC', 'LCS', 'LPL', 'MSI', 'NACL', 'PCS'])
| seuil | matchs | couv% | acc_reel% |
|---|---|---|---|
| >=50% | 3545 | 100.0 | 68.2 |
| >=55% | 2703 | 76.2 | 72.5 |
| >=60% | 1965 | 55.4 | 75.8 |
| >=65% | 1336 | 37.7 | 80.0 |
| >=70% | 863 | 24.3 | 82.9 |
| >=75% | 518 | 14.6 | 84.0 |
| >=80% | 291 | 8.2 | 85.6 |
| >=85% | 123 | 3.5 | 87.0 |

➡️ **Recette ≥80 %** : ligues fiables + confiance **≥65%/game** → **80.0%** réel sur **1336 matchs** (38% des matchs fiables).

## Par ligue fiable (confiance ≥70%/game)
| ligue | matchs_HC | acc_HC% |
|---|---|---|
| EBL | 18 | 100.0 |
| HM | 30 | 100.0 |
| LIT | 16 | 100.0 |
| LES | 37 | 89.2 |
| ROL | 36 | 88.9 |
| TCL | 27 | 88.9 |
| LJL | 59 | 88.1 |
| LRS | 22 | 86.4 |
| LFL | 29 | 86.2 |
| LCK | 87 | 85.1 |
| LAS | 93 | 81.7 |
| HLL | 42 | 81.0 |
| RL | 15 | 80.0 |
| PRM | 84 | 79.8 |
| LRN | 38 | 78.9 |
| LCP | 34 | 76.5 |
| EWC | 67 | 76.1 |
| AL | 41 | 75.6 |
| LEC | 33 | 72.7 |
| NLC | 18 | 72.2 |
| VCS | 25 | 68.0 |

**Règle d'or** : un pick 🎯 = ligue fiable **+** favori ≥70 %/game **+** data ≥15 g **+** pas de cross-ligue. En BO3/BO5, la série amplifie encore l'avantage. ⚠️ Haute *accuracy* ≠ profit auto : à cote courte le gain est faible — la value vient des ligues **mineures fiables** (LJL, LAS, PRM, TCL...) où le book est plus mou.

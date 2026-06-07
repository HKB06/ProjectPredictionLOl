# Journal de paris — suivi draft + cotes (forward test)

But : tester sur la durée si le **modèle draft-only** (penchant donné par les champions seuls)
prédit le vainqueur **mieux que le hasard**, et s'il trouve de la **valeur** que le book rate.

Tu me donnes, par game : **ligue, équipes, sides, les 10 champions, et les cotes du book**
(avant le résultat). Je calcule le penchant draft-only + l'edge vs le book, on note le résultat.

## Règle du "pari test" (révisée après game 3)
- **1 unité (u)** sur le **côté VALUE** : celui où **edge = proba modèle − proba book (dé-viggée) > +3 %**.
  (Avant on pariait le « favori draft » ; la game 3 a montré que c'est perdant si le book le **surcote**.
  On parie donc l'**outsider quand le draft dit que c'est plus serré que la cote**.)
- Ne compte QUE les games où on a la **draft AVANT le résultat** + une **cote book**.
- ⚠️ Le draft-only est un signal **partiel** (champions seuls, sans la force d'équipe/forme).
  Quand on a un modèle d'équipe (LCK), on l'ajoute ; sinon (LEC, etc.) c'est draft-only.

## Convention dé-vig
Proba book dé-viggée = (1/cote) normalisée sur les deux camps. Ex : 1.47 / 2.45 →
(1/1.47) / ((1/1.47)+(1/2.45)) = **62.5 %**.

---

## Tableau de bord (à jour)

| Métrique | Valeur |
|---|---|
| Games loggées (avec résultat) | 3 |
| Calls draft-only corrects (vainqueur map) | **2 / 3** *(G3 = coin-flip raté)* |
| Value bets (edge > 3 %) | 1 |
| P/L value bets | **+1.45 u** |
| ROI value | +145 % *(1 pari — non significatif)* |
| (info) règle « favori draft » si suivie | −0.35 u → la **value** fait mieux |

> Rappel honnête : il faut **20-30 games minimum** pour que ces chiffres veuillent dire
> quelque chose. 2/2 = encore du bruit.

---

## Détail par game

### 2026-06-07 — LCK — KT Rolster vs Dplus Kia — Game 1  ✅ terminé
- **Sides** : KT 🔵 bleu / DK 🔴 rouge
- **Draft KT** : Rumble, Xin Zhao, Viktor, Jhin, Nautilus
- **Draft DK** : Ornn, Wukong, Orianna, Ashe, Seraphine
- **Modèle draft-only** : KT **44.6 %** / DK 55.4 % → penche **DK**
- **Cote book (après draft)** : KT 2.08 / DK 1.65 → dé-viggé KT 44.2 % / DK **55.8 %**
- **Edge** : draft-only ≈ book (~0) → **pas de valeur**, juste un call
- **Pari test** : DK @ 1.65 *(ancienne règle « favori » ; edge ~0 → pas une value bet au sens révisé)*
- **Résultat** : **DK gagne** (stomp 13-4)
- **Verdict** : call ✅ (DK) · pari test **+0.65 u**
- *Note : notre draft-only colle au book (44.6 vs 44.2) → le book price déjà la draft.*

### 2026-06-07 — LEC — Karmine Corp vs G2 Esports — Game 2  ✅ terminé
- **Sides** : KC 🔵 bleu / G2 🔴 rouge
- **Draft KC** : Renekton, Jarvan IV, Taliyah, Corki, Nami
- **Draft G2** : Gnar, Xin Zhao, Ahri, Lulu, Yunara
- **Modèle draft-only** : KC **47.7 %** / G2 52.3 % → penche **G2** (malgré G2 côté rouge !)
- **Cote book** : non relevée → N/A
- **Pari test** : N/A (pas de cote)
- **Résultat** : **G2 gagne** (14-3)
- **Verdict** : call ✅ (G2) — la draft penchait G2 assez pour compenser le côté rouge.

### 2026-06-07 — LEC — Karmine Corp vs G2 Esports — Game 3 (belle, MSI final)  ✅ terminé — KC gagne
- **Sides** : G2 🔵 bleu / KC 🔴 rouge (confirmé)
- **Draft G2** : K'Sante (BrokenBlade), Nocturne, Aurora, Xayah, Neeko
- **Draft KC** : **Zaahen** (Canna, top), Vi, Viktor, Zeri, Rakan
- **Modèle draft-only (10/10)** : G2 **51.8 %** (bleu) / KC 48.2 % · side-neutre **49.3 % G2 / 50.7 % KC**
  → **pile ou face**, mini-penchant **KC** côté champions (Zaahen pèse ~ +2.7 pts KC vs sans).
- **Cote book — map 3** : G2 1.47 / KC 2.45 → dé-viggé **G2 62.5 %** / KC 37.5 %
- **Cote book — série** : G2 1.40 / KC 2.75 → dé-viggé G2 66.3 % / KC 33.7 %
- **Edge draft** : G2 **−10.7 pts** (sur-coté) · KC **+10.7 pts** → ⭐⭐ *gros value flag sur **KC***
- **Call draft-only** : G2 sur la map (51.8 % bleu) mais c'est un **coin flip** (side-neutre → KC).
- **Pari test (règle = favori draft)** : G2 @ 1.47 — ⚠️ **−EV** (faut 68 %, on a 51.8 %) → bet à éviter.
- **Value lean (thèse draft)** : **KC @ 2.45** (+10.7 pts, ≈ **+18 % EV** par le draft-only).
- **Résultat** : **KC gagne la belle** 🏆 (G2 meilleur early/mid mais ne convertit pas les kills ; KC scale mieux — Viktor late).
- **Verdict** : ⭐ **value KC @2.45 → ✅ +1.45 u** · call map (penchait G2) ❌ coin-flip · « favori draft » (G2 @1.47) aurait **perdu**.
- *Enseignement clé : ce qui a **payé = la VALUE** (book surcotait G2 à 62.5 % vs draft ~50/50), PAS « la draft décide » :
  le draft-only ne donnait pas KC favori sur la map (il penchait G2 d'un cheveu). Le vrai edge = **book trop confiant
  sur la forme** (le 14-3). C'est CE pattern — fader un favori surcoté après une grosse game — qu'on chasse.*

---

## Patterns / leçons (à enrichir)
- *(en construction — on remplit au fil des games)*
- Hypothèse à valider : le draft-only est plus utile en **LEC** (AUC ~0.65) qu'en **LCK** (~0.55).
- Angle à surveiller : **biais de récence** du book/public (sur-coter l'équipe qui vient de gagner la game précédente).
- **Game 3 KC/G2 = cas d'école (✅ confirmé)** : book G2 62.5 % mais draft-only **~50/50 (léger KC)** → tout l'écart
  du book = **forme/momentum**. **KC a gagné** → la **value** (fader le favori surcoté) a payé **+1.45 u**. ⚠️ MAIS
  le draft ne donnait pas KC favori : l'edge venait du **book trop confiant**, pas de « la draft décide ».
  → Pattern à chasser : **fader un favori surcoté juste après une grosse game** (biais de récence du public).
- **Leçon meta** : penser aux **champions récents** (ex. **Zaahen**, top Darkin sorti 19/11/2025). Ils sont
  dans la data 2026 (Zaahen = 2887 lignes) → toujours les inclure, ils peuvent peser plusieurs points.
- **Idée (game 3) à creuser** : combiner **draft + état @10/15 (gold/kills/xp)** pour une « win prob » par étapes.
  ⚠️ ultra-précis mais **pas pariable en live** (retail) ; en revanche le **scaling de la compo** (early vs late,
  ex. Viktor/Aurora) est une feature **pré-game** exploitable → on construit les deux (recherche + pré-game).

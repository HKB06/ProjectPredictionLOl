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
| Games loggées (avec résultat) | 5 |
| Calls draft-only corrects (vainqueur map) | **4 / 5** *(seul raté = KC/G2 G3, coin-flip)* |
| **Value bets « draft vs favori surcoté »** | **2 / 2 ✅** *(KC @2.45 · VKS @2.7)* |
| P/L value bets | **+1.45 u réel** · +1.7 u VKS *would-be (marché fermé)* |
| ROI value | très +, mais **n=2 → non significatif** |
| (info) règle « favori draft » si suivie | −0.35 u → la **value** fait mieux |

> Rappel honnête : il faut **20-30 games minimum** pour conclure. 2/2 = prometteur mais **encore du bruit**.
> **Friction n°1 à résoudre** : le book **ferme le moneyline à la fin de la draft** → l'edge draft n'est captable
> qu'en **live early-game**, pas en pré-game (cf. map 3 LOS/VKS).

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

## Forward test Elo-only (ligues mineures, pré-game sans draft) — `src/models/quick_edge.py`

### 2026-06-08 — EWC OQ South America — Leviatán vs paiN Gaming — ✅ terminé — **paiN 2-0**
- **Signal** : Elo toutes ligues (pas de draft), pré-game.
- **Modèle** : Leviatán 1388 ≈ paiN 1404 → **~50/50** (coin-flip).
- **Cote book** : Leviatán 5.2 / paiN 1.12 → dé-viggé Leviatán 17.7 % / paiN **82.3 %**.
- **"Edge" affiché** : Leviatán **+30 %** → ⚠️ **signalé comme FAUX edge** (Elo aveugle sur l'OQ Sud-Am :
  les 2 équipes ~1400, book à 1.12 = sharp). **Décision : NE PAS parier.**
- **Résultat** : **paiN écrase 2-0** (kills **23-4** puis **14-2**).
- **Verdict** : ✅ **skip correct** — fader paiN aurait **perdu**. *Marché soft ≠ exploitable quand NOTRE modèle est aveugle.*

### 2026-06-08 — EMEA Masters Spring Play-In — Verdant vs GOAL — 🔴 live (map 1)
- **Modèle** : Verdant 1658 / GOAL 1571 → **Verdant 62.4 %**. Book : Verdant 1.16 / GOAL 4.3 → Verdant 78.8 %.
- **"Edge" affiché** : GOAL **+16 %** → ⚠️ **artefact favori-longshot**. **Décision : NE PAS parier** (fader un favori à 1.16).
- **Live** : Verdant mène **29-17** (kills, map 1) → le favori confirme. *(Plus de marché dispo de toute façon.)*
- **Verdict (provisoire)** : artefact correctement ignoré.

### 2026-06-08 — EMEA Masters Spring Play-In — E Wie Einfach vs NightBirds — ⏳ résultat ?
- **Le seul vrai value lean** : **NightBirds @ 2.35** (Elo **1684** > E Wie Einfach **1563** ; modèle NightBirds
  **66.8 %** vs book dé-viggé 39.3 % → +27 pts, sur cotes **équilibrées** = signal crédible, pas un artefact).
- **Statut** : à confirmer — *dis-moi le résultat pour le logger.*

### 2026-06-08 — CBLOL — Los Grandes (LOS) vs Vivo Keyd Stars (VKS) — Game 2 — ⛔ PASS (pas de value)
- **Stats écran** : LOS clairement > VKS (WR 55.8 % vs 43.5 %, forme L10 7/10 vs 2/10, **H2H 71.4 %** LOS).
- **Modèle Elo** : LOS 1577 / VKS 1442 → **LOS 68.5 %** / VKS 31.5 %.
- **Cote book** : LOS 1.42 / VKS 2.6 → dé-viggé LOS 64.7 % / VKS 35.3 % (**vig 8.9 %**).
- **Test EV (vs cote brute, pas dé-viggée !)** :
  - LOS @1.42 → il faut **70.4 %** (1/1.42), modèle 68.5 % → **EV ≈ −2.7 %** ❌
  - VKS @2.6 → il faut 38.5 %, modèle 31.5 % → **EV ≈ −18 %** ❌
- **Verdict** : **les 2 côtés sont −EV** : LOS est la meilleure équipe mais le book le price déjà ; le petit
  « lean » (+3.9 pts vs juste) **ne couvre pas la vig (~4.4 %/côté)**. → **PASS.**
- ⚠️ **Sides inconnus** = facteur décisif (splits énormes : LOS red **40 %**, VKS red **20 %**). Si LOS est rouge,
  le lean s'évapore. On ne parie pas à l'aveugle sur le side.
- **Combos** (LOS & +26.5 kills @2.2 ; VKS & durée >30min @3.1) : **non** (kills = bruit prouvé + parlay surjuté).

### 2026-06-08 — EWC OQ SA — LOS vs VKS — Map 2 (VKS mène 1-0, a stomp la map 1 **17-8**)
- **Sides map 2** : VKS 🔵 bleu / LOS 🔴 rouge.
- **Draft VKS (bleu)** : Viktor, Sion, Wukong, Lulu, Sivir
- **Draft LOS (rouge)** : Ryze, Pantheon, Gnar, Rakan, Vayne
- **Modèle draft-only** : LOS **73.8 %** (rouge) / VKS 26.2 % → la draft penche **nettement LOS** (Elo aussi : 68.5 %).
- **MAIS signaux forward** ⚠️ : LOS vient de se faire **stomp 8-17** (map 1) + LOS sur son **pire side (red 40 %)**,
  VKS sur son bon side (blue 61.5 %). Les deux signaux *situationnels* penchent **VKS**.
- **Lecture** : draft/Elo (rétrospectifs) disent LOS rebondit ; stomp+side (forward) disent VKS continue. Vraie proba
  prob. proche du **coin-flip (~50-55 % LOS)**, donc book LOS 64.7 % (@1.42) **trop haut** → **VKS @2.6 = value possible**
  *(si la cote tient toujours)*. Confiance **faible** (qualif SA chaotique).
- **RÉSULTAT map 2** : ✅ **LOS gagne 22-12** (et sur son côté **rouge** !) → série **1-1**.
- **Verdict honnête** : ✅ le **draft-only (LOS 74 %) avait raison** ; ❌ mon **overlay « value VKS » (stomp+side) aurait PERDU**.
  → Leçon : ici la **qualité d'équipe/draft a battu le « fade momentum/side »**. LOS a gagné sur son « mauvais » side
  (red) → confirme qu'il est *réellement* meilleur ; le **40 % red de VKS/LOS était du small-sample/bruit**, à ne pas surpondérer.

### 2026-06-08 — EWC OQ SA — LOS vs VKS — **Map 3 (money map, série 1-1)**
- **Sides** : VKS 🔵 bleu / LOS 🔴 rouge (inchangés).
- **Cote book map 3** : LOS **1.40** / VKS **2.7** → dé-viggé LOS **65.9 %** / VKS 34.1 % (vig 8.5 %).
- **Nos signaux** : Elo LOS 68.5 % ; draft map 2 penchait LOS 74 % (draft map 3 pas encore reçue) ; LOS a gagné la map 2
  **sur red** = il est réellement meilleur. **Mais** série 1-1 + qualif chaotique = grosse variance sur 1 map.
- **Test EV (cote brute)** :
  - LOS @1.40 → faut **71.4 %**, on a ~68.5 % (Elo) → **≈ −4 % EV** ❌ (favori court, pas de value)
  - VKS @2.7 → faut 37 %, point-estim ~31.5 % → −EV *sur le papier*, MAIS sur une **décisive chaotique** la vraie
    proba outsider est souvent > modèle → **petit VKS @2.7 = pari variance défendable**, pas plus.
- **Draft map 3** — VKS 🔵 : Jarvan IV, Ashe, Leona, Aurora, Senna · LOS 🔴 : Mel, Nocturne, Pyke, Jayce, Galio
- **Draft-only** : **VKS 64.2 %** (bleu) / LOS 35.8 % → la draft a **BASCULÉ sur VKS** (vs LOS 74 % en map 2 !).
- **Edge** : book VKS 34.1 % vs draft **64.2 %** → **+30 pts sur VKS** ⭐⭐ (pattern value = draft vs favori surcoté, cf. KC/G2 G3).
- ⭐ **CALL : VALUE BET VKS @2.7** (faut 37 %, draft dit 64 %). Conflit assumé : Elo/force penche LOS, mais la **draft de
  CETTE map** penche VKS (et map 2 = la draft avait raison). Mise **modérée** (draft-only faible + qualif chaotique).
- **RÉSULTAT** : 🏆 **VKS gagne la belle → série 1-2 VKS.** Le **draft-only (VKS 64.2 %) avait RAISON** (le book à 34.1 % se trompait).
- **Verdict** : ⭐ **call value VKS @2.7 = ✅** (would-be **+1.7 u**). → **2/2** sur le pattern « draft contredit le favori surcoté » (KC/G2 G3 + ici).
- 🔑 **FRICTION MAJEURE découverte (Hugo)** : le book a **fermé le marché DÈS la fin de la draft** → impossible de poser le pari
  une fois la draft connue. Notre edge (la draft) n'existe **qu'après** la draft, mais c'est **pile** là que le moneyline pré-game
  **disparaît**. Le book traite la draft comme de l'info et se protège. → l'exploit réaliste = **live early-game** (draft connue +
  marché live encore ouvert), **pas** le pré-game. *(Même leçon que « Aucun marché disponible » plus tôt.)*

### 2026-06-10 — EMEA Masters Spring Main Event — Los Heretics (Team Heretics Academy) vs Karmine Corp Blue — BO3 — ⏳ résultat ?
- **Signal** : Elo toutes ligues (pas de draft), pré-game. ⚠️ **cross-pool** (Heretics = Superliga LES / KCB = LFL, se croisent seulement via l'EM).
- **Modèle** : Heretics Academy **1677** (38 g) / KCB **1565** (46 g) → **Heretics 65.5 %** / KCB 34.5 %.
- **Cotes book** : Heretics **2.60** / KCB 1.45 (book1, vig 7.4 %) → dé-viggé Heretics **35.8 %** ; book2 Heretics 2.38 / KCB 1.50.
- **Edge** : Heretics **+29.7 pts** (book1) / +26.9 pts (book2) → ⭐ **value flag sur Heretics** (book met KCB favori, nous Heretics).
- **Garde-fous OK** : favori KCB à 1.45 (PAS < 1.2 → pas le piège paiN/Verdant) ; cote équilibrée + désaccord vainqueur (profil = NightBirds @2.35, notre seul vrai lean). Marge de sécu : break-even à 38.5 %.
- ⚠️ **Risques** : cross-pool (Elo Heretics gonflé si Superliga < LFL) ; KCB plus chaud que notre data (manque ses 3 W vs Verdant) ; book sharp sur la marque Karmine ; BO3 = variance.
- ⭐ **CALL : value lean Los Heretics @ 2.60 — mise MODÉRÉE (0.5-1 u)** (cross-pool = on ne charge pas). Marché pré-match = **encore ouvert** (pas la friction draft-close).
- **LIVE (20h16)** : **Map 1 Heretics 22-8 = STOMP** → série **1-0 Heretics** ; Map 2 en cours, Heretics mène **15-8**.
  - ✅ **Map 1 Winner @2.35 = GAGNÉ** · ✅ **Heretics +1.5 / ≥1 map @1.52 = SÉCURISÉ** (au moins 1 map prise) · ⏳ **Match Winner @2.60** = besoin de fermer la map 2 (ou map 3).
  - 🎯 **Le read cross-pool se valide** : la Superliga (Heretics) n'était PAS plus faible — Heretics écrase la LFL (KCB). Notre modèle battait le book (qui mettait KCB favori).
  - ⚠️ **FRICTION TIMING (Hugo)** : **PAPER, NON PARIÉ** — le match a démarré **bien avant l'heure annoncée (20h)** → fenêtre pré-match ratée. Edge réel **mais non capturé** (3e échec d'exécution après "marché fermé à la draft" et "Aucun marché dispo").
- **Statut série** : ⏳ pas finie (1-0) — *donne-moi le score final + la draft si tu l'as.*

---

## Patterns / leçons (à enrichir)
- 🔑 **CE QU'ON TIENT (2026-06-08, n=2 mais net)** : quand le **draft-only contredit franchement le favori du book**
  (écart > ~15-20 pts) sur une **ligue soft/mineure**, **fader le favori = value** → **2/2** (KC @2.45 +1.45u · VKS @2.7 +1.7u).
  L'edge = **la draft**, lue AVANT que le book ne l'ait intégrée. ⚠️ **Le hic découvert sur LOS/VKS** : le book **ferme le
  marché à la fin de la draft** → fenêtre pré-game minuscule ; **capture réaliste = live early-game** (draft connue + marché
  live encore ouvert). C'est LA prochaine brique à construire (outil live draft + état @ premières minutes).
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
- **Règle ferme (validée 2026-06-08, paiN 2-0)** : un gros « edge » Elo **contre un favori court (cote < ~1.2)**
  sur une **ligue chaotique** (OQ, qualifs) = **erreur modèle, PAS value**. Le book est sharp sur les favoris courts.
  → On ne fade un favori **QUE** sur **cotes équilibrées + désaccord de vainqueur** (cf. NightBirds 1.52/2.35),
  **jamais** sur un 1.03 / 1.12 / 1.16.
- **Friction réelle (retail)** : sur ces petits events, les marchés sont **souvent fermés / absents**
  (« Aucun marché disponible ») ou se ferment vite → même un vrai edge peut être **impariable**. À intégrer
  dans le réalisme du projet (la prévisibilité ne sert que s'il existe un **marché ouvert + liquide**).
- 🚨 **LA CONTRAINTE N°1 = L'EXÉCUTION, PAS LE MODÈLE (validé 3x)** : le modèle voit juste (KC/G2, VKS, Heretics 22-8),
  mais on a **raté la mise 3 fois** pour 3 raisons d'exécution : (1) marché fermé à la fin de draft, (2) « Aucun marché
  dispo », (3) **match lancé avant l'heure → fenêtre pré-match ratée**. → Le **goulot d'étranglement n'est plus la prédiction**
  mais le **timing de la prise de pari**. PROCHAINE BRIQUE = **watchlist pré-match** (calendrier des ligues cibles +
  read Elo calculé À L'AVANCE) pour **poser le pari des heures avant le début**, quand le marché soft est ouvert et mou.
- ✅ **BRIQUE CONSTRUITE (2026-06-11) : watchlist pré-match automatisée** (`lol-predictor/src/update/`). 1 commande
  `python -m src.update.daily` enchaîne : (1) `download_data` → rafraîchit le CSV OE depuis le Drive ;
  (2) `build_match_table` → régénère les tables ; (3) `watchlist` → calcule notre proba **Elo toutes-ligues**
  sur les matchs des **prochains jours** (calendrier via l'API **lolesports**) → écrit **`WATCHLIST.md`**
  (table avec notre proba + colonnes `Cote`/`Edge` à remplir). Planifiable 1×/j (Task Scheduler, cf. README).
  - **But** : repérer **à l'avance** un favori que le book va sur-coter → **poser tôt** (résout la contrainte n°1).
  - ⚠️ **Elo-only** (pas de draft, signal partiel) = repère, **pas** la reco finale ; on garde le **draft-edge en live**.
  - Limites connues : le CSV OE a un **quota Drive** partagé (download best-effort, sinon `git pull`) ; lolesports
    couvre les ligues **Riot** (majeures + CBLOL + EMEA Masters + régionales) mais **pas** les events tiers (ex. EWC OQ)
    → ceux-là restent en **saisie manuelle**. Quelques noms à suffixe sponsor (Cloud9 Kia…) tombent en "non couvert".

# Suivi des prédictions réelles (test à l'aveugle)

But : noter, match par match, **où le modèle a vu juste et où il s'est trompé**, sur des
matchs qu'il n'avait **pas** dans ses données d'entraînement (vrai test à l'aveugle).
On saisit la draft + les sides dans le front, on relève la prédiction, puis on compare au
résultat réel.

## Conventions de verdict

- **Vainqueur** : ✅ si le favori du modèle (>50%) gagne.
- **Total kills** : ✅ si l'erreur ≤ 3 kills (on note l'écart exact).
- **Durée** : ✅ si l'erreur ≤ 3 min (on note l'écart exact).
- **First blood / tower / dragon** : ✅ si le camp favorisé (>50% du point de vue de ce camp) correspond au réel.
  - ⚠️ First blood et first dragon sont des marchés très **bruités** (proches du pile-ou-face) : faible valeur prédictive.

---

## Tableau de score global (à jour)

| Marché | Bons | Faux | Taux | Remarque |
|---|---|---|---|---|
| Vainqueur | 3 | 0 | 100% | favori correct à chaque fois |
| Total kills (±3) | 1 | 2 | 33% | sur-estime sur les **stomps** |
| Durée (±3 min) | 1 | 2 | 33% | prédit ~32-33 min quoi qu'il arrive (la moyenne) |
| First blood | 2 | 1 | 67% | marché bruité |
| First tower | 3 | 0 | 100% | seul marché "objectif" fiable |
| First dragon | 1 | 2 | 33% | DK a pris le drake G2+G3 alors que le modèle favorisait BRION |

**Total games testées : 3**

---

## Détail par game

### 2026-06-06 — LCK — Dplus KIA vs Hanjin BRION — Game 1

- **Sides** : DK = 🔵 bleu, BRION = 🔴 rouge
- **Draft DK (bleu)** : K'Sante (top), Jarvan IV (jng), Viktor (mid), Yunara (bot), Milio (sup)
- **Draft BRION (rouge)** : Rumble (top), Naafiri (jng), Orianna (mid), Xayah (bot), Rakan (sup)
- **Priors champions (modèle)** : bleu 49.2% vs rouge 52.5% (Δ = -3.3 pts) → la draft penchait **légèrement** côté BRION.

| Marché | Prédiction (DK bleu) | Réel | Écart | Verdict |
|---|---|---|---|---|
| Vainqueur | DK **75%** / BRION 25% | **DK gagne** | — | ✅ |
| Total kills | **26.8** | **25** (DK 12 + BRION 13) | -1.8 | ✅ |
| Durée | **32.2 min** | **34:23** (~34.4) | -2.2 | ✅ (sous-estimé) |
| First blood | DK 57.4% | **BRION** | — | ❌ |
| First tower | DK 60.7% | **DK** | — | ✅ |
| First dragon | DK 40.4% (→ BRION favori) | **BRION** | — | ✅ |

**Bilan Game 1 : 5/6 marchés corrects** (seul first blood manqué — marché quasi aléatoire).

**Observations :**
- DK gagne **malgré moins de kills** (12 vs 13) → victoire par objectifs/gold. Le modèle a quand même mis DK favori net.
- Le 75% vient surtout de la **force d'équipe + côté bleu**, pas de la draft (qui était légèrement pro-BRION). 👉 Bon signe : le modèle ne recopie pas bêtement la draft.

---

### 2026-06-06 — LCK — Dplus KIA vs Hanjin BRION — Game 2

- **Sides** : DK = 🔵 bleu, BRION = 🔴 rouge
- **Draft DK (bleu)** : Tristana (top), Lee Sin (jng), Ryze (mid), Ziggs (bot), Camille (sup)
- **Draft BRION (rouge)** : Gnar (top), Pantheon (jng), Anivia (mid), Caitlyn (bot), Bard (sup)
- **Priors champions (modèle)** : bleu 47.7% vs rouge 53.7% (Δ = -6.0 pts) → draft encore **pro-BRION** (plus marquée qu'en G1).

| Marché | Prédiction (DK bleu) | Réel | Écart | Verdict |
|---|---|---|---|---|
| Vainqueur | DK **68.4%** / BRION 31.6% | **DK gagne** (stomp) | — | ✅ |
| Total kills | **27.1** | **21** (DK 16 + BRION 5) | +6.1 | ❌ (sur-estimé) |
| Durée | **33.4 min** | **22:02** (~22) | +11.4 | ❌ (sur-estimé) |
| First blood | DK 61.8% | **DK** | — | ✅ |
| First tower | DK 64.6% | **DK** | — | ✅ |
| First dragon | DK 40.2% (→ BRION favori) | **DK (KIA)** | — | ❌ |

**Bilan Game 2 : 3/6 marchés corrects** (ratés = kills + durée à cause du **stomp**, + first dragon).

**Observations :**
- **Stomp en 22 min** : GD@15 énorme pour DK (+2442 mid, +1183 top, +918 jng…). Le modèle avait bien DK favori (68.4%) mais ne pouvait pas anticiper l'**ampleur/brièveté** de l'écrasement.
- Conséquence directe du stomp : **moins de kills** (21) et **game courte** (22 min) → le modèle, qui prédit une game "moyenne" (~27 kills, ~33 min), sur-estime les deux.
- Encore une fois la draft penchait côté BRION (47.7 vs 53.7) mais DK a roulé dessus → la **force d'équipe** domine la draft.

---

### 2026-06-06 — LCK — Dplus KIA vs Hanjin BRION — Game 3

- **Sides** : DK = 🔵 bleu, BRION = 🔴 rouge
- **Draft DK (bleu)** : Sion (top), Wukong (jng), Annie (mid), Mel (bot), Shen (sup)
- **Draft BRION (rouge)** : Jayce (top), Xin Zhao (jng), Taliyah (mid), Ezreal (bot), Neeko (sup)
- **Priors champions (modèle)** : bleu 47.9% vs rouge 49.2% (Δ = -1.3 pts) → draft quasi neutre, léger pro-BRION.

| Marché | Prédiction (DK bleu) | Réel | Écart | Verdict |
|---|---|---|---|---|
| Vainqueur | DK **76%** / BRION 24% | **DK gagne** (stomp) | — | ✅ |
| Total kills | **25.9** | **19** (DK 16 + BRION 3) | +6.9 | ❌ (sur-estimé) |
| Durée | **32.4 min** | **24:07** (~24) | +8.3 | ❌ (sur-estimé) |
| First blood | DK 51.8% | **DK** (1er kill 2:36) | — | ✅ |
| First tower | DK 50.5% | **DK** (T1 14:00) | — | ✅ |
| First dragon | DK 34.1% (→ BRION favori) | **DK (KIA)** (drake ~6:00) | — | ❌ |

**Bilan Game 3 : 3/6 marchés corrects** (vainqueur ✅, FB ✅, tower ✅ ; ratés : kills ❌, durée ❌, dragon ❌).

**Observations :**
- **3e stomp** (24 min, 16-3). DK favori net (76%) → correct, mais kills et durée **encore sur-estimés** : même cause (game écrasée et courte).
- **First dragon raté pour la 2e fois de suite** : le modèle favorise BRION mais c'est **DK** qui prend le drake (G2 et G3). À creuser : un gros favori prend souvent le 1er drake → le marché dragon devrait peut-être suivre la proba de victoire.

---

## Patterns / leçons (à enrichir)

- ✅ **Point fort n°1 — Vainqueur : 3/3.** Favori correct à chaque fois (75% / 68% / 76%), y compris quand la draft penchait côté adverse → la **force d'équipe + le side** dominent.
- ✅ **Point fort n°2 — First tower : 3/3.**
- 🟡 **Durée** : le modèle prédit ~**32-33 min quoi qu'il arrive** (la moyenne). Il ne capte pas la **variance** : G1 game longue (34 min) → sous-estimé ; G2/G3 stomps (22-24 min) → sur-estimés.
- 🟡 **Total kills** : bon sur une game "normale" (G1), **sur-estimé sur les stomps** (G2, G3). Même cause : le modèle ignore si la game sera serrée ou un écrasement.
- ❌ **First dragon : 1/3.** Sur les 2 stomps DK, c'est **DK** qui a pris le 1er drake alors que le modèle favorisait BRION. Un gros favori prend souvent le 1er objectif.
- ⚠️ **Faible valeur** : first blood (bruité, ~pile-ou-face).
- 💡 **Piste forte** : kills, durée ET first dragon dépendent du **scénario** (stomp vs game serrée). Idée → **conditionner ces 3 marchés sur la proba de victoire** : gros favori (ex. 76%) ⇒ souvent stomp ⇒ game plus courte, moins de kills, et le favori prend le 1er drake. Ça corrigerait d'un coup les 3 marchés les plus faibles.

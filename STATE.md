# STATE.md — Projet DZNE / Mitigation d'erreurs quantiques

> **Status (updated 2026-07-23):** The static-model phase is **complete** — full noise-model
> family up to the composite hardware model, the DZNE-vs-PEC benchmark, and readout/TREX. Current
> work is the **drift track** (non-stationary noise via Ornstein-Uhlenbeck). See
> [README.md](README.md) for the project overview. The dated notes below are a historical lab
> notebook (in French) and describe an earlier phase.

**Date :** 12 mai 2026
**Personne :** Thomas — stage en informatique quantique, encadré par Dr Tsikas.
**Objectif global du stage (12 semaines, 3 phases) :**
1. **Sem. 0–1** — Lecture papiers fondateurs + proposal. ✅
2. **Sem. 2–5/6** — Implémentation et comparaison ZNE vs PEC sur cas-jouets. ◀ **en cours**
3. **Sem. 6–12** — ML-QEM, target **hardware IBM QPU réel**.

---

## Sources de référence

- **TBG 2017** (Temme, Bravyi, Gambetta) — ZNE + PEC, scaling temporel analogique.
- **Giurgica-Tiron et al. 2021** (DZNE) — gate-level folding, parameter noise scaling, extrapolation comme problème d'inférence, algos adaptatifs.
- Papier ML-QEM (phase 3, pas encore attaqué).
- Livre *Quantum Computing: An Applied Approach* — chap. 1–4 lus. Chap. 6 (Dev Libraries) prioritaire pour l'implémentation. Chap. 5 (hardware) pour phase 3.

---

## Théorie maîtrisée

- QEM vs QEC, positionnement NISQ.
- ZNE classique (TBG) : Lindblad → image d'interaction → série de Dyson → structure polynomiale en λ → Richardson via Lagrange.
- Canaux canoniques (amplitude damping, déphasage, dépolarisation), distinctions cohérent/incohérent, markovien/non-markovien, unital/non-unital.
- Condition de validité **NTλ ≪ 1** avec N = qubits, T = profondeur (pas interchangeables).
- Le facteur 1/k! des bornes rigoureuses est absorbé dans le grand-O des énoncés heuristiques.
- DZNE : folding gate-level (global et par couche), reformulation inférentielle (biais/variance/MSE), ansatz polynomial/exponentiel/poly-exp.

---

## Implémentation actuelle

### Stack
- **Qiskit + qiskit-aer** (`NoiseModel`, `depolarizing_error`, `amplitude_damping_error`)
- **Mitiq** validé comme lib de référence (cf. décisions plus bas)
- Cirq en lecture seule (IR interne de Mitiq)

### Notebook custom (`dzne_algorithms_2_3.ipynb`)
- Bell + dépolarisant 1% → OK
- Benchmark RB : `random_clifford_circuit(2, 10, seed=42)` composé avec son inverse, observable P(|00⟩), idéal = 1
- ZNE folding λ ∈ {1.0, 1.5, 2.0}, fit poly degré 2 → OK
- Version statistique sur 20 seeds (moyenne, std, distribution R_u/R_m) → OK
- **Algorithme 2** (poly-2 adaptatif) et **Algorithme 3** (exp adaptatif, c estimé itérativement) implémentés depuis le papier
- Test OpenEvolve sur toy 1D : gain ~16% sur oracle exp (positivité + fallbacks). Pas branché sur Qiskit réel.

---

## Décisions techniques actées

### Tooling
- **Mitiq = bibliothèque de référence pour la suite.** Validation : couvre ZNE/PEC/CDR/DDD/readout, frontend-agnostique (Qiskit/Cirq/Braket/Pennylane), `AdaExpFactory` = implémentation officielle de l'Algorithme 3, maintenue par Unitary Foundation, citée dans la littérature QEM post-2020 (y compris papier ML-QEM phase 3).
- **Alternatives évaluées et rejetées** :
  - *Qiskit Runtime `resilience_level`* : boîte noire, pas de contrôle sur scale factors / ansatz / folding. À garder pour phase 3 hardware comme baseline de production.
  - *qiskit-research* : code de recherche, pas une lib.
  - *PennyLane `mitigate_with_zne`* : wrapper qui appelle Mitiq sous le capot. Pas une alternative.
  - *Cirq natif* : trop incomplet (Mitiq utilise Cirq comme IR précisément parce qu'il ne suffit pas seul).
- **Phase 3 hardware** : comparer **Mitiq + executor Qiskit Runtime** vs **Runtime `resilience_level=2`** pour positionner par rapport au pipeline IBM production.

### Méthodologie
- Focus Qiskit principal.
- Seed systématique pour reproductibilité.
- Moyenne sur **≥ 20 circuits aléatoires** (un seul run ne dit rien).
- **Conserver le notebook custom** comme référence pédagogique — le notebook Mitiq est un complément (validation croisée + accès à PEC), pas un remplacement.

---

## En cours : notebook Mitiq DZNE + PEC

**Fichier cible :** nouveau notebook (à créer), miroir du custom mais via Mitiq.

### Structure planifiée
```
1. Setup (imports, noise model, backend Aer, executor commun)
2. Benchmark (random Clifford + inverse, observable P(|00>), N seeds)
3. DZNE avec Mitiq
   3.1 Méthodes de folding (illustration sur 1 circuit)
   3.2 Factories non-adaptatives (Richardson, Poly2, Exp)
   3.3 Factory adaptative (AdaExp ≈ Algorithme 3 du papier)
   3.4 Statistiques sur N seeds
4. PEC avec Mitiq
   4.1 OperationRepresentation cohérentes avec le NoiseModel Aer
   4.2 execute_with_pec sur 1 circuit
   4.3 Statistiques sur N seeds + overhead γ
5. Comparaison ZNE vs PEC (bias, variance, MSE, coût en shots)
6. Discussion
```

### API Mitiq — points clés

**Concept central : l'executor**
```
executor(circuit) -> float                 # ou
executor(circuits: List) -> List[float]    # batché, recommandé
```
Bruit / backend / observable / shots encapsulés **une seule fois**. Toute comparaison ZNE/PEC change juste la fonction de mitigation.

**`mitiq.zne.scaling`**
- `fold_global` — `U → U(U†U)ⁿ`, équivalent du folding global custom
- `fold_all` — folding par couche en place
- `fold_gates_at_random`, `fold_gates_from_left/right` — variantes
- Pour λ non entier : Mitiq fait du **folding partiel** (pas un arrondi). Vérifier la cohérence avec le code custom pour λ=1.5.

**`mitiq.zne.inference` — mapping avec le papier**

| Mitiq | Modèle | Adaptatif | Correspond à |
|---|---|---|---|
| `RichardsonFactory` | Lagrange → λ=0 | Non | TBG original |
| `PolyFactory(order=k)` | Poly degré k | Non | DZNE non-adaptatif poly |
| `ExpFactory(asymptote=a)` | `a + b·exp(-cλ)` | Non | DZNE non-adaptatif exp |
| `PolyExpFactory` | `a + exp(P_k(λ))` | Non | Ansatz poly-exp |
| `AdaExpFactory(steps, scale_factor, asymptote)` | Exp | **Oui** | **Algorithme 3** |
| — | Poly-2 | Oui | **Algorithme 2 — pas d'équivalent direct** |

⚠️ Le notebook custom reste **seul à implémenter l'Algorithme 2 adaptatif** → à mentionner dans le rapport comme contribution propre.

**Point d'entrée**
```
mitiq.zne.execute_with_zne(circuit, executor, factory, scale_noise) -> float
```

**PEC (sections 4–5 du notebook)**
- `mitiq.pec.OperationRepresentation` : décomposition quasi-probabiliste d'une opération idéale.
- `mitiq.pec.represent_operation_with_local_depolarizing_noise(ideal_op, noise_level)` (+ variantes amplitude damping, dephasing).
- `mitiq.pec.execute_with_pec(circuit, executor, representations, num_samples)`.
- ⚠️ **Overhead γ² = (Σ|c_i|)²** : croît exponentiellement en profondeur. Sur RB depth=10 + dépolarisant 1% c'est gérable, mais barres d'erreur PEC visiblement plus larges que ZNE à shots égaux. **C'est le point central de la comparaison.**

---

## Plan d'exécution immédiat

1. Créer le notebook Mitiq, sections 1–3 (DZNE seul).
2. **Validation croisée** sur les mêmes 20 seeds :
   - Attendu : `mine_alg3 ≈ mitiq_adaexp` (à variance MC près).
   - Attendu : `mine_alg2 < mitiq_poly2` expliqué par le mauvais modèle (poly-2 sur bruit exponentiel).
   - Attendu : `mitiq_richardson` et `mitiq_exp` (non-adaptatifs) ≈ `mitiq_adaexp` sur ce benchmark (cf. Fig. 7 du papier — adaptation ne brille pas en dépolarisant pur).
3. Métriques à logger par méthode : **bias, variance, MSE, R_u/R_m, budget shots**.
4. Sections 4–5 (PEC + comparaison) après validation DZNE.

---

## Pistes parallèles (à reprendre après notebook Mitiq)

- Alg. 2 avec modèle exponentiel — isoler bénéfice pur d'adaptation.
- Varier init de `c` dans Alg. 3 (e.g. c₀ = 5) — voir si l'adaptation compense.
- Brancher amplitude damping (non strictement exp) — tester famille poly-exponentielle.
- Brancher OpenEvolve sur problème multi-dim (folding × scale factors × ansatz × pondération), pas le toy 1D.

---

## Pièges connus (à ne pas refaire)

- Confusion N ↔ T dans NTλ ≪ 1.
- Indépendance des polynômes de Lagrange / fonction interpolée.
- Initial guess fragile en fit exponentiel : médiane + pré-fit log-linéaire pour `c`.
- Magic number `c = 1.0` hardcodé : paramétrer selon profondeur réelle.
- Un seul circuit aléatoire ≠ benchmark — toujours moyenner ≥ 20 seeds.
- OpenSolve ≠ OpenEvolve (codelion) — Tsikas visait OpenEvolve.
- **Mitiq convertit Qiskit → Cirq en interne** : source potentielle de surprise si on inspecte les circuits foldés.
- **PEC** : les `OperationRepresentation` doivent encoder **exactement** le `NoiseModel` Aer, sinon on mitige autre chose que ce qui se passe.
- **Mitiq folding partiel pour λ non entier** ≠ arrondi à l'entier impair. Vérifier l'alignement avec l'implémentation custom.

---

## Style de travail préféré

- Pédagogique, incrémental, dérivations depuis les premiers principes quand la théorie est nouvelle.
- Schémas SVG bienvenus pour concepts visuels.
- Critique constructive attendue (Thomas catch les erreurs).
- **Réponses concises, code-first quand la théorie est acquise.** Ce STATE.md fait foi — ne pas réexpliquer ce qui s'y trouve.

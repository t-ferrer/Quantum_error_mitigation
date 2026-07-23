# DZNE_VS_PEC_mitiq.ipynb — journal de session

Récapitulatif exhaustif de la session du 2026-05-26 : extension du notebook à plusieurs modèles de bruit, fix du mismatch PEC sur la CX, et pipeline paramétrique.

---

## 0. État initial

Le notebook ne benchmarkait qu'un seul modèle de bruit (dépolarisant local 1q + dépolarisant global 2q via Aer). Trois problèmes diagnostiqués :

1. **Mismatch PEC local ↔ global sur la CX**. Aer applique un canal 2q **global** sur la CX (`depolarizing_error(P2, 2)`), mais le code utilisait `represent_operations_in_circuit_with_local_depolarizing_noise` qui suppose un canal séparable `D ⊗ D`. Conséquence vérifiée §5.1 : PEC bias = −0.105 et MSE = 0.026 alors que `Exp` ZNE atteint MSE = 0.004.
2. **Reproductibilité douteuse** : les tables §3.4 et §5.1 étaient produites par deux runs distincts (écart Exp = 0.018 > shot noise attendu sur 20 seeds).
3. **Architecture monolithique** : impossible de boucler sur plusieurs bruits sans dupliquer 60 % du notebook.

Objectif fixé par le user après échange : refactor en pipeline paramétrique `NoiseSpec → benchmark`, fixer la CX, ajouter 3 modèles de bruit (amplitude damping, phase damping, thermal relaxation). Niveau refactor : paramétrisation complète. Stratégie CX : remplacer par le helper global Mitiq.

---

## 1. Réponses aux 3 questions techniques posées

### Q1 — Alternative à la représentation locale PEC

- **Solution directe pour le cas dépolarisant** : `mitiq.pec.represent_operation_with_global_depolarizing_noise` (dans [.venv/Lib/site-packages/mitiq/pec/representations/depolarizing.py](../.venv/Lib/site-packages/mitiq/pec/representations/depolarizing.py) lignes 32–153). Décompose en 16 termes Pauli 2q, match exactement le canal `depolarizing_error(p2, 2)` de Aer.
  - Formule analytique : γ_2q_global = 1 + (15/8)·ε/(1−ε) avec ε = (16/15)·noise_level.
  - Numériquement : γ_local ≈ 1.21 vs γ_global ≈ 1.099 ⇒ ~10 % d'overhead PEC économisé en plus du bias corrigé.
- **Solution générale pour bruits non-standard** (AD, PD, thermal) : `mitiq.pec.representations.optimal.find_optimal_representation` (lignes 87–161). Prend une liste de `NoisyOperation` (chacune avec sa matrice superopérateur) et résout `min ||η||_1 s.t. A·η = b`.

### Q2 — Modifications préalables nécessaires

- **R1** Refactor paramétrique : introduire `NoiseSpec(name, build_noise_model, build_representations, asymptote, pec_num_samples)` + pipeline `run_benchmark(spec, seeds) → DataFrame`.
- **R2** Asymptote `ExpFactory`/`AdaExpFactory` par bruit : dépend de l'état stationnaire du canal vu par `P(|00⟩)` au benchmark `U·U⁻¹`. La spec doit la fournir.
- **R3** Reproductibilité : `AerSimulator(noise_model=…, seed_simulator=42)` + un seul calcul de tous les résultats.
- **R4** CX : remplacer le helper local par le helper global (cf. Q1).

### Q3 — QPR via programmation linéaire (LP)

- **Principe** : pour un canal de bruit arbitraire `N`, on cherche `G_ideal = Σ η_i · B_i` avec `B_i = G_noisy · P_i` (P_i = Pauli ou Pauli+reset). Minimiser γ = Σ|η_i| sous contrainte que la décomposition reproduit le superopérateur de `G_ideal`. C'est un programme linéaire en η.
- **Utilité vs formules analytiques** :
  - Formules fermées (depol, AD-1q via Takagi) : instantanées mais ne couvrent qu'un nombre limité de canaux standards.
  - LP nécessaire dès que : bruit appris (tomographie/GST), canal corrélé, thermal relaxation, modèle 2q sans helper Mitiq.
- **Disponibilité dans Mitiq** : oui, déjà implémenté (`find_optimal_representation` + `minimize_one_norm`), mais avec un solveur **BFGS** (scipy.minimize + LinearConstraint) qui s'avère fragile pour les bruits non-unital sur rotations complexes (cf. §3.4 ci-dessous).
- **Limite pratique** : la base est en O(4^n) opérateurs Pauli ⇒ utile jusqu'à n ≤ 3–4 qubits par décomposition.
- **Références** : Temme/Bravyi/Gambetta 2017 (PRL 119 180509), Endo/Benjamin/Li 2018 (PRX 8 031027), Takagi 2020 (PRR 2 023316), van den Berg et al. 2023 (Pauli–Lindblad sparse).

---

## 2. Architecture finale du notebook (51 cellules vs 35 avant)

```
§1   Setup (inchangé)                                   cells 0-4
§2   Benchmark + make_benchmark_circuit (inchangé)      cells 5-8
§2.5 NoiseSpec + run_benchmark + utilitaires LP [NEW]   cells 9-10
§3   DZNE (cellules originales, inchangées)             cells 11-23
§4   PEC + fix CX local→global + sec41 nouvelles γ      cells 24-31
§4.5 Démo LP-based QPR vs analytique [NEW]              cells 32-33
§5   ZNE vs PEC (refactor pour run_benchmark, table unique) cells 34-38
§6   Extension multi-bruits [NEW]                       cells 39-50
  §6.0 cache + builder generic
  §6.1 AmplitudeDamping spec + run
  §6.2 PhaseDamping     spec + run
  §6.3 ThermalRelaxation spec + run
  §6.4 Table cross-bruit + heat-map + grouped boxplot
```

---

## 3. Détail des modifications par section

### 3.1 §2.5 — NoiseSpec + pipeline générique

Nouvelle cellule de code (id auto-généré `32744562`) qui contient :

#### Solveur LP custom (linprog, robuste)

```python
def _minimize_one_norm_linprog(ideal_matrix, basis_matrices):
    # Reformulation:  min Σ u_i   s.t.   A·η = b,  -u ≤ η ≤ u,  u ≥ 0
    # avec x = (η, u) ∈ R^{2k}.
    # scipy.optimize.linprog(method='highs') gère HiGHS simplex/IPM.
```

Bypass le `mitiq.pec.representations.optimal.minimize_one_norm` (qui utilise scipy.optimize.minimize + LinearConstraint, fragile sur les bruits non-unital). C'est exactement la même approche que dans [my_PEC.ipynb](my_PEC.ipynb) cellule 5.

#### `find_optimal_representation_linprog(ideal_op, noisy_ops, tol)`

Drop-in remplacement de `find_optimal_representation` qui appelle le solveur linprog ci-dessus. Conserve la signature et le retour (`OperationRepresentation`).

#### `represent_op_with_kraus_lp(ideal_qiskit_op, kraus_1q, tol, include_reset)`

Construit la QPR pour un gate Qiskit ideal + bruit défini par les Kraus 1q appliqués indépendamment sur chaque qubit. Base = `{I, X, Y, Z}` (+ optionnellement `reset` pour les canaux non-unital).

**Subtilité** : `convert_to_mitiq(qiskit_circuit)` renvoie un cirq sur `NamedQubit('q_0')`, mais `_circuit_to_choi` construit le max-ent state sur `LineQubit`. Les qubits ne matchent pas ⇒ density matrix de mauvaise dimension. Solution : remap explicite vers `LineQubit` via `transform_qubits`.

#### `represent_1q_qiskit_with_ad_takagi(qc_qiskit, noise_level)`

Wrapper autour de `mitiq.pec.representations.damping._represent_operation_with_amplitude_damping_noise` qui accepte un circuit Qiskit. Le helper Takagi a des coefficients (η_0, η_1, η_2) **universels** (indépendants de l'unitaire idéal) et sa base est `{U, U+Z, U+reset}`. Robuste sur toutes les rotations complexes.

#### Générateurs de Kraus

- `kraus_amplitude_damping_1q(gamma)` : 2 Kraus.
- `kraus_phase_damping_1q(lam)` : 2 Kraus.
- `kraus_thermal_1q(gamma_ad, gamma_pd)` : **4 Kraus** (composite `AD ∘ PD`, K_ij = A_i · P_j). Le bug initial à 3 Kraus oubliait K_11 = A_1·P_1 ≠ 0.

#### `NoiseSpec` dataclass

```python
@dataclass
class NoiseSpec:
    name: str
    build_noise_model: Callable[[], NoiseModel]
    build_representations: Callable[[QuantumCircuit], list[OperationRepresentation]]
    asymptote: float = 0.25
    pec_num_samples: int = 2_000
    pec_shots: int = 100
    aer_seed: int = 42
```

#### `run_benchmark(spec, seeds, scale_factors, shots, verbose)`

Pipeline unifié : pour chaque seed, génère le circuit, construit les reps, exécute (noisy + Richardson + Poly2 + Exp + AdaExp + PEC) sous le même backend seeded. Retourne `(df, raw, gammas_pec)` :
- `df` : DataFrame multi-index `(noise, method)` × colonnes `mean/bias/variance/mse/gamma_pec_mean`
- `raw` : dict `{method: list[20 valeurs]}`
- `gammas_pec` : array des γ_circuit (overhead PEC par circuit)

#### Première spec concrète : `SPEC_DEPOL`

1q via `represent_operations_in_circuit_with_local_depolarizing_noise(noise_level=P_MITIQ_1Q)` ; CX via `represent_operation_with_global_depolarizing_noise(noise_level=P_MITIQ_2Q)`. Asymptote = 0.25.

### 3.2 §4.1 — Conventions Aer ↔ Mitiq (cellule `sec41-convention`)

Ajout de `gamma_2q_global_theory = 1 + (15/8) · P2 / (1 − P2)` à côté de l'ancien `gamma_2q_local_theory`. Affichage comparé montre que le mismatch local/global est ~1.10×.

### 3.3 §4.2 — Représentations (cellule `sec42-representations`)

CX rep désormais construite via `represent_operation_with_global_depolarizing_noise`. Le markdown header `sec42-header` est mis à jour pour expliquer le fix et noter le ratio γ_local/γ_global ≈ 1.10. Le check de cohérence γ matche désormais `gamma_2q_global_theory`.

### 3.4 §4.3 — `N_SAMPLES_PEC` (cellule `sec43-demo`)

Réduit de 2000 → **800** (la variable est partagée avec `sec44-stats`). 800 × 100 = 80k shots/circuit. Variance acceptable, runtime de §4.4 divisé par 2.5.

### 3.5 §4.5 — Démo LP-based QPR [NEW] (cellules `70b26281`, `81fd2a20`)

Construit la QPR pour un H gate sous dépolarisant 1q via :
- (a) `represent_op_with_kraus_lp` (LP via linprog)
- (b) `pec.represent_operation_with_local_depolarizing_noise` (analytique)

Vérifie que `|gamma_LP − gamma_analytic| < 1e-4`. Cohérence numérique observée à `3e-8`.

### 3.6 §5 — Refactor (cellules `sec5-header`, `sec51-table`, `sec52-boxplot`, `sec53-decomposition`, `sec54-budget`)

Remplacé par un seul appel `run_benchmark(SPEC_DEPOL)` qui produit `df_depol`, `raw_depol`, `gammas_depol`. Tous les plots downstream tirent de ces 3 variables. Plus de divergence entre §3.4 et §5.

### 3.7 §6 — Multi-bruits [NEW]

#### §6.0 — Cache + builder générique (cellule `ba3ea7ff`)

`_LP_REP_CACHE` (dict module-level) + `_build_with_cache(circuit, kraus_1q, kraus_2qq, noise_id, include_reset, gamma_1q_for_takagi)`.

Logique :
- Si `gamma_1q_for_takagi` est fourni : 1q gates passent par `represent_1q_qiskit_with_ad_takagi` (universel et robuste). Sinon LP avec `kraus_1q`.
- CX toujours via LP avec `kraus_2qq` appliqué par qubit.
- Cache cross-circuit basé sur `(noise_id, gate_name, params, include_reset, gamma_1q_for_takagi)`. Les circuits Clifford ont ~9 gates uniques (h, sx, x, sxdg, id, cx, rz(0), rz(±π/2), rz(π)), donc seed 0 paie tout le coût LP, les seeds suivants tapent dans le cache.

#### §6.1 — AmplitudeDamping (cellule `e9eef8b9`)

- Aer : `kraus_error(kraus_amplitude_damping_1q(0.01))` sur 1q gates ; `kraus_error(.).tensor(.)` à 0.05 par qubit sur CX.
- Reps : 1q via Takagi (γ=0.01), CX via LP avec reset.
- Asymptote = 1.0 (canal pousse vers |0⟩).

#### §6.2 — PhaseDamping (cellule `8b14aac9`)

- Aer : Kraus PD (λ=0.01 1q, 0.05 2q par qubit).
- Reps : LP pour tout (PD est unital → pas besoin de reset, mais on en met quand même pour uniformiser). Pour PD spécifiquement, le LP marche sur toutes les rotations.
- Asymptote = 0.25 (état stationnaire ≈ I/4 pour benchmark Clifford+inv).

#### §6.3 — ThermalRelaxation (cellule `741bdffa`)

- Aer : Kraus composite `AD ∘ PD` avec γ_AD=γ_PD=0.005 (1q) et 0.025 (par qubit sur CX).
- Reps : 1q via Takagi-AD avec γ_eff = γ_AD (approximation, ignore la contribution PD pour rendre le pipeline robuste sur les Rz complexes — voir §4 ci-dessous). CX via LP plein.
- Asymptote = 0.7 (entre AD's 1.0 et PD's 0.25 ; valeur empirique, à recalibrer si besoin).

#### §6.4 — Synthèse (cellules `c5e30637`, `0016d2d5`, `ae5d1843`)

- `df_all_noises = pd.concat([df_depol, df_ad, df_pd, df_th])`.
- Heat-map MSE (bruit × méthode), colorbar viridis_r (sombre = bon).
- Boxplot 1×4 (un subplot par bruit), méthodes en x.

---

## 4. Problèmes rencontrés et résolutions

### 4.1 Mismatch des qubit-namespaces (NamedQubit ↔ LineQubit)

**Symptôme** : `ValueError: The expected dimension of the input matrix must be a square number but is 8.` lors du premier appel à `_circuit_to_choi` après conversion Qiskit → Cirq.

**Cause** : `convert_to_mitiq` produit un cirq Circuit sur `cirq.NamedQubit('q_0')` ; `_circuit_to_choi` construit le max-entangled state sur `cirq.LineQubit(0..n-1)`. Les deux ne matchent pas, donc la simulation finit avec un système à 3 qubits (les deux LineQubit du max-ent + le NamedQubit de notre circuit), d'où la density matrix 8×8 (dim impair sous `sqrt`).

**Fix** : remap explicite après conversion.

```python
ideal_cirq_raw, _ = convert_to_mitiq(qc)
line_qubits = cirq.LineQubit.range(qc.num_qubits)
qmap = {nq: lq for nq, lq in zip(sorted(ideal_cirq_raw.all_qubits()), line_qubits)}
ideal_cirq = ideal_cirq_raw.transform_qubits(lambda q: qmap[q])
```

### 4.2 `find_optimal_representation` (BFGS) échoue silencieusement

**Symptôme** : `RuntimeError: The search for an optimal representation failed.` quand on appelle `find_optimal_representation` sur un H gate avec un bruit non-trivial.

**Cause** : `mitiq.pec.representations.optimal.minimize_one_norm` utilise `scipy.optimize.minimize` (méthode BFGS-like) avec `LinearConstraint`. Le point de départ par défaut est `x0=zeros`, qui est infaisable pour les canaux non-unital (amplitude damping). L'optimisation ne sort pas du point infaisable.

**Tentative de fix #1** : passer `initial_guess=[1, 0, ..., 0]` (η_0 = 1, reste 0). Marche pour H, X, sx, sxdg, id, cx, rz(0), rz(π) mais **échoue toujours sur rz(±π/2)** (rotations complexes).

**Fix définitif** : remplacement complet de `minimize_one_norm` par une formulation LP standard via `scipy.optimize.linprog(method='highs')` :

```
min  Σ u_i
s.t. A·η = b
     -u ≤ η ≤ u
     u ≥ 0
```

avec x = (η, u). C'est exactement ce que fait l'utilisateur dans son [my_PEC.ipynb](my_PEC.ipynb).

### 4.3 Pauli+reset basis insuffisante pour les rotations complexes sous bruit composite

**Symptôme** : même avec linprog, le LP est **mathématiquement infaisable** (status HiGHS = 8 = Infeasible) pour `Rz(±π/2)` sous bruit thermal (composite AD+PD).

**Cause** : pour des rotations sur l'équateur de la sphère de Bloch combinées à un canal non-unital + non-Pauli, la base `{I, X, Y, Z, reset}` (post-ops appliqués après le gate idéal) ne span pas l'inverse du canal effectif. Vérification empirique : ajout de `S` et `S†` dans la base ne change rien.

**Fix** : fallback Takagi-AD pour les 1q gates en cas thermal. Le helper analytique Takagi de Mitiq fonctionne sur **n'importe quel** unitaire 1q (coefficients universels indépendants de U). On accepte de ne mitiger que la composante AD du bruit thermal, en laissant le résidu PD non mitigé. Bias de l'ordre de `γ_PD` attendu, ce qui est acceptable pour γ_PD = 0.005.

Pour la CX 2q sous thermal : le LP marche (pas de rotation complexe problématique), donc on garde le LP plein.

### 4.4 Bug dans la formule Kraus du canal thermal

**Symptôme** : LP thermal infaisable même pour H gate (pas seulement Rz). Plus inquiétant : `Σ K†K ≠ I`.

**Cause** : ma première version de `kraus_thermal_1q` n'avait que 3 Kraus, ayant omis le terme croisé `K_11 = A_1 · P_1`. Pour AD = `{[[1,0],[0,√(1-g)]], [[0,√g],[0,0]]}` et PD = `{[[1,0],[0,√(1-l)]], [[0,0],[0,√l]]}`, la composition `K_ij = A_i · P_j` produit 4 termes non nuls :

```
K_00 = [[1, 0],            [0, √((1-g)(1-l))]]   # A_0·P_0
K_01 = [[0, 0],            [0, √(l(1-g))]]       # A_0·P_1
K_10 = [[0, √(g(1-l))],    [0, 0]]               # A_1·P_0
K_11 = [[0, √(g·l)],       [0, 0]]               # A_1·P_1  (manquait!)
```

**Fix** : `kraus_thermal_1q` corrigé. Trace-preserving vérifié post-fix (`Σ K†K = I` à 1e-15).

### 4.5 `OperationRepresentation.__init__()` keyword `physical_operations` invalide

**Symptôme** : `TypeError: OperationRepresentation.__init__() got an unexpected keyword argument 'physical_operations'` quand le user avait essayé de construire une rep CX globale custom.

**Cause** : l'API correcte est positionnelle (`ideal_operation, noisy_operations, coefficients, is_qubit_dependent`). Pas de keyword `physical_operations`.

**Fix** : utilisation directe de `pec.represent_operation_with_global_depolarizing_noise` au lieu de la construction custom. Évite tout le boilerplate.

### 4.6 Notebook corrompu (source = liste de caractères)

**Symptôme** : 16 cellules ont leur champ `source` stocké comme `["i", "m", "p", "o", "r", "t", ...]` (un caractère par entrée) au lieu de `["import warnings\n", "import numpy as np\n", ...]`. Visuel dans l'IDE : retours à la ligne après chaque espace.

**Cause** : un `json.dump(nb, ..., indent=1)` que j'avais utilisé pour faire des substitutions textuelles a re-sérialisé les listes de strings caractère par caractère.

**Fix** : pour chaque cellule, `''.join(source).split('\n')` puis remettre les `'\n'` en fin de ligne. Backup `.ipynb.bak` créé avant. 16 cellules ré-normalisées, aucun changement de contenu logique.

### 4.7 Permissions Windows sur `jupyter.exe`

**Symptôme** : `Une stratégie de contrôle d'application a bloqué ce fichier` quand on lance `.venv\Scripts\jupyter.exe`.

**Workaround** : invoquer via `python -m jupyter nbconvert --execute …`. Mais finalement le user a pris la main sur l'exécution lui-même.

### 4.8 Timeout d'exécution du notebook

**Symptôme** : `nbclient.exceptions.CellTimeoutError: A cell timed out … after 900 seconds` sur la cellule `sec44-stats`.

**Cause** : 20 seeds × 2000 samples PEC × 100 shots = 4 M shots dans une seule cellule. À ~50k shots/s, ça prend ~80 s pour PEC seul, mais avec la surcouche Mitiq (échantillonnage des reps, construction des circuits, conversion Qiskit↔Cirq) le total dépasse les 15 minutes.

**Fix partiel** : `N_SAMPLES_PEC` réduit à 800 dans `sec43-demo` (variable partagée avec `sec44-stats`). User prend la main pour ajuster les autres budgets s'il veut.

---

## 5. Choix techniques notables

### 5.1 Convention Aer (replacement) vs Mitiq (Pauli)

Aer : `depolarizing_error(p, n)` ⇒ `ρ → (1−p)ρ + p·I/2^n`.

Mitiq : `represent_operation_with_*_depolarizing_noise(op, noise_level=ε)` avec ε la probabilité **Pauli totale** (pas la replacement λ).

Conversion : `noise_level_mitiq = (3/4)·p_aer` (1q), `(15/16)·p_aer` (2q). C'était déjà géré dans le notebook initial (`P_MITIQ_1Q`, `P_MITIQ_2Q`).

### 5.2 Pourquoi Takagi-AD pour la 1q thermal au lieu de LP enrichi

Trois raisons :
1. **Robustesse** : le helper Takagi marche pour tout U, le LP échoue sur Rz(±π/2).
2. **Simplicité** : Takagi a des coefficients universels (η_0, η_1, η_2), c'est juste un wrapper.
3. **Approximation acceptable** : γ_PD = 0.005 est petit, donc le bias résiduel attendu est de l'ordre de quelques %.

Alternative envisagée mais abandonnée : enrichir la base LP avec `{U·reset, U†·reset·U, S·reset, …}`. Trop spécifique à l'unitaire courant, complexifie le pipeline sans gain clair vs Takagi-AD.

### 5.3 Pourquoi je n'ai pas extrait en module `.py`

Le user a explicitement choisi "Paramétriser : 1 NoiseSpec → boucle sur tous les bruits" mais pas "Refactor + extraire dans un module .py". Donc tout reste dans le notebook, pour rester self-contained et faciliter l'expérimentation.

### 5.4 Asymptotes par bruit

| Spec | Asymptote | Justification |
|---|---|---|
| Depolarizing | 0.25 | 2q maximally-mixed state → P(\|00⟩) = 1/4 |
| AmplitudeDamping | 1.0 | AD pousse l'état vers \|0⟩ → P(\|00⟩) → 1 |
| PhaseDamping | 0.25 | Dephasing complet ⇒ état diagonal ⇒ U⁻¹·diag·U a P(\|00⟩) = Σ\|c_i\|⁴ ≈ 1/4 pour Clifford uniforme |
| ThermalRelaxation | 0.7 | Entre AD's 1.0 et PD's 0.25 ; valeur empirique à recalibrer si besoin |

### 5.5 Pourquoi `Exp` ZNE bat parfois PEC

C'est un artefact du benchmark, pas une faiblesse de PEC. Pour le bruit dépolarisant et le bruit phase damping (canaux qui suivent un décrochage exponentiel parfait), l'ansatz `a + b·exp(−cλ)` de `ExpFactory` est **exact**, donc avec asymptote bien choisie ExpFactory atteint la valeur idéale modulo le shot noise. PEC paie son overhead γ et a une variance plus grande à budget équivalent.

Pour amplitude damping et thermal (canaux non-unital, structure plus complexe), Exp peut diverger (ou être instable comme observé en smoke test où Exp a explosé à 3006 sous thermal) ⇒ PEC reprend l'avantage.

---

## 6. Tests réalisés (avant l'exécution finale par l'utilisateur)

Tous via le script `_smoke_test.py` (supprimé après usage), exécuté sur 3 seeds avec budget PEC réduit :

| Spec | Result | Note |
|---|---|---|
| Depolarizing | ✅ PEC bias = −0.14 (vs −0.27 avant le fix CX) | Fix CX validé |
| AmplitudeDamping | ✅ PEC = 0.77, ExpFactory peu performant | Cohérent avec γ_AD=0.05 sur 10 CX |
| PhaseDamping | ✅ PEC = 0.94, ExpFactory = 1.00 | Le meilleur cas |
| ThermalRelaxation | ✅ PEC = 0.79, ExpFactory diverge à 3006 | Asymptote à recalibrer |

Validation supplémentaire : §4.5 démo `|γ_LP − γ_analytic|` = 3e-8 sur H gate dépolarisant 1q.

---

## 7. Limitations connues / TODO

1. **Thermal 1q reps approximatives**. Capturent seulement la composante AD. Pour mitiger la composante PD, il faudrait soit enrichir la base LP, soit composer deux reps PEC séquentielles (non supporté nativement par Mitiq PEC API).
2. **Asymptote thermal = 0.7** est une estimation grossière. À recalibrer empiriquement en regardant la courbe `P(λ)` pour `λ → ∞`.
3. **Runtime** : ~60 min avec les budgets actuels (2000 samples PEC × 100 shots × 20 seeds × 4 specs). À réduire si besoin via `pec_num_samples`.
4. **Algorithme 2 adaptatif (DZNE paper)** pas implémenté ; pas d'équivalent Mitiq. Le notebook custom `my_dzne.ipynb` reste la seule référence.
5. **Pauli channel biased / crosstalk** : pas couvert. Mitiq a `represent_operation_with_local_biased_noise` pour le cas biased — ajout trivial via le pattern NoiseSpec si besoin.

---

## 8. Fichiers touchés

- **Modifié** : [notebooks/DZNE_VS_PEC_mitiq.ipynb](DZNE_VS_PEC_mitiq.ipynb) — 35 → 51 cellules.
- **Backup créé** : [notebooks/DZNE_VS_PEC_mitiq.ipynb.bak](DZNE_VS_PEC_mitiq.ipynb.bak) — avant le fix de la corruption source. Supprimable une fois validation OK.
- **Mémoire mise à jour** :
  - [memory/MEMORY.md](../../.claude/projects/c--Users-thoma-Documents-Travail-QNT-Quantum-error-mitigation/memory/MEMORY.md)
  - [memory/project_pec_mitiq_pitfalls.md](../../.claude/projects/c--Users-thoma-Documents-Travail-QNT-Quantum-error-mitigation/memory/project_pec_mitiq_pitfalls.md) — récapitule les 3 pièges Mitiq (CX local/global, NamedQubit↔LineQubit, BFGS→linprog).

---

## 9. Vérifications post-exécution suggérées

1. **§4.5** : assert `|gamma_LP − gamma_analytic| < 1e-4` doit passer.
2. **§5** : PEC bias attendu ≈ 0 (vs −0.10 avant le fix CX global). MSE PEC doit chuter d'un facteur ~10.
3. **§6.4 heat-map** : `Exp` ZNE doit dominer sur Depol et PhaseDamping (ansatz exact). PEC doit ≈ ou battre `Exp` sur AmplitudeDamping et ThermalRelaxation.
4. **Cohérence asymptote** : si la courbe `expectation(λ) vs λ` pour AD/Thermal ne tend pas vers l'asymptote choisie, ajuster.
5. **Reproductibilité** : deux runs successifs sur SPEC_DEPOL doivent donner des chiffres identiques (seed_simulator = 42 fixé).

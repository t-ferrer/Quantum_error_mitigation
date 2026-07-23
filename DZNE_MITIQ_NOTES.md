# DZNE_VS_PEC_mitiq.ipynb — Notes de construction

**Date :** 12 mai 2026  
**Fichier produit :** `notebooks/DZNE_VS_PEC_mitiq.ipynb`

---

## Ce qui a été fait

Création des sections 1, 2 et 3 du notebook Mitiq, miroir de `my_dzne.ipynb` pour validation croisée. 22 cellules exécutées sans erreur.

---

## Section 1 — Setup

### Noise model
Identique au notebook custom :
```python
p1 = 0.01   # portes 1-qubit : h, rz, sx, x, sxdg, id
p2 = 0.05   # portes 2-qubit : cx
```

### Executor
Mitiq 1.0.0 est transparent frontend : il passe des circuits Qiskit directement à l'executor (pas de conversion Cirq visible). L'executor reçoit le circuit foldé en Qiskit, ajoute `measure_all()`, et exécute sur Aer sans retranspilation.

```python
def executor(circuit) -> float:
    qc = circuit.copy()
    qc.measure_all()
    counts = noisy_backend.run(qc, shots=SHOTS).result().get_counts()
    return counts.get('00', 0) / SHOTS
```

---

## Section 2 — Benchmark

Circuit Clifford aléatoire × son inverse → P(|00⟩) = 1 idéalement.

```python
def make_benchmark_circuit(seed):
    rc = random_clifford_circuit(num_qubits=2, num_gates=10, seed=seed)
    circ = rc.compose(rc.inverse())
    return transpile(circ, basis_gates=BASIS_GATES, optimization_level=0)
```

**Baseline noisy sur 20 seeds :** mean ≈ 0.54, std ≈ 0.08

---

## Section 3 — DZNE avec Mitiq

### 3.1 Méthodes de folding disponibles dans Mitiq 1.0.0

| Fonction | Comportement | Fonctionne à λ non entier |
|---|---|---|
| `fold_global` | U → U(U†U)^n global | Oui (folding partiel) |
| `fold_all` | Toutes les portes en place | Seulement aux entiers impairs |
| `fold_gates_at_random` | Portes aléatoires | Oui |

`fold_global` choisi comme méthode de référence (cohérence avec le custom).

### 3.2 Factories non-adaptatives

| Mitiq | Modèle | Correspond à |
|---|---|---|
| `RichardsonFactory` | Lagrange → λ=0 | TBG original |
| `PolyFactory(order=2)` | Polynôme deg. 2 | DZNE non-adaptatif poly |
| `ExpFactory(asymptote=0.25)` | `a + b·exp(-cλ)` | DZNE non-adaptatif exp |

Asymptote théorique pour bruit dépolarisant 2-qubits : 1/4 = 0.25 (état maximalement mélangé).

### 3.3 AdaExpFactory (≈ Algorithme 3)

```python
AdaExpFactory(steps=4, scale_factor=2.0, asymptote=0.25)
```

API Mitiq 1.0.0 :
- `factory.get_scale_factors()` — scale factors choisis adaptativement
- `factory.get_expectation_values()` — valeurs mesurées
- `factory.get_zero_noise_limit()` — estimateur à λ=0

### 3.4 Résultats sur 20 seeds

| Méthode | Moyenne | Biais | Variance | MSE |
|---|---|---|---|---|
| Noisy | 0.537 | −0.463 | 0.006 | 0.221 |
| Richardson | 0.865 | −0.135 | 0.029 | 0.047 |
| Poly2 | 0.881 | −0.119 | 0.028 | 0.043 |
| **Exp** | **0.995** | **−0.005** | **0.001** | **0.001** |
| AdaExp | 1.008 | +0.008 | 0.003 | 0.003 |

Lecture : `Exp` domine car l'ansatz exponentiel est le vrai modèle du bruit dépolarisant. `AdaExp` ≈ `Algorithme 3` du custom (à variance MC près), conforme à la Fig. 7 du papier DZNE.

---

## Pièges rencontrés et résolus

### 1. Folding génère des portes hors noise model

**Problème :** Passer un circuit Qiskit brut à `execute_with_zne` → `fold_global` génère des portes `ry`, `u3`, `rx` (non-standard) aux scale factors non entiers (1.5, 2.0). Ces portes sont absentes du noise model, le bruit ne croît pas linéairement → extrapolation divergente (valeurs de 2 à 7 observées).

**Fix :** Pré-transpiler le circuit vers les basis gates avec `optimization_level=0` **avant** de le passer à Mitiq.

```python
# AVANT : ne pas faire ça
circ = random_clifford_circuit(...).compose(rc.inverse())
zne.execute_with_zne(circ, ...)  # FAUX — génère ry/u3/rx

# APRÈS : pré-transpiler
circ = transpile(circ, basis_gates=['h','rz','sx','x','cx','id','sxdg'], optimization_level=0)
zne.execute_with_zne(circ, ...)  # OK
```

### 2. `sxdg` absent du noise model

**Problème :** `fold_global` génère `sxdg` (inverse de `sx`) lors du folding. Si `sxdg` n'est pas dans le noise model, ces portes passent sans bruit → asymétrie qui corrompt l'extrapolation.

**Fix :** Ajouter `sxdg` au noise model avec le même taux que `sx`.

```python
BASIS_GATES = ['h', 'rz', 'sx', 'x', 'cx', 'id', 'sxdg']  # sxdg ajouté
nm.add_all_qubit_quantum_error(depolarizing_error(P1, 1), ['h', 'rz', 'sx', 'x', 'sxdg', 'id'])
```

### 3. `optimization_level > 0` vide le circuit

**Problème :** Le circuit benchmark est `U·U⁻¹ = I`. Avec `optimization_level=1`, le transpileur reconnaît l'identité et simplifie → circuit vide → P(|00⟩) = 1.0 partout, pas de bruit observable.

**Fix :** Toujours `optimization_level=0` pour les circuits de type U·U⁻¹.

### 4. `ply` manquant

**Problème :** `mitiq.interface.mitiq_qiskit` ne se charge pas → `ModuleNotFoundError: No module named 'ply'`.

**Fix :** `pip install ply` dans le venv du projet.

### 5. API `AdaExpFactory` changée dans Mitiq 1.0.0

**Ancienne API (< 1.0) :** `factory.in_values`, `factory.out_values`  
**Nouvelle API (1.0.0) :** `factory.get_scale_factors()`, `factory.get_expectation_values()`

---

## Ce qui reste (sections 4–5)

- **Section 4 — PEC :** `mitiq.pec.represent_operation_with_local_depolarizing_noise` + `execute_with_pec`
- **Section 5 — Comparaison :** bias/variance/MSE/coût en shots ZNE vs PEC

Point clé à anticiper : l'overhead γ² du PEC croît exponentiellement avec la profondeur — les barres d'erreur PEC seront visiblement plus larges que ZNE à shots égaux sur ce benchmark (depth=10, 2 qubits). C'est le résultat central attendu de la comparaison.

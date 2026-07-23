# Toy DZNE avec OpenEvolve

Un exemple minimaliste pour comprendre le flow OpenEvolve sur un problème
de Zero-Noise Extrapolation simulé analytiquement (sans Qiskit).

## Problème

On simule la mesure d'un observable à différents scale factors, avec un
bruit de la forme `E(λ) = exp(-25 * λ)` plus du bruit de sampling.
La vraie valeur (sans bruit) est `1.0`. Le programme doit extrapoler à
`λ = 0` à partir des mesures.

## Architecture

```
toy_dzne/
├── initial_program.py   # extrapolation linéaire naïve (point de départ)
├── evaluator.py         # simule les mesures et calcule l'erreur
├── config.yaml          # config OpenEvolve (Gemini Flash gratuit)
├── oracle_program.py    # référence: fit exponentiel (juste pour comparer)
└── README.md            # ce fichier
```

## Baseline

| Programme               | Erreur moyenne | Commentaire                      |
| ----------------------- | -------------- | -------------------------------- |
| `initial_program.py`    | 0.229          | extrapolation linéaire (départ)  |
| `oracle_program.py`     | 0.094          | fit exponentiel (cible plausible)|

OpenEvolve devrait converger vers une variante du second, voire mieux
si elle trouve une combinaison robuste.

## Lancer

### 1. Installation

```bash
pip install openevolve numpy scipy
```

### 2. Clé API Gemini (gratuite)

```bash
# Récupère une clé sur https://aistudio.google.com/apikey
export OPENAI_API_KEY="ta-clé-google-ai-studio"
```

### 3. Évaluer le programme initial (avant évolution)

```bash
python evaluator.py initial_program.py
# -> erreur moyenne: 0.22879
```

### 4. Lancer OpenEvolve

```bash
python -m openevolve.cli initial_program.py evaluator.py \
    --config config.yaml --iterations 100
```

L'évolution va générer un dossier `openevolve_output/` avec les
checkpoints. À la fin, le meilleur programme est dans
`openevolve_output/best/best_program.py`.

### 5. Vérifier l'amélioration

```bash
python evaluator.py openevolve_output/best/best_program.py
```

## Quoi observer

- Les premières itérations : OpenEvolve teste des polynômes d'ordre 2, 3...
- Vers l'itération 20-40 : il devrait découvrir l'ansatz exponentiel
- Au-delà : il peut tenter des stratégies plus sophistiquées
  (poly-exponentiel, fit en log, robust regression...)

## Adapter à ta vraie pipeline

Pour passer à un vrai DZNE Qiskit/Mitiq, il suffit de remplacer la
fonction `noisy_measurement` dans `evaluator.py` par :

```python
def noisy_measurement(scale_factor, rng):
    folded = mitiq.zne.scaling.fold_global(circuit, scale_factor)
    job = simulator.run(folded, shots=SHOTS, noise_model=noise_model)
    return job.result().get_counts()  # puis calcul de <O>
```

Tout le reste (initial_program, config) reste identique.

"""
Évaluateur pour le toy DZNE.

Simule un observable mesuré à différents niveaux de bruit (modèle
exponentiel typique du bruit dépolarisant pondéré par la profondeur du
circuit), puis demande au programme d'extrapoler à bruit nul.
La métrique est l'erreur absolue moyenne.
"""

import importlib.util

import numpy as np

# Paramètres du modèle de bruit (fixés pour la reproductibilité)
TRUE_VALUE = 1.0  # E(0) = valeur idéale
BASE_NOISE = 0.02  # lambda physique
CIRCUIT_DEPTH = 25  # facteur c dans exp(-c * lambda)
SHOTS = 4000  # budget de mesure par scale factor
SCALE_FACTORS = [1.0, 1.5, 2.0, 2.5, 3.0]
N_TRIALS = 50  # nombre de tirages indépendants pour moyenner


def noisy_measurement(scale_factor: float, rng: np.random.Generator) -> float:
    """
    Simule la mesure d'un observable au scale factor donné.

    Modèle: E(lambda) = TRUE_VALUE * exp(-depth * lambda) + bruit de sampling.
    Le bruit de sampling a une variance ~1/SHOTS (limite gaussienne).
    """
    lam = BASE_NOISE * scale_factor
    expected = TRUE_VALUE * np.exp(-CIRCUIT_DEPTH * lam)
    sampling_noise = rng.normal(0.0, 1.0 / np.sqrt(SHOTS))
    return float(expected + sampling_noise)


def evaluate(program_path: str) -> dict:
    """
    Charge le programme à évaluer, exécute N_TRIALS extrapolations
    avec différents seeds, et retourne les métriques.

    OpenEvolve maximise les métriques, on retourne donc l'erreur en négatif.
    """
    # Chargement dynamique du programme candidat
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    prog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prog)

    errors = []
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(seed=trial)
        measured = [noisy_measurement(s, rng) for s in SCALE_FACTORS]

        try:
            estimate = prog.extrapolate_to_zero_noise(SCALE_FACTORS, measured)
            err = abs(estimate - TRUE_VALUE)
            # Pénalise les retours non finis (NaN, inf) ou aberrants
            if not np.isfinite(err) or err > 10.0:
                err = 10.0
        except Exception:
            err = 10.0
        errors.append(err)

    errors = np.array(errors)
    return {
        "neg_mean_error": -float(errors.mean()),
        "neg_max_error": -float(errors.max()),
        "neg_std_error": -float(errors.std()),
        # Score composite: priorité à l'erreur moyenne, pénalise pire cas
        "combined_score": -float(errors.mean() + 0.3 * errors.max()),
    }


if __name__ == "__main__":
    # Test rapide en local: on évalue le programme initial
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "initial_program.py"
    metrics = evaluate(path)
    print(f"Évaluation de {path}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")
    print(f"  -> erreur moyenne: {-metrics['neg_mean_error']:.5f}")

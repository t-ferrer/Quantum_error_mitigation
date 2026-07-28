"""Per-qubit Kraus generators (used for both Aer noise and PEC reps)."""

import numpy as np


# ---------------------------------------------------------------------------
# Per-qubit Kraus generators (used both for Aer noise and PEC reps)
# ---------------------------------------------------------------------------
def kraus_amplitude_damping_1q(gamma: float) -> list[np.ndarray]:
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def kraus_phase_damping_1q(lam: float) -> list[np.ndarray]:
    return [
        np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex),
        np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex),
    ]


def kraus_thermal_1q(gamma_ad: float, gamma_pd: float) -> list[np.ndarray]:
    """Composite AD(γ_ad) ∘ PD(γ_pd) per qubit. K_ij = A_i · P_j  =>  4 Kraus ops."""
    g, l = gamma_ad, gamma_pd
    return [
        np.array([[1, 0], [0, np.sqrt((1 - g) * (1 - l))]], dtype=complex),  # A0·P0
        np.array([[0, 0], [0, np.sqrt(l * (1 - g))]], dtype=complex),  # A0·P1
        np.array([[0, np.sqrt(g * (1 - l))], [0, 0]], dtype=complex),  # A1·P0
        np.array([[0, np.sqrt(g * l)], [0, 0]], dtype=complex),  # A1·P1
    ]

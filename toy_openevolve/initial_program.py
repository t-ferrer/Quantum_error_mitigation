"""
Toy DZNE: extrapolation à bruit nul.

OpenEvolve va faire évoluer la fonction extrapolate_to_zero_noise pour
améliorer l'estimation. Version initiale: extrapolation linéaire basique
(sous-optimale quand le bruit est exponentiel).
"""
import numpy as np


# EVOLVE-BLOCK-START
def extrapolate_to_zero_noise(scale_factors, measured_values):
    """
    À partir de mesures à différents niveaux de bruit, estime la valeur
    de l'observable à bruit nul.

    Parameters
    ----------
    scale_factors : list[float]
        Les facteurs c_j tels que lambda_j = c_j * lambda_base.
        Toujours triés en ordre croissant, c_0 = 1.0.
    measured_values : list[float]
        E(c_j * lambda) avec bruit de sampling.

    Returns
    -------
    float
        Estimation de E(0) = valeur idéale sans bruit.
    """
    # Extrapolation linéaire (Richardson d'ordre 1)
    coeffs = np.polyfit(scale_factors, measured_values, deg=1)
    return float(np.polyval(coeffs, 0.0))
# EVOLVE-BLOCK-END

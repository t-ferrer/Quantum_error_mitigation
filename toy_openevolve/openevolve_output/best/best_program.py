"""
Toy DZNE: extrapolation à bruit nul.

OpenEvolve va faire évoluer la fonction extrapolate_to_zero_noise pour
améliorer l'estimation. Version initiale: extrapolation linéaire basique
(sous-optimale quand le bruit est exponentiel).
Cette version utilise une extrapolation exponentielle pour mieux
modéliser le bruit dépolarisant.
"""
import numpy as np
from scipy.optimize import curve_fit

# Define the exponential ansatz: E(lambda) = a + b * exp(-c * lambda)
def exponential_ansatz(x, a, b, c):
    return a + b * np.exp(-c * x)

# EVOLVE-BLOCK-START
def extrapolate_to_zero_noise(scale_factors, measured_values):
    """
    À partir de mesures à différents niveaux de bruit, estime la valeur
    de l'observable à bruit nul en utilisant une extrapolation exponentielle.

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
    try:
        # Fit the data to the exponential ansatz
        # Initial guess for parameters can help convergence.
        # a: expected value at zero noise, can be approximated by the highest fidelity measurement.
        # b: amplitude of the noise decay.
        # c: decay rate, related to the noise strength.
        initial_guess = [measured_values[0], measured_values[0] - measured_values[-1], 1.0]
        # Add bounds to the fit: a >= 0, b >= 0, c >= 0.
        # The exact bounds might need tuning, but these are reasonable starting points.
        # 'a' is the extrapolated value, 'b' is the noise amplitude, 'c' is the decay rate.
        # We expect 'a' to be close to measured_values[0].
        # We expect 'b' to be non-negative (noise adds to the signal).
        # We expect 'c' to be non-negative (noise increases with scale_factors).
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]) # Lower and upper bounds for a, b, c
        params, _ = curve_fit(exponential_ansatz, scale_factors, measured_values, p0=initial_guess, maxfev=5000, bounds=bounds)
        a, b, c = params

        # Extrapolate to zero noise (lambda = 0)
        # E(0) = a + b * exp(-c * 0) = a + b
        # The extrapolated_value is 'a' itself if we interpret 'a' as E(0) and 'b*exp(-c*x)' as the noise term.
        # However, the current code sums a and b, which implies the ansatz is E(lambda) = a_noise + b_noise * exp(-c*lambda)
        # and E(0) = a_noise + b_noise. Let's stick to the current interpretation for now.
        extrapolated_value = a + b # This assumes the ansatz is E(x) = a + b*exp(-c*x), and E(0) = a + b.
        
        # Ensure the result is a finite float.
        if not np.isfinite(extrapolated_value):
            # Fallback to linear extrapolation if exponential fit fails unexpectedly
            coeffs = np.polyfit(scale_factors, measured_values, deg=1)
            return float(np.polyval(coeffs, 0.0))
        
        return float(extrapolated_value)

    except (RuntimeError, ValueError):
        # If curve_fit fails (e.g., due to bad data or convergence issues),
        # try a fallback strategy.

        # Fallback 1: Logarithmic extrapolation
        # Assumes E(lambda) ~ E_0 * exp(-c * lambda)
        # Then log(E) ~ log(E_0) - c * lambda. This is linear in log-space.
        # We need to handle cases where measured_values are non-positive.
        positive_mask = np.array(measured_values) > 0
        if np.sum(positive_mask) >= 2: # Need at least two points to fit a line
            try:
                log_measured_values = np.log(np.array(measured_values)[positive_mask])
                log_scale_factors = np.array(scale_factors)[positive_mask]
                
                # Fit a line to log(E) vs lambda
                coeffs_log = np.polyfit(log_scale_factors, log_measured_values, deg=1)
                # Extrapolate to lambda=0 in log-space: log(E_0) = coeffs_log[1]
                log_extrapolated_value = coeffs_log[1] 
                # Convert back: E_0 = exp(log(E_0))
                extrapolated_value_log = np.exp(log_extrapolated_value)
                
                # Ensure the result is a finite float.
                if np.isfinite(extrapolated_value_log):
                    return float(extrapolated_value_log)
            except (RuntimeError, ValueError):
                pass # If log fit fails, proceed to next fallback

        # Fallback 2: Linear extrapolation (as before)
        try:
            coeffs_linear = np.polyfit(scale_factors, measured_values, deg=1)
            return float(np.polyval(coeffs_linear, 0.0))
        except (RuntimeError, ValueError):
            pass # If linear fit fails, proceed to next fallback

        # Fallback 3: Return the best measured value as a last resort.
        return float(measured_values[0])
# EVOLVE-BLOCK-END

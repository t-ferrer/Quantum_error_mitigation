"""Sanity check: ce que donnerait un fit exponentiel idéal."""
import numpy as np
from scipy.optimize import curve_fit


def extrapolate_to_zero_noise(scale_factors, measured_values):
    """Fit exponentiel: E(lam) = a + b * exp(-c * lam)."""
    def model(x, a, b, c):
        return a + b * np.exp(-c * x)
    try:
        popt, _ = curve_fit(
            model, scale_factors, measured_values,
            p0=[0.0, 1.0, 0.5], maxfev=5000
        )
        return float(model(0.0, *popt))
    except Exception:
        return float(np.polyfit(scale_factors, measured_values, 1)[1])

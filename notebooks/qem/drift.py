"""Inter-run parameter drift (case B): Ornstein-Uhlenbeck trajectories that modulate the
composite noise model's parameters BETWEEN runs (non-stationary noise across executions).

Physics. On real superconducting devices T1/T2, the coherent/Pauli residual, and readout
fidelities fluctuate slowly (minutes-hours: 1/f, TLS, calibration drift). Timescale separation:
a circuit runs in ms, so the channel is FROZEN within a run and only changes BETWEEN runs. This
is the granularity used in the literature (Dasgupta/Humble arXiv:2308.14756 -> 13 datasets at ~2h
intervals; PEC-under-nonstationarity arXiv:2404.13269). It invalidates static PEC's exact-channel
assumption (QPRs built from NOMINAL params become stale -> biased), motivating re-learning.

Design (see project roadmap 'drift phase'):
  - ONE composite model (thermal o mixed-unitary + readout), NOT three separate models: drift is
    a modulation of PARAMETERS, not a new channel structure. The thermal o mu composition (and the
    cx factorisation) are already validated to 1e-16 in noise_models.py -- we only move the numbers.
  - N INDEPENDENT OU trajectories, one per physically-distinct mechanism ({T1},{T2},{gate eps},
    {readout}) -- they fluctuate via different processes but COEXIST in one device/run -> one model.
  - Orthogonal ADD-ON: SPEC_COMPOSITE and _composite_readout_noise_model stay byte-identical (the
    static DZNE-vs-PEC benchmark and its inspect.getsource cache key are preserved). This builder is
    a SEPARATE, parameterised twin -- at nominal args it reproduces the static model exactly
    (checked in smoke_drift_ou.py).
  - Two RNGs, two roles: the OU trajectory is SEEDED (replay identical drift for static-vs-adaptive
    comparisons); the Aer SHOT rng stays free/seedless (the anti-bias PEC fix). Do not conflate.

Ornstein-Uhlenbeck (in units of RUNS):  dx = -(1/tau)(x - mu) dt + sigma dW.
  - mean-reverting: a hardware parameter fluctuates but does not wander off -> stationary
    N(mu, sigma_stat^2), a BOUNDED distribution (unlike a random walk whose variance diverges).
  - autocorrelation ~ exp(-|dt|/tau): tau = 'how fast calibration goes stale'. Lorentzian
    spectrum; a sum of OU processes with different tau approximates 1/f (the TLS-ensemble picture),
    so OU is the atomic building block we start from.
"""

from dataclasses import dataclass, replace

import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

from .config import _GATES_2Q
from .noise_models import (
    EPS_MU_1Q,
    EPS_MU_2Q,
    GATE_TIME_S,
    T1_S,
    T2_S,
    _MU_PHYS_1Q,
    _mu_pauli_error,
)
from .readout import READOUT_REALISTIC, ReadoutSpec

__all__ = [
    "OUParams",
    "ou_trajectory",
    "DriftConfig",
    "DRIFT_DEFAULT",
    "NoiseParams",
    "NOMINAL_PARAMS",
    "drift_params",
    "composite_noise_model_from_params",
    "build_drift_noise_model",
    "build_drift_gate_model",
]


# ---------------------------------------------------------------------------
# 1. The scalar Ornstein-Uhlenbeck process
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OUParams:
    """One scalar OU process, time measured in RUNS.

    Exact-discretisation update (preserves the stationary N(mu, sigma_stat^2) at ANY step, unlike
    Euler):   x_{t+1} = mu + (x_t - mu) * a + sigma_stat * sqrt(1 - a^2) * z,   a = exp(-1/tau),
    with sigma_stat = sigma_rel * |mu| and z ~ N(0, 1).
    """

    mu: float  # mean-reversion target = calibrated nominal value
    tau: float  # correlation time, IN RUNS (how fast calibration goes stale)
    sigma_rel: float  # stationary std as a fraction of |mu|
    lo: float = 0.0  # output clip floor (physical validity, e.g. T1 > 0, prob >= 0)
    hi: float = np.inf  # output clip ceiling


def ou_trajectory(p: OUParams, n_runs: int, rng: np.random.Generator) -> np.ndarray:
    """Sample an OU path of length `n_runs`, started from the STATIONARY distribution.

    Starting from stationarity (not from mu) means run 0 is already a 'random moment' -> the
    ensemble over runs has no calibration transient. Clipping is applied to the OUTPUT only (the
    dynamics stay pure Gaussian so the stationary stats are exact); at the default params the tail
    reaches [lo, hi] with negligible probability, so clipping is a safety rail, not a bias.
    """
    a = np.exp(-1.0 / p.tau)
    sigma_stat = p.sigma_rel * abs(p.mu)
    step_sd = sigma_stat * np.sqrt(1.0 - a * a)
    x = np.empty(n_runs)
    x[0] = p.mu + sigma_stat * rng.standard_normal()  # draw from stationary N(mu, sigma_stat^2)
    for t in range(1, n_runs):
        x[t] = p.mu + (x[t - 1] - p.mu) * a + step_sd * rng.standard_normal()
    return np.clip(x, p.lo, p.hi)


# ---------------------------------------------------------------------------
# 2. Drift configuration: one OU per mechanism
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriftConfig:
    """Independent OU processes, one per physically-distinct drift mechanism. They coexist in ONE
    device -> a single composite NoiseModel per run (not one model per mechanism).

    `gate_scale`/`readout_scale` are MULTIPLIERS (mu=1) applied to the nominal magnitudes so the
    STRUCTURAL ratios stay fixed: gate_scale scales eps_1q AND eps_2q together (the 5x 2q/1q ratio
    is preserved), readout_scale scales every p0/p1 (the readout asymmetry is preserved). Only the
    empirically-volatile magnitudes drift; structural ratios (eta, p0/p1 shape) do not.
    """

    t1: OUParams
    t2: OUParams
    gate_scale: OUParams
    readout_scale: OUParams


# Defaults: tau ~ 5 runs (correlation across a handful of executions); sigma_rel ~ 10-15%, with T2
# the most volatile (pure dephasing / TLS). Deviations of this size are consistent with the >25%
# SPAM excursions reported on hardware (arXiv:2404.13269). All tunable.
DRIFT_DEFAULT = DriftConfig(
    t1=OUParams(mu=T1_S, tau=5.0, sigma_rel=0.10, lo=1e-9),
    t2=OUParams(mu=T2_S, tau=5.0, sigma_rel=0.15, lo=1e-9),
    gate_scale=OUParams(mu=1.0, tau=5.0, sigma_rel=0.15, lo=0.0),
    readout_scale=OUParams(mu=1.0, tau=5.0, sigma_rel=0.15, lo=0.0),
)


# ---------------------------------------------------------------------------
# 3. Per-run noise parameter vector theta_t
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseParams:
    """The parameter vector theta_t that builds ONE composite noise model for run t."""

    t1: float
    t2: float
    eps_1q: float
    eps_2q: float
    readout: ReadoutSpec


NOMINAL_PARAMS = NoiseParams(T1_S, T2_S, EPS_MU_1Q, EPS_MU_2Q, READOUT_REALISTIC)


def _scale_readout(spec: ReadoutSpec, scale: float) -> ReadoutSpec:
    """Scale every p0/p1 by `scale`, clipping each to [0, 0.49] so p0_i + p1_i <= 0.98 < 1 -> the
    twirled eigenvalue lambda_i = 1 - p0_i - p1_i stays > 0 (TREX-invertible)."""
    clip = lambda x: min(max(x * scale, 0.0), 0.49)  # noqa: E731
    return replace(
        spec,
        p0=tuple(clip(a) for a in spec.p0),
        p1=tuple(clip(b) for b in spec.p1),
        name=f"{spec.name}Drift",
    )


def drift_params(
    n_runs: int,
    seed: int = 0,
    config: DriftConfig = DRIFT_DEFAULT,
    readout_nominal: ReadoutSpec = READOUT_REALISTIC,
) -> list[NoiseParams]:
    """A reproducible drift trajectory: one NoiseParams theta_t per run.

    SEEDED on purpose (replay identical drift for a fair static-vs-adaptive comparison); the shot
    rng stays free elsewhere. Enforces the Aer validity constraint T2 <= 2*T1 per run.
    """
    rng = np.random.default_rng(seed)
    t1 = ou_trajectory(config.t1, n_runs, rng)
    t2 = ou_trajectory(config.t2, n_runs, rng)
    gscale = ou_trajectory(config.gate_scale, n_runs, rng)
    rscale = ou_trajectory(config.readout_scale, n_runs, rng)
    out = []
    for i in range(n_runs):
        t2_i = min(t2[i], 2.0 * t1[i])  # Aer requires T2 <= 2*T1 (thermal_relaxation_error)
        out.append(
            NoiseParams(
                t1=float(t1[i]),
                t2=float(t2_i),
                eps_1q=EPS_MU_1Q * float(gscale[i]),
                eps_2q=EPS_MU_2Q * float(gscale[i]),
                readout=_scale_readout(readout_nominal, float(rscale[i])),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 4. Parameterised composite noise model (twin of _composite_readout_noise_model)
# ---------------------------------------------------------------------------
def composite_noise_model_from_params(
    t1: float,
    t2: float,
    eps_1q: float,
    eps_2q: float,
    readout: ReadoutSpec | None = None,
) -> NoiseModel:
    """Parameterised twin of noise_models._composite_readout_noise_model().

    SEPARATE from the static builder ON PURPOSE: editing that one would invalidate the static
    composite cache (key = inspect.getsource). With nominal args (+READOUT_REALISTIC) this
    reproduces the static composite+readout model EXACTLY (checked in smoke_drift_ou.py).
    """
    nm = NoiseModel()
    mu_1q = _mu_pauli_error(eps_1q)
    for g in _MU_PHYS_1Q:  # sx/sxdg/x/h ; rz/id virtual (zero duration) -> no noise
        th_g = thermal_relaxation_error(t1, t2, GATE_TIME_S[g])
        nm.add_all_qubit_quantum_error(th_g.compose(mu_1q), [g])
    # CX: per-qubit thermal (tensored) composed with per-qubit biased-Pauli (tensored).
    th_cx = thermal_relaxation_error(t1, t2, GATE_TIME_S["cx"])
    mu_2q = _mu_pauli_error(eps_2q)
    nm.add_all_qubit_quantum_error(
        th_cx.tensor(th_cx).compose(mu_2q.tensor(mu_2q)), list(_GATES_2Q)
    )
    if readout is not None:
        readout.add_readout_error(nm)
    return nm


def build_drift_noise_model(p: NoiseParams) -> NoiseModel:
    """The composite+readout NoiseModel for a single run's parameter vector theta_t."""
    return composite_noise_model_from_params(
        p.t1, p.t2, p.eps_1q, p.eps_2q, p.readout
    )


def build_drift_gate_model(p: NoiseParams) -> NoiseModel:
    """Gate-only composite (thermal o mixed-unitary, NO readout) for run theta_t.

    Use for the gate-mitigation-under-drift study (Composite_drift.ipynb) so there is NO readout
    floor -- mirrors the static split Composite_noise_DZNE (gate-only) vs Composite_readout_TREX
    (full stack). The drifting readout carried in `p` is simply ignored here; it drives the TREX
    study (Composite_drift_TREX.ipynb) instead.
    """
    return composite_noise_model_from_params(p.t1, p.t2, p.eps_1q, p.eps_2q, readout=None)

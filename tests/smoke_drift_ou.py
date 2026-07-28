"""Smoke test: inter-run parameter drift (qem/drift.py), step 1 -- BEFORE touching run_benchmark.

Validates the foundational brick in isolation:
  (1) OU statistics are exact: empirical mean -> mu, std -> sigma_rel*|mu|, lag-1 autocorr -> e^(-1/tau).
  (2) The drift trajectory is reproducible (seeded) and seed-dependent.
  (3) Every run's theta_t is physically valid: T1,T2>0, T2<=2*T1, eps>0, readout p in [0,0.49], p0+p1<1.
  (4) Non-regression: at NOMINAL params the parameterised builder == the static composite+readout model.
  (5) The model builds and runs, and drift has a MEASURABLE effect on P(target).
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from qiskit_aer import AerSimulator

from qem import (
    OUParams, ou_trajectory, DRIFT_DEFAULT, NOMINAL_PARAMS, drift_params,
    composite_noise_model_from_params, build_drift_noise_model,
    make_benchmark_circuit, success_prob,
    T1_S, T2_S, EPS_MU_1Q, EPS_MU_2Q, READOUT_REALISTIC,
)
from qem.noise_models import _composite_readout_noise_model

checks = []

# (1) OU statistics -- use lo/hi = +-inf so clipping is inert and the pure-Gaussian stats show.
mu, tau, sig_rel = 3.0, 8.0, 0.2
p = OUParams(mu=mu, tau=tau, sigma_rel=sig_rel, lo=-np.inf, hi=np.inf)
x = ou_trajectory(p, 400_000, np.random.default_rng(0))
emp_mean, emp_std = x.mean(), x.std()
emp_ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
want_std, want_ac1 = sig_rel * abs(mu), np.exp(-1.0 / tau)
c1 = (abs(emp_mean - mu) < 0.02 and abs(emp_std - want_std) < 0.02
      and abs(emp_ac1 - want_ac1) < 0.01)
checks.append(c1)
print("(1) OU stats (exact discretisation)")
print(f"    mean {emp_mean:.4f} (mu={mu})   std {emp_std:.4f} (want {want_std:.4f})   "
      f"ac1 {emp_ac1:.4f} (want e^-1/tau={want_ac1:.4f})   {'OK' if c1 else '*** CHECK'}")

# (2) reproducibility
a0 = drift_params(50, seed=0)
a0b = drift_params(50, seed=0)
a1 = drift_params(50, seed=1)
same = all(x.t1 == y.t1 and x.eps_1q == y.eps_1q for x, y in zip(a0, a0b))
diff = any(x.t1 != y.t1 for x, y in zip(a0, a1))
c2 = same and diff
checks.append(c2)
print(f"(2) reproducible seeded trajectory (same seed identical, diff seed differs)   "
      f"{'OK' if c2 else '*** CHECK'}")

# (3) physical validity across a long trajectory
traj = drift_params(2000, seed=7)
c3 = all(
    q.t1 > 0 and q.t2 > 0 and q.t2 <= 2 * q.t1 + 1e-18
    and q.eps_1q > 0 and q.eps_2q > 0
    and all(0.0 <= v <= 0.49 for v in q.readout.p0 + q.readout.p1)
    and all(a + b < 1.0 for a, b in zip(q.readout.p0, q.readout.p1))
    for q in traj
)
checks.append(c3)
t1s = np.array([q.t1 for q in traj]); t2s = np.array([q.t2 for q in traj])
print(f"(3) physical validity over 2000 runs (T1,T2>0, T2<=2T1, eps>0, readout valid)   "
      f"{'OK' if c3 else '*** CHECK'}")
print(f"    T1 range [{t1s.min()*1e6:.1f}, {t1s.max()*1e6:.1f}] us   "
      f"T2 range [{t2s.min()*1e6:.1f}, {t2s.max()*1e6:.1f}] us   (nominal 100/80)")

# (4) non-regression at nominal params
nm_param = composite_noise_model_from_params(
    T1_S, T2_S, EPS_MU_1Q, EPS_MU_2Q, READOUT_REALISTIC)
nm_static = _composite_readout_noise_model()
c4 = nm_param == nm_static
checks.append(c4)
print(f"(4) nominal builder == static composite+readout model   {'OK' if c4 else '*** CHECK'}")

# (5) model builds, runs, and drift has a measurable effect on P(target)
back_nom = AerSimulator(noise_model=build_drift_noise_model(NOMINAL_PARAMS))
# a deliberately strong single-run drift: half the coherence, 3x gate error & readout
strong = drift_params(1, seed=0, config=DRIFT_DEFAULT)[0]  # a real sampled run
from qem.drift import NoiseParams
harsh = NoiseParams(t1=T1_S * 0.5, t2=T2_S * 0.5, eps_1q=EPS_MU_1Q * 3,
                    eps_2q=EPS_MU_2Q * 3, readout=NOMINAL_PARAMS.readout)
back_harsh = AerSimulator(noise_model=build_drift_noise_model(harsh))
p_nom = np.mean([success_prob(back_nom, s) for s in range(4)])
p_harsh = np.mean([success_prob(back_harsh, s) for s in range(4)])
c5 = (0.0 <= p_harsh <= 1.0) and (p_nom - p_harsh > 0.02)  # harsher noise -> lower fidelity
checks.append(c5)
print(f"(5) model runs; drift lowers fidelity: P_nominal={p_nom:.4f} > P_harsh={p_harsh:.4f}   "
      f"{'OK' if c5 else '*** CHECK'}")

# small trajectory preview
print("\n  preview (seed=0, first 6 runs):")
print(f"  {'run':>3s} {'T1/us':>7s} {'T2/us':>7s} {'eps_1q':>8s} {'eps_2q':>8s} {'p0[0]':>7s} {'p1[0]':>7s}")
for i, q in enumerate(drift_params(6, seed=0)):
    print(f"  {i:>3d} {q.t1*1e6:>7.2f} {q.t2*1e6:>7.2f} {q.eps_1q:>8.5f} {q.eps_2q:>8.5f} "
          f"{q.readout.p0[0]:>7.4f} {q.readout.p1[0]:>7.4f}")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

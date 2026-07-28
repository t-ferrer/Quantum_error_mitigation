"""Smoke test: drift-aware run_benchmark (qem/benchmark.py), step 2.

Validates the plumbing that rebuilds the backend PER RUN with theta_t, WITHOUT breaking the static
path, and demonstrates the scientific payoff (stale-rep PEC bias under drift):
  (A) Non-regression: feeding CONSTANT nominal models per-run == the static single-backend path
      (the per-run-rebuild machinery is equivalent when the params don't actually drift).
  (B) Length guard: a drift_models list whose length != #seeds raises ValueError.
  (C1) Varying-drift path: a genuinely time-varying OU trajectory (a DIFFERENT model each run) runs
       end-to-end and, on average, degrades the raw (Noisy) signal.
  (C2) Stale-rep PEC bias: on a fixed "stale calibration" channel (gate noise 2x, coherence 0.6x,
       readout unchanged so the readout floor cancels in the comparison), static PEC keeps its
       NOMINAL gate reps -> they under-correct the drifted channel -> |bias| and MSE clearly rise
       vs the nominal-channel PEC. This is the signature that motivates re-learning (PEA / NML).

Full stack: SPEC_COMPOSITE_READOUT (thermal o mixed-unitary + realistic readout).
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np

from qem import (
    run_benchmark, SPEC_COMPOSITE_READOUT,
    drift_params, build_drift_noise_model, DriftConfig, OUParams,
    NoiseParams, NOMINAL_PARAMS, T1_S, T2_S, EPS_MU_1Q, EPS_MU_2Q,
)
from qem.noise_models import _composite_readout_noise_model

spec = SPEC_COMPOSITE_READOUT
NAME = spec.name
checks = []


def row(df, method):
    r = df.loc[(NAME, method)]
    return float(r["bias"]), float(r["variance"]), float(r["mse"])


# (A) non-regression: constant nominal models per-run must match the static path (Noisy mean).
sA = list(range(6))
static_df, _, _ = run_benchmark(spec, seeds=sA, run_pec=False, verbose=False)
const_df, _, _ = run_benchmark(spec, seeds=sA, run_pec=False, verbose=False,
                               drift_models=[_composite_readout_noise_model() for _ in sA])
noisy_static = float(static_df.loc[(NAME, "Noisy"), "mean"])
noisy_const = float(const_df.loc[(NAME, "Noisy"), "mean"])
cA = abs(noisy_static - noisy_const) < 0.02
checks.append(cA)
print(f"(A) per-run-rebuild == static (constant nominal models): "
      f"Noisy {noisy_static:.4f} vs {noisy_const:.4f}  {'OK' if cA else '*** CHECK'}")

# (B) length guard
try:
    run_benchmark(spec, seeds=sA, run_pec=False, verbose=False,
                  drift_models=[_composite_readout_noise_model()] * 3)  # 3 != 6
    cB = False
except ValueError:
    cB = True
checks.append(cB)
print(f"(B) length-mismatch drift_models raises ValueError: {'OK' if cB else '*** CHECK'}")

# (C1) genuinely time-varying OU drift: a DIFFERENT model each run, runs end-to-end and degrades
#      the raw signal on average (stronger average gate/readout noise).
STRONG = DriftConfig(
    t1=OUParams(mu=T1_S, tau=3.0, sigma_rel=0.30, lo=1e-9),
    t2=OUParams(mu=T2_S, tau=3.0, sigma_rel=0.35, lo=1e-9),
    gate_scale=OUParams(mu=1.0, tau=3.0, sigma_rel=0.40, lo=0.0),
    readout_scale=OUParams(mu=1.0, tau=3.0, sigma_rel=0.40, lo=0.0),
)
sB = list(range(10))
ou_models = [build_drift_noise_model(p) for p in drift_params(len(sB), seed=0, config=STRONG)]
base_df, _, _ = run_benchmark(spec, seeds=sB, run_pec=False, verbose=False)
ou_df, _, _ = run_benchmark(spec, seeds=sB, run_pec=False, verbose=False, drift_models=ou_models)
b_base = float(base_df.loc[(NAME, "Noisy"), "bias"])
b_ou = float(ou_df.loc[(NAME, "Noisy"), "bias"])
cC1 = np.isfinite(b_ou) and b_ou < b_base  # more negative bias under drift (worse), and finite
checks.append(cC1)
print(f"(C1) varying-OU-drift path runs; raw signal degrades: Noisy bias {b_base:+.4f} -> "
      f"{b_ou:+.4f}  {'OK' if cC1 else '*** CHECK'}")

# (C2) stale-rep PEC bias on a fixed "stale calibration" channel. readout UNCHANGED so the readout
#      floor is identical in both -> the bias DIFFERENCE isolates the stale GATE reps.
sC = list(range(8))
BUDGET = 8000
bad = NoiseParams(t1=T1_S * 0.6, t2=T2_S * 0.6, eps_1q=EPS_MU_1Q * 2, eps_2q=EPS_MU_2Q * 2,
                  readout=NOMINAL_PARAMS.readout)
bad_models = [build_drift_noise_model(bad) for _ in sC]
nod_df, _, _ = run_benchmark(spec, seeds=sC, shot_budget=BUDGET, run_pec=True, verbose=False)
stale_df, _, _ = run_benchmark(spec, seeds=sC, shot_budget=BUDGET, run_pec=True, verbose=False,
                               drift_models=bad_models)

print(f"\n  {'method':10s} | {'bias(nominal)':>13s} {'mse(nominal)':>12s} | "
      f"{'bias(stale)':>12s} {'mse(stale)':>11s}")
for m in ["Noisy", "Exp", "PEC"]:
    b0, _, m0 = row(nod_df, m)
    b1, _, m1 = row(stale_df, m)
    print(f"  {m:10s} | {b0:>13.4f} {m0:>12.5f} | {b1:>12.4f} {m1:>11.5f}")

b_pec_nom, _, mse_pec_nom = row(nod_df, "PEC")
b_pec_stale, _, mse_pec_stale = row(stale_df, "PEC")
cC2 = abs(b_pec_stale) > abs(b_pec_nom) + 0.02 and mse_pec_stale > mse_pec_nom
checks.append(cC2)
print(f"\n(C2) stale reps bias PEC: |bias| {abs(b_pec_nom):.4f} -> {abs(b_pec_stale):.4f}, "
      f"MSE {mse_pec_nom:.5f} -> {mse_pec_stale:.5f}  {'OK' if cC2 else '*** CHECK'}")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

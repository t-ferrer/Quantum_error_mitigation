"""Smoke test: EXPERIMENTAL static/adaptive TREX under readout drift (qem.readout).

The experimental pipeline measures BOTH ingredients on Aer (real shot noise):
  - exec_counts  = mirror(target) through gate + TWIRLED (symmetrised) drifted readout,
  - calib_counts = |0...0> through the twirled calibration readout  -> <Z_w> = lambda_w.
Twirling is realised by simulating the SYMMETRISED channel (ReadoutSpec.twirled()) rather than
per-shot X-twirls. Validates:
  (1) .twirled() symmetrises (p0=p1=p_bar) and preserves lambda_w.
  (2) lambda_w MEASURED on |0...0> ~ analytic lambda_w (within shot noise).
  (3) adaptive TREX (calib=drifted) ~ 1.0 on an ideal gate (readout removed).
  (4) noisy < static < adaptive, and static (frozen nominal calib) deviates from 1 under drift.
  (5) FAITHFULNESS: experimental mean ~ analytic mean (trex_success_prob_from_counts), composite gate.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qem import (
    SPEC_COMPOSITE, make_benchmark_circuit, target_key, N_QUBITS,
    ReadoutSpec, READOUT_REALISTIC, READOUT_PERFECT,
    trex_success_prob_from_counts, trex_success_prob_experimental,
)
from qem.readout import _zw_from_counts

checks = []
SH = 100_000
DRIFT = ReadoutSpec("drift", p0=(0.03, 0.04), p1=(0.05, 0.06))  # worse-than-nominal readout


def mirror_counts(backend, seed, shots=SH):
    qc = make_benchmark_circuit(seed).copy(); qc.measure_all()
    return backend.run(qc, shots=shots).result().get_counts()


def calib_counts(readout, shots=SH):
    qc = QuantumCircuit(N_QUBITS); qc.measure_all()  # |0...0> through readout-only backend
    return AerSimulator(noise_model=readout.readout_noise_model()).run(qc, shots=shots).result().get_counts()


# (1) twirled() symmetrises and preserves lambda_w
tw = READOUT_REALISTIC.twirled()
pbar = [(a + b) / 2 for a, b in zip(READOUT_REALISTIC.p0, READOUT_REALISTIC.p1)]
c1 = (tw.p0 == tw.p1 and all(abs(a - b) < 1e-12 for a, b in zip(tw.p0, pbar))
      and all(abs(tw.lambda_support(w) - READOUT_REALISTIC.lambda_support(w)) < 1e-12
              for w in [(0,), (1,), (0, 1)]))
checks.append(c1)
print(f"(1) twirled() symmetrises & preserves lambda_w   {'OK' if c1 else '*** CHECK'}")

# (2) measured lambda_w ~ analytic
zc = _zw_from_counts(calib_counts(DRIFT.twirled()), SH, N_QUBITS)
lam_meas, lam_true = zc[(0, 1)], DRIFT.lambda_support((0, 1))
c2 = abs(lam_meas - lam_true) < 0.01
checks.append(c2)
print(f"(2) measured lambda_ZZ={lam_meas:.4f} ~ analytic {lam_true:.4f}   {'OK' if c2 else '*** CHECK'}")

# (3)/(4) ideal gate + twirled drifted readout: adaptive~1, static deviates, noisy floor
tgt = target_key(0)
exec_c = mirror_counts(AerSimulator(noise_model=DRIFT.twirled().readout_noise_model()), 0)
p_adaptive = trex_success_prob_experimental(exec_c, calib_counts(DRIFT.twirled()), tgt, SH, SH)
p_static = trex_success_prob_experimental(exec_c, calib_counts(READOUT_REALISTIC.twirled()), tgt, SH, SH)
p_noisy = trex_success_prob_experimental(exec_c, calib_counts(READOUT_PERFECT), tgt, SH, SH)
c3 = abs(p_adaptive - 1.0) < 0.02
c4 = (p_noisy < 0.99 and p_noisy < p_static < p_adaptive + 0.01 and abs(p_static - 1) > 0.005)
checks += [c3, c4]
print(f"(3) adaptive~1.0: {p_adaptive:.4f}   {'OK' if c3 else '*** CHECK'}")
print(f"(4) noisy {p_noisy:.4f} < static {p_static:.4f} < adaptive {p_adaptive:.4f}   {'OK' if c4 else '*** CHECK'}")

# (5) faithfulness: experimental static ~ analytic static, composite gate (nontrivial <Z_w>)
gate_bk = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model())         # gate only (analytic input)
nm = SPEC_COMPOSITE.build_noise_model(); DRIFT.twirled().add_readout_error(nm)
exec_bk = AerSimulator(noise_model=nm)                                         # gate + twirled drift readout
calib_static = calib_counts(READOUT_REALISTIC.twirled())
exp_vals, ana_vals = [], []
for s in range(8):
    tgt_s = target_key(s)
    ana_vals.append(trex_success_prob_from_counts(mirror_counts(gate_bk, s), SH, tgt_s, DRIFT, READOUT_REALISTIC))
    exp_vals.append(trex_success_prob_experimental(mirror_counts(exec_bk, s), calib_static, tgt_s, SH, SH))
exp_m, ana_m = float(np.mean(exp_vals)), float(np.mean(ana_vals))
c5 = abs(exp_m - ana_m) < 0.015
checks.append(c5)
print(f"(5) experimental mean {exp_m:.4f} ~ analytic mean {ana_m:.4f} (static, composite gate)   "
      f"{'OK' if c5 else '*** CHECK'}")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

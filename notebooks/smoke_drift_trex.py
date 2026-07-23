"""Smoke test: analytic static/adaptive TREX under readout drift (qem.readout).

trex_success_prob_from_counts computes the TREX-corrected P(target) from GATE-ONLY counts + a
(true, calib) readout pair. Validates:
  (1) projector identity: calib == true (any spec) -> ratio 1 -> exactly the raw gate P(target).
  (2) readout-only, ideal gate: adaptive (calib=true) == 1 ; noisy (calib=perfect) < 1 ; static
      (calib=nominal) == 1 at nominal but deviates once the readout drifts.
  (3) symmetric-readout cross-check: analytic noisy == direct Aer success_prob (twirled==untwirled).
  (4) mitiq cross-check: analytic ADAPTIVE (= gate-only P) ~ mitiq's real trex_success_prob on the
      composite+readout backend (both remove readout -> land at the gate floor).
  (5) staleness: as the readout drifts stronger, |static - 1| grows while adaptive stays ~1.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from qiskit_aer import AerSimulator

from qem import (
    SPEC_COMPOSITE, make_benchmark_circuit, target_key, success_prob,
    benchmark_circuit_cirq, trex_success_prob, mitiq_measurement_executor,
    trex_success_prob_from_counts, ReadoutSpec, READOUT_REALISTIC, READOUT_PERFECT, READOUT_SYM,
    build_drift_noise_model, NOMINAL_PARAMS,
)

checks = []
SH = 40_000


def counts_of(backend, seed, shots=SH):
    qc = make_benchmark_circuit(seed).copy(); qc.measure_all()
    return backend.run(qc, shots=shots).result().get_counts()


tgt = target_key(0)
gate_bk = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model())   # composite gate, no readout
ideal_bk = AerSimulator()

# (1) projector identity: calib == true -> raw gate P(target)
cg = counts_of(gate_bk, 0)
p_direct = cg.get(tgt, 0) / SH
p_ident = trex_success_prob_from_counts(cg, SH, tgt, READOUT_REALISTIC, READOUT_REALISTIC)
c1 = abs(p_direct - p_ident) < 1e-9
checks.append(c1)
print(f"(1) projector identity (calib==true == raw P): {p_direct:.5f} vs {p_ident:.5f}  "
      f"{'OK' if c1 else '*** CHECK'}")

# (2) readout-only on an IDEAL gate (point mass on target)
ci = counts_of(ideal_bk, 0)
drift_ro = ReadoutSpec("drift", p0=(0.03, 0.04), p1=(0.05, 0.06))  # drifted (worse) readout
adaptive = trex_success_prob_from_counts(ci, SH, tgt, drift_ro, drift_ro)
noisy = trex_success_prob_from_counts(ci, SH, tgt, drift_ro, READOUT_PERFECT)
static_nom = trex_success_prob_from_counts(ci, SH, tgt, READOUT_REALISTIC, READOUT_REALISTIC)
static_stale = trex_success_prob_from_counts(ci, SH, tgt, drift_ro, READOUT_REALISTIC)
c2 = (abs(adaptive - 1) < 1e-9 and noisy < 0.99 and abs(static_nom - 1) < 1e-9
      and abs(static_stale - 1) > 0.005)
checks.append(c2)
print(f"(2) readout-only: adaptive={adaptive:.4f} noisy={noisy:.4f} static@nom={static_nom:.4f} "
      f"static_stale={static_stale:.4f}  {'OK' if c2 else '*** CHECK'}")

# (3) symmetric-readout cross-check: analytic noisy == direct Aer (twirled==untwirled for p0==p1)
direct_sym = success_prob(AerSimulator(noise_model=READOUT_SYM.readout_noise_model()), 0, SH)
analytic_sym = trex_success_prob_from_counts(ci, SH, tgt, READOUT_SYM, READOUT_PERFECT)
c3 = abs(direct_sym - analytic_sym) < 0.01
checks.append(c3)
print(f"(3) symmetric cross-check: direct Aer={direct_sym:.4f} vs analytic={analytic_sym:.4f}  "
      f"{'OK' if c3 else '*** CHECK'}")

# (4) mitiq cross-check: analytic adaptive (= gate floor) ~ real mitiq TREX on gate+readout backend
comp_ro = AerSimulator(noise_model=build_drift_noise_model(NOMINAL_PARAMS))  # composite gate + readout
analytic_adaptive = trex_success_prob_from_counts(cg, SH, tgt, READOUT_REALISTIC, READOUT_REALISTIC)
mitiq_trex = trex_success_prob(benchmark_circuit_cirq(0),
                               mitiq_measurement_executor(comp_ro, SH), tgt,
                               num_randomizations=48, random_state=0)
c4 = abs(analytic_adaptive - mitiq_trex) < 0.03
checks.append(c4)
print(f"(4) mitiq cross-check: analytic adaptive={analytic_adaptive:.4f} vs mitiq TREX={mitiq_trex:.4f} "
      f"(both = gate floor)  {'OK' if c4 else '*** CHECK'}")

# (5) staleness grows with readout drift; adaptive stays ~1
print("(5) staleness sweep (ideal gate, static calib = nominal):")
print(f"    {'scale':>6s} {'adaptive':>9s} {'static':>8s} {'noisy':>8s}")
prev = 0.0
mono = True
for scale in [1.0, 1.5, 2.0, 3.0]:
    ro = ReadoutSpec("d", tuple(scale * p for p in READOUT_REALISTIC.p0),
                     tuple(scale * p for p in READOUT_REALISTIC.p1))
    a = trex_success_prob_from_counts(ci, SH, tgt, ro, ro)
    s = trex_success_prob_from_counts(ci, SH, tgt, ro, READOUT_REALISTIC)
    nz = trex_success_prob_from_counts(ci, SH, tgt, ro, READOUT_PERFECT)
    dev = abs(s - 1.0)
    mono = mono and (dev >= prev - 1e-12) and abs(a - 1) < 1e-9
    prev = dev
    print(f"    {scale:>6.1f} {a:>9.4f} {s:>8.4f} {nz:>8.4f}")
checks.append(mono)
print(f"    adaptive~1 & |static-1| non-decreasing  {'OK' if mono else '*** CHECK'}")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

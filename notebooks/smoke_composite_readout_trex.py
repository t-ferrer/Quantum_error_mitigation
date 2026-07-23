"""Smoke test: full-stack mitigation on composite gate noise + realistic readout.

Full hardware model = SPEC_COMPOSITE gate noise (ThermalRealistic o MixedUnitary) PLUS a
realistic-readout add-on layer (READOUT_REALISTIC, IBM-like ~1-2% asymmetric). Validates the
COMPLEMENTARITY of the two mitigation families:

  - noisy (gate+readout)       : worst.
  - ZNE-only (gate mitigation) : removes gate noise but LEAVES the readout floor -- folding does
    not scale the measurement error, so the readout bias is scale-independent and survives the
    extrapolation.
  - TREX-only (readout mit.)   : removes readout but LEAVES gate noise -> lands at the gate-only
    level (~ composite noisy P without readout).
  - TREX o ZNE                 : TREX-corrected executor fed into ZNE -> both removed -> ~1.0.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from qiskit_aer import AerSimulator
from mitiq import zne
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import ExpFactory, ExtrapolationError

from qem import (SPEC_COMPOSITE, make_benchmark_circuit, benchmark_circuit_cirq, mitiq_executor,
                 mitiq_measurement_executor, trex_success_prob, trex_mitigated_executor,
                 target_key, success_prob, IDEAL)
from qem.readout import READOUT_REALISTIC

SEEDS = range(4)
NR = 12
SHOTS = 8000
SF = [1.0, 1.5, 2.0, 3.0, 5.0]  # 5 points -> the non-adaptive ExpFactory(asymptote=None) fit is
#   robust here (it converged 20/20 on the composite benchmark; only AdaExp failed 3/20). The
#   exponential ansatz matches the gate-noise decay toward the mixed fixed point, so it recovers
#   ~1.0 -- unlike Poly2, which under-corrects over a short scale range.


def _zne(circ, executor):
    try:
        return zne.execute_with_zne(circ, executor, scale_noise=fold_global,
                                    factory=ExpFactory(scale_factors=SF, asymptote=None))
    except ExtrapolationError:
        return float("nan")


# composite gate noise + realistic readout add-on (SPEC_COMPOSITE stays gate-only)
nm_both = SPEC_COMPOSITE.build_noise_model()
READOUT_REALISTIC.add_readout_error(nm_both)
both = AerSimulator(noise_model=nm_both)
gate = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model())  # gate-only reference

print(f"readout REALISTIC: p0={READOUT_REALISTIC.p0}  p1={READOUT_REALISTIC.p1}  "
      f"(assignment err ~{[round(50*(a+b),2) for a, b in zip(READOUT_REALISTIC.p0, READOUT_REALISTIC.p1)]} %)")
print(f"{'seed':4s} {'noisy':>8s} {'gate_only':>9s} {'ZNE_only':>9s} {'TREX_only':>9s} {'TREXoZNE':>9s}")

rows = []
for s in SEEDS:
    circ = make_benchmark_circuit(s)
    tgt = target_key(s)
    noisy = success_prob(both, s)
    gate_only = success_prob(gate, s)
    zne_only = _zne(circ, mitiq_executor(both, s, SHOTS))                       # readout floor
    trex_only = trex_success_prob(benchmark_circuit_cirq(s),                    # gate floor
                                  mitiq_measurement_executor(both, SHOTS), tgt,
                                  num_randomizations=NR, random_state=s)
    trex_zne = _zne(circ, trex_mitigated_executor(both, tgt, SHOTS,             # both removed
                                                  num_randomizations=NR, random_state=s))
    rows.append((noisy, gate_only, zne_only, trex_only, trex_zne))
    print(f"{s:<4d} {noisy:8.4f} {gate_only:9.4f} {zne_only:9.4f} {trex_only:9.4f} {trex_zne:9.4f}")

a = np.array(rows)
noisy_m, gate_m, zne_m, trex_m, tz_m = np.nanmean(a, axis=0)
print(f"\nmeans: noisy={noisy_m:.4f}  gate_only={gate_m:.4f}  ZNE_only={zne_m:.4f}  "
      f"TREX_only={trex_m:.4f}  TREXoZNE={tz_m:.4f}  (ideal={IDEAL})")
c1 = zne_m > noisy_m and zne_m < tz_m - 0.015  # gate removed, but readout floor keeps it < full stack
c2 = abs(trex_m - gate_m) < 0.02               # TREX removes readout -> back to gate-only level
c3 = tz_m > 0.98 and tz_m > zne_m and tz_m > trex_m  # both removed -> ~1, beats either single tool
print(f"  (1) ZNE-only leaves readout floor (noisy < ZNE < TREXoZNE)? {c1}  (ZNE {zne_m:.3f})")
print(f"  (2) TREX-only ~ gate_only (|diff|<0.02)?                    {c2}  (diff {trex_m-gate_m:+.4f})")
print(f"  (3) TREXoZNE recovers ~1 and beats both singles?           {c3}  (TREXoZNE {tz_m:.3f})")
print(f"  {'OK' if c1 and c2 and c3 else '*** CHECK'}")

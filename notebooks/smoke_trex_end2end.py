"""Smoke test 4/5 for TREX: end-to-end readout-error mitigation of P(target).

Ties everything together on the actual mirror-circuit benchmark quantity P(target):

 (A) Endianness / observable sanity (NO noise): the observable path
     <projector_observable(target)> estimated by TREX must reproduce the DIRECT
     counts-based P(target) (~IDEAL=1.0). This catches any Qiskit<->Mitiq qubit/endianness
     mismatch in mitiq_measurement_executor or projector_observable BEFORE noise muddies it.
 (B) Readout-only noise (no gate noise): TREX must remove the readout bias and recover
     P(target) ~ IDEAL, for BOTH symmetric AND asymmetric readout. The asymmetric case is
     the real test -- twirling is what makes it work (cf. smoke_trex_lambda test 3).

TREX only mitigates READOUT error; here there is no gate noise, so the recovered value
should sit at the ideal up to shot noise.
"""

import numpy as np
from qiskit_aer import AerSimulator

from qem import (
    IDEAL,
    benchmark_circuit_cirq,
    make_benchmark_circuit,
    mitiq_executor,
    mitiq_measurement_executor,
    target_key,
    trex_success_prob,
)
from qem.readout import READOUT_SYM, READOUT_ASYM

SEEDS = range(6)
NUM_RAND = 32
SHOTS = 8192


print("=== (A) noiseless: TREX observable path reproduces direct P(target) ===")
ideal = AerSimulator()
gaps = []
for s in SEEDS:
    tkey = target_key(s)
    direct = mitiq_executor(ideal, s, 20000)(make_benchmark_circuit(s))  # P(target) from counts
    trex_val = trex_success_prob(
        benchmark_circuit_cirq(s), mitiq_measurement_executor(ideal, SHOTS), tkey,
        num_randomizations=NUM_RAND, random_state=s,
    )
    gaps.append(abs(direct - trex_val))
    print(f"  seed {s}: direct P={direct:.4f}  TREX(no-noise)={trex_val:.4f}  target={tkey}")
print(f"  max|direct - TREX| = {max(gaps):.4f}   "
      f"({'OK' if max(gaps) < 0.02 else '*** FAIL: endianness/observable bug'})")


for spec in (READOUT_SYM, READOUT_ASYM):
    print(f"\n=== (B) {spec.name}  p0={spec.p0} p1={spec.p1}: TREX removes readout bias ===")
    backend = AerSimulator(noise_model=spec.readout_noise_model())
    noisy_list, trex_list = [], []
    for s in SEEDS:
        noisy = mitiq_executor(backend, s, 20000)(make_benchmark_circuit(s))  # biased noisy P
        trex_val = trex_success_prob(
            benchmark_circuit_cirq(s), mitiq_measurement_executor(backend, SHOTS), target_key(s),
            num_randomizations=NUM_RAND, random_state=s,
        )
        noisy_list.append(noisy)
        trex_list.append(trex_val)
        print(f"  seed {s}: noisy={noisy:.4f}  TREX={trex_val:.4f}  (ideal={IDEAL})")
    nb = float(np.mean(noisy_list)) - IDEAL
    tb = float(np.mean(trex_list)) - IDEAL
    ok = abs(tb) < 0.03 and abs(tb) < abs(nb)
    print(f"  mean bias:  noisy={nb:+.4f}   TREX={tb:+.4f}   "
          f"({'OK: TREX unbiased & better' if ok else '*** CHECK'})")

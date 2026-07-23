"""Localize the +0.06 composite-PEC bias: channels vs run-path.

Per-gate reps reconstruct the ISOLATED Aer channels to ~1e-7 (smoke_composite_pec_unbias.py),
yet end-to-end PEC is biased +0.06 (confirmed at high samples). So the bias is NOT per-gate
channel inexactness -- it is composition/run-path.

Decisive test: run the SAME PEC but with an EXACT density-matrix executor (no shot noise; same
composite noise model). PEC combines these exactly.
  - If exact-executor PEC ~ 1.0  -> channels compose exactly; the shot-run bias comes from the
    executor's run path (e.g. transpilation of inserted y/z into NOISED basis gates, or reset).
  - If exact-executor PEC ~ +0.06 -> the noise applied to the sampled (insertion-bearing) circuit
    differs from the per-gate isolated channels -> a genuine composition mismatch.
"""

import numpy as np
from mitiq import pec
from qiskit_aer import AerSimulator

from qem import SPEC_COMPOSITE, make_benchmark_circuit, target_key, mitiq_executor, IDEAL
from qem.benchmark import _assert_all_ops_represented

_NM = SPEC_COMPOSITE.build_noise_model()
_DM = AerSimulator(method="density_matrix", noise_model=_NM)
_DM_IDEAL = AerSimulator(method="density_matrix")


def exact_prob(circuit, target, sim) -> float:
    qc = circuit.copy()
    qc.save_density_matrix()
    rho = sim.run(qc).result().data()["density_matrix"]
    return float(rho.probabilities_dict().get(target, 0.0))


def make_exact_executor(seed, sim=_DM):
    tgt = target_key(seed)
    return lambda circuit: exact_prob(circuit, tgt, sim)


print("=== sanity: ideal density-matrix P(target) ~ 1 ===")
for s in (0, 2):
    circ = make_benchmark_circuit(s)
    print(f"  seed {s}: ideal P(target)={exact_prob(circ, target_key(s), _DM_IDEAL):.4f}  "
          f"noisy P(target)={exact_prob(circ, target_key(s), _DM):.4f}")

print("\n=== PEC with EXACT density-matrix executor (no shot noise) ===")
for s in (0, 2):
    circ = make_benchmark_circuit(s)
    reps = SPEC_COMPOSITE.build_representations(circ)
    _assert_all_ops_represented(circ, reps)
    ex = make_exact_executor(s)
    val, data = pec.execute_with_pec(
        circ, ex, representations=reps, num_samples=1500, full_output=True
    )
    print(f"  seed {s}: exact-exec PEC={float(val):.4f} +/- {data['pec_error']:.4f}  "
          f"bias={float(val) - IDEAL:+.4f}")

print("\n(Compare to the shot-based high-sample run: seed0 +0.055, seed2 +0.079.)")
print("exact-exec ~0  => run-path/transpilation bias.   exact-exec ~+0.06 => composition.")

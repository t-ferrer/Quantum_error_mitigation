"""Where does the +0.06 composite-PEC bias REALLY come from?

smoke_composite_pec_unbias.py showed the per-gate reps reconstruct the EXACT Aer channels to
~1e-7 -> per-gate PEC is unbiased. Yet the 20-seed run shows PEC bias +0.06. Two suspects:
  (A) the bias is a low-sample artifact (vanishes at high num_samples), or
  (B) mitiq's sampled circuits noise the inserted Paulis differently than the isolated channels
      (e.g. inserted y/z get decomposed into NOISED basis gates).

Test (B): inspect what mitiq actually samples -- gate names of a sampled circuit.
Test (A): high-sample end-to-end PEC on a couple of seeds (low Monte-Carlo error).
"""

import numpy as np
from mitiq import pec
from mitiq.pec.sampling import sample_circuit
from qiskit_aer import AerSimulator

from qem import SPEC_COMPOSITE, make_benchmark_circuit, mitiq_executor, IDEAL
from qem.benchmark import _assert_all_ops_represented

# SEEDLESS on purpose. NOTE (added after this file was written): the answer turned out to be
# NEITHER (A) nor (B) -- the +0.06 came from the fixed `seed_simulator=42` this test itself used.
# A frozen seed restarts every backend.run() from the same noise RNG stream, correlating PEC's
# thousands of importance samples -> biased estimator. Fixed in run_benchmark; fixed here too.
backend = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model())

# ---------------------------------------------------------------------------
print("=== (B) what gate names appear in a mitiq-sampled PEC circuit? ===")
circ0 = make_benchmark_circuit(0)
reps = SPEC_COMPOSITE.build_representations(circ0)
_assert_all_ops_represented(circ0, reps)
print("  original circuit gate names :", dict(circ0.count_ops()))
sampled, sign, norm = sample_circuit(circ0, reps, num_samples=1)
s0 = sampled[0]
try:
    from mitiq.interface import convert_to_mitiq
    cirq_circ, _ = convert_to_mitiq(s0)
    names = {}
    for op in cirq_circ.all_operations():
        g = str(op.gate)
        names[g] = names.get(g, 0) + 1
    print("  sampled circuit op types    :", names)
except Exception as e:
    print("  (cirq introspection failed:", e, ")")
# If the sampled circuit is qiskit, count ops directly
if hasattr(s0, "count_ops"):
    print("  sampled circuit count_ops   :", dict(s0.count_ops()))
print("  -> if 'y'/'z'/'reset' appear as-is, Aer leaves them UNNOISED (matches the per-gate")
print("     reconstruction). If they were decomposed into sx/rz/x, they'd be (partly) noised.")

# ---------------------------------------------------------------------------
print("\n=== (A) high-sample end-to-end PEC (low MC error) -- is the +0.06 bias real? ===")
for s in (0, 2):
    circ = make_benchmark_circuit(s)
    reps = SPEC_COMPOSITE.build_representations(circ)
    ex = mitiq_executor(backend, s, 200)            # 200 shots/sample
    val, data = pec.execute_with_pec(
        circ, ex, representations=reps, num_samples=6000, full_output=True
    )
    noisy = mitiq_executor(backend, s, 20000)(circ)
    print(f"  seed {s}: noisy={noisy:.4f}  PEC={float(val):.4f} +/- {data['pec_error']:.4f}  "
          f"(ideal={IDEAL})  -> bias={float(val)-IDEAL:+.4f}")
print("  (6000 samples x 200 shots -> pec_error ~0.015; if PEC stays ~+0.06 it's a REAL bias,")
print("   if it drops to ~0 the earlier estimate was low-sample noise.)")

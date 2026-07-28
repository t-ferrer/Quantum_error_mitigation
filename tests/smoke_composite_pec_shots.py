"""Few-shots-per-sample PEC bias? Disentangle num_samples vs shots/sample.

Conflicting evidence on seed 0:
  exact-executor (no shots, 1500 samp) = 1.010   <- true value
  15 reps @ 300 samples x 100 shots     = 1.019
  1 run  @ 3000 samples x 10 shots      = 1.182   <- much higher!

Same seed, same reps -> the only difference is shots/sample (10 vs 100 vs inf). Test directly:
fix num_samples, vary shots, several reps each, and compare the MEAN. If shots=10 mean >> shots=100
mean ~ exact, then few-shots-per-sample introduces a POSITIVE PEC bias (so the fix is MORE shots
per sample, i.e. AGAINST the few-shots/many-samples split -- not a rep change).
"""

import numpy as np
from mitiq import pec
from qiskit_aer import AerSimulator

from qem import SPEC_COMPOSITE, make_benchmark_circuit, mitiq_executor

backend = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model())  # independent draws


def pec_runs(seed, num_samples, shots, reps):
    circ = make_benchmark_circuit(seed)
    repsr = SPEC_COMPOSITE.build_representations(circ)
    vals = []
    for _ in range(reps):
        ex = mitiq_executor(backend, seed, shots)
        vals.append(float(pec.execute_with_pec(
            circ, ex, representations=repsr, num_samples=num_samples)))
    return np.mean(vals), np.std(vals), [round(v, 3) for v in vals]


for seed in (0, 2):
    print(f"--- seed {seed} (true value from exact-exec ~ {1.010 if seed==0 else 1.026}) ---")
    for ns, sh, reps in [(3000, 10, 4), (3000, 100, 4), (3000, 1000, 2)]:
        m, s, vals = pec_runs(seed, ns, sh, reps)
        print(f"  num_samples={ns:5d} shots/sample={sh:5d} | mean={m:.4f} (bias {m-1:+.4f}) "
              f"std={s:.4f} | runs={vals}")
print("\nIf mean rises as shots/sample falls -> few-shots PEC bias confirmed (fix: more shots/sample).")

"""Is the composite-PEC '+0.06 bias' actually ESTIMATOR VARIANCE?

Evidence so far: per-gate reps reconstruct exact Aer channels to 1e-7; shot and density-matrix
executors agree to 1e-5; yet single PEC runs scatter from +0.005 to +0.079 -- all reporting
pec_error ~0.01-0.02. That scatter >> reported error => mitiq's pec_error UNDERESTIMATES the true
variance, and a lone estimate of +0.06 is just a high draw.

Test: repeat PEC many times (independent RNG) at the notebook budget (300 samples x 100 shots) and
compare the empirical std of the estimate to mitiq's self-reported pec_error. Also report the mean
(~ the TRUE bias). If mean ~0 and empirical std >> pec_error, the '+0.06' was variance, not bias.
"""

import numpy as np
from mitiq import pec
from qiskit_aer import AerSimulator

from qem import SPEC_COMPOSITE, make_benchmark_circuit, mitiq_executor, IDEAL

backend = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model())  # no fixed seed -> independent draws
N_REP = 15

for s in (0, 2):
    circ = make_benchmark_circuit(s)
    reps = SPEC_COMPOSITE.build_representations(circ)
    ests, errs = [], []
    for _ in range(N_REP):
        ex = mitiq_executor(backend, s, 100)
        val, data = pec.execute_with_pec(
            circ, ex, representations=reps, num_samples=300, full_output=True
        )
        ests.append(float(val)); errs.append(data["pec_error"])
    ests = np.array(ests)
    sem = ests.std() / np.sqrt(N_REP)
    print(f"seed {s}: {N_REP} reps @300x100 | mean={ests.mean():.4f} "
          f"(bias {ests.mean() - IDEAL:+.4f} +/- {sem:.4f}) | "
          f"empirical std={ests.std():.4f} | mitiq pec_error~{np.mean(errs):.4f} | "
          f"range=[{ests.min():.3f},{ests.max():.3f}]")

print("\nIf empirical std >> mitiq pec_error and mean-bias ~0 (within SEM): the '+0.06' was")
print("ESTIMATOR VARIANCE (underestimated by mitiq), not a systematic bias. Fix = more samples")
print("(revert to the low-variance 3000x10 split), NOT a rep change.")

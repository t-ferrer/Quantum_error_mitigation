"""The +0.06 PEC bias lives in the SHOT executor's run-path, not the reps.

(PEC with an exact density-matrix executor is ~unbiased: +0.01/+0.03.) Here we evaluate the
SAME mitiq-sampled circuits with BOTH executors to isolate exactly where they diverge:

  est_shot - est_dm = norm * mean_i sign_i (shot(c_i) - dm(c_i))

If est_shot is biased (+0.06) and est_dm is not, the per-circuit P(target) must differ
systematically between the two run-paths -> we then look at WHICH sampled circuits differ
(those carrying inserted Paulis / reset) to name the mechanism.
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from mitiq.pec.sampling import sample_circuit

from qem import SPEC_COMPOSITE, make_benchmark_circuit, target_key, IDEAL
from qem.config import BASIS_GATES

_NM = SPEC_COMPOSITE.build_noise_model()
_SHOT = AerSimulator(noise_model=_NM)
_DM = AerSimulator(method="density_matrix", noise_model=_NM)


def shot_prob(circuit, target, shots=4000):
    qc = circuit.copy(); qc.measure_all()
    counts = _SHOT.run(qc, shots=shots).result().get_counts()
    return counts.get(target, 0) / shots


def dm_prob(circuit, target):
    qc = circuit.copy(); qc.save_density_matrix()
    rho = _DM.run(qc).result().data()["density_matrix"]
    return float(rho.probabilities_dict().get(target, 0.0))


# direct evidence: does transpiling an inserted Pauli to basis gates turn it into NOISED gates?
print("=== does y/z/reset decompose into noised basis gates? ===")
from qiskit import QuantumCircuit
for g in ("y", "z"):
    qc = QuantumCircuit(1); getattr(qc, g)(0)
    qct = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    print(f"  {g} -> basis {dict(qct.count_ops())}  (noised gates: sx/sxdg/x/h)")
print(f"  BASIS_GATES = {BASIS_GATES}; AerSimulator runs y/z NATIVELY unless transpiled.")

print("\n=== same sampled circuits, two executors (seed 0, 2) ===")
for s in (0, 2):
    circ = make_benchmark_circuit(s)
    reps = SPEC_COMPOSITE.build_representations(circ)
    tgt = target_key(s)
    out = sample_circuit(circ, reps, num_samples=1200)
    sampled, signs, norm = out[0], out[1], out[2]

    shot_vals = np.array([shot_prob(c, tgt) for c in sampled])
    dm_vals = np.array([dm_prob(c, tgt) for c in sampled])
    signs = np.array(signs, dtype=float)

    est_shot = norm * np.mean(signs * shot_vals)
    est_dm = norm * np.mean(signs * dm_vals)
    per_circ = shot_vals - dm_vals
    print(f"  seed {s}: norm={norm:.3f}  est_shot={est_shot:.4f} (bias {est_shot-IDEAL:+.4f})  "
          f"est_dm={est_dm:.4f} (bias {est_dm-IDEAL:+.4f})")
    print(f"           mean(shot-dm) over circuits = {per_circ.mean():+.5f}  "
          f"(shot noise alone -> ~0); |mean| signals a run-path channel difference")

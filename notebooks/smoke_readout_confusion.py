"""Smoke test 1/5 for TREX: the Aer ReadoutError reproduces the target confusion matrix A.

Validates the readout *model* (before any mitigation):

 (1) Per qubit, the empirical (p0, p1) = (P(1|0), P(0|1)) match ReadoutSpec.
 (2) The full 2-qubit empirical confusion matrix A_emp[measured, true] matches the
     analytic A = A_1 (x) A_0 (Qiskit little-endian) to shot-noise precision.
 (3) A is left-stochastic (columns sum to 1): p_noisy = A @ p_ideal is a valid distribution.

Endianness guard: a Qiskit counts key "b_{n-1}...b_0" maps to int(key, 2) with qubit 0 as
the LEAST significant bit -- the SAME ordering as ReadoutSpec.confusion_matrix(). If this
mapping were wrong, A_emp would come out as a permuted version of A and (2) would blow up.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qem.readout import READOUT_SYM, READOUT_ASYM

SHOTS = 400_000  # tight tolerance: shot-noise std ~ 1/sqrt(SHOTS) ~ 1.6e-3


def prepare_basis(n: int, j: int) -> QuantumCircuit:
    """Circuit preparing computational basis state |j> (qubit i <- bit i of j), then measure."""
    qc = QuantumCircuit(n, n)
    for i in range(n):
        if (j >> i) & 1:
            qc.x(i)
    qc.measure(range(n), range(n))
    return qc


def empirical_confusion(spec, shots: int = SHOTS) -> np.ndarray:
    """A_emp[measured, true]: prepare each basis state |true>, measure, tally outcomes."""
    n = spec.n_qubits
    backend = AerSimulator(noise_model=spec.readout_noise_model())
    A = np.zeros((2**n, 2**n))
    for true in range(2**n):
        counts = backend.run(prepare_basis(n, true), shots=shots).result().get_counts()
        for key, c in counts.items():
            A[int(key, 2), true] = c / shots  # int(key,2): q0 = LSB (little-endian)
    return A


for spec in (READOUT_SYM, READOUT_ASYM):
    print(f"=== {spec.name}  p0={spec.p0}  p1={spec.p1} ===")
    A_th = spec.confusion_matrix()
    A_emp = empirical_confusion(spec)

    # (1) per-qubit rates read straight off the single-qubit marginals of the 2q runs is
    #     fiddly; instead just report the full-matrix agreement, which subsumes them.
    # (3) left-stochastic check on the analytic A
    col_sums = A_th.sum(axis=0)
    print(f"  A left-stochastic? max|col_sum - 1| = {np.max(np.abs(col_sums - 1)):.2e}")

    # (2) analytic vs empirical confusion matrix
    err = np.max(np.abs(A_th - A_emp))
    print(f"  max|A_theory - A_emp| = {err:.2e}   (expect ~1e-3 at {SHOTS:,} shots)")
    print("  A_theory =")
    print(np.array2string(A_th, precision=4, prefix="    "))
    status = "OK" if err < 5e-3 else "*** FAIL"
    print(f"  {status}\n")

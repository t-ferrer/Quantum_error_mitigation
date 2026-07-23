"""Smoke test 2/5 for TREX: X-twirling symmetrizes the confusion matrix.

The core theoretical move of TREX. Sample a random bitstring s, apply X^s just before
measurement, and classically correct the outcome by y = x XOR s. Averaged over s this
maps the confusion matrix A -> the twirled matrix

    A* = (1/2^n) sum_s P_s A P_s ,   P_s[a,b] = 1 iff a = b XOR s.

Claims validated here:

 (1) A* factorizes per qubit into a SYMMETRIC bit-flip with p_bar_i = (p0_i + p1_i)/2:
        A* == kron_i confusion_matrix_1q(p_bar_i, p_bar_i).
 (2) A* is symmetric (A* == A*^T), even when the raw A is NOT (asymmetric readout).
     -> the state-independent OFFSET that asymmetric readout puts on <Z> is removed;
        what remains is a pure eigenvalue scaling (that eigenvalue is measured in test 3).
 (3) The empirical twirl (random s, XOR post-processing, on the Aer backend) reproduces
     the analytic A* to shot-noise precision -> the procedure TREX runs is faithful.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qem.readout import READOUT_SYM, READOUT_ASYM, confusion_matrix_1q


def perm_xor(n: int, s: int) -> np.ndarray:
    """Permutation matrix P_s with P_s[a,b] = 1 iff a = b XOR s (bit-flip by pattern s)."""
    P = np.zeros((2**n, 2**n))
    for b in range(2**n):
        P[b ^ s, b] = 1.0
    return P


def twirled_confusion_analytic(spec) -> np.ndarray:
    """A* = (1/2^n) sum_s P_s A P_s, brute-forced over all 2^n twirl patterns."""
    n = spec.n_qubits
    A = spec.confusion_matrix()
    Astar = np.zeros_like(A)
    for s in range(2**n):
        P = perm_xor(n, s)
        Astar += P @ A @ P
    return Astar / 2**n


def twirled_confusion_symmetrized(spec) -> np.ndarray:
    """Predicted A* = kron_i confusion_matrix_1q(p_bar_i, p_bar_i), p_bar=(p0+p1)/2."""
    M = np.array([[1.0]])
    for a, b in zip(spec.p0, spec.p1):
        pbar = 0.5 * (a + b)
        M = np.kron(confusion_matrix_1q(pbar, pbar), M)  # q0 least significant
    return M


def twirled_confusion_emp(spec, n_twirls=128, shots=2000, seed=0) -> np.ndarray:
    """Monte-Carlo twirl on the Aer backend: random s, apply X^s, measure, XOR the outcome."""
    n = spec.n_qubits
    backend = AerSimulator(noise_model=spec.readout_noise_model())
    rng = np.random.default_rng(seed)
    Astar = np.zeros((2**n, 2**n))
    for true in range(2**n):
        for _ in range(n_twirls):
            s = int(rng.integers(0, 2**n))
            qc = QuantumCircuit(n, n)
            for i in range(n):
                if (true >> i) & 1:
                    qc.x(i)
            for i in range(n):  # twirl: X^s before measurement
                if (s >> i) & 1:
                    qc.x(i)
            qc.measure(range(n), range(n))
            counts = backend.run(qc, shots=shots).result().get_counts()
            for key, c in counts.items():
                y = int(key, 2) ^ s  # classical XOR correction
                Astar[y, true] += c
        Astar[:, true] /= n_twirls * shots
    return Astar


for spec in (READOUT_SYM, READOUT_ASYM):
    print(f"=== {spec.name}  p0={spec.p0}  p1={spec.p1} ===")
    A = spec.confusion_matrix()
    Astar = twirled_confusion_analytic(spec)
    Astar_pred = twirled_confusion_symmetrized(spec)
    Astar_emp = twirled_confusion_emp(spec)

    asym_raw = np.max(np.abs(A - A.T))
    asym_tw = np.max(np.abs(Astar - Astar.T))
    print(f"  raw A asymmetry   max|A - A^T|   = {asym_raw:.2e}")
    print(f"  twirled asymmetry max|A* - A*^T| = {asym_tw:.2e}   (expect ~0)")

    err_pred = np.max(np.abs(Astar - Astar_pred))
    print(f"  (1) A* == kron symmetrized bit-flip:  max diff = {err_pred:.2e}")

    err_emp = np.max(np.abs(Astar - Astar_emp))
    print(f"  (3) empirical twirl vs analytic A*:   max diff = {err_emp:.2e}   (shot noise)")

    ok = asym_tw < 1e-12 and err_pred < 1e-12 and err_emp < 5e-3
    print(f"  {'OK' if ok else '*** FAIL'}\n")

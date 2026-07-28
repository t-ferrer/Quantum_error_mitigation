"""Smoke test 3/5 for TREX: the calibrated eigenvalue lambda_w matches theory.

This is the crux of TREX. For each Pauli support w (set of qubits carrying a Z), TREX
runs a CALIBRATION circuit -- prepare |0...0>, apply the SAME measurement twirl, measure,
XOR-correct -- and estimates

    lambda_w = < prod_{i in w} (-1)^{y_i} >         (true parity is +1 on |0...0>).

The corrected expectation is then <P>_true = <P>_noisy,twirled / lambda_w.

Claims validated:

 (1) Twirled calibration recovers the analytic  lambda_w = prod_{i in w} (1 - p0_i - p1_i)
     (symmetrized bit-flip eigenvalue), to shot-noise precision -- for BOTH symmetric and
     asymmetric readout.
 (2) Support factorization: lambda_{01} == lambda_0 * lambda_1.
 (3) Contrast: the UNTWIRLED calibration on |0...0> gives lambda_raw_i = 1 - 2*p0_i, which
     sees only p0 (not the (p0+p1) symmetrized rate). For asymmetric readout this is the
     WRONG eigenvalue -> dividing by it would leave a residual bias. This is *why* TREX
     twirls the calibration, not just the main circuit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qem.readout import READOUT_SYM, READOUT_ASYM

SUPPORTS = [(0,), (1,), (0, 1)]


def _mask(support) -> int:
    m = 0
    for i in support:
        m |= 1 << i
    return m


def parity(y: int, mask: int) -> float:
    """prod_{i in support}(-1)^{y_i} = (-1)^popcount(y & mask)."""
    return -1.0 if bin(y & mask).count("1") % 2 else 1.0


def lambda_exact(spec, support, twirl: bool) -> float:
    """Exact (shot-noise-free) calibrated eigenvalue from the confusion matrix.

    Column true=0 of A (twirled A* if twirl=True) is P(measured y | prepared |0..0>);
    lambda = sum_y P(y) * parity(y). twirl=False reproduces the untwirled calibration.
    """
    n = spec.n_qubits
    A = spec.confusion_matrix()
    if twirl:
        A = sum(
            _perm(n, s) @ A @ _perm(n, s) for s in range(2**n)
        ) / 2**n
    col0 = A[:, 0]
    m = _mask(support)
    return float(sum(col0[y] * parity(y, m) for y in range(2**n)))


def _perm(n: int, s: int) -> np.ndarray:
    P = np.zeros((2**n, 2**n))
    for b in range(2**n):
        P[b ^ s, b] = 1.0
    return P


def lambda_twirled_emp(spec, supports, shots=100_000) -> dict:
    """Twirled calibration on the Aer backend, for all supports at once.

    Enumerates ALL 2^n twirl patterns with EQUAL weight (the exact uniform twirl at the
    ensemble level) rather than Monte-Carlo sampling s: with only 2^n=4 patterns here,
    random sampling would leave a pattern-reweighting variance ON TOP of shot noise and
    make the check flaky. Uniform enumeration isolates pure shot noise -> a clean theory
    test. (TREX itself samples patterns; that sampling variance is a mitigation-cost
    question, exercised in the end-to-end test, not here.)
    """
    n = spec.n_qubits
    backend = AerSimulator(noise_model=spec.readout_noise_model())
    masks = {w: _mask(w) for w in supports}
    num = {w: 0.0 for w in supports}
    den = 0
    for s in range(2**n):  # uniform over all twirl patterns
        qc = QuantumCircuit(n, n)  # prepare |0...0>
        for i in range(n):  # measurement twirl X^s
            if (s >> i) & 1:
                qc.x(i)
        qc.measure(range(n), range(n))
        counts = backend.run(qc, shots=shots).result().get_counts()
        for key, c in counts.items():
            y = int(key, 2) ^ s  # XOR-correct
            for w in supports:
                num[w] += parity(y, masks[w]) * c
            den += c
    return {w: num[w] / den for w in supports}  # den = 2^n * shots (uniform twirl avg)


for spec in (READOUT_SYM, READOUT_ASYM):
    print(f"=== {spec.name}  p0={spec.p0}  p1={spec.p1} ===")
    print(f"  {'support':8s} {'analytic':>10s} {'exact_tw':>10s} {'emp_tw':>10s} "
          f"{'raw(untw)':>10s}")
    lam = lambda_twirled_emp(spec, SUPPORTS)  # empirical twirled calibration, all supports
    for w in SUPPORTS:
        an = spec.lambda_support(w)          # prod (1 - p0 - p1)
        ex = lambda_exact(spec, w, twirl=True)
        raw = lambda_exact(spec, w, twirl=False)
        print(f"  {str(w):8s} {an:10.5f} {ex:10.5f} {lam[w]:10.5f} {raw:10.5f}")

    # (1) empirical twirled calibration vs analytic
    err = max(abs(spec.lambda_support(w) - lam[w]) for w in SUPPORTS)
    # (2) support factorization on the empirical estimates
    fact = abs(lam[(0, 1)] - lam[(0,)] * lam[(1,)])
    # (3) raw calibration bias (should be > 0 only for asymmetric)
    raw_bias = max(abs(spec.lambda_support(w) - lambda_exact(spec, w, twirl=False))
                   for w in SUPPORTS)
    print(f"  (1) max|emp_tw - analytic|      = {err:.2e}   (shot noise)")
    print(f"  (2) |lam_01 - lam_0*lam_1|      = {fact:.2e}   (support factorization)")
    print(f"  (3) raw-vs-twirled lambda gap   = {raw_bias:.2e}   "
          f"({'~0: symmetric, twirl not needed' if raw_bias < 1e-9 else 'asym: twirl REQUIRED'})")
    print(f"  {'OK' if err < 5e-3 and fact < 5e-3 else '*** FAIL'}\n")

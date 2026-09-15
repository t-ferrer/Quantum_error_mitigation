"""Readout (measurement) error models + helpers, for TREX mitigation.

Readout error is a *classical* bit-flip on the measurement outcome, independent of
the gate noise: with true outcome bit b, the reported bit is flipped with probability
p0 (0->1) or p1 (1->0). It is NOT a gate channel, so it has NO PEC representation; it
is mitigated by TREX (Twirled Readout Error eXtinction).

Conventions
-----------
Per qubit i we store (p0_i, p1_i):
    p0_i = P(measure 1 | true 0)   ("excitation" readout error)
    p1_i = P(measure 0 | true 1)   ("relaxation" readout error, usually the larger one)

TREX left-stochastic confusion matrix A  (p_noisy = A @ p_ideal, A[measured, true]):
    1q:  A = [[1 - p0,   p1  ],
              [  p0  , 1 - p1]]
Aer's ReadoutError wants probabilities[true][measured] = A.T:
    ReadoutError([[1 - p0,   p0  ],
                  [  p1  , 1 - p1]])

Twirled eigenvalue (readout twirling symmetrizes p0,p1 -> p_bar = (p0+p1)/2):
    lambda_i = 1 - 2*p_bar = 1 - (p0_i + p1_i)   for a Z on qubit i,
    lambda_w = prod_{i in support w} lambda_i.
This is the exact quantity TREX estimates by calibration and divides out.
"""

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
from qiskit_aer.noise import NoiseModel, ReadoutError

__all__ = [
    "readout_error_1q",
    "confusion_matrix_1q",
    "ReadoutSpec",
    "READOUT_SYM",
    "READOUT_ASYM",
    "READOUT_REALISTIC",
    "READOUT_PERFECT",
    "trex_success_prob_from_counts",
    "trex_success_prob_experimental",
]


def readout_error_1q(p0: float, p1: float) -> ReadoutError:
    """Aer 1-qubit ReadoutError with P(1|0)=p0, P(0|1)=p1 (rows=true, cols=measured)."""
    return ReadoutError([[1.0 - p0, p0], [p1, 1.0 - p1]])


def confusion_matrix_1q(p0: float, p1: float) -> np.ndarray:
    """TREX left-stochastic A (A[measured, true]) for one qubit: p_noisy = A @ p_ideal."""
    return np.array([[1.0 - p0, p1], [p0, 1.0 - p1]])


@dataclass(frozen=True)
class ReadoutSpec:
    """Independent per-qubit bit-flip readout error, parameterised by (p0, p1) per qubit."""

    name: str
    p0: tuple[float, ...]  # P(measure 1 | true 0), one entry per qubit
    p1: tuple[float, ...]  # P(measure 0 | true 1), one entry per qubit

    def __post_init__(self):
        if len(self.p0) != len(self.p1):
            raise ValueError("p0 and p1 must have one entry per qubit (equal length)")

    @property
    def n_qubits(self) -> int:
        return len(self.p0)

    def readout_noise_model(self) -> NoiseModel:
        """Fresh NoiseModel carrying ONLY this readout error (no gate noise)."""
        return self.add_readout_error(NoiseModel())

    def add_readout_error(self, nm: NoiseModel) -> NoiseModel:
        """Attach this readout error to an existing (gate-noise) NoiseModel, in place."""
        for q, (a, b) in enumerate(zip(self.p0, self.p1)):
            nm.add_readout_error(readout_error_1q(a, b), [q])
        return nm

    def confusion_matrix(self) -> np.ndarray:
        """Full 2^n x 2^n TREX A = A_{n-1} (x) ... (x) A_0 (Qiskit little-endian, q0 = LSB)."""
        A = np.array([[1.0]])
        for a, b in zip(self.p0, self.p1):  # qubit 0 first -> ends up least significant
            A = np.kron(confusion_matrix_1q(a, b), A)
        return A

    def lambda_support(self, support) -> float:
        """Analytic twirled eigenvalue lambda_w = prod_{i in w} (1 - p0_i - p1_i)."""
        val = 1.0
        for i in support:
            val *= 1.0 - self.p0[i] - self.p1[i]
        return val

    def twirled(self) -> "ReadoutSpec":
        """The readout-TWIRLED (symmetrised) channel: p0 = p1 = (p0+p1)/2 per qubit.

        Simulating THIS channel in Aer reproduces TREX's twirled measurement statistics WITHOUT
        per-shot X-twirls -- twirling a bit-flip channel averages (p0,p1) -> p_bar, so raw counts
        through the symmetric channel ARE the twirled counts (with real shot noise). lambda_w is
        unchanged: 1 - 2*p_bar = 1 - (p0 + p1). Used by the experimental TREX pipeline.
        """
        pbar = tuple((a + b) / 2.0 for a, b in zip(self.p0, self.p1))
        return replace(self, p0=pbar, p1=pbar, name=f"{self.name}Twirled")


def trex_success_prob_from_counts(
    counts: dict, shots: int, target: str, readout_true: "ReadoutSpec", readout_calib: "ReadoutSpec"
) -> float:
    """Analytic TREX-corrected P(target) from GATE-ONLY measurement counts + a (true, calib) pair.

    Twirled readout scales each <Z_w> by lambda_w(true) = prod_{i in w}(1 - p0_i - p1_i); TREX
    divides by lambda_w(calib). Choosing `readout_calib` selects the regime, all from the SAME
    gate-only counts (so the ONLY thing that varies is the lambda_w staleness -- no twirl /
    calibration shot noise):
        - calib = READOUT_PERFECT (lambda=1) -> NO correction = the twirled NOISY value (readout floor),
        - calib = readout_true                -> ADAPTIVE TREX (exact readout removal = gate-only P),
        - calib = the frozen nominal readout  -> STATIC TREX (stale -> a lambda-ratio bias).

    `counts` is a Qiskit little-endian counts dict from the GATE-ONLY (no-readout) backend;
    P(target) = 2^-n sum_w (-1)^{t.w} (lambda_w^true / lambda_w^calib) <Z_w>_gate.
    """
    n = len(target)
    tbit = [int(target[n - 1 - i]) for i in range(n)]  # target bit of qubit i (key[-1]=qubit0)
    probs = {}  # {(x0,...,x_{n-1}): P_gate(x)}
    for key, c in counts.items():
        s = key.replace(" ", "")
        x = tuple(int(s[n - 1 - i]) for i in range(n))
        probs[x] = probs.get(x, 0.0) + c / shots
    val = 0.0
    for r in range(n + 1):
        for w in combinations(range(n), r):
            sign = -1 if sum(tbit[i] for i in w) % 2 else 1  # (-1)^{t.w}
            zw = sum(
                pr * (-1 if sum(x[i] for i in w) % 2 else 1) for x, pr in probs.items()
            )  # <Z_w>_gate
            ratio = (
                readout_true.lambda_support(w) / readout_calib.lambda_support(w) if w else 1.0
            )
            val += (1.0 / 2**n) * sign * ratio * zw
    return val


def _zw_from_counts(counts: dict, shots: int, n: int) -> dict:
    """{w: <Z_w>} for every subset w of range(n), from a Qiskit little-endian counts dict.

    <Z_w> = sum_x (-1)^{parity(x restricted to w)} P(x). The empty set gives <Z_{}> = 1 (the total
    probability). `shots` normalises the counts.
    """
    probs = {}  # {(x0,...,x_{n-1}): P(x)}
    for key, c in counts.items():
        s = key.replace(" ", "")
        x = tuple(int(s[n - 1 - i]) for i in range(n))
        probs[x] = probs.get(x, 0.0) + c / shots
    zw = {}
    for r in range(n + 1):
        for w in combinations(range(n), r):
            zw[w] = sum(
                pr * (-1 if sum(x[i] for i in w) % 2 else 1) for x, pr in probs.items()
            )
    return zw


def trex_success_prob_experimental(
    exec_counts: dict, calib_counts: dict, target: str, exec_shots: int, calib_shots: int
) -> float:
    """EXPERIMENTAL TREX-corrected P(target) from real (twirled) Aer measurements.

    Both count dicts come from the TWIRLED channel (= the readout simulated via ReadoutSpec.twirled()),
    so readout acts as a clean per-Z_w scaling lambda_w, but lambda_w is now MEASURED (shot noise), not
    analytic -> honest scatter (unlike trex_success_prob_from_counts):
        - `exec_counts`  : mirror(target) run through gate + twirled-DRIFTED readout
                           -> <Z_w>_exec = lambda_w^true * <Z_w>_gate  (+ shot noise),
        - `calib_counts` : |0...0> run through the twirled CALIBRATION readout -> <Z_w>_calib = lambda_w^calib
                           (since <Z_w>_ideal on |0...0> = +1) (+ calibration shot noise).
    TREX divides:  P(target) = 2^-n sum_w (-1)^{t.w} <Z_w>_exec / <Z_w>_calib.  The calibration readout
    selects the regime: calib = drifted -> ADAPTIVE (recalibrated per run); calib = frozen nominal ->
    STATIC (stale); calib = perfect (<Z_w>_calib = 1) -> the un-corrected NOISY floor.
    """
    n = len(target)
    tbit = [int(target[n - 1 - i]) for i in range(n)]
    zx = _zw_from_counts(exec_counts, exec_shots, n)
    zc = _zw_from_counts(calib_counts, calib_shots, n)
    val = 0.0
    for w, zx_w in zx.items():
        sign = -1 if sum(tbit[i] for i in w) % 2 else 1  # (-1)^{t.w}
        lam = zc[w] if w else 1.0  # <Z_{}> == 1 exactly; the identity term needs no correction
        val += (1.0 / 2**n) * sign * (zx_w / lam)
    return val


# Ready-made 2-qubit specs (matching the benchmark's N_QUBITS=2 mirror circuits).
# Perfect readout (lambda_w = 1): the "no correction" calib in trex_success_prob_from_counts.
READOUT_PERFECT = ReadoutSpec(name="ReadoutPerfect", p0=(0.0, 0.0), p1=(0.0, 0.0))

# Symmetric: p0 == p1, so even the *untwirled* channel scales <Z> cleanly (baseline).
READOUT_SYM = ReadoutSpec(name="ReadoutSym", p0=(0.03, 0.03), p1=(0.03, 0.03))

# Asymmetric: p1 (1->0 relaxation) > p0 (0->1), as on real hardware. Here the untwirled
# channel adds a state-independent OFFSET to <Z> (not a clean scaling); twirling is what
# restores a pure lambda scaling -> the decisive case for TREX. (Pedagogically strong effect.)
READOUT_ASYM = ReadoutSpec(name="ReadoutAsym", p0=(0.02, 0.015), p1=(0.06, 0.045))

# Physically REALISTIC readout, calibrated to modern IBM (Eagle/Heron) devices: median readout
# assignment error ~1-2% per qubit, ASYMMETRIC because T1 relaxation during the ~1-2 us measurement
# makes the 1->0 error p1=P(0|1) larger than the 0->1 error p0=P(1|0) (which is thermal excitation +
# discriminator error only). Consistent with the composite model's realistic T1=100us/T2=80us framing.
# Per-qubit: q0 assignment err ~1.75%, q1 ~2.1% -- typical spread across a device.
READOUT_REALISTIC = ReadoutSpec(name="ReadoutRealistic", p0=(0.010, 0.013), p1=(0.025, 0.030))

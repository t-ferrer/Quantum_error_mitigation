"""Probabilistic-error-cancellation representation machinery (LP + Takagi)."""

from itertools import product as iproduct

import cirq
import numpy as np
from mitiq.interface import convert_to_mitiq
from mitiq.pec.channels import _circuit_to_choi, choi_to_super, kraus_to_super
from mitiq.pec.representations.damping import (
    _represent_operation_with_amplitude_damping_noise,
)
from mitiq.pec.representations.optimal import find_optimal_representation
from mitiq.pec.types import NoisyOperation, OperationRepresentation
from mitiq.utils import matrix_to_vector
from qiskit import QuantumCircuit
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Robust L1 minimisation via scipy.linprog (replaces Mitiq's BFGS-based solver).
# Reformulation:  min  Σ u_i   s.t.   A·η = b,   -u ≤ η ≤ u,   u ≥ 0
# Mitiq's `minimize_one_norm` uses scipy.optimize.minimize + LinearConstraint,
# which is BFGS-based: it fails on complex Rz unitaries combined with non-unital
# noise (amplitude damping). linprog (HiGHS) handles those cases robustly.
# ---------------------------------------------------------------------------
def _minimize_one_norm_linprog(ideal_matrix, basis_matrices):
    ideal_real = np.hstack((np.real(ideal_matrix), np.imag(ideal_matrix)))
    basis_real = [np.hstack((np.real(m), np.imag(m))) for m in basis_matrices]
    A_eq = np.array([matrix_to_vector(m) for m in basis_real]).T  # (M, k)
    b_eq = matrix_to_vector(ideal_real)  # (M,)
    k = len(basis_matrices)
    c = np.concatenate([np.zeros(k), np.ones(k)])  # objective: sum of u
    A_eq_full = np.hstack([A_eq, np.zeros_like(A_eq)])
    A_ub = np.vstack(
        [
            np.hstack([np.eye(k), -np.eye(k)]),  # η - u ≤ 0
            np.hstack([-np.eye(k), -np.eye(k)]),  # -η - u ≤ 0
        ]
    )
    b_ub = np.zeros(2 * k)
    bounds = [(None, None)] * k + [(0, None)] * k
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq_full,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP infeasible: {res.message}")
    return res.x[:k]


def find_optimal_representation_linprog(ideal_op, noisy_ops, tol=1e-8):
    """Mitiq-compatible: builds OperationRepresentation, but solves via linprog.

    `ideal_matrix` is built with the SAME convention as the basis channel matrices
    (LineQubit-sorted choi->super), NOT from the op's intrinsic qubit order. Without
    this, a reversed-orientation gate like cx(1,0) is expressed in (control, target)
    order while the basis is in (line0, line1) order -> the LP becomes infeasible.
    """
    ideal_cirq_raw, _ = convert_to_mitiq(ideal_op)
    n_qubits = ideal_op.num_qubits
    line_qubits = cirq.LineQubit.range(n_qubits)
    qmap = {nq: lq for nq, lq in zip(sorted(ideal_cirq_raw.all_qubits()), line_qubits)}
    ideal_cirq = ideal_cirq_raw.transform_qubits(lambda q: qmap[q])
    ideal_matrix = choi_to_super(_circuit_to_choi(ideal_cirq))
    basis_matrices = [op.channel_matrix for op in noisy_ops]
    etas = _minimize_one_norm_linprog(ideal_matrix, basis_matrices)
    # 1q reps: qubit-agnostic so one rep matches the gate on EVERY qubit (q_0 and q_1).
    # 2q reps: qubit-dependent so orientation matters (we build cx(0,1) AND cx(1,0)).
    return OperationRepresentation(
        ideal_op, noisy_ops, etas.tolist(), is_qubit_dependent=(n_qubits > 1)
    )


# ---------------------------------------------------------------------------
# Generic LP-based representation for arbitrary per-qubit Kraus noise.
# Basis = ideal_op ∘ ( {I, X, Y, Z} (+ optional reset) )^n  per qubit.
# Noise = single-qubit Kraus applied INDEPENDENTLY on each qubit after the basis op.
# ---------------------------------------------------------------------------
def represent_op_with_kraus_lp(
    ideal_qiskit_op: QuantumCircuit,
    kraus_1q: list,
    tol: float = 1.0e-8,
    include_reset: bool = False,
) -> OperationRepresentation:
    """LP-based QPR for an ideal Qiskit op when per-qubit noise = `kraus_1q` Kraus.

    For NON-UNITAL noise (amplitude damping, thermal), pass `include_reset=True`.
    Note: convert_to_mitiq returns Cirq w/ NamedQubit; we re-map to LineQubit so
    that `_circuit_to_choi` (max-ent state on LineQubits) aligns.
    """
    n = ideal_qiskit_op.num_qubits
    ideal_cirq_raw, _ = convert_to_mitiq(ideal_qiskit_op)
    line_qubits = cirq.LineQubit.range(n)
    qmap = {nq: lq for nq, lq in zip(sorted(ideal_cirq_raw.all_qubits()), line_qubits)}
    ideal_cirq = ideal_cirq_raw.transform_qubits(lambda q: qmap[q])

    tags = ["i", "x", "y", "z"]
    if include_reset:
        tags.append("r")
    paulis_cirq = {"i": None, "x": cirq.X, "y": cirq.Y, "z": cirq.Z}

    implementable_qiskit, super_operators = [], []
    for combo in iproduct(tags, repeat=n):
        qi = ideal_qiskit_op.copy()
        for qi_idx, t in enumerate(combo):
            if t == "x":
                qi.x(qi_idx)
            elif t == "y":
                qi.y(qi_idx)
            elif t == "z":
                qi.z(qi_idx)
            elif t == "r":
                qi.reset(qi_idx)
        implementable_qiskit.append(qi)
        c = cirq.Circuit(ideal_cirq)
        for qi_idx, t in enumerate(combo):
            if t in ("x", "y", "z"):
                c.append(paulis_cirq[t].on(line_qubits[qi_idx]))
            elif t == "r":
                c.append(cirq.ResetChannel().on(line_qubits[qi_idx]))
        for q in line_qubits:
            c.append(cirq.KrausChannel(kraus_1q).on(q))
        super_operators.append(choi_to_super(_circuit_to_choi(c)))

    noisy_operations = [
        NoisyOperation(c, mat) for c, mat in zip(implementable_qiskit, super_operators)
    ]
    return find_optimal_representation_linprog(
        ideal_qiskit_op, noisy_operations, tol=tol
    )


def represent_1q_qiskit_with_ad_takagi(qc_qiskit: QuantumCircuit, noise_level: float):
    """Mitiq's universal Takagi 1q AD rep, applied to a Qiskit op.

    For 1q non-unital channels (AD, thermal), the LP on a Pauli+reset basis is
    infeasible for complex rotations like Rz(π/2). Takagi's analytical formula
    works for ANY 1q ideal U with coefficients (η_0, η_1, η_2) = ((1+√(1-γ))/(2(1-γ)),
    (1-√(1-γ))/(2(1-γ)), -γ/(1-γ)) on basis {U, U+Z, U+reset}.
    """
    # Matching fix: keep the ideal in the SAME qubit namespace convert_to_mitiq gives the
    # benchmark circuit (NamedQubit) -- do NOT remap to LineQubit, else the rep matches no op.
    # is_qubit_dependent=False so this single rep matches the gate on EVERY qubit (q_0, q_1).
    raw, _ = convert_to_mitiq(qc_qiskit)
    rep = _represent_operation_with_amplitude_damping_noise(
        raw, noise_level=noise_level
    )
    return OperationRepresentation(
        rep.ideal, rep.noisy_operations, rep.coeffs, is_qubit_dependent=False
    )


_LP_REP_CACHE: dict = {}


def _build_with_cache(
    circuit,
    kraus_1q,
    kraus_2qq,
    noise_id: str,
    include_reset: bool = False,
    gamma_1q_for_takagi: float | None = None,
):
    """Build per-gate QPRs, caching across seeds (LP solves are the bottleneck).

    - If `gamma_1q_for_takagi` is given, 1q gates use Mitiq's universal Takagi AD
      helper (robust on complex Rz rotations where the LP would otherwise fail).
    - Otherwise 1q gates go through the LP with `kraus_1q`.
    - 2q CX always uses the LP with `kraus_2qq` applied per qubit.
    """
    reps_1q, seen = [], set()
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier") or op.num_qubits != 1:
            continue
        key = (
            noise_id + "1",
            op.name,
            tuple(round(float(p), 10) for p in op.params),
            include_reset,
            gamma_1q_for_takagi,
        )
        if key in seen:
            continue
        seen.add(key)
        if key not in _LP_REP_CACHE:
            qc = QuantumCircuit(1)
            qc.append(op, [0])
            if gamma_1q_for_takagi is not None:
                _LP_REP_CACHE[key] = represent_1q_qiskit_with_ad_takagi(
                    qc, gamma_1q_for_takagi
                )
            else:
                _LP_REP_CACHE[key] = represent_op_with_kraus_lp(
                    qc, kraus_1q, include_reset=include_reset
                )
        reps_1q.append(_LP_REP_CACHE[key])

    # CX rep for BOTH orientations (cx(1,0) otherwise left un-mitigated -> PEC bias)
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        cx_key = (noise_id + "2", "cx", (ctrl, tgt), include_reset)
        if cx_key not in _LP_REP_CACHE:
            qc = QuantumCircuit(2)
            qc.cx(ctrl, tgt)
            _LP_REP_CACHE[cx_key] = represent_op_with_kraus_lp(
                qc, kraus_2qq, include_reset=include_reset
            )
        cx_reps.append(_LP_REP_CACHE[cx_key])
    return reps_1q + cx_reps

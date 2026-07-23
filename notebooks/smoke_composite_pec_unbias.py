"""Can we cancel the +0.06 PEC bias on the composite? (offline, no Monte-Carlo)

Root cause (smoke_composite_pec_bias.py): the LP basis models a basis op "G then Pauli P"
as  N o P o G  (noise applied ONCE, inserted P noiseless). Aer instead runs it as the
EXACT channel of the circuit -- noise after every NOISED gate. For the composite that means:

  - after the ideal gate G : composite noise (1q 35ns for 1q gates, cx 400ns per qubit for cx);
  - after an inserted X    : 1q composite noise (x IS in the noise model);
  - after an inserted Y/Z   : NOTHING (y,z are not basis gates -> Aer adds no noise);
  - after an inserted reset : NOTHING (reset not noised).

Fix under test ("noise-after-each"): build the LP basis super-operators with the noise placed
exactly where Aer puts it, solve for new coeffs, and check the reconstruction is exact w.r.t. the
REAL Aer channels:   || sum_i a_i  S_aer(impl_i)  -  S_ideal ||   should drop to ~0.

DECISIVE & cheap: if that residual ~0 for the fixed coeffs (and = the bias for the current
coeffs), the fix cancels the bias -- no end-to-end Monte-Carlo needed.
"""

from itertools import product as iproduct

import cirq
import numpy as np
from mitiq.interface import convert_to_mitiq
from mitiq.pec.channels import _circuit_to_choi, choi_to_super
from mitiq.pec.types import NoisyOperation
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, Operator, SuperOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import thermal_relaxation_error

from qem.noise_models import (
    T1_S, T2_S, GATE_TIME_S, EPS_MU_1Q, EPS_MU_2Q, _mu_pauli_error, _composite_noise_model,
)
from qem.pec_core import _minimize_one_norm_linprog, find_optimal_representation_linprog


def composite_kraus_1q(eps, t):
    qe = thermal_relaxation_error(T1_S, T2_S, t).compose(_mu_pauli_error(eps))
    return list(Kraus(qe).data)


# --- exact Aer channel of an implementable circuit (qiskit SuperOp convention) -------------
_NM = _composite_noise_model()
_SUPEROP_SIM = AerSimulator(method="superop", noise_model=_NM)


def aer_super(qc: QuantumCircuit) -> np.ndarray:
    qc2 = qc.copy()
    qc2.save_superop()
    res = _SUPEROP_SIM.run(qc2).result()
    return np.asarray(res.data()["superop"])


# ============================================================================
print("=== (0) Aer superop method includes the composite noise? ===")
qc_x = QuantumCircuit(1); qc_x.x(0)
S_aer_x = aer_super(qc_x)
# expected: noise(x) o X  -- compose via qiskit objects (same convention)
qe_x = thermal_relaxation_error(T1_S, T2_S, GATE_TIME_S["x"]).compose(_mu_pauli_error(EPS_MU_1Q))
S_expected = SuperOp(Operator(qc_x)).compose(SuperOp(qe_x))   # X first, then noise
S_expected_rev = SuperOp(qe_x).compose(SuperOp(Operator(qc_x)))
d1 = np.max(np.abs(S_aer_x - S_expected.data))
d2 = np.max(np.abs(S_aer_x - S_expected_rev.data))
print(f"  ||S_aer(x) - noise.compose order1|| = {d1:.2e}   order2 = {d2:.2e}  (one should be ~0)")
# also confirm y/z are NOT noised
for g in ("y", "z"):
    qc = QuantumCircuit(1); getattr(qc, g)(0)
    S = aer_super(qc)
    S_ideal = SuperOp(Operator(qc)).data
    print(f"  inserted {g}: ||S_aer({g}) - ideal {g}|| = {np.max(np.abs(S - S_ideal)):.2e}  "
          f"(~0 => {g} NOT noised, as expected)")


# ============================================================================
# Build a rep (coeffs aligned to a known basis-op order) in either noise-placement mode.
def build_rep(ideal_qiskit_op, kraus_after_gate, kraus_inserted, noised_inserted, mode,
              include_reset=True):
    """Return (implementable_qiskit_list, OperationRepresentation). mode: 'once' | 'each'."""
    n = ideal_qiskit_op.num_qubits
    ideal_cirq_raw, _ = convert_to_mitiq(ideal_qiskit_op)
    line = cirq.LineQubit.range(n)
    qmap = {nq: lq for nq, lq in zip(sorted(ideal_cirq_raw.all_qubits()), line)}
    ideal_cirq = ideal_cirq_raw.transform_qubits(lambda q: qmap[q])
    paulis = {"x": cirq.X, "y": cirq.Y, "z": cirq.Z}
    tags = ["i", "x", "y", "z"] + (["r"] if include_reset else [])

    impl, supers = [], []
    for combo in iproduct(tags, repeat=n):
        qi = ideal_qiskit_op.copy()
        for idx, t in enumerate(combo):
            if t == "x": qi.x(idx)
            elif t == "y": qi.y(idx)
            elif t == "z": qi.z(idx)
            elif t == "r": qi.reset(idx)
        impl.append(qi)

        c = cirq.Circuit(ideal_cirq)
        if mode == "each":
            for q in line:
                c.append(cirq.KrausChannel(kraus_after_gate).on(q))
        for idx, t in enumerate(combo):
            if t in paulis:
                c.append(paulis[t].on(line[idx]))
                if mode == "each" and t in noised_inserted:
                    c.append(cirq.KrausChannel(kraus_inserted).on(line[idx]))
            elif t == "r":
                c.append(cirq.ResetChannel().on(line[idx]))
        if mode == "once":
            for q in line:
                c.append(cirq.KrausChannel(kraus_after_gate).on(q))
        supers.append(choi_to_super(_circuit_to_choi(c)))

    noisy_ops = [NoisyOperation(cc, m) for cc, m in zip(impl, supers)]
    rep = find_optimal_representation_linprog(ideal_qiskit_op, noisy_ops)
    return impl, rep


def reconstruction_residual(impl, rep):
    """Convention-consistent (qiskit): || sum_i a_i S_aer(impl_i) - S_ideal_gate ||.

    rep.noisy_operations order == impl order. The ideal is the bare gate (impl[0] minus its
    inserted ops); we recover it as the gate with no insertion = the 'I...I' combo = impl[0].
    """
    ideal_qc = QuantumCircuit(impl[0].num_qubits)
    for instr in impl[0].data:  # impl[0] is the identity-combo: just the ideal gate
        ideal_qc.append(instr.operation, instr.qubits)
    S_ideal = SuperOp(Operator(ideal_qc)).data
    acc = np.zeros_like(S_ideal, dtype=complex)
    for a, qc in zip(rep.coeffs, impl):
        acc = acc + a * aer_super(qc)
    return np.max(np.abs(acc - S_ideal)), S_ideal


print("\n=== (1)+(2) bias of current ('once') vs fixed ('each') rep -- exact Aer reconstruction ===")
cases = [
    ("x (1q, 35ns)", (lambda: QuantumCircuit(1)), "x",
     composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S["x"]), composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S["x"])),
]
# 1q gate x
for label, mk, gate, k_after, k_ins in cases:
    qc = mk(); getattr(qc, gate)(0)
    for mode in ("once", "each"):
        impl, rep = build_rep(qc, k_after, k_ins, noised_inserted=("x",), mode=mode)
        resid, _ = reconstruction_residual(impl, rep)
        print(f"  {label:14s} mode={mode:4s}  gamma={rep.norm:.5f}  "
              f"||recon_aer - ideal|| = {resid:.2e}")

# cx: noise after cx uses cx params; inserted 1q Paulis use 1q (35ns) noise
print()
k_after_cx = composite_kraus_1q(EPS_MU_2Q, GATE_TIME_S["cx"])
k_ins_1q = composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S["x"])
for ctrl, tgt in [(0, 1)]:
    qc = QuantumCircuit(2); qc.cx(ctrl, tgt)
    for mode in ("once", "each"):
        impl, rep = build_rep(qc, k_after_cx, k_ins_1q, noised_inserted=("x",), mode=mode)
        resid, _ = reconstruction_residual(impl, rep)
        print(f"  cx({ctrl},{tgt})       mode={mode:4s}  gamma={rep.norm:.5f}  "
              f"||recon_aer - ideal|| = {resid:.2e}")

print("\nReading: 'once' residual = the per-gate PEC bias (current). 'each' residual ~0 =>")
print("the noise-after-each rep is unbiased on the REAL Aer channels (bias cancelled),")
print("at the cost of a higher gamma (overhead). If 'each' residual is NOT ~0, the inserted-")
print("gate noise model (noised_inserted) is still off.")

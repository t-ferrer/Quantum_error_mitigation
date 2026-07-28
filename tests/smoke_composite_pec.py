"""Smoke tests for PEC on the Composite (ThermalRealistic ∘ MixedUnitary) noise model.

Run BEFORE wiring PEC into SPEC_COMPOSITE. Verifies the theory:

 (1) The per-qubit composite channel FACTORIZES: Aer's 2q cx noise (th⊗th)∘(mu⊗mu)
     equals (th∘mu)⊗(th∘mu)  -> one per-qubit Kraus suffices for the cx LP rep
     (same trick as ThermalRealistic, where the per-qubit channel was tensored).
 (2) Extracting the composite 1q Kraus straight from the Aer QuantumError makes the PEC
     basis-noise match the SIMULATED noise EXACTLY (no manual composition-order risk;
     no need for the 1e-15 hand-derivation ThermalRealistic required).
 (3) The Pauli+reset LP is FEASIBLE for the physical gates sx/sxdg/x/h (1q) and cx (2q),
     with a sane overhead gamma.  rz/id (zero noise) -> identity rep via Takagi(0).
 (4) End-to-end: PEC on a few seeds is UNBIASED (PEC ~ 1.0 within +/- pec_error).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import thermal_relaxation_error
from mitiq import pec

from qem import SPEC_COMPOSITE, make_benchmark_circuit, mitiq_executor, IDEAL
from qem.benchmark import _assert_all_ops_represented
from qem.noise_models import (
    T1_S, T2_S, GATE_TIME_S, EPS_MU_1Q, EPS_MU_2Q, _mu_pauli_error, _MU_PHYS_1Q,
)
from qem.pec_core import (
    represent_op_with_kraus_lp, represent_1q_qiskit_with_ad_takagi,
)


def composite_qe_1q(eps, t):
    """Per-qubit composite noise as an Aer QuantumError: thermal(T1,T2,t) ∘ mu_pauli(eps)."""
    return thermal_relaxation_error(T1_S, T2_S, t).compose(_mu_pauli_error(eps))


def composite_kraus_1q(eps, t):
    """Exact per-qubit composite Kraus, lifted straight from the Aer QuantumError."""
    return list(Kraus(composite_qe_1q(eps, t)).data)


# ---------------------------------------------------------------------------
print("=== (1) cx composite channel factorizes per qubit ===")
th_cx = thermal_relaxation_error(T1_S, T2_S, GATE_TIME_S["cx"])
mu_cx = _mu_pauli_error(EPS_MU_2Q)
aer_2q = th_cx.tensor(th_cx).compose(mu_cx.tensor(mu_cx))  # exactly what _composite_noise_model puts on cx
S_aer = SuperOp(aer_2q).data
comp1 = composite_qe_1q(EPS_MU_2Q, GATE_TIME_S["cx"])
S_fact = SuperOp(comp1.tensor(comp1)).data
print(f"  ||S_aer_cx - (comp(x)comp)||_max = {np.max(np.abs(S_aer - S_fact)):.2e}  (expect ~1e-15)")

# ---------------------------------------------------------------------------
print("\n=== (2) extracted composite Kraus reproduces the Aer 1q channel exactly ===")
for g in _MU_PHYS_1Q:
    qe = composite_qe_1q(EPS_MU_1Q, GATE_TIME_S[g])
    kr = composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S[g])
    S_qe = SuperOp(qe).data
    S_kr = SuperOp(Kraus(kr)).data
    cptp = np.max(np.abs(sum(K.conj().T @ K for K in kr) - np.eye(2)))
    print(f"  {g:5s} n_kraus={len(kr)}  ||S_kraus - S_qe||={np.max(np.abs(S_kr - S_qe)):.2e}  "
          f"CPTP(sum K†K - I)={cptp:.2e}")

# ---------------------------------------------------------------------------
print("\n=== (3) LP feasibility + overhead gamma (physical gates only) ===")
for g in _MU_PHYS_1Q:  # sx, sxdg, x, h
    qc = QuantumCircuit(1)
    getattr(qc, g)(0)
    try:
        rep = represent_op_with_kraus_lp(
            qc, composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S[g]), include_reset=True
        )
        print(f"  {g:5s}  FEASIBLE  norm(gamma)={rep.norm:.6f}  n_terms={len(rep.coeffs)}")
    except Exception as e:
        print(f"  {g:5s}  *** {type(e).__name__}: {e}")
for ctrl, tgt in [(0, 1), (1, 0)]:
    qc = QuantumCircuit(2)
    qc.cx(ctrl, tgt)
    try:
        rep = represent_op_with_kraus_lp(
            qc, composite_kraus_1q(EPS_MU_2Q, GATE_TIME_S["cx"]), include_reset=True
        )
        print(f"  cx({ctrl},{tgt}) FEASIBLE  norm(gamma)={rep.norm:.6f}  n_terms={len(rep.coeffs)}")
    except Exception as e:
        print(f"  cx({ctrl},{tgt}) *** {type(e).__name__}: {e}")

print("\n  rz/id (zero noise) -> identity rep via Takagi(0):")
for name in ("rz", "id"):
    qc = QuantumCircuit(1)
    qc.rz(0.5, 0) if name == "rz" else qc.id(0)
    rep = represent_1q_qiskit_with_ad_takagi(qc, 0.0)
    print(f"    {name:4s} coeffs={np.round(rep.coeffs, 6)}  norm={rep.norm:.6f}")


# ---------------------------------------------------------------------------
def build_composite_reps(circuit):
    """Prototype of the SPEC_COMPOSITE.build_representations to be added to noise_models.py."""
    reps_1q, seen = [], set()
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier") or op.num_qubits != 1:
            continue
        key = (op.name, tuple(round(float(p), 10) for p in op.params))
        if key in seen:
            continue
        seen.add(key)
        qc = QuantumCircuit(1)
        qc.append(op, [0])
        if op.name in _MU_PHYS_1Q:
            rep = represent_op_with_kraus_lp(
                qc, composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S[op.name]), include_reset=True
            )
        else:  # rz / id : virtual, no noise -> exact identity rep (robust to any angle)
            rep = represent_1q_qiskit_with_ad_takagi(qc, 0.0)
        reps_1q.append(rep)
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        qc = QuantumCircuit(2)
        qc.cx(ctrl, tgt)
        cx_reps.append(
            represent_op_with_kraus_lp(
                qc, composite_kraus_1q(EPS_MU_2Q, GATE_TIME_S["cx"]), include_reset=True
            )
        )
    return reps_1q + cx_reps


print("\n=== (4) end-to-end PEC on seeds 0..2 (decisive unbiasedness test) ===")
backend = AerSimulator(noise_model=SPEC_COMPOSITE.build_noise_model(), seed_simulator=42)
for s in range(3):
    circ = make_benchmark_circuit(s)
    reps = build_composite_reps(circ)
    _assert_all_ops_represented(circ, reps)
    gamma_circ = 1.0
    for instr in circ.data:
        if instr.operation.name in ("measure", "barrier"):
            continue
        matched = [r for r in reps if len(r.ideal.all_qubits()) == instr.operation.num_qubits]
        if matched:
            gamma_circ *= matched[0].norm
    ex_pec = mitiq_executor(backend, s, 100)
    val, data = pec.execute_with_pec(
        circ, ex_pec, representations=reps, num_samples=300, full_output=True
    )
    noisy = mitiq_executor(backend, s, 10000)(circ)
    print(f"  seed {s}: noisy={noisy:.4f}  PEC={float(val):.4f} +/- {data['pec_error']:.4f}  "
          f"(ideal={IDEAL})  gamma_circ={gamma_circ:.3f}")

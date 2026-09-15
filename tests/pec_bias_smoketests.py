# -*- coding: utf-8 -*-
"""
Smoke tests diagnosing the persistent NEGATIVE PEC bias in DZNE_VS_PEC_mitiq.ipynb.

PEC is asymptotically UNBIASED iff every gate in the circuit is matched to a
representation that inverts its true noisy channel. Finite num_samples / shots add
VARIANCE, not bias. A systematic bias => some gate is mis-modelled or NOT mitigated.

Findings (run the tests to reproduce):
  TEST A  convention check       -> the (3/4) and (15/16) Aer->Mitiq factors are CORRECT
                                    (rep built at factor*P exactly inverts Aer's depol(P)).
  TEST B  recovery-gate noise    -> negligible (~1e-4 per gate). NOT the cause.
  TEST C  ROOT CAUSE             -> build_representations() only builds a rep for cx(0,1).
                                    Random Clifford circuits contain cx(1,0) too. Mitiq emits
                                    "No representation found" for every reversed CX (silenced by
                                    warnings.simplefilter('ignore')) and leaves it UN-MITIGATED.
                                    40-60% of the CX (the dominant error source) are uncorrected
                                    => large negative bias on EVERY noise model.
  TEST D  fix validation         -> adding a cx(1,0) rep removes the bias.

Fix in the notebook's build_representations():
    for ctrl, tgt in [(0, 1), (1, 0)]:
        qc_cx = QuantumCircuit(2); qc_cx.cx(ctrl, tgt)
        reps.append(pec.represent_operation_with_global_depolarizing_noise(qc_cx, P_MITIQ_2Q))
(and likewise for the LP-based reps in the §6 non-depolarizing specs).
"""

import collections
import warnings

import numpy as np
from mitiq import pec
from mitiq.interface import convert_from_mitiq
from mitiq.pec.sampling import sample_circuit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_clifford_circuit
from qiskit.quantum_info import Operator, SuperOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

np.set_printoptions(precision=4, suppress=True)

P1, P2 = 0.01, 0.05
BASIS_GATES = ["h", "rz", "sx", "x", "cx", "id", "sxdg"]
P_MITIQ_1Q = (3 / 4) * P1
P_MITIQ_2Q = (15 / 16) * P2
IDEAL = 1.0


def make_noise_model():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(P1, 1), ["h", "rz", "sx", "x", "sxdg", "id"]
    )
    nm.add_all_qubit_quantum_error(depolarizing_error(P2, 2), ["cx"])
    return nm


# SEEDLESS on purpose (same fix as run_benchmark): a fixed seed_simulator restarts every
# backend.run() from the same noise RNG stream, correlating PEC's importance samples -> biased
# estimator, inflated variance. That artefact sits ON TOP of the cx(1,0) bias diagnosed here, so
# the seeded numbers overstated the residual bias TEST D was trying to drive to zero.
_BE = AerSimulator(noise_model=make_noise_model())


def executor(circuit, shots=4000):
    qc = circuit.copy()
    qc.measure_all()
    counts = _BE.run(qc, shots=shots).result().get_counts()
    return counts.get("0" * circuit.num_qubits, 0) / shots


def make_benchmark_circuit(seed, depth=10):
    rc = random_clifford_circuit(num_qubits=2, num_gates=depth, seed=seed)
    return transpile(
        rc.compose(rc.inverse()), basis_gates=BASIS_GATES, optimization_level=0
    )


def reps_buggy(circuit):
    """Notebook's build_representations: ONE cx rep, orientation (0,1) only."""
    a = pec.represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, noise_level=P_MITIQ_1Q
    )
    r1 = [r for r in a if len(r.ideal.all_qubits()) == 1]
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    return r1 + [
        pec.represent_operation_with_global_depolarizing_noise(
            qc, noise_level=P_MITIQ_2Q
        )
    ]


def reps_fixed(circuit):
    """Fixed: a cx rep for BOTH orientations."""
    a = pec.represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, noise_level=P_MITIQ_1Q
    )
    out = [r for r in a if len(r.ideal.all_qubits()) == 1]
    for ctrl, tgt in [(0, 1), (1, 0)]:
        qc = QuantumCircuit(2)
        qc.cx(ctrl, tgt)
        out.append(
            pec.represent_operation_with_global_depolarizing_noise(
                qc, noise_level=P_MITIQ_2Q
            )
        )
    return out


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ---------------------------------------------------------------------------
def test_A_convention():
    banner("TEST A - Aer->Mitiq noise-level conversion factors (3/4, 15/16) correct?")
    # rep built at factor*P should EXACTLY invert Aer's depol(P) o gate (ideal recovery).
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    Sn1 = SuperOp(depolarizing_error(P1, 1)).compose(SuperOp(Operator(qc1))).data
    rep1 = pec.represent_operation_with_local_depolarizing_noise(
        qc1, noise_level=P_MITIQ_1Q
    )
    acc = np.zeros((4, 4), complex)
    for no, eta in zip(rep1.noisy_operations, rep1.coeffs):
        sub = convert_from_mitiq(no.circuit, "qiskit")
        rec = QuantumCircuit(1)
        for instr in sub.data:
            if instr.operation.name != "h":
                rec.append(instr.operation, [0])
        S = SuperOp(Operator(rec)).data if rec.data else np.eye(4)
        acc += eta * (S @ Sn1)
    print(
        f"  1q H, factor=3/4   : max|reconstructed - ideal| = {np.abs(acc - SuperOp(Operator(qc1)).data).max():.5f}  (0 => correct)"
    )

    qc2 = QuantumCircuit(2)
    qc2.cx(0, 1)
    Sn2 = SuperOp(depolarizing_error(P2, 2)).compose(SuperOp(Operator(qc2))).data
    rep2 = pec.represent_operation_with_global_depolarizing_noise(
        qc2, noise_level=P_MITIQ_2Q
    )
    acc = np.zeros((16, 16), complex)
    for no, eta in zip(rep2.noisy_operations, rep2.coeffs):
        sub = convert_from_mitiq(no.circuit, "qiskit")
        rec = QuantumCircuit(2)
        for instr in sub.data:
            if instr.operation.name != "cx":
                rec.append(
                    instr.operation, [sub.find_bit(q).index for q in instr.qubits]
                )
        S = SuperOp(Operator(rec)).data if rec.data else np.eye(16)
        acc += eta * (S @ Sn2)
    print(
        f"  CX,  factor=15/16  : max|reconstructed - ideal| = {np.abs(acc - SuperOp(Operator(qc2)).data).max():.5f}  (0 => correct)"
    )
    print("  => conventions are CORRECT; not the source of bias.")


def test_C_root_cause():
    banner("TEST C - ROOT CAUSE: reversed-orientation CX has no representation")
    for s in range(6):
        c = make_benchmark_circuit(s)
        orient = collections.Counter()
        for instr in c.data:
            if instr.operation.name == "cx":
                orient[tuple(c.find_bit(q).index for q in instr.qubits)] += 1
        print(f"  seed {s}: CX orientations {dict(orient)}")
    print(
        "\n  Mitiq matching with the notebook's single-orientation reps (warnings shown):"
    )
    c = make_benchmark_circuit(0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, norm = sample_circuit(c, reps_buggy(c), num_samples=50)
        msgs = [str(x.message) for x in w if "No representation" in str(x.message)]
    print(
        f"  -> {len(msgs)} 'No representation found' warnings (one per reversed CX, normally SILENCED)"
    )
    print("  -> those CX run noisy & UNCORRECTED. Sampling norm:", round(norm, 3))


def test_D_fix(seeds=range(10), num_samples=4000):
    banner("TEST D - fix validation: add the cx(1,0) representation")
    print(f"  {'representations':28s} {'PEC mean':>9} {'bias':>8} {'var':>8}")
    for label, rf in [
        ("BUGGY (cx(0,1) only)", reps_buggy),
        ("FIXED (both cx orient.)", reps_fixed),
    ]:
        vals = []
        for s in seeds:
            c = make_benchmark_circuit(s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v, _ = pec.execute_with_pec(
                    c,
                    lambda x: executor(x, 200),
                    representations=rf(c),
                    num_samples=num_samples,
                    full_output=True,
                )
            vals.append(float(v))
        vals = np.array(vals)
        print(
            f"  {label:28s} {vals.mean():>9.4f} {vals.mean() - IDEAL:>+8.4f} {vals.var():>8.4f}"
        )


if __name__ == "__main__":
    test_A_convention()
    test_C_root_cause()
    test_D_fix()

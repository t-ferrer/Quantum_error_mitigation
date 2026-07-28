"""Mirror-circuit generation and bitstring executors."""

from functools import lru_cache
from itertools import combinations

import cirq
from mitiq import MeasurementResult
from mitiq.benchmarks import generate_mirror_circuit
from mitiq.interface.conversions import convert_from_mitiq, convert_to_mitiq
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString
from qiskit import ClassicalRegister, QuantumCircuit, transpile

from .config import BASIS_GATES, CONNECTIVITY, MIRROR_NLAYERS, SHOTS, TWO_QUBIT_PROB


# Executor: P(`target` bitstring). Circuit already in basis gates -> no re-transpile.
# `target` is a Qiskit counts key (little-endian). For mirror circuits use
# `target_key(seed)` / `mitiq_executor(...)` which bind the correct bitstring per seed.
def make_executor(backend, target, shots=SHOTS):
    def executor(circuit) -> float:
        qc = circuit.copy()
        qc.measure_all()
        counts = backend.run(qc, shots=shots).result().get_counts()
        return counts.get(target, 0) / shots

    return executor


@lru_cache(maxsize=None)
def _mirror(seed: int):
    # (circuit_in_basis, qiskit_target_key) for `seed` -- memoised & deterministic.
    circ, bits = generate_mirror_circuit(
        MIRROR_NLAYERS,
        TWO_QUBIT_PROB,
        CONNECTIVITY,
        seed=seed,
        return_type="qiskit",
    )
    # Mitiq returns bits as [q0, q1, ...]; Qiskit counts keys are little-endian -> reverse.
    key = "".join(str(b) for b in bits)[::-1]
    return transpile(circ, basis_gates=BASIS_GATES, optimization_level=0), key


def make_benchmark_circuit(seed: int):
    # Pre-transpiled mirror circuit for `seed` (signature unchanged from old API).
    return _mirror(seed)[0]


def target_key(seed: int) -> str:
    # Correct (noise-free) outcome bitstring for `seed`, as a Qiskit counts key.
    return _mirror(seed)[1]


def mitiq_executor(backend, seed: int, shots: int = SHOTS):
    # Executor bound to `seed`'s correct bitstring. The bitstring is a property of
    # the seed, invariant under ZNE folding & PEC sampling (both preserve the logical
    # action), so per-seed binding is the robust route -- folding STRIPS circuit.metadata.
    return make_executor(backend, target_key(seed), shots)


def success_prob(backend, seed: int, shots: int = SHOTS) -> float:
    # One-shot helper: P(correct bitstring) for `seed` on `backend`.
    return mitiq_executor(backend, seed, shots)(make_benchmark_circuit(seed))


# ---------------------------------------------------------------------------
# TREX plumbing: a raw-bitstring executor + the P(target) projector observable.
# TREX (mitiq.experimental.trex) mitigates readout error on a Pauli <O>, so it needs
#   (a) an executor returning a MeasurementResult (raw bitstrings), not a probability, and
#   (b) an Observable. We express the benchmark quantity P(target) = <|t><t|> as a sum of
#       Z-strings so TREX corrects each eigenvalue lambda_w (readout only touches Z terms).
# ---------------------------------------------------------------------------
def to_cirq_linequbit(circuit):
    """Convert a Qiskit circuit to a Cirq circuit on LineQubit(0..n-1), for TREX.

    TREX's construct_circuits indexes twirl strings by the CIRCUIT's own qubits but the
    observable's measured qubits by LineQubit(i); the two only line up if the circuit is
    already on LineQubits. A Qiskit circuit converts to NamedQubit('q_i') -> mismatch
    (KeyError in construct_circuits). Relabeling to LineQubit here fixes that; the executor
    converts each twirled/calibration circuit back to Qiskit to run on Aer. Works for ANY
    Qiskit circuit (mirror circuit, or a ZNE-folded one). (n small -> sorted() == register order.)
    """
    cc, _ = convert_to_mitiq(circuit)
    qubit_map = {q: cirq.LineQubit(i) for i, q in enumerate(sorted(cc.all_qubits()))}
    return cc.transform_qubits(lambda q: qubit_map[q])


def benchmark_circuit_cirq(seed: int):
    """Mirror circuit for `seed` as a Cirq circuit on LineQubit(0..n-1), for TREX."""
    return to_cirq_linequbit(make_benchmark_circuit(seed))


def _measured_qubits(circuit) -> list[int]:
    """Integer indices of the qubits carrying a `measure` in `circuit` (sorted, unique)."""
    qs = {
        circuit.find_bit(instr.qubits[0]).index
        for instr in circuit.data
        if instr.operation.name == "measure"
    }
    return sorted(qs)


def mitiq_measurement_executor(backend, shots: int = SHOTS):
    """Executor returning a Mitiq MeasurementResult (raw bitstrings) -- the TREX input type.

    TREX inserts its readout twirl (X gates) + measurements into the circuit, then calls this.
    We re-measure the Mitiq-chosen qubits into a SINGLE fresh classical register
    (qubit measured[j] -> clbit j) so the counts-key endianness is unambiguous: column j of the
    returned bitstrings is qubit `measured[j]`, and qubit_indices carries those integer labels
    (which is exactly what PauliString.support() / MeasurementResult.filter_qubits() key on).
    Rebuilding the register sidesteps the Qiskit little-endian / multi-creg spacing pitfall
    (cf. the mirror-benchmark endianness note).
    """

    def executor(circuit) -> MeasurementResult:
        # TREX hands us Cirq circuits (input was Cirq); convert back to Qiskit to run on Aer.
        qc = circuit if isinstance(circuit, QuantumCircuit) else convert_from_mitiq(
            circuit, "qiskit"
        )
        measured = _measured_qubits(qc)
        base = qc.remove_final_measurements(inplace=False)
        if not measured:  # degenerate group (e.g. identity-only) -> measure everything
            measured = list(range(base.num_qubits))
        qc = base.copy()
        creg = ClassicalRegister(len(measured), "trex")
        qc.add_register(creg)
        for j, q in enumerate(measured):
            qc.measure(q, creg[j])
        counts = backend.run(qc, shots=shots).result().get_counts()
        m = len(measured)
        bitstrings = []
        for key, n in counts.items():
            s = key.replace(" ", "")  # single creg -> no spaces, but be defensive
            row = [int(s[m - 1 - j]) for j in range(m)]  # clbit j = s[m-1-j] (little-endian)
            bitstrings.extend([row] * n)
        return MeasurementResult(result=bitstrings, qubit_indices=tuple(measured))

    return executor


def projector_observable(target: str, include_identity: bool = True) -> Observable:
    """Mitiq Observable for the projector |t><t| onto Qiskit counts-key `target`.

        |t><t| = 2^-n prod_i (I + (-1)^{t_i} Z_i)
               = 2^-n sum_{w subset of qubits} (prod_{i in w} (-1)^{t_i}) Z_w .

    `target` is little-endian (target[-1] = qubit 0), matching target_key(seed). Each Z_w is a
    readout-sensitive term TREX corrects by dividing out lambda_w; the identity term (w=empty)
    has lambda=1 (readout-invariant) and just adds the constant 2^-n. With ``include_identity``
    the full projector is returned (<projector> == P(target)); set it False for TREX, which
    cannot process an empty-support Pauli (mitiq's combine_results does mean-over-empty -> NaN)
    -- use trex_success_prob, which re-adds the 2^-n offset.
    """
    n = len(target)
    tbit = {i: int(target[n - 1 - i]) for i in range(n)}  # target bit of qubit i
    norm = 1.0 / (2**n)
    terms = []
    for r in range(n + 1):
        for w in combinations(range(n), r):
            if not w and not include_identity:  # skip empty-support identity term
                continue
            sign = 1
            for i in w:
                if tbit[i]:
                    sign = -sign
            terms.append(
                PauliString(spec="Z" * len(w), support=tuple(w), coeff=norm * sign)
            )
    return Observable(*terms)


def trex_success_prob(
    circuit_cirq, executor, target: str, *, num_randomizations: int = 32, random_state=None
) -> float:
    """TREX-mitigated P(target) = <|t><t|>, readout-error-corrected.

    Wraps mitiq's execute_with_trex on the non-identity Z-terms of the projector and re-adds
    the identity offset 2^-n (its <I>=1 is exact and readout-invariant, and mitiq can't handle
    the empty-support term). `circuit_cirq` must be a LineQubit Cirq circuit (see
    benchmark_circuit_cirq) so TREX's qubit indexing lines up with the observable.
    """
    from mitiq.experimental.trex import execute_with_trex  # local: experimental namespace

    n = len(target)
    offset = 1.0 / (2**n)  # identity-term contribution to P(target)
    obs = projector_observable(target, include_identity=False)
    z_value = execute_with_trex(
        circuit_cirq, executor, obs,
        num_randomizations=num_randomizations, random_state=random_state,
    )
    return offset + float(z_value)


def trex_mitigated_executor(backend, target: str, shots: int = SHOTS, *,
                            num_randomizations: int = 32, random_state=None):
    """Float executor returning READOUT-corrected P(target) via TREX -- composable with ZNE/PEC.

    Accepts a Qiskit circuit (e.g. a ZNE-folded mirror circuit), relabels it to a LineQubit Cirq
    circuit, and returns trex_success_prob on `backend`. Because gate-noise mitigation (ZNE folding /
    PEC sampling) preserves the logical action, the target bitstring stays valid -> stacking this
    inside execute_with_zne(..., scale_noise=fold_global) gives TREX o ZNE: readout removed by TREX,
    gate noise removed by the extrapolation. (TREX only touches readout; ZNE/PEC only touch gates.)
    """
    def ex(qiskit_circuit) -> float:
        return trex_success_prob(
            to_cirq_linequbit(qiskit_circuit), mitiq_measurement_executor(backend, shots),
            target, num_randomizations=num_randomizations, random_state=random_state,
        )

    return ex

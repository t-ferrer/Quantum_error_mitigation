"""Entanglement (process) fidelity: a channel-level noise metric for the benchmark.

WHY. The benchmark's figure of merit is P(target bitstring) -- state-dependent, circuit-dependent,
and blind by construction to anything a computational-basis measurement cannot see (a channel that
dephases completely is invisible to it). Entanglement fidelity is the complementary, CHANNEL-level
quantity: purify the input rho into |psi>_RA against a fictitious reference R, let the channel act
on A ONLY, and measure the surviving overlap

    F_e(rho, N) = <psi_RA| (id_R (x) N)(|psi_RA><psi_RA|) |psi_RA>              (Schumacher 1996)

Schumacher's non-trivial result is that this does NOT depend on the purification chosen -- only on
(rho, N). For rho = I/d (the maximally mixed input, i.e. characterising a GATE rather than a state)
the purification is the maximally entangled state and F_e reduces to the Choi-state overlap that
qiskit calls `process_fidelity`, with the Kraus shortcut

    F_e = (1/d^2) sum_k |Tr K_k|^2        <- amplitudes summed BEFORE squaring: coherence-sensitive

Affinely equivalent to the average gate fidelity, F_avg = (d*F_e + 1)/(d + 1) (Horodecki et al. 1999;
simplified proof in Nielsen 2002). We expose the INFIDELITY 1 - F_e as the primary quantity: it is
the additive one (weak channels compose by summing infidelities), it is what hardware calibration
data reports, and it does not shed significant digits when small.

WHAT IT BUYS HERE.
  - A common currency across the NoiseSpec family. `p` (depolarizing), `gamma` (damping), T1/T2 and
    `eps` (biased Pauli) are structurally incomparable parameters; their entanglement infidelities
    live on one axis and add up over circuit depth. Sanity: Composite cx = 0.016863 vs
    thermal 0.006971 + mixed-unitary 0.009975 = 0.016946 (0.5% apart, second-order).
  - A quantitative reading of the `N*T*lambda << 1` validity regime, per model, as a number.
  - PEC verification: the quasi-probability decomposition is a LINEAR combination of channels and
    F_e is affine, so `qpd_effective_infidelity` checks a representation ALGEBRAICALLY (one gate,
    zero shots) instead of statistically (20 seeds x thousands of importance samples).

TWO PITFALLS, both load-bearing.
  1. NoiseModel exposes no public accessor for its errors (only `to_dict`/`noise_instructions`), so
     introspecting `_default_quantum_errors` would bind us to a private attribute. We instead read
     the channel END-TO-END out of Aer via `save_superop`: it measures what Aer ACTUALLY applies
     rather than what we believe was registered, and the same primitive scales from one gate to a
     whole circuit. (Verified identical to the private route to 0.00e+00 on Composite/sx.)
  2. A quasi-probability combination is NOT completely positive (negative coefficients), and
     `process_fidelity` defaults to `require_cp=True, require_tp=True` -- it raises on exactly the
     case we care about most. Every path that can see a QPD passes require_cp=require_tp=False.
     F_e stays a well-defined affine functional there; it just loses its probabilistic reading
     (it can exceed 1, which is precisely how over-mitigation shows up).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Choi, Operator, SuperOp, process_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as _NoiseModel

__all__ = [
    "channel_of",
    "error_channel",
    "channel_infidelity",
    "entanglement_fidelity",
    "entanglement_infidelity",
    "average_gate_infidelity",
    "gate_infidelities",
    "polarization",
    "infidelity_from_success_prob",
    "qpd_effective_infidelity",
    "gate_circuit",
    "benchmark_circuit_infidelity",
    "add_fidelity_columns",
    "FIDELITY_COLUMNS",
    "channel_asymptote",
    "fidelity_executor",
    "zne_channel_infidelity",
    "pec_channel_infidelity",
    "method_channel_infidelities",
]


# ---------------------------------------------------------------------------
# 1. Reading the channel out of Aer
# ---------------------------------------------------------------------------
def channel_of(circuit: QuantumCircuit, noise_model: "_NoiseModel | None" = None) -> SuperOp:
    """The superoperator Aer actually applies for `circuit` under `noise_model`.

    Uses the `superop` method + `save_superop`, so noise is inserted by Aer's own machinery --
    no introspection of NoiseModel internals, and the result reflects Aer's real behaviour
    (including any gate the model noises that we did not think about). Cost is set by the qubit
    count (a 2-qubit superop is 16x16), not by depth, so this stays cheap on the benchmark.
    """
    qc = circuit.copy()
    qc.save_superop()
    sim = AerSimulator(method="superop", noise_model=noise_model)
    return SuperOp(sim.run(qc).result().data(0)["superop"])


def error_channel(circuit: QuantumCircuit, noise_model: "_NoiseModel | None" = None) -> SuperOp:
    """Noise-only channel: (what Aer applies) composed with the inverse of the ideal unitary.

    `circuit` must be unitary (no measurements/resets) for `Operator(circuit)` to exist -- which
    is the case for gates and for the mirror benchmark circuits before `measure_all`.
    """
    return channel_of(circuit, noise_model).compose(SuperOp(Operator(circuit)).adjoint())


# ---------------------------------------------------------------------------
# 2. The metric itself
# ---------------------------------------------------------------------------
def channel_infidelity(channel, target=None) -> float:
    """1 - F_e for an ALREADY-BUILT channel (SuperOp/Choi/ndarray), vs `target` (default: identity).

    `require_cp/tp=False`: this is the entry point used for quasi-probability combinations, which
    are linear-but-not-CP maps. See the module docstring, pitfall 2.
    """
    if isinstance(channel, np.ndarray):
        channel = SuperOp(channel)
    return 1.0 - process_fidelity(
        Choi(channel), target=target, require_cp=False, require_tp=False
    )


def entanglement_fidelity(
    circuit: QuantumCircuit, noise_model: "_NoiseModel | None" = None
) -> float:
    """F_e of `circuit`'s error channel under `noise_model` (1.0 = noiseless)."""
    return 1.0 - entanglement_infidelity(circuit, noise_model)


def entanglement_infidelity(
    circuit: QuantumCircuit, noise_model: "_NoiseModel | None" = None
) -> float:
    """1 - F_e of `circuit`'s error channel. THE primary quantity -- additive, hardware-comparable.

    Works for a single gate or for a whole circuit: `circuit_infidelity` is just this called on the
    full mirror circuit, which is why there is only one function.
    """
    return channel_infidelity(error_channel(circuit, noise_model))


def average_gate_infidelity(entanglement_infid: float, num_qubits: int) -> float:
    """1 - F_avg from 1 - F_e, via F_avg = (d*F_e + 1)/(d + 1)  =>  1-F_avg = (d/(d+1)) * (1-F_e).

    Affine, so the two carry identical information; F_avg is the one with the direct operational
    reading ("average over Haar-random input states"), F_e the one with the useful algebra.
    """
    d = 2**num_qubits
    return entanglement_infid * d / (d + 1)


# ---------------------------------------------------------------------------
# 3. Per-gate breakdown
# ---------------------------------------------------------------------------
# Representative single-gate circuits. The angle chosen for `rz` is irrelevant to the result: the
# NoiseSpec family appends its channel AFTER the ideal gate, so the error channel -- and hence
# F_e -- is angle-independent. Fixed here only so the circuit is constructible.
_GATE_BUILDERS = {
    "id": lambda qc: qc.id(0),
    "x": lambda qc: qc.x(0),
    "y": lambda qc: qc.y(0),
    "z": lambda qc: qc.z(0),
    "h": lambda qc: qc.h(0),
    "s": lambda qc: qc.s(0),
    "sdg": lambda qc: qc.sdg(0),
    "sx": lambda qc: qc.sx(0),
    "sxdg": lambda qc: qc.sxdg(0),
    "rz": lambda qc: qc.rz(np.pi / 4, 0),
    "cx": lambda qc: qc.cx(0, 1),
    "cz": lambda qc: qc.cz(0, 1),
}
_TWO_QUBIT = {"cx", "cz"}


def gate_circuit(name: str) -> QuantumCircuit:
    """A one-gate circuit for `name`, sized to the gate's arity."""
    if name not in _GATE_BUILDERS:
        raise KeyError(f"no builder for gate '{name}'; known: {sorted(_GATE_BUILDERS)}")
    qc = QuantumCircuit(2 if name in _TWO_QUBIT else 1)
    _GATE_BUILDERS[name](qc)
    return qc


def gate_infidelities(
    noise_model: _NoiseModel, gates: "list[str] | None" = None
) -> "dict[str, float]":
    """{gate name -> 1 - F_e} for every gate the model actually noises.

    Defaults to `noise_model.noise_instructions` -- the PUBLIC accessor, and exactly the set of
    instructions Aer attaches an error to, so nothing is silently missed or invented.
    """
    names = list(noise_model.noise_instructions) if gates is None else list(gates)
    return {
        g: entanglement_infidelity(gate_circuit(g), noise_model)
        for g in names
        if g in _GATE_BUILDERS
    }


# ---------------------------------------------------------------------------
# 4. Bridge to the benchmark's own observable
# ---------------------------------------------------------------------------
# Pauli-randomised mirror circuits (Proctor et al.) twirl the circuit noise towards a global
# depolarizing channel, under which the success probability and the polarization gamma satisfy
#     P_s = gamma + (1 - gamma)/2^n      =>      gamma = (2^n * P_s - 1)/(2^n - 1)
# and the depolarizing channel of polarization gamma has F_e = gamma + (1 - gamma)/4^n.
# APPROXIMATE by construction (the effective channel is only approximately depolarizing; the exact
# MRB estimator uses the Hamming-weighted polarization, not P(success) alone). Empirically ~4% from
# the a-priori per-gate product on Composite/ThermalRealistic. Fit for reporting and figures --
# deliberately NOT asserted in the smoke tests, where it would be a flaky test of physics modelling
# rather than of code.
def polarization(p_success: float, num_qubits: int) -> float:
    """Effective global-depolarizing polarization implied by a success probability."""
    d = 2**num_qubits
    return (d * p_success - 1.0) / (d - 1.0)


def infidelity_from_success_prob(p_success: float, num_qubits: int) -> float:
    """1 - F_e implied by P(target), assuming globally-depolarizing effective noise.

    Can go NEGATIVE on over-mitigated estimates (P_s > 1); that is informative, not a bug -- it
    flags an effective map outside the physical (CP) set.
    """
    g = polarization(p_success, num_qubits)
    return (1.0 - g) * (1.0 - 1.0 / 4**num_qubits)


# ---------------------------------------------------------------------------
# 5. PEC: algebraic verification of a quasi-probability representation
# ---------------------------------------------------------------------------
def qpd_effective_infidelity(representation, noise_model: _NoiseModel) -> float:
    """1 - F_e of the map PEC effectively realises with `representation` under `noise_model`.

    PEC claims  ideal = sum_i eta_i * noisy(B_i).  Because F_e is affine, we can evaluate that claim
    directly: run every basis circuit B_i through Aer (real noise), combine the superoperators with
    the representation's own eta_i, and compare the result to the ideal unitary. Zero shots, and the
    answer is deterministic.

    Returns ~0 when the representation matches the noise, and jumps to the order of the UNMITIGATED
    infidelity when it does not -- verified as a negative control in smoke_fidelity_pec.py (matched
    3e3-1e6x suppression, mismatched 0-2x). Sign is not meaningful: the combination is non-CP, so
    the residual can land either side of zero.

    Note the residual is NOT machine-precision even when everything is correct: the representations
    model noise as acting once after the whole basis operation, whereas Aer re-applies it after each
    noised recovery gate (`x` is in the noised set, `y`/`z` are not). That mismatch is second order
    -- |eta_recovery| x eps_gate, e.g. 1.6e-05 on Depolarizing/cx -- so test on the suppression
    FACTOR, never on an absolute machine-precision tolerance.
    """
    from mitiq.interface.conversions import convert_from_mitiq

    ideal_qiskit = convert_from_mitiq(representation.ideal, "qiskit")
    acc = None
    for eta, noisy_op in zip(representation.coeffs, representation.noisy_operations):
        sop = channel_of(convert_from_mitiq(noisy_op.circuit, "qiskit"), noise_model).data
        acc = eta * sop if acc is None else acc + eta * sop
    effective = SuperOp(acc).compose(SuperOp(Operator(ideal_qiskit)).adjoint())
    return channel_infidelity(effective)


# ---------------------------------------------------------------------------
# 6. Channel-level figure of merit for each mitigation method
# ---------------------------------------------------------------------------
# The benchmark's MSE mixes the METHOD's systematic error with the ESTIMATOR's sampling noise.
# Evaluating each method's figure of merit on the CHANNEL separates them: it is deterministic and
# shot-free, so whatever remains is pure method bias. (The seed_simulator artefact took weeks to
# untangle statistically; this view answers the same question in one deterministic call.)
#
# The enabling observation is that F_e is itself an expectation value,
#     F_e = <Phi+| (id (x) N)(|Phi+><Phi+|) |Phi+>  =  <O>  with O = |Phi+><Phi+| ,
# so "ZNE of F_e" is not some new estimator -- it is the benchmark's own ZNE applied to a different
# observable. Handing mitiq an executor that returns the EXACT F_e of the folded circuit therefore
# works for every factory, ADAPTIVE ONES INCLUDED: mitiq only ever calls the executor at the scale
# factors it picks, and never learns that the value came from a superoperator instead of shots.
#
# MEASURED, not assumed: extrapolating F_e is NOT more accurate than extrapolating P(target). Over
# the benchmark's own scale factors the two are a wash (Exp: F_e wins 2.4x on MixedUnitary, loses
# 0.7x on ThermalRealistic). F_e earns its place by measuring something P(target) cannot see -- a
# computational-basis success probability is blind to pure dephasing -- not by fitting better.


def channel_asymptote(num_qubits: int) -> float:
    """The lambda -> infinity fixed point of F_e: 1/d^2 (= 1/16 for 2 qubits).

    ANY channel that discards its input, N(rho) = sigma * Tr(rho), has
        <Phi+| (I/d (x) sigma) |Phi+> = (1/d^2) Tr(sigma) = 1/d^2 ,
    independently of sigma. So the fixed point is the same whether the noise scrambles towards the
    maximally mixed state (depolarizing) or towards |0..0> (amplitude damping) -- unlike P(target),
    whose fixed point is circuit-dependent and forced `asymptote=None` on the non-unital specs
    (cf. the per-seed spread noted on SPEC_AD). Verified: at lambda=35 both SPEC_DEPOL and SPEC_AD
    sit at 0.0625 with per-seed std 0.0000.

    Worth having: on SPEC_DEPOL the free 3-parameter Exp fit fails on 4 seeds out of 6, while fixing
    the asymptote here converges everywhere and cuts the extrapolation error from 0.175 to 0.042.
    Caveat: weak-noise models are still far from the fixed point at the benchmark's scale factors,
    so this helps the FIT's conditioning rather than guaranteeing a better extrapolation.
    """
    return 1.0 / 4**num_qubits


def fidelity_executor(noise_model: "_NoiseModel | None"):
    """A mitiq executor returning the EXACT F_e of whatever circuit it is handed.

    Drop-in wherever the benchmark passes `mitiq_executor`, which is the whole point: the mitigation
    machinery is exercised unchanged, only the observable differs. Deterministic -- no shots, no
    variance -- so a ZNE run through this executor yields the method's channel-level bias alone.
    """

    def ex(circuit) -> float:
        return 1.0 - entanglement_infidelity(circuit, noise_model)

    return ex


def zne_channel_infidelity(circuit, noise_model, factory) -> float:
    """1 - F_e of the channel ZNE effectively realises, via `factory`, on `circuit`.

    Same call the benchmark makes (`execute_with_zne` + `fold_global`), with the fidelity executor
    substituted. Returns NaN if the fit fails to converge, mirroring `_zne_or_nan` in benchmark.py.

    For LINEAR extrapolators (Richardson, Poly*) this is exactly the entanglement infidelity of the
    effective map sum_i c_i N_{lambda_i}: the c_i are fixed, and F_e is affine, so extrapolating the
    fidelities equals the fidelity of the extrapolated channel. For Exp/AdaExp the fit is nonlinear,
    so no such effective channel exists -- the number is still the mitigated fidelity estimate, just
    without that second reading.
    """
    from mitiq import zne
    from mitiq.zne.inference import ExtrapolationError
    from mitiq.zne.scaling import fold_global

    try:
        return 1.0 - float(
            zne.execute_with_zne(
                circuit, fidelity_executor(noise_model), factory=factory,
                scale_noise=fold_global,
            )
        )
    except ExtrapolationError:
        return float("nan")


_EFF_GATE_CACHE: dict = {}


def pec_channel_infidelity(circuit, representations, noise_model) -> float:
    """1 - F_e of the channel PEC effectively realises over the WHOLE `circuit`.

    PEC's quasi-probability identity is per-gate, so the circuit-level effective map is the ORDERED
    COMPOSITION of the per-gate effective maps sum_i eta_i S_i -- no importance sampling required,
    because we are evaluating the identity rather than estimating an expectation through it. Exact
    and deterministic, where the shot-based route needs thousands of samples to resolve a residual
    of order 1e-7.

    Per-gate effective channels are cached on (gate name, params, qubits): a mirror circuit reuses
    ~10 distinct instructions, so the Aer superop calls are paid once rather than per gate.
    """
    from mitiq.interface.conversions import convert_from_mitiq

    n = circuit.num_qubits
    total = SuperOp(np.eye(4**n))
    nm_id = id(noise_model)
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier"):
            continue
        qubits = tuple(circuit.find_bit(q).index for q in instr.qubits)
        key = (nm_id, op.name, tuple(round(float(p), 10) for p in op.params), qubits, n)
        if key not in _EFF_GATE_CACHE:
            rep = _match_representation(representations, op, qubits)
            if rep is None:
                raise ValueError(
                    f"no PEC representation matches '{op.name}' on qubits {qubits}; "
                    "the gate would be left un-mitigated (cf. the cx(1,0) bug)"
                )
            acc = None
            for eta, noisy_op in zip(rep.coeffs, rep.noisy_operations):
                sub = convert_from_mitiq(noisy_op.circuit, "qiskit")
                embedded = QuantumCircuit(n)
                embedded.compose(sub, qubits=list(qubits), inplace=True)
                s = channel_of(embedded, noise_model).data
                acc = eta * s if acc is None else acc + eta * s
            _EFF_GATE_CACHE[key] = acc
        # circuit order: everything so far, THEN this gate. `.compose` is explicit about that;
        # SuperOp's `@` has the opposite convention and silently reverses the circuit.
        total = total.compose(SuperOp(_EFF_GATE_CACHE[key]))
    ideal = SuperOp(Operator(circuit))
    return channel_infidelity(total.compose(ideal.adjoint()))


def _match_representation(representations, op, qubits):
    """The representation for `op`, in its CANONICAL qubit order.

    Orientation is handled by the embedding, not by picking a different representation: composing a
    canonical cx(0,1) rep onto `qubits=(1,0)` maps rep-qubit 0 -> circuit qubit 1, which IS cx(1,0),
    recovery Paulis and all. So we deliberately take the canonical rep and let `compose(..., qubits)`
    place it -- selecting the pre-reversed cx(1,0) rep here would reverse it a second time.

    (Mitiq itself needs both orientations because its matcher is qubit-dependent and cannot re-embed;
    that is why noise_models.py builds the pair. Here only the canonical one is used.)
    """
    from mitiq.interface.conversions import convert_from_mitiq

    for rep in representations:
        if len(rep.ideal.all_qubits()) != op.num_qubits:
            continue
        ideal_qk = convert_from_mitiq(rep.ideal, "qiskit")
        rep_instr = ideal_qk.data[0]
        if rep_instr.operation.name != op.name:
            continue
        if len(rep_instr.operation.params) != len(op.params) or not all(
            abs(float(a) - float(b)) < 1e-10
            for a, b in zip(rep_instr.operation.params, op.params)
        ):
            continue
        rep_qubits = tuple(ideal_qk.find_bit(q).index for q in rep_instr.qubits)
        if rep_qubits == tuple(range(op.num_qubits)):  # canonical order only
            return rep
    return None


def method_channel_infidelities(spec, seeds, drift_models=None) -> "dict[str, float]":
    """{method -> channel-level 1 - F_e}, averaged over `seeds`. Deterministic, shot-free.

    Mirrors run_benchmark's method list and its factory settings, with ONE deliberate difference:
    the Exp/AdaExp asymptote is `channel_asymptote(n)` = 1/d^2, not `spec.asymptote`. The latter is
    the fixed point of P(target) (0.25 for 2 qubits, or None where it varies per seed); F_e has its
    own, and unlike P(target)'s it is the same 1/d^2 for unital and non-unital noise alike, so it is
    always available -- no free-asymptote fallback is needed here.

    Richardson, Poly2 and Exp all read the SAME scale factors, so the folded-circuit fidelity curve
    is built ONCE per seed and the three extrapolate off it -- verified identical to routing each
    through execute_with_zne (max deviation 6e-17). That removes a 3x redundancy on the dominant
    cost and takes 20 seeds from ~20 s to ~7 s. Only AdaExp keeps the full path, since it chooses
    its own scale factors adaptively.

    Still far too slow for cached_benchmark's millisecond reloads, hence the separate on-disk cache
    in qem/cache.py; run_benchmark computes it inline, where it is 0.2% of the run.
    """
    from mitiq.zne.inference import (
        AdaExpFactory, ExpFactory, ExtrapolationError, PolyFactory, RichardsonFactory,
    )
    from mitiq.zne.scaling import fold_global

    from .circuits import make_benchmark_circuit
    from .config import ADA_STEPS, SCALE_FACTORS

    sf = spec.scale_factors or list(SCALE_FACTORS)
    models = (
        list(drift_models) if drift_models is not None
        else [spec.build_noise_model()] * len(seeds)
    )
    acc = {k: [] for k in ("Noisy", "Richardson", "Poly2", "Exp", "AdaExp")}
    run_pec = spec.build_representations is not None
    if run_pec:
        acc["PEC"] = []

    def _or_nan(fn):
        try:
            return 1.0 - float(fn())
        except ExtrapolationError:  # as _zne_or_nan does: one bad fit must not kill the sweep
            return float("nan")

    for seed, nm in zip(seeds, models):
        circ = make_benchmark_circuit(seed)
        a = channel_asymptote(circ.num_qubits)
        acc["Noisy"].append(entanglement_infidelity(circ, nm))
        fes = [1.0 - entanglement_infidelity(fold_global(circ, l), nm) for l in sf]
        acc["Richardson"].append(_or_nan(lambda: RichardsonFactory.extrapolate(sf, fes)))
        acc["Poly2"].append(_or_nan(lambda: PolyFactory.extrapolate(sf, fes, order=2)))
        acc["Exp"].append(_or_nan(lambda: ExpFactory.extrapolate(sf, fes, asymptote=a)))
        acc["AdaExp"].append(
            zne_channel_infidelity(
                circ, nm, AdaExpFactory(steps=ADA_STEPS, scale_factor=2.0, asymptote=a)
            )
        )
        if run_pec:
            acc["PEC"].append(
                pec_channel_infidelity(circ, spec.build_representations(circ), nm)
            )
    return {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in acc.items()}


# ---------------------------------------------------------------------------
# 7. Plumbing into the benchmark DataFrame
# ---------------------------------------------------------------------------
# infid_channel   the method's systematic error on the CHANNEL: exact, deterministic, zero shots
# infid_psucc     the same method read off its P(target) estimate, through the polarization bridge
#
# The pair is the point. infid_psucc alone would be worthless -- it is an invertible function of
# `mean`, so it is blind to exactly what P(target) is blind to. Beside its exact counterpart it
# becomes the empirical half of a bias/variance split: infid_channel is the method bias, and the gap
# between the two is sampling variance plus the depolarizing-bridge approximation. The names say
# which is which, so the two can never be read as the same kind of number.
#
# There is no separate "noise level" column: that is just the Noisy row of infid_channel.
FIDELITY_COLUMNS = ("infid_channel", "infid_psucc")


def benchmark_circuit_infidelity(
    seeds, noise_model=None, drift_models=None
) -> float:
    """Mean exact entanglement infidelity of the mirror circuits over `seeds`.

    This is the A-PRIORI noise level of the benchmark: computed from the channel Aer applies to the
    whole circuit, with no shots and no estimator involved. It is the honest x-axis against which
    mitigation quality should be plotted -- and it is exact, unlike the per-gate approximations
    (on Composite: exact 0.1856, vs 0.2074 summing per-gate infidelities and 0.1882 taking their
    product; the gate-wise routes ignore how the channels actually interleave).

    Under drift, pass `drift_models` (one per seed, aligned to `seeds` as in run_benchmark): each
    run then gets its own theta_t and the mean is over the trajectory.
    """
    from .circuits import make_benchmark_circuit  # local: keeps the metric module standalone

    if drift_models is not None:
        if len(drift_models) != len(seeds):
            raise ValueError(
                f"drift_models has {len(drift_models)} entries but there are {len(seeds)} seeds"
            )
        models = list(drift_models)
    else:
        models = [noise_model] * len(seeds)
    vals = [
        entanglement_infidelity(make_benchmark_circuit(s), m) for s, m in zip(seeds, models)
    ]
    return float(np.mean(vals))


def add_fidelity_columns(df, spec, seeds, drift_models=None, channel=None):
    """Return `df` with `infid_channel` and `infid_psucc` attached (no-op if both already present).

    `channel` accepts a precomputed {method -> value} mapping, which is how cached_benchmark feeds
    in the disk-cached channel numbers instead of spending ~20 s recomputing them on every reload.
    Pass None to compute them here (what run_benchmark does, where 20 s is 0.2% of the run).

    infid_psucc is pure post-processing of the already-computed `mean`, so it costs nothing and can
    always be backfilled onto old pickles -- the cache key does not hash run_benchmark's source, so
    those entries stay valid and should never be discarded over a derived column.
    """
    from .circuits import make_benchmark_circuit

    if all(c in df.columns for c in FIDELITY_COLUMNS):
        return df
    n = make_benchmark_circuit(seeds[0]).num_qubits
    if channel is None:
        channel = method_channel_infidelities(spec, seeds, drift_models=drift_models)
    out = df.copy()
    methods = out.index.get_level_values("method")
    out["infid_channel"] = [
        round(channel[m], 9) if m in channel and channel[m] == channel[m] else float("nan")
        for m in methods
    ]
    out["infid_psucc"] = [
        round(infidelity_from_success_prob(float(m), n), 6) if m == m else float("nan")
        for m in out["mean"]
    ]
    return out

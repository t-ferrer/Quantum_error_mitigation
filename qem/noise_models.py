"""All NoiseSpec instances: depolarizing, amplitude/phase damping, thermal (gamma and realistic)."""

import numpy as np
from mitiq import pec
from mitiq.pec.types import OperationRepresentation
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus
from qiskit_aer.noise import NoiseModel as _NoiseModel
from qiskit_aer.noise import (
    depolarizing_error,
    kraus_error,
    pauli_error,
    thermal_relaxation_error,
)
from mitiq.pec.representations import represent_operation_with_local_biased_noise

from .benchmark import NoiseSpec
from .config import _GATES_1Q, _GATES_2Q, P1, P2, P_MITIQ_1Q, P_MITIQ_2Q
from .kraus import kraus_amplitude_damping_1q, kraus_phase_damping_1q, kraus_thermal_1q
from .pec_core import (
    _LP_REP_CACHE,
    _build_with_cache,
    represent_1q_qiskit_with_ad_takagi,
    represent_op_with_kraus_lp,
)
from .readout import READOUT_REALISTIC


# ---------------------------------------------------------------------------
# 1st concrete spec : DEPOLARIZING (with the CX global fix)
# ---------------------------------------------------------------------------
def _depol_noise_model() -> _NoiseModel:
    nm_ = _NoiseModel()
    nm_.add_all_qubit_quantum_error(
        depolarizing_error(P1, 1), ["h", "rz", "sx", "x", "sxdg", "id"]
    )
    nm_.add_all_qubit_quantum_error(depolarizing_error(P2, 2), ["cx"])
    return nm_


def _depol_build_reps(circuit: QuantumCircuit) -> list[OperationRepresentation]:
    # 1q local (== global on 1q)
    all_reps = pec.represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, noise_level=P_MITIQ_1Q
    )
    reps_1q = [r for r in all_reps if len(r.ideal.all_qubits()) == 1]
    # CX rep for BOTH orientations (cx(1,0) otherwise left un-mitigated -> PEC bias)
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        qc_cx = QuantumCircuit(2)
        qc_cx.cx(ctrl, tgt)
        cx_reps.append(
            pec.represent_operation_with_global_depolarizing_noise(
                qc_cx, noise_level=P_MITIQ_2Q
            )
        )
    return reps_1q + cx_reps


SPEC_DEPOL = NoiseSpec(
    name="Depolarizing",
    build_noise_model=_depol_noise_model,
    build_representations=_depol_build_reps,
    asymptote=0.25,  # 2q maximally-mixed state -> P(|00>) = 1/4
)


# §6.1  Amplitude damping spec
GAMMA_AD_1Q = 0.01


GAMMA_AD_2Q = 0.05  # per qubit on each CX


_kraus_ad_1q = kraus_amplitude_damping_1q(GAMMA_AD_1Q)


_kraus_ad_2qq = kraus_amplitude_damping_1q(GAMMA_AD_2Q)


def _ad_noise_model() -> _NoiseModel:
    err_1q = kraus_error(_kraus_ad_1q)
    err_2q_local = kraus_error(_kraus_ad_2qq)
    err_2q = err_2q_local.tensor(err_2q_local)
    nm_ = _NoiseModel()
    nm_.add_all_qubit_quantum_error(err_1q, list(_GATES_1Q))
    nm_.add_all_qubit_quantum_error(err_2q, list(_GATES_2Q))
    return nm_


# 1q reps via Takagi (universal, robust); 2q CX via LP.
def _ad_build_reps(circuit) -> list[OperationRepresentation]:
    return _build_with_cache(
        circuit,
        _kraus_ad_1q,
        _kraus_ad_2qq,
        "AD",
        include_reset=True,
        gamma_1q_for_takagi=GAMMA_AD_1Q,
    )


SPEC_AD = NoiseSpec(
    name="AmplitudeDamping",
    build_noise_model=_ad_noise_model,
    build_representations=_ad_build_reps,
    asymptote=None,  # per-seed P(|00>) varies (std=0.18 at lambda=50) because U^-1 acts on the
    # AD-relaxed state -> per-seed fixed point depends on Clifford. Empirical:
    # asymptote=None drops Exp MSE 0.10 -> 0.014 (x7).
    scale_factors=[
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
    ],  # >=4 points required for the 3-param free fit
)


# §6.2  Phase damping spec
LAMBDA_PD_1Q = 0.01


LAMBDA_PD_2Q = 0.05


_kraus_pd_1q = kraus_phase_damping_1q(LAMBDA_PD_1Q)


_kraus_pd_2qq = kraus_phase_damping_1q(LAMBDA_PD_2Q)


def _pd_noise_model() -> _NoiseModel:
    err_1q = kraus_error(_kraus_pd_1q)
    err_2q_local = kraus_error(_kraus_pd_2qq)
    err_2q = err_2q_local.tensor(err_2q_local)
    nm_ = _NoiseModel()
    nm_.add_all_qubit_quantum_error(err_1q, list(_GATES_1Q))
    nm_.add_all_qubit_quantum_error(err_2q, list(_GATES_2Q))
    return nm_


def _pd_build_reps(circuit) -> list[OperationRepresentation]:
    reps_1q, seen = [], set()
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier") or op.num_qubits != 1:
            continue
        key = ("PD1", op.name, tuple(round(float(p), 10) for p in op.params))
        if key in seen:
            continue
        seen.add(key)
        if key in _LP_REP_CACHE:
            reps_1q.append(_LP_REP_CACHE[key])
        else:
            qc = QuantumCircuit(1)
            qc.append(op, [0])
            rep = represent_op_with_kraus_lp(qc, _kraus_pd_1q)
            _LP_REP_CACHE[key] = rep
            reps_1q.append(rep)
    # CX rep for BOTH orientations (cx(1,0) otherwise left un-mitigated -> PEC bias)
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        cx_key = ("PD2", "cx", (ctrl, tgt))
        if cx_key not in _LP_REP_CACHE:
            qc = QuantumCircuit(2)
            qc.cx(ctrl, tgt)
            _LP_REP_CACHE[cx_key] = represent_op_with_kraus_lp(qc, _kraus_pd_2qq)
        cx_reps.append(_LP_REP_CACHE[cx_key])
    return reps_1q + cx_reps


SPEC_PD = NoiseSpec(
    name="PhaseDamping",
    build_noise_model=_pd_noise_model,
    build_representations=_pd_build_reps,
    asymptote=0.25,
)


# §6.3  Thermal relaxation spec (AD + PD composite, 4 Kraus operators)
GAMMA_TH_AD_1Q = 0.005
GAMMA_TH_PD_1Q = 0.005


GAMMA_TH_AD_2Q = 0.025
GAMMA_TH_PD_2Q = 0.025


_kraus_th_1q = kraus_thermal_1q(GAMMA_TH_AD_1Q, GAMMA_TH_PD_1Q)


_kraus_th_2qq = kraus_thermal_1q(GAMMA_TH_AD_2Q, GAMMA_TH_PD_2Q)


def _th_noise_model() -> _NoiseModel:
    err_1q = kraus_error(_kraus_th_1q)
    err_2q_local = kraus_error(_kraus_th_2qq)
    err_2q = err_2q_local.tensor(err_2q_local)
    nm_ = _NoiseModel()
    nm_.add_all_qubit_quantum_error(err_1q, list(_GATES_1Q))
    nm_.add_all_qubit_quantum_error(err_2q, list(_GATES_2Q))
    return nm_


# 1q reps: Takagi AD approximation (treats AD part exactly, ignores PD contribution).
# 2q CX rep: full thermal LP.
def _th_build_reps(circuit) -> list[OperationRepresentation]:
    return _build_with_cache(
        circuit,
        _kraus_th_1q,
        _kraus_th_2qq,
        "TH",
        include_reset=True,
        gamma_1q_for_takagi=GAMMA_TH_AD_1Q,
    )


SPEC_TH = NoiseSpec(
    name="ThermalRelaxation",
    build_noise_model=_th_noise_model,
    build_representations=_th_build_reps,
    asymptote=None,  # per-seed asymptote variance (std~0.13) too high for a single global value;
    # let the 3-param fit determine `a` per circuit. Requires >=4 scale_factors.
    scale_factors=[
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
    ],  # 5 points to stabilise ExpFactory on composite noise
)


# Section 6.4  Realistic thermal relaxation: thermal_relaxation_error(T1, T2, gate_time)
T1_S = 100e-6  # 100 us
T2_S = 80e-6  # 80 us   (T2 <= 2*T1 constraint satisfied)
GATE_TIME_S = {  # IBM-style durations, in seconds
    "rz": 0.0,
    "id": 0.0,  # rz virtual -> no noise added
    "sx": 35e-9,
    "sxdg": 35e-9,
    "x": 35e-9,
    "h": 35e-9,
    "cx": 400e-9,  # 2q long -> dominant relaxation
}


def _th_real_noise_model() -> _NoiseModel:
    nm_ = _NoiseModel()
    for g in _GATES_1Q:  # ('h','rz','sx','x','sxdg','id')
        nm_.add_all_qubit_quantum_error(
            thermal_relaxation_error(T1_S, T2_S, GATE_TIME_S[g]), [g]
        )
    err_cx_1q = thermal_relaxation_error(T1_S, T2_S, GATE_TIME_S["cx"])
    nm_.add_all_qubit_quantum_error(err_cx_1q.tensor(err_cx_1q), list(_GATES_2Q))
    return nm_


# PEC for ThermalRealistic.
# Aer applies thermal_relaxation_error(T1,T2,t) per gate. That zero-temperature, T2<=T1
# channel is EXACTLY the composite AD(gamma_ad) o PD(gamma_pd) for the gammas derived below
# (verified to machine precision: ||S_AD.PD - S_qiskit||_max ~ 1e-15 for every gate time).
# So PEC inverts it with no model mismatch -> unbiased.
#
#   AD population decay:  1 - gamma_ad = exp(-t/T1)
#   Total coherence must decay as exp(-t/T2); AD already gives exp(-t/2T1), so PD supplies
#   the rest:  sqrt(1-gamma_pd) = exp(-t/T2 + t/2T1)  =>  gamma_pd = 1 - exp(-2t/T2 + t/T1).
def _thermal_gammas(t1: float, t2: float, t: float) -> tuple[float, float]:
    if t <= 0.0:
        return 0.0, 0.0  # virtual gate (rz/id): zero duration -> identity channel
    gamma_ad = 1.0 - np.exp(-t / t1)
    gamma_pd = 1.0 - np.exp(-2.0 * t / t2 + t / t1)
    return gamma_ad, gamma_pd


def _th_real_build_reps(circuit) -> list[OperationRepresentation]:
    """Per-gate thermal QPRs from each gate's own duration (gammas vary by gate).

    1q reps are EXACT (full AD+PD via LP), unlike SPEC_TH which uses the AD-only Takagi
    approximation. This is feasible because the only complex gate, rz, carries zero noise
    here (gate_time=0): the noised physical gates sx/sxdg/x/h are all LP-feasible, while
    rz/id (gamma=0) get an exact identity rep via Takagi (robust to any rz angle, where the
    Pauli+reset LP would be infeasible).
    """
    reps_1q, seen = [], set()
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier") or op.num_qubits != 1:
            continue
        key = ("THR1", op.name, tuple(round(float(p), 10) for p in op.params))
        if key in seen:
            continue
        seen.add(key)
        if key not in _LP_REP_CACHE:
            g_ad, g_pd = _thermal_gammas(T1_S, T2_S, GATE_TIME_S[op.name])
            qc = QuantumCircuit(1)
            qc.append(op, [0])
            if g_ad == 0.0 and g_pd == 0.0:
                # zero-duration gate (rz/id) -> identity rep; Takagi(0) = exact identity for
                # ANY 1q U, sidestepping the LP infeasibility on arbitrary rz rotations.
                _LP_REP_CACHE[key] = represent_1q_qiskit_with_ad_takagi(qc, 0.0)
            else:
                _LP_REP_CACHE[key] = represent_op_with_kraus_lp(
                    qc, kraus_thermal_1q(g_ad, g_pd), include_reset=True
                )
        reps_1q.append(_LP_REP_CACHE[key])

    # CX rep for BOTH orientations (cx(1,0) otherwise left un-mitigated -> PEC bias).
    g_ad_cx, g_pd_cx = _thermal_gammas(T1_S, T2_S, GATE_TIME_S["cx"])
    kraus_cx = kraus_thermal_1q(g_ad_cx, g_pd_cx)
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        cx_key = ("THR2", "cx", (ctrl, tgt))
        if cx_key not in _LP_REP_CACHE:
            qc = QuantumCircuit(2)
            qc.cx(ctrl, tgt)
            _LP_REP_CACHE[cx_key] = represent_op_with_kraus_lp(
                qc, kraus_cx, include_reset=True
            )
        cx_reps.append(_LP_REP_CACHE[cx_key])
    return reps_1q + cx_reps


SPEC_TH_REAL = NoiseSpec(
    name="ThermalRealistic",
    build_noise_model=_th_real_noise_model,
    build_representations=_th_real_build_reps,
    asymptote=None,  # per-seed fixed point varies -> 3-param fit
    scale_factors=[1.0, 1.5, 2.0, 3.0, 5.0],  # >=4 points for the free fit (cf. AD/TH)
)


# Section 6.5  Mixed-unitary noise: a biased Pauli channel (combined depol + dephasing)
# A Pauli channel IS a mixed-unitary channel (convex combo of Pauli unitaries). Mitiq ships
# an *analytic* QPR for exactly this channel, so PEC needs no LP machinery (unlike AD/thermal).
# Channel (Strikis 2021):  D(eps) = (1-eps)[I] + eps*( eta/(eta+1) Z + 1/(3(eta+1))(X+Y+Z) )
# eta=0 -> pure depolarizing ; eta->inf -> pure dephasing. eta>0 adds an X/Y/Z asymmetry
# that plain depolarizing lacks.
#
# REALISTIC parameterisation (toward the composite hardware model):
#   - eps small: hardware coherent/biased-Pauli residual after calibration is well below the
#     incoherent budget (IBM CX total ~7e-3). Per-gate eps_1q=1e-3, eps_2q=5e-3 keeps the
#     noisy bias ~ -0.1, in the same range as ThermalRealistic (not the elevated pedagogical
#     0.01/0.05 that made this the dominant noise source).
#   - eta=4: noise biased toward dephasing (Z), as on real devices (T2<T1, ZZ terms).
#   - Noise on PHYSICAL gates only: rz/id are virtual (zero-duration) on hardware -> NO error,
#     matching ThermalRealistic's GATE_TIME rz=id=0. PEC builds an eps=0 (identity) rep for
#     those so they are left untouched without tripping the all-ops-represented guardrail.
EPS_MU_1Q = 1e-3
EPS_MU_2Q = 5e-3  # per qubit on each CX
ETA_MU = 4.0  # bias toward dephasing
_MU_PHYS_1Q = tuple(g for g in _GATES_1Q if g not in ("rz", "id"))  # noised (sx,sxdg,x,h)


def _mu_pauli_error(eps: float):
    """Aer pauli_error matching Mitiq's biased channel D(eps, ETA_MU) (probs sum to 1)."""
    p_x = p_y = eps / (3 * (ETA_MU + 1))
    p_z = eps * ETA_MU / (ETA_MU + 1) + eps / (3 * (ETA_MU + 1))
    return pauli_error([("I", 1 - eps), ("X", p_x), ("Y", p_y), ("Z", p_z)])


def _mu_noise_model() -> _NoiseModel:
    err_1q = _mu_pauli_error(EPS_MU_1Q)
    err_2q_local = _mu_pauli_error(EPS_MU_2Q)
    err_2q = err_2q_local.tensor(err_2q_local)
    nm_ = _NoiseModel()
    nm_.add_all_qubit_quantum_error(err_1q, list(_MU_PHYS_1Q))  # rz/id virtual -> no noise
    nm_.add_all_qubit_quantum_error(err_2q, list(_GATES_2Q))
    return nm_


def _mu_build_reps(circuit) -> list[OperationRepresentation]:
    # 1q reps: analytic biased rep, qubit-agnostic so one matches the gate on every qubit.
    # Virtual gates (rz/id) get eps=0 -> identity rep (coeff [1,0,0,0], norm 1): they carry no
    # noise, so PEC must leave them untouched (an eps>0 rep would over-correct phantom noise).
    reps_1q, seen = [], set()
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier") or op.num_qubits != 1:
            continue
        key = ("MU1", op.name, tuple(round(float(p), 10) for p in op.params))
        if key in seen:
            continue
        seen.add(key)
        if key not in _LP_REP_CACHE:
            eps = EPS_MU_1Q if op.name in _MU_PHYS_1Q else 0.0
            qc = QuantumCircuit(1)
            qc.append(op, [0])
            _LP_REP_CACHE[key] = represent_operation_with_local_biased_noise(
                qc, eps, ETA_MU, is_qubit_dependent=False
            )
        reps_1q.append(_LP_REP_CACHE[key])
    # CX rep for BOTH orientations (cx(1,0) otherwise left un-mitigated -> PEC bias)
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        cx_key = ("MU2", "cx", (ctrl, tgt))
        if cx_key not in _LP_REP_CACHE:
            qc = QuantumCircuit(2)
            qc.cx(ctrl, tgt)
            _LP_REP_CACHE[cx_key] = represent_operation_with_local_biased_noise(
                qc, EPS_MU_2Q, ETA_MU, is_qubit_dependent=True
            )
        cx_reps.append(_LP_REP_CACHE[cx_key])
    return reps_1q + cx_reps


SPEC_MIXED_UNITARY = NoiseSpec(
    name="MixedUnitary",
    build_noise_model=_mu_noise_model,
    build_representations=_mu_build_reps,
    asymptote=0.25,  # unital & mixing (depol component) -> maximally-mixed fixed point,
    # P(|00>)=1/4 EXACTLY for every seed (unlike non-unital AD/thermal, which need asymptote=None).
    # Keeping it fixed = 2-param Exp fit, more stable on this weak noise.
    scale_factors=[1.0, 1.5, 2.0, 3.0, 5.0],  # >=4 points: weak near-linear signal needs them
)


# Section 6.6  Composite hardware model: ThermalRealistic ∘ MixedUnitary (biased Pauli).
# The realistic device picture stacks two physically-distinct mechanisms per gate:
#   - incoherent T1/T2 relaxation (ThermalRealistic, §6.4, non-unital), and
#   - the biased-Pauli residual left after calibration (MixedUnitary, §6.5).
# NO separate depolarizing_error is added: the MixedUnitary channel ALREADY contains a
# depolarizing component -- its symmetric term 1/(3(eta+1))(X+Y+Z) IS depolarizing (eta=0 ->
# pure depol), with eta=4 adding the extra Z (dephasing) bias. A standalone depol channel would
# double-count the isotropic Pauli error.
#
# Both components act on PHYSICAL gates only (sx, sxdg, x, h, cx); rz/id are virtual (zero
# duration, eps=0) -> no noise, consistent with BOTH parent models. The two Aer QuantumErrors
# are composed per gate (thermal first, then the biased-Pauli kick) into a single channel.
def _composite_noise_model() -> _NoiseModel:
    nm_ = _NoiseModel()
    mu_1q = _mu_pauli_error(EPS_MU_1Q)
    for g in _MU_PHYS_1Q:  # ('sx','sxdg','x','h')  -- rz/id virtual -> no noise
        th_g = thermal_relaxation_error(T1_S, T2_S, GATE_TIME_S[g])
        nm_.add_all_qubit_quantum_error(th_g.compose(mu_1q), [g])
    # CX: per-qubit thermal (tensored) composed with per-qubit biased-Pauli (tensored).
    th_cx = thermal_relaxation_error(T1_S, T2_S, GATE_TIME_S["cx"])
    th_cx2 = th_cx.tensor(th_cx)
    mu_cx2 = _mu_pauli_error(EPS_MU_2Q).tensor(_mu_pauli_error(EPS_MU_2Q))
    nm_.add_all_qubit_quantum_error(th_cx2.compose(mu_cx2), list(_GATES_2Q))
    return nm_


# PEC for the composite. The per-qubit composite noise channel is lifted STRAIGHT from the Aer
# QuantumError (thermal ∘ mu_pauli) via qiskit's Kraus(), so the PEC basis-noise matches the
# simulated noise to machine precision -- no hand-derivation/verification needed (unlike SPEC_TH_REAL
# which had to check ‖S_AD∘PD − S_qiskit‖∞~1e-15). The cx channel (th⊗th)∘(mu⊗mu) factorizes EXACTLY
# as comp⊗comp (verified 1e-16 in smoke_composite_pec.py), so one per-qubit Kraus drives the cx LP.
def _composite_kraus_1q(eps: float, t: float) -> list:
    """Exact per-qubit composite Kraus: thermal_relaxation_error(T1,T2,t) ∘ mu_pauli(eps)."""
    qe = thermal_relaxation_error(T1_S, T2_S, t).compose(_mu_pauli_error(eps))
    return list(Kraus(qe).data)


def _composite_build_reps(circuit) -> list[OperationRepresentation]:
    """Per-gate composite QPRs (thermal ∘ mixed-unitary), cached across seeds.

    - 1q physical gates (sx/sxdg/x/h): full composite LP, include_reset for the non-unital
      thermal part; the Aer-extracted Kraus makes the inversion exact (channel match).
    - rz/id (zero noise): identity rep via Takagi(0) -- robust to any rz angle, where the
      Pauli+reset LP would be infeasible (cf. SPEC_TH_REAL, [[project-pec-mitiq-pitfalls]]).
    - CX (both orientations): per-qubit composite LP (the cx channel factorizes as comp⊗comp).

    These reps are EXACT: they reconstruct the Aer channels to ~1e-7 (smoke_composite_pec_unbias.py)
    and PEC is unbiased (bias +0.004). (The "noise-once" inserted-Pauli approximation is harmless
    here because the correction coeffs are tiny.) The +0.1 "bias" seen earlier was an artefact of a
    fixed seed_simulator in the benchmark -> fixed by the seedless PEC backend in run_benchmark.
    """
    reps_1q, seen = [], set()
    for instr in circuit.data:
        op = instr.operation
        if op.name in ("measure", "barrier") or op.num_qubits != 1:
            continue
        key = ("CMP1", op.name, tuple(round(float(p), 10) for p in op.params))
        if key in seen:
            continue
        seen.add(key)
        if key not in _LP_REP_CACHE:
            qc = QuantumCircuit(1)
            qc.append(op, [0])
            if op.name in _MU_PHYS_1Q:  # sx/sxdg/x/h carry noise -> composite LP
                _LP_REP_CACHE[key] = represent_op_with_kraus_lp(
                    qc, _composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S[op.name]),
                    include_reset=True,
                )
            else:  # rz/id virtual (no noise) -> exact identity rep
                _LP_REP_CACHE[key] = represent_1q_qiskit_with_ad_takagi(qc, 0.0)
        reps_1q.append(_LP_REP_CACHE[key])

    # CX rep for BOTH orientations (cx(1,0) otherwise left un-mitigated -> PEC bias).
    kraus_cx = _composite_kraus_1q(EPS_MU_2Q, GATE_TIME_S["cx"])
    cx_reps = []
    for ctrl, tgt in [(0, 1), (1, 0)]:
        cx_key = ("CMP2", "cx", (ctrl, tgt))
        if cx_key not in _LP_REP_CACHE:
            qc = QuantumCircuit(2)
            qc.cx(ctrl, tgt)
            _LP_REP_CACHE[cx_key] = represent_op_with_kraus_lp(
                qc, kraus_cx, include_reset=True
            )
        cx_reps.append(_LP_REP_CACHE[cx_key])
    return reps_1q + cx_reps


SPEC_COMPOSITE = NoiseSpec(
    name="Composite",
    build_noise_model=_composite_noise_model,
    build_representations=_composite_build_reps,
    asymptote=None,  # the non-unital thermal component makes the per-seed P(|00>) fixed point
    # vary -> let Exp/AdaExp fit `a` freely (3-param fit), as in ThermalRealistic.
    # PEC is unbiased here (bias +0.004) since run_benchmark uses a seedless backend for PEC;
    # the earlier apparent +0.1 "bias" was an artefact of the fixed seed_simulator, NOT the reps.
    scale_factors=[1.0, 1.5, 2.0, 3.0, 5.0],  # >=4 points required for the free-asymptote fit
)


# Section 6.7  FULL hardware model: composite gate noise + realistic READOUT (add-on layer).
# The complete device picture = gate noise (composite, §6.6) AND readout error at measurement.
# Neither ZNE nor PEC removes readout (folding does not scale the measurement error; PEC has no
# readout representation), so on THIS spec ZNE/PEC hit a readout FLOOR -- that is the point (TREX
# is the complementary tool that removes it). SPEC_COMPOSITE stays gate-only so its DZNE-vs-PEC
# benchmark is unchanged; this spec adds READOUT_REALISTIC on top and REUSES the composite gate
# reps (readout is not a gate op -> left unmitigated -> the all-ops-represented guardrail is fine).
def _composite_readout_noise_model() -> _NoiseModel:
    nm_ = _composite_noise_model()
    READOUT_REALISTIC.add_readout_error(nm_)  # IBM-like ~1-2% asymmetric readout, on every qubit
    return nm_


SPEC_COMPOSITE_READOUT = NoiseSpec(
    name="CompositeReadout",
    build_noise_model=_composite_readout_noise_model,
    build_representations=_composite_build_reps,  # gate reps only -> ZNE/PEC leave the readout floor
    asymptote=None,  # non-unital thermal part -> free-asymptote Exp/AdaExp fit (as SPEC_COMPOSITE)
    scale_factors=[1.0, 1.5, 2.0, 3.0, 5.0],
)

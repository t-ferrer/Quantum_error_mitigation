"""Diagnose the composite PEC overshoot: is it the 'noise-once' LP approximation?

The LP basis models a basis op "G then Pauli P" as  N o P o G  (noise applied ONCE, P
noiseless). On Aer the same circuit runs as  N o P o N o G  (every physical gate, incl. the
inserted Pauli, is noised). So:

  recon_model  = sum_i a_i (N o P_i o G)        -- equals ideal G by LP construction (exact)
  recon_actual = sum_i a_i (N o P_i o N o G)    -- what PEC ACTUALLY realises on Aer

The gap recon_actual - ideal is the per-gate PEC bias. We quantify it on the 1q physical gates
(exact super-operators, column-stacking convention, no Monte-Carlo).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Kraus, Operator
from qiskit_aer.noise import thermal_relaxation_error

from qem.noise_models import T1_S, T2_S, GATE_TIME_S, EPS_MU_1Q, _mu_pauli_error
from qem.pec_core import represent_op_with_kraus_lp

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
# reset-to-|0> Kraus
R_KRAUS = [np.array([[1, 0], [0, 0]], dtype=complex), np.array([[0, 1], [0, 0]], dtype=complex)]


def sup_u(U):
    """Column-stacking super-operator of a unitary: vec(U rho U^dag) = (conj(U) (x) U) vec(rho)."""
    return np.kron(np.conj(U), U)


def sup_kraus(ks):
    return sum(np.kron(np.conj(K), K) for K in ks)


def composite_kraus_1q(eps, t):
    qe = thermal_relaxation_error(T1_S, T2_S, t).compose(_mu_pauli_error(eps))
    return list(Kraus(qe).data)


# basis super-operators in iproduct(["i","x","y","z","r"]) order -- matches represent_op_with_kraus_lp
PAULI_SUPS = [sup_u(I2), sup_u(X), sup_u(Y), sup_u(Z), sup_kraus(R_KRAUS)]
PAULI_NAMES = ["I", "X", "Y", "Z", "reset"]


def ideal_super_of(qc):
    return sup_u(Operator(qc).data)


def diagnose(gate_name):
    qc = QuantumCircuit(1)
    getattr(qc, gate_name)(0)
    kraus = composite_kraus_1q(EPS_MU_1Q, GATE_TIME_S[gate_name])
    N = sup_kraus(kraus)              # composite 1q noise super
    SG = ideal_super_of(qc)           # ideal gate super
    ideal = SG

    rep = represent_op_with_kraus_lp(qc, kraus, include_reset=True)
    a = np.array(rep.coeffs, dtype=float)
    assert len(a) == 5, f"expected 5 coeffs, got {len(a)}"

    recon_model = sum(ai * (N @ P @ SG) for ai, P in zip(a, PAULI_SUPS))
    recon_actual = sum(ai * (N @ P @ N @ SG) for ai, P in zip(a, PAULI_SUPS))

    err_model = np.max(np.abs(recon_model - ideal))      # ~0 confirms basis order + LP exactness
    err_actual = np.max(np.abs(recon_actual - ideal))    # the per-gate PEC bias magnitude

    # effect on a population: <0| eff(|+><+|) |0> vs ideal, a proxy for the bias sign
    plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    vec_plus = plus.reshape(-1, order="F")
    p0_ideal = (ideal @ vec_plus).reshape(2, 2, order="F")[0, 0].real
    p0_eff = (recon_actual @ vec_plus).reshape(2, 2, order="F")[0, 0].real

    print(f"  {gate_name:5s} coeffs(I,X,Y,Z,r)={np.round(a,5)}  "
          f"err_model={err_model:.1e}  err_actual={err_actual:.2e}  "
          f"P0(+): ideal={p0_ideal:.5f} eff={p0_eff:.5f} (d={p0_eff-p0_ideal:+.5f})")


print("=== per-1q-gate PEC bias: noise-once model vs noise-after-each actual ===")
for g in ("x", "sx", "h", "sxdg"):
    diagnose(g)

print("\nInterpretation: err_model ~ 0 => LP exact & basis order correct.")
print("err_actual > 0 (and d != 0) => the inserted Pauli's OWN noise (ignored by the")
print("noise-once LP) shifts the realised channel -> the source of the PEC overshoot.")

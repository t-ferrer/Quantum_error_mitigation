"""Smoke tests for PEC on the ThermalRealistic noise model.

(1) Verify the AD∘PD composite (kraus_thermal_1q) derived from (T1,T2,t) reproduces
    Qiskit's thermal_relaxation_error(T1,T2,t) super-operator EXACTLY -> PEC unbiased.
(2) Verify the per-gate gammas and that the Takagi rep at gamma=0 (rz/id) is identity.
"""

import numpy as np
from qiskit_aer.noise import thermal_relaxation_error

from qem.kraus import kraus_thermal_1q

T1_S = 100e-6
T2_S = 80e-6
GATE_TIME_S = {
    "rz": 0.0, "id": 0.0,
    "sx": 35e-9, "sxdg": 35e-9, "x": 35e-9, "h": 35e-9,
    "cx": 400e-9,
}


def _thermal_gammas(t1, t2, t):
    if t <= 0.0:
        return 0.0, 0.0
    gamma_ad = 1.0 - np.exp(-t / t1)
    gamma_pd = 1.0 - np.exp(-2.0 * t / t2 + t / t1)
    return gamma_ad, gamma_pd


def kraus_to_super(kraus):
    """Column-stacking superoperator: sum_k conj(K) ⊗ K."""
    return sum(np.kron(np.conj(K), K) for K in kraus)


def qiskit_super(t):
    err = thermal_relaxation_error(T1_S, T2_S, t)
    # Aer QuantumError -> Kraus list
    from qiskit.quantum_info import Kraus
    kr = Kraus(err).data
    return kraus_to_super(kr)


print("=== (1) super-operator match: AD∘PD composite vs Qiskit thermal_relaxation_error ===")
for g, t in GATE_TIME_S.items():
    g_ad, g_pd = _thermal_gammas(T1_S, T2_S, t)
    S_mine = kraus_to_super(kraus_thermal_1q(g_ad, g_pd))
    S_qk = qiskit_super(t)
    err = np.max(np.abs(S_mine - S_qk))
    print(f"  {g:4s} t={t*1e9:6.1f}ns  gamma_ad={g_ad:.3e} gamma_pd={g_pd:.3e}  "
          f"||S_mine - S_qiskit||_max = {err:.2e}")

print("\n=== (2) Takagi rep at gamma=0 should be identity (coeff [1,0,0]) ===")
from qiskit import QuantumCircuit
from qem.pec_core import represent_1q_qiskit_with_ad_takagi

for name in ("rz", "id", "sx"):
    qc = QuantumCircuit(1)
    if name == "rz":
        qc.rz(0.5, 0)
    elif name == "id":
        qc.id(0)
    else:
        qc.sx(0)
    g_ad, _ = _thermal_gammas(T1_S, T2_S, GATE_TIME_S[name])
    rep = represent_1q_qiskit_with_ad_takagi(qc, g_ad)
    print(f"  {name:4s} gamma_ad={g_ad:.3e}  coeffs={np.round(rep.coeffs,6)}  norm={rep.norm:.6f}")

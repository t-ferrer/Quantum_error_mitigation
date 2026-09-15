"""Smoke test: entanglement (process) fidelity (qem/fidelity.py), step 1 -- the metric alone.

Every expected value below is an INDEPENDENT CLOSED FORM derived from the Kraus operators, never a
second call into qiskit. Testing `process_fidelity` against `process_fidelity` would assert nothing;
the point is to pin our extraction path (Aer save_superop -> error channel -> Choi overlap) against
textbook algebra, on the actual NoiseSpecs the benchmark runs.

Closed forms used (d = 2^n, F_e = (1/d^2) sum_k |Tr K_k|^2):
  depolarizing(p)        1 - F_e = p (d^2 - 1)/d^2
  amplitude damping(g)         F_e = (1 + sqrt(1-g))^2 / 4
  phase damping(l)             F_e = [(1 + sqrt(1-l))^2 + l] / 4
  thermal AD(g) o PD(l)        F_e = [(1 + sqrt((1-g)(1-l)))^2 + l(1-g)] / 4
  unitary Rz(theta)      1 - F_e = sin^2(theta/2)
  tensor product               F_e(N1 (x) N2) = F_e(N1) F_e(N2)
  Nielsen/Horodecki          F_avg = (d F_e + 1)/(d + 1)

Checks:
  (1) limiting cases: no noise -> 0; complete dephasing -> exactly 1/2; coherent Rz -> sin^2(theta/2)
  (2)-(5) the closed form of each NoiseSpec family, 1-qubit AND 2-qubit (tensored)
  (6) Nielsen relation, cross-checked against qiskit's independently-implemented average_gate_fidelity
  (7) composite additivity: thermal + mixed-unitary infidelities add to second order
  (8) affinity of F_e on a quasi-probability combination (the property PEC verification rests on)
  (9) polarization bridge: round-trip and agreement with a global depolarizing channel of known gamma
"""

import warnings; warnings.filterwarnings("ignore")

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Choi, Kraus, Operator, SuperOp, average_gate_fidelity
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error

from qem import (
    SPEC_AD, SPEC_COMPOSITE, SPEC_DEPOL, SPEC_MIXED_UNITARY, SPEC_PD, SPEC_TH,
    average_gate_infidelity, channel_infidelity, entanglement_infidelity,
    gate_circuit, gate_infidelities, infidelity_from_success_prob, polarization,
)
from qem.noise_models import (
    GAMMA_AD_1Q, GAMMA_AD_2Q, GAMMA_TH_AD_1Q, GAMMA_TH_AD_2Q, GAMMA_TH_PD_1Q,
    GAMMA_TH_PD_2Q, LAMBDA_PD_1Q, LAMBDA_PD_2Q,
)
from qem.config import P1, P2

checks = []
TOL = 1e-9  # the extraction path is exact algebra; only float round-off separates us from theory


def report(num, label, got, want, tol=TOL):
    ok = abs(got - want) < tol
    checks.append(ok)
    print(f"({num}) {label}")
    print(f"    got {got:.12f}   want {want:.12f}   |d| {abs(got-want):.2e}   "
          f"{'OK' if ok else '*** CHECK'}")
    return ok


# --- closed forms, written out independently of qem/ and of qiskit ---------
def fe_depol(p, d):
    return 1.0 - p * (d**2 - 1) / d**2


def fe_ad(g):
    return (1.0 + np.sqrt(1.0 - g)) ** 2 / 4.0


def fe_pd(lam):
    return ((1.0 + np.sqrt(1.0 - lam)) ** 2 + lam) / 4.0


def fe_thermal(g, lam):
    return ((1.0 + np.sqrt((1.0 - g) * (1.0 - lam))) ** 2 + lam * (1.0 - g)) / 4.0


# ---------------------------------------------------------------------------
# (1) limiting cases
# ---------------------------------------------------------------------------
print("(1) limiting cases")
noiseless = entanglement_infidelity(gate_circuit("sx"), None)
c = abs(noiseless) < TOL
checks.append(c)
print(f"    no noise                    1-F_e = {noiseless:.2e}   {'OK' if c else '*** CHECK'}")

# complete dephasing: Kraus {I/sqrt2, Z/sqrt2} -> F_e = (1/4)(|Tr I/sqrt2|^2 + 0) = 1/2 EXACTLY.
# The channel is perfect on |0>,|1> and maximally destructive on |+>: precisely the blind spot of a
# computational-basis success probability, which is why this case is the load-bearing one.
nm_deph = NoiseModel()
nm_deph.add_all_qubit_quantum_error(pauli_error([("I", 0.5), ("Z", 0.5)]), ["id"])
got = entanglement_infidelity(gate_circuit("id"), nm_deph)
c = abs(got - 0.5) < TOL
checks.append(c)
print(f"    complete dephasing          1-F_e = {got:.12f}   want 0.5   {'OK' if c else '*** CHECK'}")

# coherent unitary error: no noise model at all, just Rz(theta) compared against the identity.
theta = 0.37
qc_rz = QuantumCircuit(1); qc_rz.rz(theta, 0)
got = channel_infidelity(SuperOp(Operator(qc_rz)))
want = np.sin(theta / 2) ** 2
c = abs(got - want) < TOL
checks.append(c)
print(f"    coherent Rz({theta})           1-F_e = {got:.12f}   want sin^2(t/2)={want:.12f}   "
      f"{'OK' if c else '*** CHECK'}")

# ---------------------------------------------------------------------------
# (2)-(5) one closed form per NoiseSpec family, 1q and 2q
# ---------------------------------------------------------------------------
inf_depol = gate_infidelities(SPEC_DEPOL.build_noise_model())
report(2, f"depolarizing 1q (p={P1}): p(d^2-1)/d^2, d=2",
       inf_depol["sx"], 1 - fe_depol(P1, 2))
report("2b", f"depolarizing 2q (p={P2}): p(d^2-1)/d^2, d=4  [global, not per-qubit]",
       inf_depol["cx"], 1 - fe_depol(P2, 4))

inf_ad = gate_infidelities(SPEC_AD.build_noise_model())
report(3, f"amplitude damping 1q (g={GAMMA_AD_1Q}): (1+sqrt(1-g))^2/4",
       inf_ad["sx"], 1 - fe_ad(GAMMA_AD_1Q))
report("3b", f"amplitude damping 2q (g={GAMMA_AD_2Q}/qubit): tensor -> F_e^2",
       inf_ad["cx"], 1 - fe_ad(GAMMA_AD_2Q) ** 2)

inf_pd = gate_infidelities(SPEC_PD.build_noise_model())
report(4, f"phase damping 1q (l={LAMBDA_PD_1Q}): [(1+sqrt(1-l))^2+l]/4",
       inf_pd["sx"], 1 - fe_pd(LAMBDA_PD_1Q))
report("4b", f"phase damping 2q (l={LAMBDA_PD_2Q}/qubit): tensor -> F_e^2",
       inf_pd["cx"], 1 - fe_pd(LAMBDA_PD_2Q) ** 2)

inf_th = gate_infidelities(SPEC_TH.build_noise_model())
report(5, f"thermal AD({GAMMA_TH_AD_1Q}) o PD({GAMMA_TH_PD_1Q}) 1q: 4-Kraus closed form",
       inf_th["sx"], 1 - fe_thermal(GAMMA_TH_AD_1Q, GAMMA_TH_PD_1Q))
report("5b", f"thermal 2q ({GAMMA_TH_AD_2Q}/{GAMMA_TH_PD_2Q} per qubit): tensor -> F_e^2",
       inf_th["cx"], 1 - fe_thermal(GAMMA_TH_AD_2Q, GAMMA_TH_PD_2Q) ** 2)

# ---------------------------------------------------------------------------
# (6) Nielsen / Horodecki relation
# ---------------------------------------------------------------------------
# Cross-check against qiskit's average_gate_fidelity, which is implemented from the average-over-
# states side -- an independent route to the same number, so this genuinely tests the relation.
print("(6) Nielsen relation  F_avg = (d F_e + 1)/(d + 1)")
ok6 = True
for spec, gate, n in ((SPEC_DEPOL, "sx", 1), (SPEC_DEPOL, "cx", 2),
                      (SPEC_COMPOSITE, "sx", 1), (SPEC_COMPOSITE, "cx", 2)):
    nm = spec.build_noise_model()
    fe_infid = entanglement_infidelity(gate_circuit(gate), nm)
    ours = average_gate_infidelity(fe_infid, n)
    from qem.fidelity import error_channel
    theirs = 1 - average_gate_fidelity(Choi(error_channel(gate_circuit(gate), nm)))
    good = abs(ours - theirs) < 1e-9
    ok6 &= good
    print(f"    {spec.name:<14} {gate:<3} n={n}   ours {ours:.12f}   qiskit {theirs:.12f}   "
          f"{'OK' if good else '*** CHECK'}")
checks.append(ok6)

# ---------------------------------------------------------------------------
# (7) composite additivity
# ---------------------------------------------------------------------------
# Composite = thermal o mixed-unitary. Infidelities add to FIRST order, so the residual must be
# second order in the individual infidelities -- assert exactly that, not an arbitrary tolerance.
print("(7) additivity: 1-F_e(thermal o mixed-unitary) ~ sum of the two")
from qem.noise_models import SPEC_TH_REAL
i_comp = gate_infidelities(SPEC_COMPOSITE.build_noise_model())
i_th = gate_infidelities(SPEC_TH_REAL.build_noise_model())
i_mu = gate_infidelities(SPEC_MIXED_UNITARY.build_noise_model())
ok7 = True
for gate in ("sx", "cx"):
    tot, a, b = i_comp[gate], i_th[gate], i_mu[gate]
    resid = abs(tot - (a + b))
    bound = 5 * a * b  # second-order term, with a factor-5 margin
    good = resid < bound
    ok7 &= good
    print(f"    {gate:<3} composite {tot:.8f}   thermal {a:.8f} + mixed-u {b:.8f} = {a+b:.8f}")
    print(f"        residual {resid:.2e}  <  2nd-order bound 5*a*b = {bound:.2e}   "
          f"{'OK' if good else '*** CHECK'}")
checks.append(ok7)

# ---------------------------------------------------------------------------
# (8) affinity of F_e on a quasi-probability combination
# ---------------------------------------------------------------------------
# The property every PEC verification rests on: F_e(sum eta_i B_i) = sum eta_i F_e(B_i), for eta of
# ANY sign. Built here from an analytic depolarizing QPD, independent of qem/pec_core.py, so this
# tests the metric's algebra and not the LP solver.
print("(8) affinity  F_e(sum eta_i B_i) = sum eta_i F_e(B_i)  on a signed combination")
p = 0.1
eta_I, eta_P = 1 + 3 * p / (4 * (1 - p)), -p / (4 * (1 - p))
nm_d = NoiseModel(); nm_d.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["id"])
basis, etas = [], [eta_I, eta_P, eta_P, eta_P]
from qem.fidelity import channel_of
for recovery in (None, "x", "y", "z"):
    qc = QuantumCircuit(1); qc.id(0)   # the noised "ideal" op (only `id` carries noise here)
    if recovery is not None:           # recovery Pauli, noiseless: x/y/z are not in the noised set
        getattr(qc, recovery)(0)       # (appending `id` here would noise the channel TWICE)
    basis.append(channel_of(qc, nm_d).data)
lhs = 1 - channel_infidelity(SuperOp(sum(e * b for e, b in zip(etas, basis))))
rhs = sum(e * (1 - channel_infidelity(SuperOp(b))) for e, b in zip(etas, basis))
c8a = abs(lhs - rhs) < 1e-12
c8b = abs(lhs - 1.0) < 1e-12   # and the QPD does invert the channel exactly
checks.append(c8a and c8b)
print(f"    eta = [{eta_I:+.6f}, {eta_P:+.6f} x3]   1-norm gamma = {abs(eta_I)+3*abs(eta_P):.6f}")
print(f"    F_e(combination) {lhs:.14f}   sum eta_i F_e(B_i) {rhs:.14f}   "
      f"|d| {abs(lhs-rhs):.2e}   {'OK' if c8a else '*** CHECK'}")
print(f"    and equals F_e(ideal) = 1 (the QPD inverts the channel)   "
      f"{'OK' if c8b else '*** CHECK'}")

# ---------------------------------------------------------------------------
# (9) polarization bridge
# ---------------------------------------------------------------------------
# For a GLOBAL depolarizing channel of polarization g the mapping is exact (that is the model it
# assumes), so it can be asserted here. On real mirror-circuit data it is only ~4% accurate, which
# is why it is used for reporting and never asserted -- see the note in qem/fidelity.py.
print("(9) polarization bridge  P_s -> gamma -> 1-F_e")
ok9 = True
for n in (1, 2, 3):
    for g in (1.0, 0.9, 0.5):
        p_s = g + (1 - g) / 2**n
        got_g = polarization(p_s, n)
        got_i = infidelity_from_success_prob(p_s, n)
        want_i = (1 - g) * (1 - 1 / 4**n)
        good = abs(got_g - g) < 1e-12 and abs(got_i - want_i) < 1e-12
        ok9 &= good
        if not good:
            print(f"    n={n} gamma={g}: got {got_g:.12f}/{got_i:.12f} want {g}/{want_i:.12f} *** CHECK")
print(f"    9 (n, gamma) combinations, round-trip and closed form   {'OK' if ok9 else '*** CHECK'}")
checks.append(ok9)

# --- per-spec summary table (reporting, not asserted) ----------------------
print("\n  entanglement infidelity per gate, across the NoiseSpec family:")
print(f"  {'spec':<20}{'1q (sx)':>14}{'2q (cx)':>14}{'1-F_avg cx':>14}")
for spec in (SPEC_DEPOL, SPEC_AD, SPEC_PD, SPEC_TH, SPEC_TH_REAL,
             SPEC_MIXED_UNITARY, SPEC_COMPOSITE):
    d = gate_infidelities(spec.build_noise_model())
    print(f"  {spec.name:<20}{d.get('sx', float('nan')):>14.6f}{d.get('cx', float('nan')):>14.6f}"
          f"{average_gate_infidelity(d.get('cx', float('nan')), 2):>14.6f}")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

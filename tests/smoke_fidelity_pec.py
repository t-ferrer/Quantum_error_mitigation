"""Smoke test: ALGEBRAIC verification of the PEC representations, via entanglement fidelity.

WHY THIS EXISTS. The README claims the quasi-probability representations match the Aer NoiseModel to
machine precision. Until now that claim was only checked STATISTICALLY -- PEC bias ~0.004 over 20
seeds x thousands of importance samples: slow, shot-noisy, and it does not localise the fault. But
PEC asserts a LINEAR identity between channels,

        ideal  =  sum_i eta_i * noisy(B_i)

and F_e is affine, so the identity can be evaluated directly: run each basis circuit B_i through
Aer, combine the superoperators with the representation's own eta_i, compare to the ideal unitary.
One gate, zero shots, deterministic answer.

This is the test that would have caught both historical PEC bugs immediately -- the missing cx(1,0)
orientation (silent un-mitigation, bias -0.28) and the BFGS solver failing on non-unital noise --
without running the benchmark. The existing `_assert_all_ops_represented` guardrail checks that a
representation EXISTS; this one checks that it is CORRECT.

WHAT IS ASSERTED, and why it is not "residual < 1e-12":
  The representations model the noise as acting ONCE after the whole basis operation, while Aer
  re-applies it after every noised recovery gate (`x` is in the noised gate set, `y`/`z` are not).
  The mismatch is genuine and second-order -- |eta_recovery| x eps_gate -- so the honest assertions
  are (a) a large SUPPRESSION FACTOR versus the unmitigated channel, and (b) a residual bounded by
  the second-order product (gamma - 1) x eps_gate. An absolute machine-precision tolerance would be
  wrong physics and would fail for a reason that is not a bug.

ONE KNOWN, DELIBERATE APPROXIMATION IS PINNED RATHER THAN EXEMPTED.
  SPEC_TH's 1-qubit representations use Mitiq's Takagi amplitude-damping helper (the LP is
  infeasible for complex Rz rotations under non-unital noise), which inverts the AD component only
  -- see the comment at noise_models.py `_th_build_reps`, "ignores PD contribution". This test
  measures what that costs: the residual is EXACTLY the phase-damping-only infidelity,
  1 - [(1+sqrt(1-l))^2 + l]/4 = 0.001251566418500 for l = GAMMA_TH_PD_1Q, to machine precision.
  So we assert that value tightly instead of loosening the suppression threshold. If someone later
  replaces Takagi with a full thermal representation, this check fails and says exactly why --
  which is the useful failure, not a silent pass.

Checks:
  (1) matched spec: suppression >= 100x on every 1q gate and on cx, for the whole NoiseSpec family
  (2) both cx orientations behave identically (the cx(1,0) fix, verified algebraically)
  (3) the residual is second-order: |residual| < (gamma - 1) * eps_gate
  (4) the pinned SPEC_TH 1q residual equals the phase-damping-only infidelity
  (5) NEGATIVE CONTROL: representations of spec A against the noise of spec B collapse to <= 10x.
      Without this the suite could pass while asserting nothing -- it proves the test can fail.

Out of scope: SPEC_COMPOSITE_READOUT. It reuses the composite GATE representations by design and
readout is not a gate operation, so there is nothing extra for a channel-level check to verify.
"""

import warnings; warnings.filterwarnings("ignore")

import numpy as np
from mitiq.interface.conversions import convert_from_mitiq
from qiskit import QuantumCircuit

from qem import (
    SPEC_AD, SPEC_COMPOSITE, SPEC_DEPOL, SPEC_MIXED_UNITARY, SPEC_PD, SPEC_TH,
    entanglement_infidelity, qpd_effective_infidelity,
)
from qem.noise_models import GAMMA_TH_PD_1Q, SPEC_TH_REAL

checks = []
SPECS = (SPEC_DEPOL, SPEC_AD, SPEC_PD, SPEC_TH, SPEC_TH_REAL, SPEC_MIXED_UNITARY, SPEC_COMPOSITE)
GATES_1Q = ("sx", "x", "h")
MIN_SUPPRESSION = 100.0     # matched pairs measure 1e3-1e6x; mismatched measure 0-2x
MAX_MISMATCH_SUPPRESSION = 10.0

# Pinned residuals for representations that are approximate BY DESIGN (see the module docstring).
# Key: (spec name, gate arity). Value: the exact residual the approximation must leave behind.
_PD_ONLY_INFIDELITY = 1.0 - (
    (1.0 + np.sqrt(1.0 - GAMMA_TH_PD_1Q)) ** 2 + GAMMA_TH_PD_1Q
) / 4.0
KNOWN_RESIDUALS = {("ThermalRelaxation", 1): _PD_ONLY_INFIDELITY}


def _reps_for(spec, gate_circuit_1q):
    """(1q rep, [2q reps]) for `spec`, built from a single-gate circuit so matching is unambiguous."""
    reps = spec.build_representations(gate_circuit_1q)
    one = [r for r in reps if len(r.ideal.all_qubits()) == 1]
    two = [r for r in reps if len(r.ideal.all_qubits()) == 2]
    return one, two


def _ideal_circuit(rep) -> QuantumCircuit:
    return convert_from_mitiq(rep.ideal, "qiskit")


def _orientation(rep) -> str:
    qc = _ideal_circuit(rep)
    instr = qc.data[0]
    c, t = (qc.find_bit(q).index for q in instr.qubits)
    return f"cx({c},{t})"


def _assess(rep, noise_model):
    """(residual, unmitigated infidelity, suppression factor, 1-norm gamma) for one representation."""
    resid = qpd_effective_infidelity(rep, noise_model)
    raw = entanglement_infidelity(_ideal_circuit(rep), noise_model)
    gamma = float(np.abs(np.asarray(rep.coeffs, dtype=float)).sum())
    supp = raw / abs(resid) if abs(resid) > 0 else np.inf
    return resid, raw, supp, gamma


# ---------------------------------------------------------------------------
# (1)-(3) matched representations, across the whole family
# ---------------------------------------------------------------------------
print("(1)-(4) matched spec: QPD suppression, cx orientations, second-order residual")
print(f"  {'spec':<18}{'gate':<10}{'unmitigated':>13}{'residual':>12}{'suppress':>11}"
      f"{'gamma':>9}  bound/pinned")
ok1 = ok2 = ok3 = ok4 = True
for spec in SPECS:
    nm = spec.build_noise_model()
    for gname in GATES_1Q:
        qc = QuantumCircuit(2)
        getattr(qc, gname)(0)
        one, two = _reps_for(spec, qc)
        targets = [(gname, 1, one[0])]
        if gname == "sx":  # the 2q reps are spec-wide, so build them once per spec
            targets += [(_orientation(r), 2, r) for r in two]
        cx_resid = []
        for label, arity, rep in targets:
            resid, raw, supp, gamma = _assess(rep, nm)
            pinned = KNOWN_RESIDUALS.get((spec.name, arity))
            if pinned is not None:
                # approximate BY DESIGN: assert the exact residual it must leave, not a threshold
                good = abs(abs(resid) - pinned) < 1e-12
                ok4 &= good
                note = f"pin {pinned:.3e}"
            else:
                bound = (gamma - 1.0) * raw     # |eta_recovery| x eps_gate, generously bounded
                good_s, good_b = supp >= MIN_SUPPRESSION, abs(resid) < bound
                ok1 &= good_s
                ok3 &= good_b
                good = good_s and good_b
                note = f"{bound:>8.1e}"
            if label.startswith("cx"):
                cx_resid.append(abs(resid))
            print(f"  {spec.name:<18}{label:<10}{raw:>13.6f}{resid:>12.2e}{supp:>10.0f}x"
                  f"{gamma:>9.5f}  {note}  {'OK' if good else '*** CHECK'}")
        if len(cx_resid) == 2:
            # both orientations must be suppressed the SAME way; a broken cx(1,0) would show as a
            # residual orders of magnitude apart (that bug left the gate entirely un-mitigated).
            lo, hi = min(cx_resid), max(cx_resid)
            same = hi <= max(10 * lo, 1e-12)
            ok2 &= same
            if not same:
                print(f"      *** cx orientations disagree: {lo:.2e} vs {hi:.2e}")
checks += [ok1, ok2, ok3, ok4]
print(f"  (1) every suppression >= {MIN_SUPPRESSION:.0f}x                {'OK' if ok1 else '*** CHECK'}")
print(f"  (2) cx(0,1) and cx(1,0) suppressed alike     {'OK' if ok2 else '*** CHECK'}")
print(f"  (3) residual < (gamma-1)*eps_gate (2nd order){'  OK' if ok3 else '  *** CHECK'}")
print(f"  (4) pinned Takagi-AD residual = PD-only infidelity {_PD_ONLY_INFIDELITY:.12f}"
      f"  {'OK' if ok4 else '*** CHECK'}")

# ---------------------------------------------------------------------------
# (5) negative control
# ---------------------------------------------------------------------------
# A verification suite that cannot fail is worthless. Here the representations of one spec are
# applied to another spec's noise: the QPD is then solving the wrong inverse and the suppression
# must collapse. Measured: matched 3e3-1e6x, mismatched 0-2x -- an order of magnitude of clearance
# either side of the 100x threshold used above.
print("\n(4) NEGATIVE CONTROL: reps of spec A vs noise of spec B (must collapse)")
control = (SPEC_DEPOL, SPEC_TH_REAL, SPEC_COMPOSITE)
qc_cx = QuantumCircuit(2); qc_cx.sx(0)
rep_cache = {sp.name: _reps_for(sp, qc_cx)[1][0] for sp in control}
print(f"  {'reps of':<18}{'noise of':<18}{'residual':>12}{'suppression':>13}")
ok4 = True
for sp_rep in control:
    for sp_noise in control:
        resid, raw, supp, _ = _assess(rep_cache[sp_rep.name], sp_noise.build_noise_model())
        matched = sp_rep is sp_noise
        good = (supp >= MIN_SUPPRESSION) if matched else (supp <= MAX_MISMATCH_SUPPRESSION)
        ok4 &= good
        tag = "  <- matched" if matched else ""
        print(f"  {sp_rep.name:<18}{sp_noise.name:<18}{resid:>12.2e}{supp:>12.0f}x{tag}"
              f"{'' if good else '   *** CHECK'}")
checks.append(ok4)
print(f"  (5) matched suppress >= {MIN_SUPPRESSION:.0f}x, mismatched <= {MAX_MISMATCH_SUPPRESSION:.0f}x"
      f"   {'OK' if ok4 else '*** CHECK'}")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

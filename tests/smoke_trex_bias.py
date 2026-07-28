"""Smoke test 5/5 for TREX: dissect the small POSITIVE offset of TREX'd P(target).

TREX recovers P(target) up to a small +offset on ASYMMETRIC readout. This is the expected
finite-sample ratio-estimator bias, NOT a bug and NOT the Aer seed_simulator artifact. Matches
van den Berg, Minev, Temme, Phys. Rev. A 105, 032620 (2022), Sec. IV, which estimates
<Z~_s> as f(D1,s)/f(D0,s) -- explicitly "of the form x_hat/y_hat" -- a ratio estimator whose
finite-N bias is controlled by the number of randomizations (sample complexity ~ 1/lambda^2).

Confirms:
 (1) offset is ~0 for SYMMETRIC readout (per-pattern calibration factor is constant -> no
     denominator variance) and positive for ASYMMETRIC (per-pattern factor swings -> denom
     variance -> 1/x convexity bias);
 (2) offset SHRINKS with num_randomizations N but is essentially FLAT in shots (finite-N effect,
     not shot noise);
 (3) mitiq's combine_results uses AVERAGE-OF-RATIOS mean_s(n_s/lam_s); the paper's estimator is
     RATIO-OF-AVERAGES mean_s(n_s)/mean_s(lam_s). Identical for symmetric readout, paper slightly
     lower for asymmetric; both -> 0 as N -> inf.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from qiskit_aer import AerSimulator
from mitiq.experimental.trex import execute_with_trex
from mitiq.experimental.trex.trex_utils import xor_bitstrings

from qem import (benchmark_circuit_cirq, mitiq_measurement_executor, projector_observable,
                 target_key, trex_success_prob)
from qem.readout import READOUT_SYM, READOUT_ASYM

SEEDS = list(range(20))


def trex_offset(spec, nr, shots):
    be = AerSimulator(noise_model=spec.readout_noise_model())  # seedless: no fixed-seed artifact
    vals = np.array([
        trex_success_prob(benchmark_circuit_cirq(s), mitiq_measurement_executor(be, shots),
                          target_key(s), num_randomizations=nr, random_state=s)
        for s in SEEDS
    ])
    return vals.mean() - 1.0, vals.std() / len(SEEDS) ** 0.5


# ---------------------------------------------------------------------------
print("=== (1)+(2) offset vs shots (flat) and vs num_randomizations (shrinks), sym vs asym ===")
print(f"  {'':14s} {'sym offset':>16s} {'asym offset':>16s}")
print("  -- vs shots (nr=32 fixed): NOT shrinking with shots -> not shot noise --")
for shots in (1000, 16000):
    so, ss = trex_offset(READOUT_SYM, 32, shots)
    ao, asem = trex_offset(READOUT_ASYM, 32, shots)
    print(f"  shots={shots:<8d} {so:+.4f}+/-{ss:.4f}  {ao:+.4f}+/-{asem:.4f}")

print("  -- vs num_randomizations (shots=4096 fixed): asym shrinks toward 0 --")
asym_by_nr = {}
for nr in (1, 8, 64):
    so, ss = trex_offset(READOUT_SYM, nr, 4096)
    ao, asem = trex_offset(READOUT_ASYM, nr, 4096)
    asym_by_nr[nr] = ao
    print(f"  N={nr:<11d} {so:+.4f}+/-{ss:.4f}  {ao:+.4f}+/-{asem:.4f}")

ok_asym_specific = asym_by_nr[1] > trex_offset(READOUT_SYM, 1, 4096)[0] + 0.005
ok_shrinks = asym_by_nr[64] < asym_by_nr[1]
print(f"  asymmetry-specific (asym N=1 >> sym)? {ok_asym_specific};  shrinks with N? {ok_shrinks}")


# ---------------------------------------------------------------------------
# (3) mitiq (average-of-ratios) vs paper (ratio-of-averages), on the SAME raw TREX data.
def _parity_mean(mr, support):
    bits = mr.filter_qubits(support)
    return float(np.mean([(-1) ** np.sum(b) for b in bits]))


def both_estimators(spec, nr=8, shots=20000):
    be = AerSimulator(noise_model=spec.readout_noise_model())
    mit, aor, roa = [], [], []
    for s in SEEDS:
        n = len(target_key(s))
        obs = projector_observable(target_key(s), include_identity=False)
        val, data = execute_with_trex(benchmark_circuit_cirq(s), mitiq_measurement_executor(be, shots),
                                      obs, num_randomizations=nr, random_state=s, full_output=True)
        tw, cal, rs = data["twirled_results"], data["calibration_results"], data["randomization_strings"]
        q2i = {q: i for i, q in enumerate(sorted(obs._qubits()))}
        a = r = 0.0
        for gi, group in enumerate(obs.groups):
            measured = sorted(group._qubits_to_measure())
            for pauli in group.elements:
                sup = sorted(pauli.support())
                ns = np.array([_parity_mean(xor_bitstrings(tw[gi * nr + ri],
                              np.array([sv[q2i[q]] for q in measured], dtype=np.int64)), sup)
                              for ri, sv in enumerate(rs)])
                ls = np.array([_parity_mean(xor_bitstrings(cal[ri], sv), sup) for ri, sv in enumerate(rs)])
                c = float(np.real(pauli.coeff))
                a += c * np.mean(ns / ls)             # average-of-ratios (mitiq)
                r += c * (np.mean(ns) / np.mean(ls))  # ratio-of-averages (paper)
        off = 1.0 / 2 ** n
        mit.append(val + off); aor.append(a + off); roa.append(r + off)
    m = lambda x: np.mean(x) - 1.0
    return m(mit), m(aor), m(roa)


print("\n=== (3) mitiq average-of-ratios vs paper ratio-of-averages (nr=8, shots=20000) ===")
for spec in (READOUT_ASYM, READOUT_SYM):
    mit, aor, roa = both_estimators(spec)
    print(f"  {spec.name:11s}  mitiq={mit:+.5f}  replica(avg-of-ratios)={aor:+.5f}  "
          f"paper(ratio-of-avgs)={roa:+.5f}")
    tag = "identical (sym)" if abs(mit - roa) < 1e-4 else "paper <= mitiq (asym)"
    print(f"    replica matches mitiq? {abs(mit-aor)<1e-6};  {tag}")

print("\nDONE")

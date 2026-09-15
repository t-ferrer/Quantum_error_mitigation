"""Smoke test: the channel-level fidelity columns in run_benchmark / cached_benchmark (phase 3).

Verifies the PLUMBING, not the metric (smoke_fidelity.py) and not the physics. The a-priori-vs-
measured agreement is PRINTED, never asserted: the polarization bridge assumes globally-depolarizing
effective noise and is only ~4% accurate on real mirror-circuit data, so asserting it would test
physics modelling rather than code.

The two columns are a pair, and the pairing is the point:
    infid_channel   the method's systematic error on the channel -- exact, deterministic, no shots
    infid_psucc     the same method read off its P(target) estimate, via the polarization bridge
infid_psucc alone carries no information beyond `mean` (it is an invertible function of it). Beside
its exact counterpart it becomes the empirical half of a bias/variance split.

Checks:
  (1) both columns present; both vary by method (infid_channel is per-method, NOT a noise constant)
  (2) infid_psucc is exactly the polarization bridge applied to each row's own `mean`
  (3) the Noisy row of infid_channel equals an independent per-seed circuit-infidelity computation
  (4) ORDERING: PEC's channel bias is orders of magnitude below Noisy, and below every ZNE method
      -- the headline result, and the one that a shot-based benchmark cannot resolve
  (5) add_fidelity_columns backfills a legacy DataFrame, accepts a precomputed channel mapping,
      and is idempotent (this is what lets cached_benchmark upgrade old pickles instead of
      discarding a 20-minute run)
  (6) the drift path averages over the theta_t trajectory rather than a nominal model
  (7) cached_benchmark surfaces the columns, and its channel cache makes the second call instant
"""

import warnings; warnings.filterwarnings("ignore")

import time

import numpy as np

from qem import (
    FIDELITY_COLUMNS, SPEC_COMPOSITE, add_fidelity_columns, benchmark_circuit_infidelity,
    build_drift_gate_model, drift_params, entanglement_infidelity, infidelity_from_success_prob,
    make_benchmark_circuit, method_channel_infidelities, run_benchmark,
)

checks = []
SEEDS = list(range(4))
BUDGET = 4000  # plumbing test: cheap, and none of the sampled numbers are asserted

df, _, _ = run_benchmark(
    SPEC_COMPOSITE, seeds=SEEDS, shot_budget=BUDGET, run_pec=True, verbose=False
)

# ---------------------------------------------------------------------------
print("(1) both columns present, both method-dependent")
c1 = all(c in df.columns for c in FIDELITY_COLUMNS)
c1 &= df["infid_channel"].nunique() > 1   # per-method, not a single noise constant
c1 &= df["infid_psucc"].nunique() > 1
checks.append(c1)
print(f"    {list(FIDELITY_COLUMNS)}   infid_channel spans {df['infid_channel'].nunique()} values"
      f"   {'OK' if c1 else '*** CHECK'}")

# ---------------------------------------------------------------------------
print("(2) infid_psucc == polarization bridge applied to each row's mean")
n = make_benchmark_circuit(SEEDS[0]).num_qubits
c2 = True
for (_, method), row in df.iterrows():
    want = round(infidelity_from_success_prob(float(row["mean"]), n), 6)
    if abs(float(row["infid_psucc"]) - want) > 1e-9:
        c2 = False
        print(f"    *** {method}: got {row['infid_psucc']} want {want}")
checks.append(c2)
print(f"    {len(df)} rows re-derived from `mean` independently   {'OK' if c2 else '*** CHECK'}")

# ---------------------------------------------------------------------------
print("(3) Noisy row of infid_channel == mean of the per-seed exact circuit infidelities")
nm = SPEC_COMPOSITE.build_noise_model()
per_seed = [entanglement_infidelity(make_benchmark_circuit(s), nm) for s in SEEDS]
want = float(np.mean(per_seed))
got = float(df.loc[(SPEC_COMPOSITE.name, "Noisy"), "infid_channel"])
c3 = abs(got - want) < 1e-8
checks.append(c3)
print(f"    got {got:.8f}   want {want:.8f}   per-seed spread "
      f"[{min(per_seed):.4f}, {max(per_seed):.4f}]   {'OK' if c3 else '*** CHECK'}")

# ---------------------------------------------------------------------------
print("(4) ordering: PEC's channel bias sits orders of magnitude below Noisy and every ZNE")
ch = {m: float(df.loc[(SPEC_COMPOSITE.name, m), "infid_channel"]) for m in
      df.index.get_level_values("method")}
pec, noisy = abs(ch["PEC"]), abs(ch["Noisy"])
zne_best = min(abs(ch[m]) for m in ("Richardson", "Poly2", "Exp", "AdaExp"))
c4 = pec < noisy / 1000 and pec < zne_best
checks.append(c4)
for m in ("Noisy", "Richardson", "Poly2", "Exp", "AdaExp", "PEC"):
    print(f"      {m:<11} {ch[m]:>+12.3e}")
print(f"    PEC {pec:.2e} < Noisy/1000 {noisy/1000:.2e} and < best ZNE {zne_best:.2e}   "
      f"{'OK' if c4 else '*** CHECK'}")

# ---------------------------------------------------------------------------
print("(5) backfill: legacy DataFrame, precomputed channel mapping, idempotence")
legacy = df.drop(columns=list(FIDELITY_COLUMNS))
precomputed = method_channel_infidelities(SPEC_COMPOSITE, SEEDS)
back = add_fidelity_columns(legacy, SPEC_COMPOSITE, SEEDS, channel=precomputed)
c5 = all(c in back.columns for c in FIDELITY_COLUMNS)
c5 &= np.allclose(back["infid_channel"], df["infid_channel"])
c5 &= np.allclose(back["infid_psucc"], df["infid_psucc"])
again = add_fidelity_columns(back, SPEC_COMPOSITE, SEEDS)   # no-op
c5 &= np.allclose(again["infid_channel"], back["infid_channel"])
checks.append(c5)
print(f"    dropped -> backfilled -> identical, re-applying is a no-op   "
      f"{'OK' if c5 else '*** CHECK'}")

# ---------------------------------------------------------------------------
print("(6) drift path averages over the theta_t trajectory")
models = [build_drift_gate_model(p) for p in drift_params(len(SEEDS), seed=0)]
dfd, _, _ = run_benchmark(
    SPEC_COMPOSITE, seeds=SEEDS, shot_budget=BUDGET, run_pec=False, verbose=False,
    drift_models=models,
)
want_d = benchmark_circuit_infidelity(SEEDS, drift_models=models)
got_d = float(dfd.loc[(SPEC_COMPOSITE.name, "Noisy"), "infid_channel"])
c6 = abs(got_d - want_d) < 1e-8 and abs(got_d - got) > 1e-6   # and NOT the nominal value
checks.append(c6)
print(f"    drift {got_d:.6f} (independent {want_d:.6f})   static {got:.6f}   "
      f"differs as it must   {'OK' if c6 else '*** CHECK'}")

try:
    benchmark_circuit_infidelity(SEEDS, drift_models=models[:-1])
    c6b = False
except ValueError:
    c6b = True
checks.append(c6b)
print(f"    wrong-length drift_models raises ValueError   {'OK' if c6b else '*** CHECK'}")

# ---------------------------------------------------------------------------
print("(7) cached_benchmark: columns present, and the channel cache makes reload instant")
from qem.cache import CACHE_DIR, cached_benchmark
if any(CACHE_DIR.glob("Composite_*.pkl")):
    cached_benchmark(SPEC_COMPOSITE, verbose=False)          # warm both caches
    t0 = time.time()
    dfc, _, _ = cached_benchmark(SPEC_COMPOSITE, verbose=False)
    dt = time.time() - t0
    c7 = all(c in dfc.columns for c in FIDELITY_COLUMNS) and dt < 2.0
    checks.append(c7)
    print(f"    columns present, warm reload {dt:.3f}s (< 2s)   {'OK' if c7 else '*** CHECK'}")
else:
    print("    (no Composite cache on disk -> skipped, not a failure)")

# --- reporting only --------------------------------------------------------
noisy_p = float(df.loc[(SPEC_COMPOSITE.name, "Noisy"), "infid_psucc"])
print(f"\n  Noisy: channel {got:.6f}  vs  P(target) bridge {noisy_p:.6f}"
      f"   -> {100*abs(noisy_p-got)/got:.1f}% apart")
print("  NOT asserted: the bridge assumes globally-depolarizing effective noise.")

print(f"\n{'ALL OK' if all(checks) else '*** SOME CHECKS FAILED'}  ({sum(checks)}/{len(checks)})")

"""NoiseSpec dataclass and the unified DZNE-vs-PEC benchmark driver."""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from mitiq import pec, zne
from mitiq.pec.types import OperationRepresentation
from mitiq.zne.inference import (
    AdaExpFactory,
    ExpFactory,
    ExtrapolationError,
    PolyFactory,
    RichardsonFactory,
)
from mitiq.zne.scaling import fold_global
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as _NoiseModel

from .circuits import make_benchmark_circuit, mitiq_executor
from .config import (
    ADA_STEPS,
    IDEAL,
    PEC_NUM_SAMPLES,
    PEC_SHOTS,
    SCALE_FACTORS,
    SEEDS,
    SHOT_BUDGET,
    SHOTS,
)


# ---------------------------------------------------------------------------
# NoiseSpec dataclass + pipeline
# ---------------------------------------------------------------------------
@dataclass
class NoiseSpec:
    name: str
    build_noise_model: Callable[[], _NoiseModel]
    build_representations: "Callable[[QuantumCircuit], list[OperationRepresentation]] | None" = None
    asymptote: float | None = (
        0.25  # set to None to let ExpFactory/AdaExp fit `a` freely (3-param fit)
    )
    pec_num_samples: int = PEC_NUM_SAMPLES  # only used in the shot_budget=None fallback
    pec_shots: int = PEC_SHOTS  # see config.PEC_SHOTS for the few-shots/many-samples rationale
    aer_seed: int = 42
    scale_factors: Optional[list[float]] = None  # falls back to run_benchmark's default


def _assert_all_ops_represented(circuit, reps):
    """Guardrail: raise if any gate lacks a PEC representation.

    A missing rep (e.g. the cx(1,0) orientation) makes Mitiq SILENTLY leave the gate
    un-mitigated -> biased PEC. We use Mitiq's own matching (sample one circuit) and
    surface the 'No representation found' warning as a hard error instead of swallowing it.
    """
    from mitiq.pec.sampling import sample_circuit

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sample_circuit(circuit, reps, num_samples=1)
    missing = [str(x.message) for x in caught if "No representation" in str(x.message)]
    if missing:
        raise ValueError(
            f"PEC guardrail: {len(missing)} operation(s) have NO representation and "
            f"would be left un-mitigated (biasing PEC). First: {missing[0]}"
        )


def _zne_or_nan(circ, executor, factory):
    """Run ZNE; return NaN if the extrapolation fit fails to converge.

    Free-asymptote Exp/AdaExp fits can fail to converge on very weak, near-linear
    noise (e.g. the realistic thermal model) and raise ExtrapolationError. We record
    NaN for that (seed, method) rather than aborting the whole multi-seed benchmark.
    """
    try:
        return zne.execute_with_zne(
            circ, executor, factory=factory, scale_noise=fold_global
        )
    except ExtrapolationError:
        return float("nan")


def run_benchmark(
    spec: NoiseSpec,
    seeds: list[int] = SEEDS,
    scale_factors: list[float] = SCALE_FACTORS,
    shots: int = SHOTS,
    shot_budget: "int | None" = SHOT_BUDGET,
    run_pec: bool = True,
    verbose: bool = True,
    drift_models: "list[_NoiseModel] | None" = None,
):
    """Single-pass benchmark: noisy + 4 ZNE methods + PEC, on `seeds` circuits.

    EQUAL-COST method comparison: when `shot_budget` is set (default), every method
    spends ~shot_budget shots per circuit, so MSE gaps reflect the METHOD, not the
    budget:
      - non-adaptive ZNE : shot_budget // len(scale_factors) shots per lambda
      - AdaExp           : shot_budget // ada_steps shots per step (AdaExp does
                           `steps` evaluations, not len(scale_factors))
      - PEC              : shot_budget // pec_shots samples
      - Noisy            : shot_budget shots (no-mitigation reference at cost B)
    Set shot_budget=None to fall back to fixed `shots` / spec.pec_num_samples.

    Inter-run DRIFT: pass `drift_models` (one NoiseModel per seed, aligned to `seeds` order) to
    rebuild the backend PER RUN with theta_t (non-stationary noise). spec.build_noise_model is then
    unused; spec still supplies the NOMINAL PEC reps (stale vs the drifted channel -- by design) and
    the asymptote/scale_factors. Each per-run backend stays SEEDLESS (the anti-bias PEC fix).
    """
    # SINGLE SEEDLESS backend for ALL methods (uniform benchmark). A fixed seed_simulator makes
    # every backend.run() restart from the same noise RNG stream; for PEC -- thousands of
    # importance-sample runs -- that correlates the noise across samples and BIASES the estimator
    # (verified: composite PEC bias +0.13 seeded -> +0.004 seedless, variance ~25x lower). ZNE/noisy
    # were unbiased either way, but running them seedless too puts every method on the same footing
    # (independent noise per run -> honest variance, uniform ZNE-vs-PEC comparison). Reproducibility
    # is provided by the on-disk cache, which freezes each computed (df, raw, gammas).
    # `spec.aer_seed` is now vestigial (kept only so existing cache keys stay unchanged).
    if drift_models is not None and len(drift_models) != len(seeds):
        raise ValueError(
            f"drift_models has {len(drift_models)} entries but there are {len(seeds)} seeds; "
            "pass one per-run NoiseModel, aligned to `seeds` order."
        )
    # No drift -> one shared seedless backend (as before). Drift -> rebuilt PER RUN in the loop
    # below (theta_t), still SEEDLESS so PEC's importance samples see independent noise per run.
    static_backend = (
        None
        if drift_models is not None
        else AerSimulator(noise_model=spec.build_noise_model())
    )
    eff_scale_factors = (
        spec.scale_factors if spec.scale_factors is not None else scale_factors
    )
    ada_steps = ADA_STEPS

    if shot_budget is not None:
        zne_shots = max(1, shot_budget // len(eff_scale_factors))
        ada_shots = max(1, shot_budget // ada_steps)
        pec_samples = max(1, shot_budget // spec.pec_shots)
        noisy_shots = shot_budget
    else:
        zne_shots = ada_shots = noisy_shots = shots
        pec_samples = spec.pec_num_samples

    _methods = ["Noisy", "Richardson", "Poly2", "Exp", "AdaExp"] + (
        ["PEC"] if run_pec else []
    )
    res = {k: [] for k in _methods}
    gammas_pec = []

    if verbose:
        print(
            f"[{spec.name}]  seeds={len(seeds)}  asymptote={spec.asymptote}  "
            f"scale_factors={eff_scale_factors}"
        )
        if shot_budget is not None:
            print(
                f"  shot_budget={shot_budget:,}/circuit/method  ->  "
                f"ZNE {zne_shots}x{len(eff_scale_factors)}, AdaExp {ada_shots}x{ada_steps}, "
                f"PEC {pec_samples} samples x {spec.pec_shots} shots"
            )
        else:
            print(f"  fixed: shots={shots}, PEC samples={spec.pec_num_samples}")

    for i, s in enumerate(seeds):
        circ = make_benchmark_circuit(s)
        # Drift: theta_t backend for THIS run (seedless); else the shared static backend.
        backend = (
            static_backend
            if drift_models is None
            else AerSimulator(noise_model=drift_models[i])
        )
        ex_zne = mitiq_executor(backend, s, zne_shots)
        ex_ada = mitiq_executor(backend, s, ada_shots)
        ex_noisy = mitiq_executor(backend, s, noisy_shots)
        if run_pec:
            ex_pec = mitiq_executor(backend, s, spec.pec_shots)
            reps = spec.build_representations(circ)
            # Crude PEC overhead proxy: product of per-arity rep norms across gates
            gamma_circuit = 1.0
            for instr in circ.data:
                if instr.operation.name in ("measure", "barrier"):
                    continue
                matched = [
                    r
                    for r in reps
                    if len(r.ideal.all_qubits()) == instr.operation.num_qubits
                ]
                if matched:
                    gamma_circuit *= matched[0].norm
            gammas_pec.append(gamma_circuit)

        res["Noisy"].append(ex_noisy(circ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res["Richardson"].append(
                _zne_or_nan(
                    circ, ex_zne, RichardsonFactory(scale_factors=eff_scale_factors)
                )
            )
            res["Poly2"].append(
                _zne_or_nan(
                    circ, ex_zne, PolyFactory(scale_factors=eff_scale_factors, order=2)
                )
            )
            res["Exp"].append(
                _zne_or_nan(
                    circ,
                    ex_zne,
                    ExpFactory(scale_factors=eff_scale_factors, asymptote=spec.asymptote),
                )
            )
            res["AdaExp"].append(
                _zne_or_nan(
                    circ,
                    ex_ada,
                    AdaExpFactory(
                        steps=ada_steps, scale_factor=2.0, asymptote=spec.asymptote
                    ),
                )
            )
        if run_pec:
            # PEC: do NOT swallow warnings (a 'No representation found' = un-mitigated gate).
            _assert_all_ops_represented(circ, reps)
            val, _ = pec.execute_with_pec(
                circ,
                ex_pec,
                representations=reps,
                num_samples=pec_samples,
                full_output=True,
            )
            res["PEC"].append(float(val))
        if verbose and (s + 1) % 5 == 0:
            print(f"  seed {s + 1}/{len(seeds)} done")

    rows = []
    for name, vals in res.items():
        arr = np.asarray(vals, dtype=float)
        n_fail = int(np.isnan(arr).sum())  # extrapolation failures recorded as NaN
        if n_fail and verbose:
            print(f"  [warn] {name}: extrapolation failed on {n_fail}/{len(arr)} seeds "
                  f"-> stats over the {len(arr) - n_fail} that converged")
        # nan-aware stats: a method that fails on some seeds doesn't poison the rest
        mean = float(np.nanmean(arr)) if n_fail < len(arr) else float("nan")
        bias = mean - IDEAL
        var = float(np.nanvar(arr)) if n_fail < len(arr) else float("nan")
        rows.append(
            {
                "noise": spec.name,
                "method": name,
                "mean": round(mean, 4),
                "bias": round(bias, 5),
                "variance": round(var, 5),
                "mse": round(bias**2 + var, 5),
                "n_fail": n_fail,
                "gamma_pec_mean": round(float(np.mean(gammas_pec)), 3)
                if name == "PEC"
                else None,
            }
        )
    df = pd.DataFrame(rows).set_index(["noise", "method"])
    if verbose:
        print(df.to_string())
        print()
    return df, res, np.array(gammas_pec)

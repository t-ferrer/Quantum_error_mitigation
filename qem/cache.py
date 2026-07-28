"""Disk caching for run_benchmark results, so they survive kernel restarts.

`cached_benchmark` is a drop-in replacement for `run_benchmark`: it returns the same
`(df, raw, gammas)` tuple, but the first call computes + pickles the result and every
later call (even after a kernel restart) loads it from disk in milliseconds.

The cache key captures every parameter that affects the result AND the source code of
the spec's `build_noise_model` / `build_representations`, so editing a noise model (or
its parameters) automatically invalidates the stale entry.
"""
import hashlib
import inspect
import pickle
from pathlib import Path

from .benchmark import SHOT_BUDGET, run_benchmark
from .config import SCALE_FACTORS, SEEDS, SHOTS

CACHE_DIR = Path(__file__).resolve().parent.parent / "qem_cache"


def _key(spec, seeds, scale_factors, shots, shot_budget, run_pec):
    src = inspect.getsource(spec.build_noise_model)
    if spec.build_representations is not None:
        src += inspect.getsource(spec.build_representations)
    payload = repr((
        spec.name, spec.asymptote, spec.pec_num_samples, spec.pec_shots, spec.aer_seed,
        tuple(spec.scale_factors) if spec.scale_factors else None,
        tuple(seeds),
        tuple(scale_factors) if scale_factors else None,
        shots, shot_budget, run_pec, src,
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def cached_benchmark(spec, seeds=SEEDS, *, scale_factors=SCALE_FACTORS, shots=SHOTS,
                     shot_budget=SHOT_BUDGET, run_pec=True, force=False,
                     cache_dir=CACHE_DIR, verbose=True):
    """Run (and persist) `run_benchmark`, or load a matching cached result from disk.

    Pass `force=True` to recompute and overwrite the cached entry.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    key = _key(spec, seeds, scale_factors, shots, shot_budget, run_pec)
    path = cache_dir / f"{spec.name}_{key}.pkl"
    if path.exists() and not force:
        if verbose:
            print(f"[cache] loaded '{spec.name}' from {path.name} "
                  f"(delete it or pass force=True to recompute)")
        return pickle.loads(path.read_bytes())
    result = run_benchmark(spec, seeds=seeds, scale_factors=scale_factors, shots=shots,
                           shot_budget=shot_budget, run_pec=run_pec, verbose=verbose)
    path.write_bytes(pickle.dumps(result))
    if verbose:
        print(f"[cache] saved '{spec.name}' -> {path.name}")
    return result

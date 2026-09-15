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
from .fidelity import (
    FIDELITY_COLUMNS, add_fidelity_columns, method_channel_infidelities,
)

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


def cached_channel_infidelities(spec, seeds=SEEDS, *, force=False, cache_dir=CACHE_DIR):
    """Disk-cached `method_channel_infidelities` -- a few hundred bytes, ~20 s to compute.

    Kept in its OWN entry rather than folded into the benchmark pickle for two reasons: it is
    deterministic (no shots, so no reason to tie it to a particular sampling run), and the existing
    benchmark pickles predate it. Its key deliberately omits shots/shot_budget -- the channel-level
    numbers do not depend on the sampling budget -- but includes the scale factors and the noise
    source, which do.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    src = inspect.getsource(spec.build_noise_model)
    if spec.build_representations is not None:
        src += inspect.getsource(spec.build_representations)
    payload = repr((spec.name, tuple(seeds),
                    tuple(spec.scale_factors) if spec.scale_factors else None, src))
    key = hashlib.sha1(payload.encode()).hexdigest()[:12]
    path = cache_dir / f"channel_{spec.name}_{key}.pkl"
    if path.exists() and not force:
        return pickle.loads(path.read_bytes())
    result = method_channel_infidelities(spec, seeds)
    path.write_bytes(pickle.dumps(result))
    return result


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
        df, raw, gammas = pickle.loads(path.read_bytes())
        # BACKFILL the fidelity columns onto pickles written before that metric existed. The key
        # does NOT hash run_benchmark's source, so those entries stay valid -- and they should: a
        # derived column never justifies discarding a 20-minute run. The channel numbers come from
        # their own cache (first call ~20 s, then instant) so reloads stay in the millisecond range
        # this function exists for. No-op when the columns are already there.
        if not all(c in df.columns for c in FIDELITY_COLUMNS):
            df = add_fidelity_columns(
                df, spec, seeds, channel=cached_channel_infidelities(
                    spec, seeds, cache_dir=cache_dir)
            )
        return df, raw, gammas
    # Compute the channel numbers through THEIR cache and hand them to run_benchmark rather than
    # letting it recompute inline (`with_channel=False`): same single computation, but it also lands
    # on disk. Without this the channel entry was only ever written on the cache-HIT path, so a
    # `force=True` sweep left six specs with no channel cache and paid ~12 s again on the next load.
    channel = cached_channel_infidelities(spec, seeds, force=force, cache_dir=cache_dir)
    result = run_benchmark(spec, seeds=seeds, scale_factors=scale_factors, shots=shots,
                           shot_budget=shot_budget, run_pec=run_pec, verbose=verbose,
                           channel=channel)
    path.write_bytes(pickle.dumps(result))
    if verbose:
        print(f"[cache] saved '{spec.name}' -> {path.name}")
    return result

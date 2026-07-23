# Quantum Error Mitigation — ZNE · PEC · TREX under realistic noise

A reproducible benchmark suite comparing the main **quantum error mitigation (QEM)** techniques
across increasingly realistic noise models — from the toy depolarizing channel up to a composite
"hardware" model (`thermal ∘ mixed-unitary + readout`), and then under **calibration drift**
(non-stationary noise between executions).

Stack: **Qiskit / qiskit-aer** (simulation + noise models), **Mitiq** (ZNE/PEC/TREX), plus an
in-house layer (`qem/`) for building PEC representations and driving the benchmark.

---

## What this project does

- **DZNE (ZNE) vs PEC** — unified comparison of `Richardson / Poly2 / Exp / AdaExp` against PEC,
  at **equal shot budget** (so MSE gaps reflect the method, not the budget).
- **Exact PEC** — the quasi-probability representations are built to match the Aer `NoiseModel`
  **to machine precision** (Kraus extracted straight from Aer; custom L1 solver via
  `scipy.linprog`/HiGHS, more robust than Mitiq's BFGS on non-unital noise).
- **TREX** — readout-error mitigation (twirled readout), complementary to ZNE/PEC (neither of
  which removes the readout floor).
- **Drift track** — Ornstein-Uhlenbeck drift of the device parameters (T1, T2, gate residual,
  readout) between runs, which invalidates static PEC's exact-channel assumption → motivating
  re-learning. This is the research phase currently in progress.

### Noise models covered (`qem/noise_models.py`)

| Spec | Channel | Unital? | PEC |
|---|---|---|---|
| `SPEC_DEPOL` | Depolarizing | yes | analytic |
| `SPEC_AD` / `SPEC_PD` | Amplitude / Phase damping | no / yes | LP + Takagi |
| `SPEC_TH` / `SPEC_TH_REAL` | Thermal (γ and realistic T1/T2) | no | LP (exact) |
| `SPEC_MIXED_UNITARY` | Biased Pauli (depol + dephasing) | yes | analytic |
| `SPEC_COMPOSITE` | `thermal ∘ mixed-unitary` (hardware model) | no | LP (exact) |
| `SPEC_COMPOSITE_READOUT` | composite + readout | no | LP + readout floor |

Per-method metrics logged: **bias, variance, MSE, overhead γ, shot budget**.

---

## Repository layout

```
notebooks/
├── qem/                     Core package (imported by every notebook)
│   ├── noise_models.py      The NoiseSpecs (depol, AD, PD, thermal, composite, readout)
│   ├── pec_core.py          PEC representations: L1 linprog solver + Takagi helper
│   ├── benchmark.py         run_benchmark: noisy + 4 ZNE + PEC, N seeds, equal budget
│   ├── drift.py             Inter-run OU drift (non-stationary noise)
│   ├── readout.py           Readout models + TREX
│   ├── circuits.py          Mirror circuits (Proctor et al.) + Mitiq executors
│   ├── kraus.py             1-qubit Kraus generators
│   ├── config.py            Global parameters (shots, seeds, budget, gates)
│   └── cache.py             On-disk cache of benchmark results
├── DZNE_VS_PEC_mitiq.ipynb  ZNE vs PEC benchmark (depolarizing case, cross-validation)
├── Composite_noise_DZNE.ipynb   ZNE vs PEC on the composite model
├── Composite_drift*.ipynb   ZNE/PEC/TREX under calibration drift
├── TREX_readout_mitigation.ipynb  Readout mitigation
├── smoke_*.py               18 validation scripts (test harness)
└── qem_cache/               Pickled benchmark results (git-ignored, regenerated)
toy_openevolve/              LLM-guided extrapolation experiment (appendix)
refs/                        Reference papers
presentations/               Slides (only the final PDF is versioned)
STATE.md                     Lab notebook / technical decisions
```

---

## Installation

Python **3.12**.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

---

## Quickstart

The package is imported as `qem` and lives in `notebooks/qem/`, so notebooks and smoke tests run
**from the `notebooks/` directory**:

```bash
cd notebooks

# 1. Sanity check that everything runs (fast):
python smoke_drift_ou.py            # expect 5/5 OK

# 2. Run a benchmark in 3 lines:
python -c "
from qem import SPEC_COMPOSITE, cached_benchmark
df, raw, gammas = cached_benchmark(SPEC_COMPOSITE)   # 20 seeds, equal shot budget
print(df)
"
```

`cached_benchmark` pickles the result into `qem_cache/`: the first call computes, later calls
reload in milliseconds. The cache is auto-invalidated when the noise-model source code changes.

For interactive exploration and figures, open the notebooks (`jupyter notebook`).

---

## Project status

- ✅ **Static model** — complete family of noise models (up to the composite hardware model),
  DZNE-vs-PEC benchmark + readout/TREX. Done.
- 🔬 **Drift track (in progress)** — time-varying noise (OU) that breaks static ZNE/PEC; next
  steps: noise-model learning, PEA, VQA/QISMET.
- ⏳ **Hardware target** — running on a real IBM QPU (Mitiq vs Qiskit Runtime `resilience_level`).

See [STATE.md](STATE.md) for the detailed technical decisions and known pitfalls.

---

## References

- Temme, Bravyi, Gambetta (2017) — ZNE + PEC.
- Giurgica-Tiron et al. (2020) — DZNE (gate-level folding, adaptive extrapolation).
- Proctor et al., PRL (2022) — Pauli-randomized mirror circuits (benchmark).
- van den Berg et al. — TREX (Twirled Readout Error eXtinction).
- Dasgupta & Humble (arXiv:2308.14756), PEC-under-nonstationarity (arXiv:2404.13269) — drift.

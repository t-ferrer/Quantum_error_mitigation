"""Global parameters and derived constants shared across the benchmark."""

import networkx as nx

# Global parameters
P1 = 0.01  # 1-qubit gate depolarizing
P2 = 0.05  # CX gate depolarizing
SHOTS = 10_000
N_QUBITS = 2
DEPTH = 10
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
IDEAL = 1.0  # ideal P(|00>) for U.U^-1
SCALE_FACTORS = [1.0, 1.5, 2.0]

# --- Budget knobs (equal-cost comparison, run_benchmark) ---
# Single source of truth: change one of these and every NoiseSpec follows.
SHOT_BUDGET = 30_000  # shots / circuit / method for the equal-cost comparison
PEC_SHOTS = (
    100  # few shots / many QPD samples -> pec_samples = SHOT_BUDGET // PEC_SHOTS.
)
# At fixed budget B=N*shots the shot-noise term is set by B alone, while the dominant
# sign-variance term ~gamma^2/N only shrinks with N. So 3000x10 beats 300x100 (same B).
ADA_STEPS = 4  # AdaExp evaluations -> ada_shots = SHOT_BUDGET // ADA_STEPS
PEC_NUM_SAMPLES = (
    200  # NB: only used in the shot_budget=None fallback; IGNORED in equal-cost mode
)
#                        (there PEC uses SHOT_BUDGET // PEC_SHOTS samples, not this value).

# 'sxdg' added because fold_global generates the inverse of sx
BASIS_GATES = ["h", "rz", "sx", "x", "cx", "id", "sxdg"]

# Gate sets the noise models act on (shared by every NoiseSpec factory)
_GATES_1Q = ("h", "rz", "sx", "x", "sxdg", "id")
_GATES_2Q = ("cx",)

# Pauli-randomized mirror circuit knobs (Proctor et al., PRL 2022)
MIRROR_NLAYERS = 6
TWO_QUBIT_PROB = 0.7
CONNECTIVITY = nx.complete_graph(N_QUBITS)

# Aer (replacement) -> Mitiq (Pauli) convention
P_MITIQ_1Q = (3 / 4) * P1  # 0.0075
P_MITIQ_2Q = (15 / 16) * P2  # 0.046875
gamma_1q_theory = (1 + P1 / 2) / (1 - P1)
gamma_2q_local_theory = ((1 + 2 * P_MITIQ_2Q / 3) / (1 - 4 * P_MITIQ_2Q / 3)) ** 2
gamma_2q_global_theory = 1 + (15 / 8) * P2 / (1 - P2)

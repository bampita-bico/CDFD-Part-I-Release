"""
AFL Principle: J_max = min(C_i)
The maximum sustainable throughput of any network equals its weakest constraint.
Produces: afl_principle.png
"""
import os, sys, pathlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cdfd")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from minimal_engine import State, step, compute_life_number

# Build a 1-D chain of N nodes with varying constraint strengths.
# Theoretical J_max = min(C_i); we verify this numerically.

N = 10
trials = 50
rng = np.random.default_rng(0)

theoretical, simulated = [], []

for _ in range(trials):
    C_values = rng.uniform(0.5, 5.0, N)
    j_theory = float(np.min(C_values))

    # Simulate on a 1×N grid
    state = State(nx=1, ny=N, seed=int(rng.integers(1000)))
    state.C[0, :] = C_values
    state.phi[:] = 1.0

    for __ in range(200):
        step(state, dt=0.01)

    # Steady-state flux = mean(Φ/C) as a proxy for throughput
    j_sim = float(np.mean(state.phi / state.C))
    theoretical.append(j_theory)
    simulated.append(j_sim)

theoretical = np.array(theoretical)
simulated   = np.array(simulated)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(theoretical, simulated, alpha=0.7, color="#2176AE", s=40)
lim = (0, max(theoretical.max(), simulated.max()) * 1.1)
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.set_xlabel("Theoretical $J_{max} = \\min(C_i)$")
ax.set_ylabel("Simulated steady-state $J$")
ax.set_title("AFL Principle: Throughput bounded by weakest constraint")
ax.legend()
plt.tight_layout()
out = pathlib.Path(__file__).parent.parent / "afl_principle.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")

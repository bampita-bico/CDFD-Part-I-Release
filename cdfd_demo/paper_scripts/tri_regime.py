"""
Tri-Regime Bioenergetics
Demonstrates that the Life Number Λ crosses 1 as the three transport coefficients
(σ_e, σ_p, S) move from bottleneck → balanced → optimised.
Produces: tri_regime.png
"""
import os, sys, pathlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cdfd")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from minimal_engine import State, step, compute_life_number

# Sweep electron-transport capacity (σ_e proxy: inverse mean C)
# while keeping source/dissipation fixed.
constraint_levels = np.linspace(0.2, 5.0, 30)  # low C = high σ_e = high transport
lambdas = []

for c_level in constraint_levels:
    state = State(nx=32, ny=32, seed=7)
    state.C[:] = c_level
    state.phi[:] = 1.0

    for _ in range(300):
        step(state, dt=0.01, source=0.05, dissipation=0.01)

    lam = compute_life_number(state)
    lambdas.append(lam)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(constraint_levels, lambdas, color="#E63946", lw=2)
ax.axhline(1.0, color="k", lw=1, ls="--", label="Λ = 1 (life threshold)")
ax.fill_between(constraint_levels, lambdas, 1.0,
                where=[l > 1 for l in lambdas], alpha=0.15, color="green", label="Sustained life (Λ > 1)")
ax.fill_between(constraint_levels, lambdas, 1.0,
                where=[l < 1 for l in lambdas], alpha=0.15, color="red",   label="Non-living (Λ < 1)")
ax.set_xlabel("Constraint level C  (low C → high electron transport σ_e)")
ax.set_ylabel("Life Number Λ")
ax.set_title("Tri-Regime Model: Λ threshold separates living from non-living")
ax.legend()
plt.tight_layout()
out = pathlib.Path(__file__).parent.parent / "tri_regime.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")

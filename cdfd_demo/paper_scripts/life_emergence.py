"""
Life-number trajectory: Λ < 1 -> Λ ≈ 1 -> Λ > 1

Tracks a normalized public-capacity trajectory as energy input, electron
transport, and proton coherence rise together. The values are dimensionless and
illustrative; the figure is intended as a reproducible equation check, not as an
empirical validation claim.
Produces: life_emergence.png
"""
import os, sys, pathlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cdfd")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import PALETTE, apply_style

apply_style()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


times = np.linspace(0.0, 12.0, 360)

# Dimensionless normalized public-capacity terms.
energy_input = 0.35 + 0.75 * sigmoid(0.90 * (times - 4.0))
electron_capacity = 0.45 + 0.95 * sigmoid(0.80 * (times - 5.2))
proton_coherence = 0.72 + 0.28 * sigmoid(0.75 * (times - 6.0))
constraint_load = 1.42 - 0.44 * sigmoid(0.70 * (times - 5.4))
surface_response = 0.72 + 0.28 * sigmoid(0.80 * (times - 6.2))
maintenance_cost = 0.74

lambdas = (
    energy_input
    * electron_capacity
    * proton_coherence
    / (constraint_load * maintenance_cost)
)
psi_s = (energy_input / constraint_load) * surface_response

cross_idx = np.flatnonzero(lambdas >= 1.0)
t_cross = float(times[cross_idx[0]]) if cross_idx.size else None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)

ax1.axhspan(0.0, 1.0, color=PALETTE["light_red"], alpha=0.65, label=r"$\Lambda < 1$")
ax1.axhspan(1.0, 2.3, color=PALETTE["light_green"], alpha=0.65, label=r"$\Lambda > 1$")
ax1.plot(times, lambdas, color=PALETTE["green"], lw=2.1)
ax1.axhline(1.0, color="#222222", lw=1.0, ls="--")
if t_cross is not None:
    ax1.axvline(t_cross, color=PALETTE["gray"], ls=":", lw=1.0)
    ax1.text(
        t_cross + 0.18,
        0.14,
        rf"$\Lambda = 1$ at t = {t_cross:.1f}",
        fontsize=9,
        color=PALETTE["gray"],
    )
ax1.set_ylim(0.0, 2.25)
ax1.set_ylabel(r"Life Number  $\Lambda$")
ax1.set_title(r"Normalized Life-Number Trajectory")
ax1.legend(loc="upper left", frameon=True)

ax2.plot(times, psi_s, color=PALETTE["blue"], lw=2.1)
ax2.axhline(1.0, color="#222222", lw=1.0, ls="--", label=r"$\Psi_s = 1$")
if t_cross is not None:
    ax2.axvline(t_cross, color=PALETTE["gray"], ls=":", lw=1.0)
ax2.set_xlabel("Time t")
ax2.set_ylabel(r"Equilibrium proxy  $\Psi_s$")
ax2.set_ylim(0.0, 1.25)
ax2.legend(loc="upper left", frameon=True)

plt.tight_layout()
out = pathlib.Path(__file__).parent.parent / "life_emergence.png"
plt.savefig(out)
print(f"Saved {out}")

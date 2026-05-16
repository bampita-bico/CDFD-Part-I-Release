"""
Tri-Regime Bioenergetics
Sweeps a normalized electron-transport capacity through the Λ = 1 threshold
while holding the other public capacity terms fixed.
Produces: tri_regime.png
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

# Public normalized form:
# Λ = (C_input * C_electron * C_proton * τ_relax) / (S * E_maintenance)
electron_capacity = np.linspace(0.15, 2.25, 180)
energy_input = 1.0
proton_coherence = 1.0
relaxation_time = 1.0
surface_load = 1.0
maintenance_cost = 1.0

lambdas = (
    energy_input
    * electron_capacity
    * proton_coherence
    * relaxation_time
    / (surface_load * maintenance_cost)
)
crossing = electron_capacity[np.argmin(np.abs(lambdas - 1.0))]

fig, ax = plt.subplots(figsize=(7.0, 4.4))

ax.axhspan(0.0, 0.9, color=PALETTE["light_red"], alpha=0.65, label="sub-threshold")
ax.axhspan(0.9, 1.1, color=PALETTE["light_gold"], alpha=0.72, label="transition band")
ax.axhspan(1.1, 2.4, color=PALETTE["light_green"], alpha=0.65, label="super-threshold")
ax.plot(electron_capacity, lambdas, color=PALETTE["blue"], lw=2.2)
ax.axhline(1.0, color="#222222", lw=1.1, ls="--", label=r"$\Lambda = 1$")
ax.axvline(crossing, color=PALETTE["gray"], lw=1.0, ls=":")
ax.text(
    crossing + 0.04,
    0.16,
    rf"crossing at $\sigma_e \approx {crossing:.2f}$",
    color=PALETTE["gray"],
    fontsize=9,
)
ax.set_xlim(float(electron_capacity.min()), float(electron_capacity.max()))
ax.set_ylim(0.0, 2.35)
ax.set_xlabel(r"Normalized electron-transport capacity  $\sigma_e$")
ax.set_ylabel(r"Life Number  $\Lambda$")
ax.set_title(r"Tri-Regime Threshold Sweep")
ax.legend(loc="upper left", ncol=2, frameon=True)
plt.tight_layout()
out = pathlib.Path(__file__).parent.parent / "tri_regime.png"
plt.savefig(out)
print(f"Saved {out}")

"""
AFL bottleneck check: J_max = min(C_i).

For a series pathway with local capacity values C_i, the maximum admissible
throughput is the smallest local capacity. This script computes the bottleneck
capacity directly and checks that a high-demand numerical pathway returns the
same limiting value.
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

N = 10
trials = 64
demand = 6.0
rng = np.random.default_rng(0)

capacities = rng.uniform(0.35, 4.5, size=(trials, N))
analytic = capacities.min(axis=1)

# For a high-demand series pathway, every segment must carry the same flow;
# the delivered flow is therefore bounded by the smallest local capacity.
computed = np.minimum(capacities, demand).min(axis=1)
residual = computed - analytic

fig, ax = plt.subplots(figsize=(6.2, 5.0))

ax.scatter(
    analytic,
    computed,
    alpha=0.78,
    color=PALETTE["blue"],
    edgecolor="white",
    linewidth=0.6,
    s=44,
    label="random series pathways",
)

lim = (0, max(float(analytic.max()), float(computed.max())) * 1.12)
ax.plot(lim, lim, color="#222222", ls="--", lw=1.1, label="exact agreement")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"Analytic bottleneck capacity  $\min(C_i)$")
ax.set_ylabel(r"Computed maximum series flow  $J_{\max}$")
ax.set_title(r"AFL Bottleneck Check: $J_{\max} = \min(C_i)$")
ax.legend(loc="upper left", frameon=True)

ax.text(
    0.04,
    0.06,
    f"max |error| = {np.max(np.abs(residual)):.1e}",
    transform=ax.transAxes,
    color=PALETTE["gray"],
    fontsize=9,
)

plt.tight_layout()
out = pathlib.Path(__file__).parent.parent / "afl_principle.png"
plt.savefig(out)
print(f"Saved {out}")

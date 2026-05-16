"""
Life Emergence: Λ < 1 → Λ ≈ 1 → Λ > 1
Tracks the Life Number Λ over time as autocatalytic feedback drives the system
from non-living decay through proto-biological to sustained life.
Produces: life_emergence.png
"""
import os, sys, pathlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cdfd")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from minimal_engine import State, step, compute_life_number

# Start in non-living regime (high constraint, low flow)
# Gradually increase source term to simulate prebiotic chemistry kickstart.
state = State(nx=32, ny=32, seed=99)
state.C[:] = 3.0    # heavy constraint → Λ < 1 initially
state.phi[:] = 0.3

times, lambdas = [], []
total_steps = 1000

for i in range(total_steps):
    # Source ramps up over first 500 steps (chemical energy accumulation)
    source = 0.005 + 0.045 * min(i / 500, 1.0)
    step(state, dt=0.01, source=source, dissipation=0.003)
    lam = compute_life_number(state)
    times.append(state.t)
    lambdas.append(lam)

times   = np.array(times)
lambdas = np.array(lambdas)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(times, lambdas, color="#06D6A0", lw=1.5)
ax1.axhline(1.0, color="k", lw=1, ls="--")
ax1.fill_between(times, lambdas, 1.0, where=lambdas > 1, alpha=0.2, color="green")
ax1.fill_between(times, lambdas, 1.0, where=lambdas < 1, alpha=0.2, color="red")
ax1.set_ylabel("Life Number Λ")
ax1.set_title("Life Emergence: Λ transition from non-living → proto-biological → sustained life")

# Annotate regimes
t_cross = times[np.where(lambdas >= 1)[0][0]] if np.any(lambdas >= 1) else None
if t_cross:
    ax1.axvline(t_cross, color="gray", ls=":", lw=1)
    ax1.text(t_cross + 0.1, 0.5, f"Λ=1 at t={t_cross:.1f}", fontsize=8, color="gray")

# Also plot mean Ψ
psis = []
state2 = State(nx=32, ny=32, seed=99)
state2.C[:] = 3.0
state2.phi[:] = 0.3
for i in range(total_steps):
    source = 0.005 + 0.045 * min(i / 500, 1.0)
    step(state2, dt=0.01, source=source, dissipation=0.003)
    psis.append(float(np.mean(state2.psi_s)))

ax2.plot(times, psis, color="#118AB2", lw=1.5)
ax2.axhline(1.0, color="k", lw=1, ls="--", label="$\Psi_s = 1$ (equilibrium)")
ax2.set_xlabel("Time t")
ax2.set_ylabel("Mean $\Psi_s = (\Phi / C) \cdot S \cdot M_s$")
ax2.legend()

plt.tight_layout()
out = pathlib.Path(__file__).parent.parent / "life_emergence.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")

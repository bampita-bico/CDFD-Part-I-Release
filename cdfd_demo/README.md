# CDFD Public Minimal Demo

Welcome to the public-facing demonstration of the **Constraint-Driven Flux Dynamics (CDFD)** framework.

This directory is a small, runnable proof-of-concept for the public $\Psi_s = (\Phi / C) \cdot S \cdot M_s$ equations. It is not the full research archive, not the full paper source tree, and not a product.

## What This Demo Does

The demo exposes only:

1. A minimal NumPy equation model in `minimal_engine.py`.
2. Three figure-producing scripts in `paper_scripts/`.
3. A Docker path that runs those scripts without the private app stack.

The full workspace contains the broader paper archive and active code. This demo should be presented only as a reproducible public equation check.

## Public Demo Instructions
This minimal script exercises the core mathematics of the Adaptive Surface/Vacuum framework ($\Psi_s = (\Phi / C) \cdot S \cdot M_s$) without describing any non-public implementation details.

One command, three figures:
```bash
docker compose up
```

Figures are written to this directory:
| File | What it proves |
|------|---------------|
| `afl_principle.png` | Throughput J bounded by weakest constraint: J_max = min(C_i) |
| `tri_regime.png` | Life Number Λ crosses 1 as electron transport capacity increases |
| `life_emergence.png` | Λ transition: non-living → proto-biological → sustained life |

### Core Equations Proved Herein
**Flow evolution:** $\partial\Phi/\partial t = \nabla\cdot(1/C \cdot \nabla\Phi) + S - D$
**Constraint evolution:** $\partial C/\partial t = \alpha|\Phi| - \beta C + \gamma\nabla^2C$
**Equilibrium:** $\Psi_s = (\Phi / C) \cdot S \cdot M_s$

No internet required. `minimal_engine.py` is ~100 lines of NumPy — read it to verify the math.

### Numerical Verification Scripts
The main runnable public scripts are `afl_principle.py`, `tri_regime.py`, and `life_emergence.py`. Other `supplementary_*` files in this demo are legacy/minimal mirrors and should not be presented as the complete proof scripts from the private paper archive unless they are regenerated from the active source tree.

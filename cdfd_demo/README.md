# CDFD Public Demo

This directory contains a small, reproducible public demonstration for the
**Constraint-Driven Flux Dynamics (CDFD)** framework.

The demo is intentionally narrow: it exposes a minimal NumPy equation model,
three figure-generation scripts, and a Docker path for rerunning the public
checks. It is not the full research archive or the full paper source tree.
The notation follows CDFL, the public CDFD flow-constraint-memory language; the
larger reusable software target should be referred to as the CDFD Runtime.

## Author

Steve Bico Mujjabi, MD  
ORCID: https://orcid.org/0009-0001-0556-5516

## Scope

Included:

1. A minimal NumPy equation model in `minimal_engine.py`.
2. Three figure-producing scripts in `paper_scripts/`.
3. A Docker path that runs those scripts without the private app stack.

The figures should be read as reproducible equation illustrations and numerical
sanity checks, not as empirical validation of the CDFD hypotheses.

## Run

One command regenerates the three public figures:
```bash
docker compose up
```

The figures are written to this directory:

| File | Purpose |
|------|---------------|
| `afl_principle.png` | Checks the series-path bottleneck relation `J_max = min(C_i)`. |
| `tri_regime.png` | Sweeps a normalized transport capacity through the `Lambda = 1` threshold. |
| `life_emergence.png` | Shows an illustrative normalized trajectory crossing `Lambda = 1`. |

## Public Equations Represented

- Flow evolution: `dPhi/dt = div((1/C) grad Phi) + S - D`
- Constraint evolution: `dC/dt = alpha |Phi| - beta C + gamma laplacian(C)`
- Adaptive proxy: `Psi_s = (Phi / C) * S * M_s`
- Life Number: `Lambda = (C_input * C_electron * C_proton * tau_relax) / (S * E_maintenance)`

In this demo, `Phi` is normalized drive, `C` is normalized resistance or
constraint, `S` is a surface-response/load term, `M_s` is retained structural
memory, and `tau_relax` is the relaxation timescale used by the normalized Life
Number check. These are dimensionless public-demo variables, not calibrated
empirical constants.

No internet is required. `minimal_engine.py` is a compact NumPy implementation
intended for inspection and reproducibility.

## License

This public demo is licensed under CC BY 4.0. See `LICENSE`.

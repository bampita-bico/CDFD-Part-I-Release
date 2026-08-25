# Finite-Thickness `T(2,n)` Filament Optimization: A Constrained Numerical Non-Result

**Date:** 2026-08-24  
**Status:** reproducible computational non-result; not a physical theory paper.

## Question

Can a fixed finite-dimensional optimization of smooth `T(2,n)` Fourier curves
produce dense-grid-validated, unit-thickness equilibria from which a length or
energy scaling could be estimated? This question is deliberately narrower than
the former CDFT mass claim. The computation is a geometric proxy, not a
Faddeev--Niemi field-energy calculation and not a model of particle masses.

## Frozen method and acceptance rule

The implementation is
[`scripts/constrained_torus_knot_optimizer.py`](scripts/constrained_torus_knot_optimizer.py).
For odd `n` in `{3, 5, 7, 9, 11, 13}`, it represents a torus-knot initial
curve with `n + 2` Fourier modes and uses SLSQP to minimize polygonal curve
length subject to all nonlocal pairs on a finite constraint grid satisfying
`d_ij >= 1`.

A case is accepted only when all of the following were defined *before* the
run:

1. SLSQP reports success.
2. The normalized length has a relative range no greater than `5e-4` over the
   final five recorded iterates.
3. The dense validation-grid minimum nonlocal distance is in `[0.995, 1.005]`.
4. The sampled constraint residual is at least `-5e-6`.

The dense validation grid is six times the constraint-grid resolution. Passing
the finite constraint grid alone is not treated as a certificate for the
continuous curve.

## Reproduction record

On 2026-08-24 the archived unmodified script was rerun with its default fixed
settings. The exact command, script, CSV/JSON summaries, coefficients, and
iteration histories are retained in the separate local archive created that
day.

The rerun reproduced the archived values and acceptance verdicts. All six
cases were rejected; the fitted-power-law routine reported
`insufficient_converged_points` with count zero.

| `n` | Solver success | Dense minimum distance | Accepted | Rejection reason |
|---:|:---:|---:|:---:|---|
| 3 | yes | 0.786493 | no | dense thickness violation |
| 5 | yes | 0.983796 | no | dense thickness violation |
| 7 | yes | 0.966314 | no | dense thickness violation and no plateau |
| 9 | yes | 0.980437 | no | dense thickness violation |
| 11 | no | 0.835678 | no | solver failure, dense thickness violation, no plateau |
| 13 | yes | 0.824891 | no | dense thickness violation and no plateau |

The archived full result record is
[`outputs/torus_knot_optimizer/torus_knot_constrained_optimizer_summary.csv`](outputs/torus_knot_optimizer/torus_knot_constrained_optimizer_summary.csv)
and the associated JSON, coefficients, and iteration histories are in the same
directory.

## Result

This optimization does **not** support a finite-thickness equilibrium for any
tested member of this family under the stated representation, sampling, and
acceptance rule. Consequently it does not estimate an ideal-knot ropelength,
does not estimate a vortex-field energy, and does not test or support an
`n^0.75` mass law.

The previous comparison line `L/D ~= 5.66 + 3.57n` is retained only as an
external geometric ropelength reference. It is not an energy law, and a visual
similarity after calibration at `n = 3` does not constitute a physical
derivation.

## Limits and next legitimate step

This result does not prove that no finite-thickness `T(2,n)` minimizer exists.
It shows that this sampled SLSQP/Fourier workflow did not certify one. A
positive follow-up would need a continuous-thickness certificate or a much
stronger adaptive contact treatment, mesh-refinement convergence, multiple
independent initializations, and validation by an independent implementation.
Only then would a geometric length study be justified. A field-theory energy
claim would additionally require a specified energy functional and a separate
converged field solve.

## Publication boundary

This note can serve as a transparent supplement or negative computational
methods report after independent code review. It is not evidence for CDFT,
particle-family assignments, or a universal cross-domain law. The binding
release-wide reading rule remains
[`../Part_I_Fundamental_Physics/CORRECTION_STATUS_2026-08-18.md`](../Part_I_Fundamental_Physics/CORRECTION_STATUS_2026-08-18.md).

# Part I: Fundamental Physics

This directory contains the Part I physics sequence for the
Constraint-Driven Field Theory / Constraint-Driven Flux Dynamics project.

## Author

Steve Bico Mujjabi, MD  
ORCID: https://orcid.org/0009-0001-0556-5516

## Contents

- `papers/` - REVTeX 4-2 LaTeX sources for Papers I-XII.
- `PDFs/` - compiled submission PDFs.
- `figures/` - paper figures referenced by the REVTeX source files.
- `notebooks/supplementary_*.py` - deterministic supplementary scripts.
- `outputs/` - generated figures, tables, gate checks, and interactive panels.
- `references.bib` - shared bibliography.
- `CLAIM_STATUS.md` - claim-level status and uncertainty map.
- `MUJJABI_LAWS_AND_TESTS.md` - named laws, principles, tests, and vacuum
  engineering status.
- `REPRODUCIBILITY.md` - commands used to rebuild and verify the archive.
- `requirements-fullstack.txt` - optional acceleration and autodiff stack for
  scripts and notebooks.

## Scope

Papers I-X form the formal CDFT physics sequence. They develop the internal
flow/constraint/vortex-knot model, its algebraic identities, and its numerical
consistency checks.

Papers XI-XII are bridge papers. They extend the vocabulary toward cosmology
and collective matter, but they are explicitly hypothesis-level until they
produce independent observational or laboratory discriminants.

## Technical Preface

Part I is framed as a move from treating selected physical constants as
empirical inputs toward asking whether they can be recovered as equilibrium
conditions of a constrained transport medium. In that framing, the vacuum is not
presented as empty background, but as an effective medium with capacity,
responsiveness, and possible memory variables. The central Part I claim is
therefore model-level: if the CDFD transport-medium assumptions hold, the
fine-structure ratio can be read as the aspect ratio of a stable vortex
regulator. Paper I now states the CODATA 2022 target
`alpha^-1 = 137.035999177(21)` and the reproduced CDFD recovery
`chi = 137.035999177` near the beginning of the manuscript.

Historical comparisons in the source notes are treated only as intellectual
lineage. The release does not claim to replace established theories. It states
where the CDFD reading agrees with standard formulas, where it remains an
internal consistency check, and where future tests would have to outperform
QED, general relativity, or established laboratory-domain models.

## Current Integration Note

The material from `MISSING.docx` is integrated as a professional release update,
not as a thirteenth paper. The final edit plants the material as a chain:
Paper I states the laws, Papers II-IX develop symmetry/topology/mass/boundary
consequences, Paper X becomes the vacuum-engineering control-law bridge, Paper
XI carries cosmic-scale memory and capacity tests, and Paper XII closes with
the Mujjabi Falsification Program.

The public release keeps the final architecture in `MUJJABI_LAWS_AND_TESTS.md`
and the uncertainty boundaries in `CLAIM_STATUS.md`. The working extraction map
from `MISSING.docx` is an editorial artifact and is not part of the public
release surface.

## Series Flow

| Paper | Series role |
|---|---|
| I | States the Mujjabi Capacity Law, Adaptive Operating Ratio, and Stability Attractor for `1/alpha`. |
| II | Turns the regulator vortex into `Z_3` generation symmetry and the Koide theorem. |
| III | Interprets topology, chirality, and charge as constraint-history structure. |
| IV | Converts density, capacity, and responsiveness into a vacuum equation-of-state problem. |
| V | Closes the lepton chirality phase in the equilibrium `S*M_s -> 1` limit. |
| VI | Extends the structure into the torus-knot hierarchy. |
| VII | Reads mass sums as conserved transport/action budgets. |
| VIII | Produces the spectrum and prediction ledger. |
| IX | Defines exclusion and boundary principles so the theory can say no. |
| X | States vacuum engineering as control of `Phi`, `C`, `S`, and `M_s`. |
| XI | Moves capacity and memory tests to cosmology and the dark-sector bridge. |
| XII | Gives the Mujjabi Tests and laboratory falsification program. |

## Public CDFD Runtime Relationship

The public CDFD Runtime is now the reusable implementation target for
CDFL, the flow-constraint-memory notation used here. Part I does not depend on
hidden runtime behavior: its manuscript claims are reproduced by the paper-local
supplementary scripts, notebooks, and generated outputs in this archive. The
runtime is the broader implementation path for future domain simulations; this
release is the audited physics-paper reproduction layer.

## Scientific Python Stack

The Part I scripts use the full public scientific stack where it materially
supports a paper-local check: NumPy, SciPy, Pandas, Matplotlib, SymPy,
Statsmodels, scikit-learn, optional Numba, and optional Torch/JAX autodiff.
Not every paper needs every library; the rule is to use the stack where it
provides an independent symbolic, numerical, statistical, acceleration, or
autodiff check rather than adding decorative dependencies.

## Interactive Panels

`make_interactive_panels.py` generates offline HTML panels from the released
figures and CSV outputs. The panels are viewers over the full-stack outputs; the
scientific work remains in the supplementary scripts and notebooks.

## Review Status

These manuscripts are not peer reviewed. The supplementary scripts reproduce
the calculations and checks internal to the manuscripts; they do not constitute
empirical validation of the physical hypotheses.

## Submission Note

For APS-style archival submission, include the compiled PDFs, REVTeX sources,
`references.bib`, supplementary scripts, generated outputs, and the two
orientation files `CLAIM_STATUS.md` and `REPRODUCIBILITY.md`.

# Part I: Fundamental Physics

This directory contains the Part I physics sequence for the
Constraint-Driven Field Theory / Constraint-Driven Flux Dynamics project.

## Author

Steve Bico Mujjabi, MD  
ORCID: https://orcid.org/0009-0001-0556-5516

## Contents

- `papers/` - neutral LaTeX `article` sources for Papers I-XII.
- `PDFs/` - compiled submission PDFs.
- `figures/` - paper figures referenced by the LaTeX source files.
- `notebooks/supplementary_*.py` - deterministic supplementary scripts.
- `outputs/` - generated figures, tables, gate checks, and interactive panels.
- `references.bib` - shared bibliography.
- `CLAIM_STATUS.md` - claim-level status and uncertainty map.
- `CORRECTION_STATUS_2026-08-18.md` - binding correction notice for claims that
  are not established physics.
- `ARCHIVE_NOTICE_2026-08-24.md` - retained release scope and the separate
  archive boundary for working material.
- `standalone_notes/` - retained corrected algebra note and finite-thickness
  computational non-result.
- `methods/` - shared auditable toy-model declaration protocol.
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

The original Part I model is framed as a move from treating selected physical constants as
empirical inputs toward asking whether they can be recovered as equilibrium
conditions of a constrained transport medium. In that framing, the vacuum is not
presented as empty background, but as an effective medium with capacity,
responsiveness, and possible memory variables. This is an unconfirmed model
proposal, not an independent recovery of the fine-structure ratio. Paper I
states the CODATA 2022 target
`alpha^-1 = 137.035999177(21)` and the reproduced CDFD recovery
`chi = 137.035999177` near the beginning of the manuscript.

Historical comparisons in the source notes are treated only as intellectual
lineage. The release does not claim to replace established theories. It states
where the CDFD reading agrees with standard formulas, where it remains an
internal consistency check, and where future tests would have to outperform
QED, general relativity, or established laboratory-domain models.

## Archive Integration Note

The following describes the historical manuscript architecture. The correction
status, not this architecture summary, governs the current scientific reading.
The material from `MISSING.docx` was integrated as a release update, not as a
thirteenth paper. The final edit planted the material as a chain:
Paper I states the laws, Papers II-IX develop symmetry/topology/mass/boundary
consequences, Paper X becomes the vacuum-engineering control-law bridge, Paper
XI carries cosmic-scale memory and capacity tests, and Paper XII closes with
the Mujjabi Falsification Program.

The uncertainty boundaries in `CLAIM_STATUS.md` govern the public release. The
working extraction map from `MISSING.docx` is an editorial artifact and is not
part of the public release surface.

## Series Flow

| Paper | Series role |
|---|---|
| I | Introduces the named capacity, operating-ratio, and stability-attractor notation. |
| II | Turns the regulator vortex into `Z_3` generation symmetry and the Koide theorem. |
| III | Interprets topology, chirality, and charge as constraint-history structure. |
| IV | Converts density, capacity, and responsiveness into a vacuum equation-of-state problem. |
| V | Closes the lepton chirality phase in the equilibrium `S*M_s -> 1` limit. |
| VI | Extends the structure into the torus-knot hierarchy. |
| VII | Reads mass sums as conserved transport/action budgets. |
| VIII | Produces the spectrum and prediction ledger. |
| IX | Defines exclusion and boundary principles so the theory can say no. |
| X | Presents vacuum engineering as a hypothetical control programme for `Phi`, `C`, `S`, and `M_s`. |
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

The active Part I claim boundary is
`CORRECTION_STATUS_2026-08-18.md`, read together with `CLAIM_STATUS.md`.
The original PDFs remain archival records and must not be cited as established
physics where that notice withdraws or demotes a claim.

## Submission Note

For neutral journal submission, include the compiled PDFs, article-class
sources, `references.bib`, supplementary scripts, generated outputs, and the
two orientation files `CLAIM_STATUS.md` and `REPRODUCIBILITY.md`.

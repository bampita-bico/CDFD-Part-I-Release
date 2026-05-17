# Part I: Fundamental Physics

This directory contains the Part I physics sequence for the
Constraint-Driven Field Theory / Constraint-Driven Flux Dynamics project.

## Author

Steve Bico Mujjabi, MD  
ORCID: https://orcid.org/0009-0001-0556-5516

## Contents

- `papers/` - LaTeX sources for Papers I-XII.
- `PDFs/` - compiled submission PDFs.
- `notebooks/supplementary_*.py` - deterministic supplementary scripts.
- `outputs/` - generated figures, tables, gate checks, and interactive panels.
- `references.bib` - shared bibliography.
- `CLAIM_STATUS.md` - claim-level status and uncertainty map.
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
regulator.

Historical comparisons in the source notes are treated only as intellectual
lineage. The release does not claim to replace established theories. It states
where the CDFD reading agrees with standard formulas, where it remains an
internal consistency check, and where future tests would have to outperform
QED, general relativity, or established laboratory-domain models.

## Current Integration Note

The material from `MISSING.docx` is integrated as a professional release update,
not as a thirteenth paper. Paper I now carries the front-door interpretation of
the adaptive operating ratio, vacuum memory, and the conditional status of the
fine-structure numerical recovery. Paper XII now carries the falsification
roadmap for vacuum-memory and extreme-field tests. The middle papers already
use the shared equilibrium notation and are not padded with repeated text.

See `MISSING_INTEGRATION_PLAN.md` for the source-to-paper map and claim
guardrails.

## Public Universal Engine Relationship

The public CDFD Universal Engine is now the reusable implementation target for
the flow-constraint-memory notation used here. Part I does not depend on hidden
engine behavior: its manuscript claims are reproduced by the paper-local
supplementary scripts, notebooks, and generated outputs in this archive. The
engine is the broader runtime for future domain simulations; this release is the
audited physics-paper reproduction layer.

## Scientific Python Stack

The Part I scripts use the full public scientific stack where it materially
supports a paper-local check: NumPy, SciPy, Pandas, Matplotlib, SymPy,
Statsmodels, scikit-learn, optional Numba, and optional Torch/JAX autodiff.
Not every paper needs every library; the rule is to use the stack where it
provides an independent symbolic, numerical, statistical, acceleration, or
autodiff check rather than adding decorative dependencies.

## Review Status

These manuscripts are not peer reviewed. The supplementary scripts reproduce
the calculations and checks internal to the manuscripts; they do not constitute
empirical validation of the physical hypotheses.

## Submission Note

For archival submission, include the compiled PDFs, LaTeX sources,
`references.bib`, supplementary scripts, generated outputs, and the two
orientation files `CLAIM_STATUS.md` and `REPRODUCIBILITY.md`.

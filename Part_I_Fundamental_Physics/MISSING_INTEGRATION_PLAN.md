# MISSING.docx Integration Plan

Last edited: 2026-05-17.

## Editorial Decision

The new material should be integrated into the existing twelve-paper Part I
sequence rather than becoming a thirteenth paper. The argument is strongest when
Paper I states the central fine-structure and adaptive-vacuum framework early,
and Paper XII closes the sequence with falsifiable laboratory targets.

## Material To Integrate

| Source theme from `MISSING.docx` | Integration target | Editorial treatment |
|---|---|---|
| Adaptive operating ratio, responsiveness, and memory | Paper I and series README | Keep as the shared CDFD language; define memory as a local state/history variable, not as metaphor. |
| Twelve-decimal recovery of `1/alpha` | Paper I | Keep as an internal numerical consistency result conditional on the stated stiffness calibration. Do not present it as independent empirical proof. |
| Rankine/Gaussian/core-profile stress tests | Paper I open problems | Add as a required robustness program unless and until reproducible code is added. Do not import unsupported numerical tables. |
| Quantum decoherence as vacuum hysteresis | Paper I open problems and Paper XII tests | Present as a hypothesis with a possible relaxation-time parameter, not as a settled derivation. |
| Experimental roadmap | Paper XII | Add guarded falsification targets with success and failure conditions. |
| GitHub/Zenodo versioning guidance | Release workflow only | Keep out of manuscripts. Update DOI fields only after a new Zenodo version exists. |
| Claims about surpassing historical figures or "magic numbers" | Exclude | Replace with professional, evidence-based language. |

## Files Edited

- `papers/01_Vortex_Stability_in_a_Constrained_Transport_Vacuum.tex`
- `papers/12_Friction_Superconductivity_Plasma_and_Transport_Mysteries.tex`
- `README.md`
- `CLAIM_STATUS.md`

## Release Guardrails

- The manuscripts should distinguish internal consistency checks from empirical
  validation.
- The adaptive memory parameter `M_s` should be treated as an explicit state
  variable whose relaxation law remains open.
- Experimental targets should be framed as discriminants against QED, general
  relativity, and established laboratory-domain models, not as observed effects.
- No new DOI, badge, or Zenodo version DOI should be written until Zenodo assigns
  one after a new GitHub release.

# MISSING.docx Full Integration Plan

Last edited: 2026-05-17.

## Source Audit

The source document reviewed for this integration is:

`/home/bampita/Projects/CDFD/MISSING.docx`

The DOCX package contains one main document body, no separate comments,
footnotes, endnotes, headers, or footers, and two embedded PNG figures. A fresh
`pandoc` extraction produced 1,050 Markdown lines. OCR on the two images showed
that they are low-resolution plot screenshots: one energy-stability plot versus
aspect ratio, and one convergence plot near the fine-structure point. Because
the figures do not include reproducible data or code, they are treated as source
notes rather than imported numerical evidence.

## Editorial Decision

The new material should be integrated into the existing twelve-paper Part I
sequence rather than becoming a thirteenth paper. The argument is strongest when
Paper I states the central fine-structure and adaptive-vacuum framework early,
and Paper XII closes the sequence with falsifiable laboratory targets.

## Full DOCX Disposition

| DOCX material | Integration target | Editorial treatment |
|---|---|---|
| ORCID/profile review and 12-paper overview | README and Paper I framing | Retain author details and the idea that the twelve papers form one chain; remove conversational phrasing. |
| Adaptive operating ratio, responsiveness, and memory | Paper I and series README | Keep as shared CDFD notation; define memory as a local state/history variable, not as metaphor. |
| Twelve-decimal recovery of `1/alpha` | Paper I | Keep as an internal numerical consistency result conditional on the stated stiffness calibration. Do not present it as independent empirical proof. |
| Rankine, Gaussian, hollow, and finite-core stress tests | Paper I open problems | Add as a required robustness program unless and until reproducible code is added. Do not import unsupported numerical tables. |
| Vacuum-memory kernel and relaxation time | Paper I and Paper XII | Use a guarded kernel with relaxation time `tau_M`; make measurement or bounds an explicit open problem. |
| Quantum decoherence as vacuum hysteresis | Paper I operational dictionary and Paper XII tests | Present as a proposed residual mechanism beyond standard environmental decoherence, not as a settled replacement. |
| Planck constant relation `h = Phi_c tau_v` | Paper I open problems | Treat as a target derivation for Paper III, not as a completed derivation. |
| Charge, mass, and gravity interpretations | Paper I operational dictionary and claim status | Keep as proposed CDFD readings; mark mass/gravity extensions as conditional. |
| Transport-medium vs Copenhagen comparison | Paper I | Convert from rhetorical comparison into an operational dictionary with status notes. |
| Draft abstract/numerical appendix for a thirteenth paper | Paper I and Paper XII | Integrate the core content into the existing sequence; no thirteenth paper. |
| Suggested numerical core-profile table | Integration plan only | Do not import as evidence because no generating code or provenance was present in the DOCX. |
| GitHub/Zenodo DOI versioning guidance | Release workflow only | Keep the concept DOI stable; update version DOI fields only after Zenodo assigns a new version. |
| Historical-person comparison language | Exclude from manuscripts | Replace with professional lineage language: the work extends existing physics vocabulary and must earn status through tests. |
| High-impact README wording | README metadata only | Distill to sober summary language; remove "magic numbers", "that era ends", and "universal operating system" phrasing. |
| Experimental roadmap / eponymous test labels | Paper XII | Convert to descriptive falsification targets: pump-probe hysteresis, extreme-field alpha sensitivity, and rotating-source residuals. |

## Files Edited

- `papers/01_Vortex_Stability_in_a_Constrained_Transport_Vacuum.tex`
- `papers/12_Friction_Superconductivity_Plasma_and_Transport_Mysteries.tex`
- `README.md`
- `CLAIM_STATUS.md`
- root `README.md`
- `.zenodo.json`
- `CITATION.cff`

## Release Guardrails

- The manuscripts should distinguish internal consistency checks from empirical
  validation.
- The adaptive memory parameter `M_s` should be treated as an explicit state
  variable whose relaxation law remains open.
- Experimental targets should be framed as discriminants against QED, general
  relativity, and established laboratory-domain models, not as observed effects.
- The embedded figures are not release evidence until their data and generating
  scripts are available.
- No new DOI, badge, or Zenodo version DOI should be written until Zenodo assigns
  one after a new GitHub release.

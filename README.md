# CDFD Part I: Fundamental Physics

[![DOI](https://zenodo.org/badge/1241207680.svg)](https://doi.org/10.5281/zenodo.21420219)

This release contains the public Part I physics archive for the
Constraint-Driven Field Theory / Constraint-Driven Flux Dynamics project. The
manuscript sources are maintained as neutral LaTeX `article`-class files.

## Author

Steve Bico Mujjabi, MD<br>
Independent Researcher<br>
CDFD framework: Steve Bico Mujjabi and VuraLabs<br>
Kampala, Uganda<br>
ORCID: https://orcid.org/0009-0001-0556-5516

Institutional home: **VuraLabs**

## Keywords

Constraint-Driven Field Theory; Constraint-Driven Flux Dynamics; CDFT; CDFD;
CDFD Runtime; CDFL; constrained-flow language;
fundamental physics; theoretical physics; mathematical physics; computational
physics; particle physics; quantum foundations; field theory; vacuum structure;
vacuum equation of state; vacuum engineering; topological physics; knot theory;
torus knots; vortex dynamics; adaptive vacuum; vacuum memory; vacuum
hysteresis; Mujjabi Tests; falsifiable predictions; Koide formula; lepton
masses; fine-structure constant; reproducible research; open science.

## GitHub Topics

`cdfd`, `cdft`, `theoretical-physics`, `mathematical-physics`,
`cdfd-runtime`, `cdfl`, `computational-physics`, `quantum-foundations`, `field-theory`,
`vacuum-structure`, `topological-physics`, `knot-theory`, `vortex-dynamics`,
`vacuum-engineering`, `mujjabi-tests`, `fine-structure-constant`,
`koide-formula`, `reproducible-research`,
`open-science`.

## Contents

- `Part_I_Fundamental_Physics/` - Part I article-class manuscripts, PDFs, figure
  assets, supplementary scripts, generated outputs, bibliography, claim-status
  notes, and reproducibility instructions.
- `cdfd_demo/` - small public equation demo with Docker support.
- `CITATION.cff` - GitHub citation metadata.
- `.zenodo.json` - Zenodo deposit metadata.
- `requirements.txt` - Python dependencies for the public supplementary scripts.
- `environment.yml` - Conda environment for the public science stack.
- `LICENSE` - CC BY 4.0 release license.
- `LICENSE_BOUNDARY.md` - explains why the scholarly Part I archive can use
  CC BY 4.0 while the public CDFD Runtime uses AGPL-3.0-or-later.

## Review Status

These manuscripts are archived as research/preprint materials and are not peer
reviewed. The supplementary scripts reproduce the internal calculations, figures,
tables, and consistency checks. They do not establish empirical validation of the
physical hypotheses.

## Final Part I Architecture

The final edit turns Part I into a named adaptive-vacuum program rather than a
loose sequence of papers. Paper I states the Mujjabi Capacity Law, the Mujjabi
Adaptive Operating Ratio, and the Mujjabi Stability Attractor for the
fine-structure ratio. Papers II-IX develop the symmetry, topology, mass-budget,
spectrum, and boundary rules. Paper X states the vacuum-engineering control
principle. Papers XI-XII define the Mujjabi Tests that can falsify or support
the memory, alpha-jitter, and pressure-gradient claims.

## Suggested Reading Order

1. `Part_I_Fundamental_Physics/README.md`
2. `Part_I_Fundamental_Physics/CLAIM_STATUS.md`
3. `Part_I_Fundamental_Physics/REPRODUCIBILITY.md`
4. `Part_I_Fundamental_Physics/PDFs/`

## Reproducibility

See `Part_I_Fundamental_Physics/REPRODUCIBILITY.md` for rebuild and verification
commands.

## Public CDFD Runtime

The CDFD Runtime is now public in the main CDFD workspace as the
general runtime for CDFL, the flow-constraint-memory state grammar. This Part I
release remains a scholarly archive: its paper-local Python scripts,
notebooks, and outputs reproduce the figures and tables cited by the physics
papers.

Cite and license the two layers separately until a combined runtime release DOI
exists.

## License

This release is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0). See `LICENSE` and
`LICENSE_BOUNDARY.md`.

## Citation

For this release family, cite:

Steve Bico Mujjabi, MD. CDFD Part I: Fundamental Physics. Zenodo.
https://doi.org/10.5281/zenodo.20250820

After a new GitHub release is archived by Zenodo, cite the version DOI shown on
that Zenodo record for the exact released snapshot.

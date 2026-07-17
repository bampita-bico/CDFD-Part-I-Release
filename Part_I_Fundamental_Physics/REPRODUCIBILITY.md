# Reproducibility

Last local verification: 2026-07-17.

## Environment Used

- Python: `3.14.4`
- LaTeX: `latexmk`, `pdflatex`, `bibtex`, REVTeX 4-2, and `apsrev4-2`
  available on path
- Required Python stack: see `../requirements.txt` and `requirements-fullstack.txt`
- Optional acceleration backends observed during verification:
  - `numba=ok` (`0.65.1`)
  - `torch=missing`
  - `jax=missing`

The optional Torch/JAX checks are written as optional gates. Their absence did
not fail the current verification run.

If the local Matplotlib config directory is not writable, set
`MPLCONFIGDIR=/tmp/matplotlib-cdfd-part-i` before running the scripts. The
2026-05-20 verification completed with Matplotlib using temporary cache
directories under `/tmp`.

## Stack Policy

The supplementary scripts and notebooks use the full scientific Python stack
where it adds a real independent check:

- SymPy for symbolic identities and derivatives.
- SciPy for roots, integrals, optimizers, signal checks, and regressions.
- Pandas for released tables and gate files.
- Matplotlib for deterministic figures.
- Statsmodels and scikit-learn for statistical and parity checks.
- Numba, Torch, and JAX as optional acceleration/autodiff cross-checks.

Dependencies are not added merely to make a paper look computational. If a
manuscript does not cite a numerical result, it does not need an output or
notebook beyond the relevant reproducibility path.

## Rebuild Supplementary Outputs

From the repository root:

```bash
python Part_I_Fundamental_Physics/notebooks/supplementary_I.py
python Part_I_Fundamental_Physics/notebooks/supplementary_II.py
python Part_I_Fundamental_Physics/notebooks/supplementary_III.py
python Part_I_Fundamental_Physics/notebooks/supplementary_IV.py
python Part_I_Fundamental_Physics/notebooks/supplementary_V.py
python Part_I_Fundamental_Physics/notebooks/supplementary_VI.py
python Part_I_Fundamental_Physics/notebooks/supplementary_VII.py
python Part_I_Fundamental_Physics/notebooks/supplementary_VIII.py
python Part_I_Fundamental_Physics/notebooks/supplementary_IX.py
python Part_I_Fundamental_Physics/notebooks/supplementary_X.py
python Part_I_Fundamental_Physics/notebooks/supplementary_XI.py
python Part_I_Fundamental_Physics/notebooks/supplementary_XII.py
```

Expected result: each script exits successfully and writes figures, tables, and
check files under `Part_I_Fundamental_Physics/outputs/paper_*`.

## Check Summary Gates

The current verification run found no failed summary gates in:

- `outputs/paper_III/checks_summary_gate.csv`
- `outputs/paper_IV/checks_summary_IV_gate.csv`
- `outputs/paper_V/checks_summary_V_gate.csv`
- `outputs/paper_VI/checks_summary_VI_gate.csv`
- `outputs/paper_VII/checks_summary_VII_gate.csv`
- `outputs/paper_VIII/checks_summary_VIII_gate.csv`
- `outputs/paper_IX/checks_summary_IX_gate.csv`
- `outputs/paper_X/checks_summary_X_gate.csv`
- `outputs/paper_XI/checks_summary_XI_gate.csv`
- `outputs/paper_XII/checks_summary_XII_gate.csv`

Papers I and II report their pass/fail checks directly in generated check files
and script output.

Paper I also writes the `MISSING.docx` core-profile robustness guardrail:

- `outputs/paper_I/table4_core_profile_stress_tests.csv`
- `outputs/paper_I/checks_core_profile_stress_tests.csv`
- `outputs/paper_I/fig1e_core_profile_stress_test.pdf`

## Mujjabi Tests and Vacuum Engineering Outputs

The final Part I edit names a falsification program, but it does not pretend
that the laboratory protocols have already been run. The current release
contains operational definitions and guardrail tables; future engine-facing work
should add executable protocol templates for:

- pump-probe vacuum hysteresis and `tau_M` bounds;
- extreme-field alpha-jitter spectroscopy residuals;
- rotating-source gravitational residual controls;
- decoherence-memory residuals after standard open-system terms;
- transport-threshold tests in plasma, superconductivity, friction, jamming,
  and fracture.

Those protocols should write machine-readable prediction tables before any
claim is treated as empirical evidence.

## Rebuild PDFs

From `Part_I_Fundamental_Physics`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/cdfd_parti_revtex_build papers/*.tex
```

The current local REVTeX build completed for all 12 papers with no failed
papers and no missing citation or reference warnings.

The submission-facing PDFs are copied to `PDFs/`.

## What This Verifies

The reproducibility scripts verify that the archive can regenerate its stated
tables, figures, and internal consistency checks. They do not verify that the
physical hypotheses are true in nature.

---
title: "An auditable declaration protocol for toy cross-domain models"
date: 24 August 2026
status: "methods-paper draft; evaluation incomplete"
---

# Summary

Cross-domain model proposals often mix a renamed variable, a numerical output,
and a scientific interpretation in one narrative. That makes a basic question
hard to answer: what was actually declared before a result was seen? This paper
proposes a small declaration record for toy models. The record requires a
domain question, named native baseline, declared inputs and state, equations,
calibration status, outcome, finite-output audit, falsification criterion, and
executable provenance. Its purpose is to make a model easy to inspect and easy
to reject when its stated criterion fails.

The protocol does not claim that a shared vocabulary identifies a shared
mechanism. It treats terms such as drive, capacity, recovery, or constraint as
labels whose scientific value depends entirely on a domain-native measurement,
comparison, and held-out test.

# Record structure

The machine-readable schema is
[`model-declaration.schema.json`](model-declaration.schema.json). A valid
record must state:

1. A question and at least one field-native baseline.
2. The exact scope of the model and an explicit list of non-claims.
3. Inputs, state variables, equations, and whether calibration is absent,
   prespecified, or fitted.
4. Outcomes, the scope and result of a finite-output audit, and a criterion
   capable of falsifying the intended claim.
5. Repository, revision, command, and run-bundle location sufficient for a
   reader to recover the computational record.

The example record documents a failed finite-thickness knot optimization. It
records the failure rather than normalizing the failed curves into a usable
scaling law. This is the intended use case: preserving a negative result with
enough context to prevent later reinterpretation.

# Relationship to existing practice

The record is not a replacement for domain standards. SBML supports exchange
and reuse of biological models; SED-ML describes simulation experiments across
tools; and COPASI supplies biochemical-network simulation and analysis. Those
systems are preferable wherever their semantics and workflows apply. The
present record instead adds claim scope, calibration status, and a compact
falsification field around a small locally executed model. It is also narrower
than general research-object packaging: it does not prescribe a storage format,
solver, ontology, or metadata catalog.

The associated CDFD Runtime can create deterministic result envelopes,
finite-output audits, and run bundles for CDFL examples, but the protocol is
not tied to that runtime. A record can point to a notebook, another simulator,
or experimental analysis code, provided that the provenance is precise and the
scientific baseline is named.

# Claim boundaries

A completed declaration shows that a computation was recorded and passed or
failed a stated computational check. A finite-output audit does not establish
convergence, calibration, construct validity, causality, clinical utility, or
external generalization. Similarly, an empirical record remains incomplete
until its data source, preregistration, comparison model, and held-out analysis
are documented.

This distinction matters particularly for cross-domain work. Similar ratios
can occur because mature fields already use drive-to-resistance or
drive-to-threshold quantities. Such a correspondence is at most a teaching or
bookkeeping analogy until a pre-specified extension improves prediction beyond
the field's own variables.

# Evaluation required before publication

This is not yet a publishable methods contribution. Before submission, the
schema must be evaluated on independent records with at least three kinds of
evidence: inter-reviewer agreement about the claim boundary and calibration
status; an audit showing that omitted provenance or post-hoc fitting is detected
more reliably than in unstructured reports; and user feedback from researchers
outside this project. The protocol should be rejected or revised if it adds
documentation without improving reviewability or reproducibility.

# Availability and archival disposition

The current schema, example, and consolidation register are in this directory.
The CDFD Part I--IV archives remain historical records with their own
correction notices. They are not evidence that the protocol has been validated,
and they should not be submitted as a set of separate universal-law papers.

## Sources

- [SBML specification and resources](https://sbml.org/)
- [SED-ML specification and resources](https://sed-ml.org/)
- [COPASI documentation](https://copasi.org/)
- [Research Object Crate specification](https://www.researchobject.org/ro-crate/)
- [FAIR principles](https://www.go-fair.org/fair-principles/)

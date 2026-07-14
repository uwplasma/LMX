# Contributing to LMX

Contributions are welcome when they preserve LMX's central rule: evidence comes
before a validation, differentiability, or performance claim.

## Before changing code

1. Read [plan.md](plan.md) for the current milestone and scope boundary.
2. Check the [documentation index](docs/index.md),
   [example catalog](examples/catalog.toml), and
   [benchmark matrix](docs/benchmark_matrix.md) for the capability's stability
   and existing unit, verification, and workflow evidence.
3. Discuss changes that alter equations, normalization, boundary conditions,
   benchmark tolerances, or the stable API in a GitHub issue before implementation.

## Development setup

```bash
git clone https://github.com/uwplasma/LMX.git
cd LMX
uv venv
uv pip install -e '.[dev,docs]'
```

Run focused tests while developing, then the complete gate:

```bash
.venv/bin/python scripts/run_full_test_suite.py
```

The complete portable gate must retain at least 95% branch coverage and finish
within 600 seconds per supported-Python job. New or changed logic should have
complete branch coverage unless a documented numerical or platform constraint
prevents it.

## Evidence required with a capability

Every shipped capability needs:

- a deterministic unit/branch test;
- a physical invariant, manufactured solution, analytical comparison, or other
  verification test;
- a public-API workflow test;
- equations, units, assumptions, supported parameter envelope, and failure
  behavior in the documentation;
- provenance and checksums for external reference data.

Production-size sweeps, Docker solvers, accelerators, and clusters run in their
named external lanes. Their parsers, schemas, small fixtures, and failure paths
remain in the portable gate.

## Support and security

Use GitHub Discussions or an issue for installation questions, reproducible
bugs, and proposed capabilities. Include the LMX commit, Python/JAX versions,
platform and device, the smallest reproducer, expected and observed behavior,
and relevant residual or conservation diagnostics. Research-stage features are
supported only within their documented evidence envelope.

Report vulnerabilities through GitHub's private security-advisory interface for
`uwplasma/LMX`; never publish exploit details, credentials, or private data in an
issue. Security fixes target the latest release and `main`. Treat user-provided
TOML, NPZ, CSV, JSON, VTK, and external-solver data as untrusted input.

## Benchmarks and tolerances

Freeze a versioned benchmark specification before examining production results.
Never relax a tolerance retrospectively to obtain a pass. A threshold may change
only when reference uncertainty, discretization analysis, or a corrected physical
definition justifies it, with the old and new evidence retained.

## Pull requests

Keep each pull request reviewable and identify:

- the plan package it advances;
- public behavior and numerical contracts changed;
- focused and complete test commands and results;
- benchmark records regenerated or intentionally unchanged;
- documentation, migration, and provenance updates;
- cold/warm timing and memory effects for solver changes.

Do not commit generated fields, movies, meshes, build products, or large figure
bundles. Use the release-asset process documented in the developer guide.

## Authorship and credit

Authorship is based on substantive intellectual contribution, not repository
access, job title, or funding alone. Publications should discuss authorship early
and use the CRediT taxonomy to record conceptualization, methodology, software,
validation, analysis, data curation, visualization, supervision, and writing.
Code and data contributions that do not meet a publication's authorship criteria
must still receive appropriate acknowledgement and repository credit. Conflicts
are resolved by the maintainers using the contribution record and the target
venue's authorship policy.

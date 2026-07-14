# External benchmark comparisons

External solvers and experiments provide independent evidence; they are not
dependencies of an ordinary LMX installation. LMX keeps adapters, frozen input
specifications, and compact observables in Git while raw solver trees and field
dumps remain local or in versioned releases.

## FreeMHD

The closed-channel parity lane audits FreeMHD dictionaries before constructing
the matching LMX case. It checks geometry, material properties, magnetic field,
Hartmann number, inlet drive, wall conductivities, mesh, and observable
normalization.

Run against a local installation or container:

```bash
.venv/bin/python scripts/run_freemhd_parity_suite.py --help
```

The same command materializes an audited Benchmark-A smoke case without
starting FreeMHD. Keep the case under the external FreeMHD workspace, not in
the LMX checkout:

```bash
.venv/bin/python scripts/run_freemhd_parity_suite.py \
  --materialize shercliff \
  --freemhd-install-dir /Users/rogerio/local/tests/freemhd_install \
  --output /Users/rogerio/local/tests/freemhd_install/external_cases/shercliff-ha20
```

Run that exact materialized case through the external Docker installation:

```bash
cd /Users/rogerio/local/tests/freemhd_install
./run_case.sh /workspace/external_cases/shercliff-ha20 4
```

The `/workspace/...` path is the container-visible form of the mounted host
path. Do not use `run_shercliff.sh` or `run_hunt.sh` for a materialized case:
those convenience wrappers recopy the original demos. The case manifest,
OpenFOAM logs, VTK, reconstructed fields, and raw time histories remain in the
external workspace. Only compact, checksummed acceptance records belong in
LMX; large reusable artifacts belong in a versioned release.

To compare an already materialized reference:

```bash
.venv/bin/python -m examples.freemhd_closed_channel_observable_parity --help
```

The frozen Benchmark A aggregate is
`benchmarks/results/benchmark-a-acceptance.json`. At its finest audited mesh,
velocity, Lorentz-force, and pressure-gradient observables for Shercliff and
Hunt are below the 1% finite-grid target. Current closure and power identities
pass independently.

The original Docker examples were found to describe physical `Ha = 1000`, not
the nominal `Ha = 20` comparison. Those outputs are historical workflow data,
not accepted parity evidence. The audited case specification is authoritative.

## Matched Benchmark B contract

Benchmark-B acceptance no longer trusts a record-supplied
`exact_case_match` boolean. The validator compares the LMX and FreeMHD
equations, nondimensional groups, geometry, field mapping, wall model,
boundaries and drive, mesh coordinates, stopping rules, observable, and
normalization. It also requires current specification, source, input,
evaluator, and output hashes, then recomputes the pressure-comparison gates.
An `acceptance_role` separates a bounded `harness-smoke` record from B1 or B2
production evidence; a smoke record cannot promote itself.

This machinery is a gate, not a parity result. B2 production remains blocked:
LMX's present stationwise model omits the convective inertia in FreeMHD's
finite-inertia momentum equation, and fixed-flow/Neumann axial treatment has
not been shown equivalent to FreeMHD's inlet-flow/outlet-pressure treatment.
Reconcile both formulations before creating a tiny matched B2 smoke record;
do not launch a medium or production FreeMHD campaign before that record
passes.

## ALEX experiments

Digitized pipe and square-duct pressure references live in
`benchmarks/references/`; frozen case definitions live in `benchmarks/specs/`.
Use:

```bash
.venv/bin/python examples/pipe_reference_comparison_demo.py --help
.venv/bin/python examples/fringing_benchmark_demo.py --help
```

These cases remain research-stage until experimental observable, mesh/time,
conservation, and steady-convergence gates pass together.

## Q2D-MHDfoam and other OpenFOAM data

`lmx.external_validation` parses line profiles, force histories, probes, VTK
fields, and case dictionaries from an external Q2D-MHDfoam tree. The former
in-repository Docker build context was removed because it duplicated a large,
version-specific external code installation. Point the adapters at a separately
managed reference tree instead.

No turbulent Q2D parity claim is currently made. A valid comparison must match
domain, topology, forcing, Hartmann friction, Reynolds number, averaging window,
and observable definition before numerical errors are evaluated.

## Comparison contract

For every external record, retain:

1. citation and source checksum;
2. dimensional inputs and nondimensional groups;
3. geometry, mesh, boundaries, initial state, and drive mode;
4. exact observable definition and normalization;
5. convergence and averaging protocol;
6. source and dependency fingerprints;
7. source, input, evaluator, and output checksums;
8. acceptance role, quantitative tolerance, and a machine-recomputed result.

Primary acceptance uses observables the reference actually resolves. Images,
streamline resemblance, or a single centerline value may support diagnosis but
cannot replace current, power, convergence, and uncertainty gates.

## Data policy

Small independently sourced tables may live in `benchmarks/references/` with
provenance. Large meshes, complete OpenFOAM/FreeMHD cases, raw transient fields,
restarts, logs, VTK, and generated movies stay in an external workspace or a
release with checksums. This keeps cloning and installation lightweight without
discarding reproducibility.

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
python scripts/run_freemhd_parity_suite.py --help
```

To compare an already materialized reference:

```bash
python -m examples.freemhd_closed_channel_observable_parity --help
```

The frozen Benchmark A aggregate is
`benchmarks/results/benchmark-a-acceptance.json`. At its finest audited mesh,
velocity, Lorentz-force, and pressure-gradient observables for Shercliff and
Hunt are below the 1% finite-grid target. Current closure and power identities
pass independently.

The original Docker examples were found to describe physical `Ha = 1000`, not
the nominal `Ha = 20` comparison. Those outputs are historical workflow data,
not accepted parity evidence. The audited case specification is authoritative.

## ALEX experiments

Digitized pipe and square-duct pressure references live in
`benchmarks/references/`; frozen case definitions live in `benchmarks/specs/`.
Use:

```bash
python examples/pipe_reference_comparison_demo.py --help
python examples/fringing_benchmark_demo.py --help
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
7. quantitative tolerance and a machine-readable pass/fail summary.

Primary acceptance uses observables the reference actually resolves. Images,
streamline resemblance, or a single centerline value may support diagnosis but
cannot replace current, power, convergence, and uncertainty gates.

## Data policy

Small independently sourced tables may live in `benchmarks/references/` with
provenance. Large meshes, complete OpenFOAM/FreeMHD cases, raw transient fields,
and generated movies belong in a release with checksums. This keeps cloning and
installation lightweight without discarding reproducibility.

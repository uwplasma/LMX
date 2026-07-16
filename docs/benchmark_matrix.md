# Benchmark matrix

LMX promotes a capability only when its governing equations, numerical gates,
reference data, and limitations are all explicit. Compact machine-readable
records live in `benchmarks/`; large fields and movies live in releases.

## Status

| Lane | Evidence | Acceptance | Status |
|---|---|---|---|
| A1 Hartmann duct | analytical profile, mesh convergence, current and power balance | bounded errors and second-order trend | verified |
| A2 Shercliff/Hunt ducts | analytical profiles and Samper Table I | all eight frozen rows pass | verified |
| A3 FreeMHD closed channel | audited inputs and four mesh levels | finest finite-grid observables below 1% | verified for frozen cases |
| B1 ALEX pipe fringe | digitized pressure data and 3D solver | conservation passes; steady pressure gate open | research-stage |
| B2 ALEX square fringe | digitized pressure data and 3D solver | canonical smoke and sharding pass; fresh current-formulation coarse conservation passes but steady convergence remains open | research-stage |
| Q2D turbulence | reduced-model and adapter checks | independent turbulent parity required | staged |
| Magnetic obstacle | qualitative topology checks | quantitative experimental parity required | staged |

"Verified" applies only to the listed model and parameter range. It does not
imply full FreeMHD parity, turbulence, heat transfer, free surfaces, or magnetic
induction.

## Benchmark A: fully developed ducts

The stable solver is checked against Hartmann, Shercliff, and Hunt solutions.
Each accepted case combines:

- velocity-profile and flow-rate error;
- mesh refinement and observed order;
- discrete charge conservation and wall-current closure;
- pressure, viscous, Lorentz, and Joule power identities;
- solver residual and implementation fingerprints.

At the audited `85 x 63` FreeMHD comparison mesh, Shercliff
velocity/Lorentz/pressure errors are `0.56% / 0.40% / 0.27%`; Hunt errors are
`0.58% / 0.86% / 0.26%`. The eight high-Hartmann-number Samper rows cover
Shercliff and Hunt at `Ha = 500, 5000, 10000, 15000`; all pass the frozen flow,
layer, solver, current, and power gates.

The authoritative aggregate is
`benchmarks/results/benchmark-a-acceptance.json`. Regenerate it with:

```bash
python scripts/run_samper_table_i.py
python scripts/run_samper_table_i.py \
  --freeze-summary artifacts/samper/table-i-summary.json \
  --output benchmarks/results/samper-table-i-accepted.json
python scripts/build_benchmark_a_acceptance.py
```

Fresh FreeMHD comparisons require a local FreeMHD installation or container and
are deliberately outside the portable test gate:

```bash
python scripts/run_freemhd_parity_suite.py --help
```

Finite-grid comparison and continuum inference are reported separately. A
Richardson-extrapolated observable is not presented as a raw mesh result.

## Benchmark B: fringing fields

Frozen specifications and digitized references are:

- `benchmarks/specs/alex-b1-pipe.toml`
- `benchmarks/specs/alex-b2-square.toml`
- `benchmarks/references/alex-b1-pipe.csv`
- `benchmarks/references/alex-b2-square.csv`

The portable example exercises construction, current closure, restart, and
observable extraction:

```bash
python examples/fringing_benchmark_demo.py --help
```

B2 has exact-parity axial sharding on 1/2/4 forced CPU devices. Its prior
two-RTX-A4000 result is historical: the current-source refresh exposes an
alternating shard-boundary defect and is not accepted. B1 now uses its
accepted compatible retained-modal pressure solver; experimental-observable,
mesh-ladder, and final steady-response acceptance remain open. This is not yet
a claim that the experimental pressure curve has passed.
The first fresh current-formulation coarse trajectory (`102 x 77 x 77`) and
one exact restart continuation reach step 256 in two bounded 131-second runs on
two RTX A4000 shards. All pressure/electric solves and conservation gates pass,
and the residual decreases monotonically. Its final `7.1081e-4` residual is
still above the frozen `5e-5` criterion and misses the continuation's
precommitted `5.5e-4` stop/go gate. It is not an accepted coarse result; more
stepping, independence variants, and larger meshes remain blocked while the
convergence slowdown is diagnosed.
The legacy no-inertia, stationwise-flow B2 coarse campaign passes steady, conservation, tighter
tolerance, doubled-iteration, and confirmation-wall gates. Its coarse ALEX
curve alone does not pass the frozen literature limits (`weighted RMS 1.221`,
`weighted max 3.595`, integrated error `0.139`), so it is correctly retained as
mesh evidence rather than promoted as an experimental result.
The source-identical legacy medium campaign (`152 x 113 x 113`) passes baseline,
tighter-tolerance, doubled-iteration, confirmation-wall, steady, and
conservation gates on two actual GPU shards. The tolerance and iteration deltas
are `5.781e-4` and `5.784e-4` of the frozen uncertainty; the wall-thickness
difference is `9.26e-14`. Its curve still has weighted RMS `1.324`, weighted
maximum `3.967`, and integrated error `0.206`, so medium numerical independence
is closed while experimental and three-mesh acceptance remain open.
The legacy fine campaign (`202 x 149 x 149`) passes baseline, doubled-iteration,
confirmation-wall, steady, conservation, and two-shard placement gates. Its
tight variant is preserved at a durable 48-update checkpoint with residual
`3.4770e-5`, above the `2.5e-5` target. Bounded scalar probes either plateaued
or oscillated, and componentwise Aitken diverged. The fine curve also misses all
three frozen ALEX literature limits (weighted RMS `1.389`, weighted maximum
`4.218`, integrated error `0.251`). Fine numerical independence remains open;
the tiny matched-FreeMHD smoke is complete, while production-mesh FreeMHD and
observable/model normalization are the next validation priority.

After all three source-identical mesh campaigns finish, assemble (without
rerunning) their literature, independence, refinement, and exact-case FreeMHD
gates with the existing runner:

```bash
python scripts/run_benchmark_b_independence.py \
  --acceptance-mesh coarse=artifacts/b-coarse \
  --acceptance-mesh medium=artifacts/b-medium \
  --acceptance-mesh fine=artifacts/b-fine \
  --freemhd-record B1-fringing-pipe=/release-assets/b1-freemhd.json \
  --freemhd-record B2-fringing-square=/release-assets/b2-freemhd.json
```

The compact `benchmark-b-acceptance.json` output reports uncertainty-weighted
RMS and maximum error, integrated-pressure error, successive mesh changes, all
solver/wall independence gates, and external-evidence checksums. Full fields
and restart bundles remain release assets rather than Git content.
Medium and fine B2 campaigns may explicitly pass `--prolong-restart` with the
matching coarser baseline or confirmation-wall asset. This is an initialization
optimization only; every refined solve must still converge and pass all gates.

## Required gates

Every promoted physics benchmark must define:

1. a frozen case specification and source citation;
2. independently stored reference observables;
3. mesh or time convergence where discretization error matters;
4. conservation and power identities;
5. a quantitative tolerance chosen before the final run;
6. a source, dependency, hardware, and input fingerprint;
7. a bounded reproducer and a machine-readable result.

Differentiable benchmarks additionally compare automatic derivatives with
finite differences or an independent transpose solve. Scaling benchmarks must
compare identical physics, report cold and warm timings, and prove actual shard
placement.

## Source map

| Purpose | Location |
|---|---|
| analytical cases | `lmx/cases.py`, `lmx/validation.py` |
| benchmark construction | `lmx/benchmarks.py` |
| FreeMHD input observers and audits | `lmx/freemhd.py` |
| external observable adapters | `lmx/external_validation.py` |
| frozen specs and references | `benchmarks/specs/`, `benchmarks/references/` |
| accepted compact results | `benchmarks/results/` |

The literature basis and governing equations are summarized in
[Theory](theory.md); exact current acceptance records take precedence over old
figures or prose.

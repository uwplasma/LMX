# LMX authoritative development plan

- **Status:** final authoritative execution plan
- **Baseline:** 12 July 2026, after M3 closure and accepted ALEX B1/B2
  finite-volume integration; production sharding and scaling are active
- **Scope:** `uwplasma/lmx` and the reusable solver work contributed to
  `uwplasma/SOLVAX`
- **Supersedes:** every earlier roadmap, checklist, and planning note

This file is the sole roadmap and decision record. Status statements are
authoritative only when backed by a tracked acceptance artifact from the same
source fingerprint. Commands, investigation logs, and transient measurements
belong in issues, campaign artifacts, or benchmark documentation—not here.

The current critical path is **M5 performance unblock -> M4 closure -> research
release**. M0 through M3 are closed. SOLVAX 0.7.0 is the pinned runtime
dependency and `auto` resolves deterministically to its PCG backend. Its current
CPU and RTX A4000 forward, implicit-gradient, transpose, resource, and
end-to-end Hartmann gates pass. The four-level Ha=20 FreeMHD and all-eight-row
high-Ha acceptance record remains the historical 0.5.1 M3 baseline until the
version-matched M14 physics refresh completes. Explicit native `cg` remains
available for comparison and one compatibility cycle.
Research-stage fringing, Q2D, blanket, and scaling demonstrations are not
release evidence until their milestone gates pass. Both ALEX production
branches now use the accepted nonuniform finite-volume operators, compatible
face-flux projections, fixed-flow constraints, and SOLVAX implicit solves.
They remain research-stage until numerical independence, three-level
experimental comparison, balances, and matched FreeMHD parity pass section 10.

The plan is intentionally gated. A later milestone may be prototyped, but it
may not become the default, expand a release claim, or consume the main
optimization effort until the preceding milestone exits. Failed gates are
reported as results; tolerances are changed only from reference uncertainty,
discretization analysis, or a corrected physical definition, never to make a
campaign pass.

## 1. Mission

LMX will be a compact, research-grade, JAX-native solver for incompressible,
inductionless liquid-metal MHD. It will be accurate against analytical,
literature, experimental, and FreeMHD references; end-to-end differentiable
inside a documented parameter envelope; efficient on CPU and GPU; and usable
without knowledge of its internals.

The governing rule is **evidence before scope or optimization**. A capability
is called validated only when its versioned, machine-readable acceptance record
passes. A plot, finite gradient, single-grid comparison, or qualitative match
is supporting evidence, never an acceptance gate.

Work proceeds in this dependency order:

1. keep the complete portable quality gate reproducible and below ten minutes;
2. close fully developed duct verification and matched FreeMHD parity;
3. simplify the core without changing the verified numerical contracts;
4. move generic solver machinery to SOLVAX and validate implicit gradients;
5. close literature, experimental, and FreeMHD fringing-field validation;
6. keep production runs within their runtime budgets and demonstrate CPU/GPU
   strong scaling before launching expensive validation campaigns;
7. add turbulence and heat transfer in separately validated releases.

## 2. Product boundary

### Stable target

- incompressible, low-magnetic-Reynolds-number electric-potential MHD;
- steady and transient fully developed rectangular and layered ducts;
- uniform, analytic, and tabulated imposed magnetic fields;
- insulating, thin-wall-conductance, and explicit conducting-wall models;
- pressure-gradient and flow-rate driving;
- restart, diagnostics, compact outputs, Python API, TOML, and CLI;
- differentiable observables for parameters that pass the gradient contract.

### Research-stage until externally validated

- extruded 3D rectangular, layered, and mapped-pipe meshes;
- fringing/nonuniform magnetic fields;
- spatial multi-device decomposition;
- reduced Q2D models.

### Deferred

- free surfaces, multiphase flow, and full magnetic induction;
- turbulence/LES before its laminar and Q2D prerequisites pass;
- heat transfer and buoyancy before the 3D laminar solver passes Benchmark B.

FreeMHD parity means parity for overlapping inductionless, single-phase cases
with matched inputs and observables. It does not mean reproducing all FreeMHD
features.

## 3. Current verified baseline

### Portable quality gate

The sole complete local gate is:

```bash
uv run --locked --extra dev python scripts/run_full_test_suite.py
```

CI may invoke the runner directly only after the shared setup action has
synchronized this same locked `dev` extra.

The source-matched endpoint gate at commit `b42403f` collects 907 tests: 899
pass and 8 optional external-reference tests skip. Python 3.10.20 with JAX
0.6.2 reaches 95.07% branch coverage in 142.8 seconds; Python 3.13.7 with JAX
0.10.2 reaches 95.06% in 196.3 seconds. The slower endpoint consumes 33% of
the hard ten-minute budget and 55% of the 360-second engineering target. The
compact record is `benchmarks/results/portable-gate-20260713.json`. Hosted
Actions did not assign runners because of an account billing/spending-limit
failure, so they must be rerun after that administrative issue is resolved;
this is not a code-gate failure. Strict documentation and deterministic
provenance also pass. No focused run, historical green run, or partially
regenerated provenance may be reported as the current gate.

The runner collects every portable test, uses up to six host-aware workers,
constrains BLAS threads, disables JAX preallocation, fails below 95% branch
coverage, warns at 450 seconds, and terminates at 600 seconds. Four-core CI
runners remain at four workers. On the 10-core reference Mac, six workers reduce
the current Python 3.10 gate from 141.1 to 113.0 seconds (`1.249x`) with the same
900 passes, 8 skips, and 95.09% coverage. The engineering target is <=360
seconds, leaving 40% headroom for slower supported machines and future tests.
No feature may evade this gate by being marked slow. External data, Docker,
GPU, and cluster runs live in explicit lanes whose schemas, parsers, dispatch,
and failure paths remain in portable CI.

### Benchmark A: Ha=20 ducts

Canonical specifications are
`benchmarks/specs/shercliff-ha20.toml` and
`benchmarks/specs/hunt-ha20.toml`. They freeze equations, dimensions,
materials, wall conductance, drive, mesh family, solver controls, reference
checksums, observables, normalization, and thresholds. FreeMHD is pinned at
commit `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5`.

The canonical near-constant-ratio ladder is 37x29, 49x37, 65x49, with 85x63
as the confirmation level. Hunt's effective meshes include four explicit wall
cells in each direction. Flow is controlled to the independently specified
case target; no profile is peak-fitted.

At 65x49, normalized errors against the processed FreeMHD slices are:

| Level/case | velocity y-cut | Lorentz y-cut | pressure gradient | balance gate |
|---|---:|---:|---:|---:|
| 65x49 Shercliff | 0.00761 | 0.00619 | 0.00791 | pass |
| 65x49 Hunt | 0.00867 | 0.01042 | 0.00557 | pass |
| 85x63 Shercliff | 0.00561 | 0.00404 | 0.00266 | pass |
| 85x63 Hunt | 0.00581 | 0.00859 | 0.00261 | pass |

The 65x49 Hunt Lorentz cut misses 1% by 0.00042. The completed 85x63
confirmation passes every raw primary gate. Hunt's last-three-level apparent
orders are 1.72 for velocity, 1.69 for Lorentz force, and 1.85 for pressure;
Shercliff's corresponding leading orders are 1.46, 1.51, and 1.89. The local
four-level artifact, compact acceptance files, and provenance manifests are
tracked and deterministically reproducible.

Current and power diagnostics are now conservative and dimensional. At 85x63,
the electrical-network residual is `1.84e-11` for Shercliff and `8.48e-12` for
Hunt, face-to-cell Lorentz transfer closes at roundoff, and the mechanical
residual is `1.08e-5` and `1.31e-8`, all below the `1e-3` balance threshold.
The corrected normalized interface-current residual is zero and `1.51e-8`.

Generalized Richardson extrapolation passes every primary quantity except the
Hunt Lorentz y-cut, whose extrapolated error is 0.01120 despite the raw finest
level passing. The processed FreeMHD slices have a measurable analytical
reference floor.
For velocity, their normalized errors against the supplied analytical solution
are 0.01542/0.00975 on the Shercliff y/z cuts and 0.00563/0.00453 for Hunt.
The corresponding 85x63 LMX errors are 0.01069/0.00664 and
0.00325/0.00250. Therefore:

- finite-grid FreeMHD agreement and analytical-continuum agreement are reported
  separately;
- Richardson extrapolation may diagnose convergence but cannot manufacture a
  pass against a finite-grid reference floor;
- the 1% threshold is not relaxed and cancellation is not promoted as parity.

Compact evidence belongs in `benchmarks/results/`; detailed interpretation
belongs in `docs/external_benchmarks.md`. Investigation history does not belong
in this plan.

### High-Hartmann-number Table I status

The accepted numerical path now includes exact zero-conductivity disconnection,
insulating-boundary ownership at wall intersections, volume-scaled potential
assembly, gauge-compatible tensor/line preconditioning, SOLVAX flexible GMRES,
JIT-cached linear solves, and a staged nonlinear tolerance with a strict final
potential solve. These are numerical contracts, not benchmark-specific tuning.
The final current, power, analytical-flow, layer-resolution, refinement, and
steady-state gates remain strict.

| Samper/Buehler row | Evidence state | Finest analytical flow error | Refinement result |
|---|---|---:|---|
| Shercliff Ha=500 | accepted | 0.241% | monotone, order 2.01, 0.134% finest change |
| Shercliff Ha=5,000 | accepted | 0.369% | monotone, order 2.02, 0.204% finest change |
| Shercliff Ha=10,000 | accepted | 0.418% | monotone, order 2.07, 0.247% finest change |
| Shercliff Ha=15,000 | accepted | 0.300% | 119x119 confirmation, order 1.96, 0.139% finest change |
| Hunt Ha=500 | accepted | 0.154% | monotone, order 1.45, 0.051% finest change |
| Hunt Ha=5,000 | accepted | 0.325% | monotone, order 1.29, 0.096% finest change |
| Hunt Ha=10,000 | accepted | 0.427% | monotone, order 1.37, 0.137% finest change |
| Hunt Ha=15,000 | accepted | 0.507% | monotone, order 1.40, 0.170% finest change |

All rows and the refreshed Ha=20 FreeMHD ladder share solver-core fingerprint
`60c67d073508d36be713148955150074ad556166d48cb8a94df330b7b1be4172`.
Their separated finite-grid, analytical-continuum, conservation/power, and
literature claims pass in `benchmarks/results/benchmark-a-acceptance.json`.
Intermediate and failed campaign JSON files are diagnostic only and must never
be cited as release evidence.

## 4. Evidence and benchmark contract

Every accepted benchmark has a versioned YAML or TOML specification with:

- equations, approximations, sign conventions, and nondimensionalization;
- geometry, materials, wall models, imposed field, drive, and dimensionless
  groups;
- boundary/initial conditions, potential gauge, mesh family, layer resolution,
  and time/solver tolerances;
- reference title/DOI, license, checksum, source, and extraction method;
- primary integral observables and secondary field cuts;
- spatial/time convergence procedure and balance definitions;
- tolerances derived from analytical precision, experimental uncertainty, or a
  documented cross-code error budget;
- runtime lane, backend, code/environment identifiers, command, and artifact
  checksums.

Evidence is weighted in this order:

1. discrete identities and manufactured solutions;
2. analytical solutions and asymptotic limits;
3. refinement with conservation and power closure;
4. independent matched-code parity;
5. experimental or published benchmark data.

PDFs are cited and checksummed, not vendored. Git contains compact licensed
observables, schemas, and provenance. Large cases, meshes, VTK fields, movies,
and figure bundles belong in a checksummed GitHub/Zenodo release.

### Benchmark A acceptance

Required coverage:

- manufactured potential, momentum, and pressure operators on uniform,
  stretched, and discontinuous-conductivity meshes;
- formal observed order for smooth Hartmann problems;
- Hartmann, Shercliff, and Hunt velocity, potential/current, flow, pressure,
  current closure, and Joule/Lorentz/mechanical power;
- Samper/Buehler Table I integral flow for Ha=500, 5,000, 10,000, and 15,000
  on boundary-layer-resolved meshes;
- matched FreeMHD Shercliff/Hunt studies on at least three refinements plus a
  confirmation level when the finest primary result is within 20% of its
  tolerance.

The gate passes only when:

- case and reference checksums match and sampling transformations are recorded;
- normalized flow and each primary profile error are <=1%;
- refinement is monotonic with a reliable observed order, or the analytical
  reference floor is explicitly quantified;
- charge, mass, interface-current, and power residuals are <=0.1% and at least
  ten times smaller than the accepted observable error;
- results are independent of stopping limits; and
- no per-profile fitted normalization is used.

### Benchmark B acceptance

Implement the published fringing cases, not only bounded synthetic responses:

- pipe: Ha=6600, interaction parameter 10700, wall conductance 0.027;
- square duct: Ha=2900, interaction parameter 540, wall conductance 0.07.

Primary evidence is pressure drop between published stations. Secondary
evidence is flow conservation, potential/current closure, wall current, and
published velocity/current profiles. `div(B)` must converge to zero and the
field's magnetostatic assumptions must be documented. Freeze tolerances before
production. Require three meshes, iteration/time independence, uncertainty-
aware experimental agreement, and one matched FreeMHD case.

### Later benchmarks

- C: Q2D turbulent Type-I/Type-II regimes and published statistics;
- D: 3D turbulence or magnetic-obstacle statistics;
- E: coupled energy/buoyancy and Nusselt trends.

Each is a separate release tier. Internal decay curves and movies remain
demonstrations until external gates pass.

## 5. Test and CI contract

“Full battery” means every portable test for every shipped feature, not a
curated subset. It must pass on every pull request in <=600 seconds for each
supported Python job.

| Lane | Contents | Cadence | Hard limit |
|---|---|---|---:|
| portable-full | all unit, API, CLI, I/O, examples, manufactured, analytical, bounded physics, adapter, docs-contract, and branch-coverage tests | every PR, Python min/max | 600 s/job |
| docs | strict build, links, examples, public API | every PR | 600 s/job |
| external-data | checksummed literature data and artifact regeneration | nightly/manual | recorded |
| FreeMHD | pinned Docker build/run and matched reports | scheduled/release | recorded |
| accelerator | CPU/GPU equivalence, gradients, memory, warm performance | scheduled/release | recorded |
| scaling | multi-device/multi-host strong scaling | release candidate | recorded |

Portable-full rules:

- >=95% branch coverage over `lmx`; new or changed code targets 100%;
- no marker exclusions, network, sleeps, optional binaries, or large datasets;
- deterministic seeds and explicit x64/precision policy;
- default per-test timeout 120 seconds; any test over 20 seconds gets a runtime
  owner and optimization issue;
- synthetic fixtures cover every external format and failure mode;
- every public feature maps to unit, invariant/verification, and user-workflow
  tests in `provenance/features.json`;
- runtime, collection count, skips, coverage, and slowest tests are retained;
- warn at 450 seconds, fail at 600, and open a remediation issue at two
  consecutive runs above 80% of either budget;
- dependency updates and minimum/maximum Python versions run the same gate;
- flaky retries are forbidden in the required gate.

Future growth is handled by making tests cheaper, sharing compiled fixtures,
and moving only nonportable evidence to its proper lane—not by testing fewer
functions. Release claims require the corresponding external lanes to be green
at the same commit and checksums.

Every new capability must therefore ship with three layers of portable
evidence: a small deterministic unit/branch test, a physical invariant or
manufactured/analytical test, and a public-API workflow test. Expensive
parameter sweeps are represented in the full gate by minimal boundary-value
sets and synthetic external formats; their production-size campaigns run in
the named external lane. JAX compilation is amortized through shared shapes and
fixtures, subprocess tests are reserved for true process-boundary behavior,
and duplicate numerical campaigns are consolidated behind one tested runner.
Quarterly, and before each release, record collection time, compile time,
slowest tests, coverage by module, and peak memory. A new test that pushes the
gate above 360 seconds must include an offsetting runtime improvement or a
documented reason and owner; the 600-second limit is never the planning target.

## 6. Architecture and slimming contract

The maintained core converges toward the responsibilities `case`, `mesh`,
`fields`, `operators`, `solve`, `diagnostics`, `io`, `benchmarks`, and a thin
`cli`; this is not a requirement for one file per name.

- one implementation per numerical concept;
- pure JAX array kernels separated from I/O, plotting, printing, and campaigns;
- explicit shapes, dtypes, gauge, boundary conditions, normalization, and
  convergence policy;
- solver outputs are pytrees of arrays plus compact diagnostics;
- no new module over 1,000 lines;
- <=30 deliberate stable root exports after one deprecation release;
- plotting under `lmx.viz`; campaign/manuscript tooling outside the core;
- examples call only public APIs and contain no hidden tuning.

The measured pre-slimming surface was 45 top-level Python modules, about 32,000
package lines, 285 root exports, and 96 top-level Python/TOML examples. The
current working-tree audit has 45 modules, 9,007 maintained core/facade lines,
35,967 total classified lines, 30 stable root exports, and 11 curated examples.
The tracked inventory is regenerated after each accepted milestone and must
remain within the numeric limits below. The M3 adapter and Benchmark B operator
work are included in that inventory rather than exempted from it.
These counts make slimming a product requirement, not cosmetic cleanup. Every
item is classified as stable core, research-stage extension, compatibility
facade, campaign, visualization, or generated/archival material. Advanced APIs
belong in explicit submodules and campaigns do not define the stable package
surface.

The architecture-release target remains <=15,000 maintained core lines and
<=10 MB checkout excluding Git history and release assets. The count is
measured by a checked-in script with explicit inclusions; moving code between
files or hiding it from the counter is not progress. If validated capabilities
make 15,000 lines unattainable, M2 may exit only with a reviewed exception that
states the irreducible subsystem, its tests, and a revised numeric ceiling.
These targets are achieved behind characterization tests without reducing
accepted physics, coverage, or reproducibility.

## 7. SOLVAX and differentiability contract

Generic linear algebra belongs in SOLVAX. MHD discretization, material
interfaces, boundary assembly, and physics balances remain in LMX.

SOLVAX 0.5.1 is released at commit
`e348c0b4a1b9995c3e33ceb11c04f93e7aa48e63` and LMX pins that exact revision.
Its matrix-free pytree PCG provides real and complex Hermitian solves,
fixed-shape residual history, explicit convergence/breakdown status, and an
implicit `custom_linear_solve` path that preserves forward diagnostics while
using independent transpose controls. The standalone release passed 203 tests,
97.76% package branch coverage, 100% PCG branch coverage, focused x64 tests,
strict documentation, package build, and macOS/Linux CI. GPU promotion remains
a downstream LMX M3 gate; it was not claimed by the CPU release.

The gauge-compatible tensor decomposition, conducting-region detection,
material topology, and MHD convergence policy remain in LMX. Structured
line/ADI or multigrid primitives move to SOLVAX only when they have a genuinely
generic contract, tests, documentation, and a versioned release; LMX never
depends on an unreleased branch tip.

LMX integrates SOLVAX as `linear_solver = "solvax_pcg"`, and `auto` selects it.
The tracked x64 CPU comparison on a 64 x 64 problem passes with field relative
difference `1.54e-12`, gradient relative error `1.13e-15`, warm-time ratio
`0.748`, and compiler temporary-memory ratio `0.625`. The RTX A4000 record
passes with `1.54e-12`, `1.13e-16`, `0.230`, and `1.000`, respectively, plus an
independent transpose residual of `2.54e-13`. The four-level Ha=20 FreeMHD
ladder and all eight high-Ha Table I rows pass at solver-core fingerprint
`b6cd09aaabffbe40f7b361b28760e42c08d1129ac64413b0c587136a7518c383`;
Shercliff Ha=15,000 uses the 119 x 119 confirmation. The combined record is
`benchmarks/results/solvax-pcg-acceptance.json` and has
`m3_promotion_pass=true` with no blockers. Duplicate native machinery is
removed only after one deprecation cycle.

Each accepted gradient requires converged implicit differentiation, reported
primal/transpose residuals, a centered finite-difference step ladder (and
complex-step where valid), a truncation plateau, and an independently verified
primal observable. The supported envelope covers drive, field amplitude/shape,
conductivity, viscosity, wall conductance, and smooth geometry. Topology
changes, clipping/limiters, iteration-count branches, and unsupported callbacks
are explicitly nonsmooth.

## 8. Performance and scaling contract

Optimize only verified Benchmark A/B paths. Reports record global problem size,
tolerances, precision, device/interconnect, JAX/jaxlib/XLA, compilation, median
warm time over >=5 repeats, memory/device, iterations, communication, transfers,
and physics/balance equivalence.

Implementation order is profiling; eliminating host synchronization and
redundant materialization; `vmap` for independent ensembles; named spatial
sharding; then justified halo collectives and multi-host restart.

Acceptance requires one CPU and one GPU baseline, numerical equivalence across
device counts, and a target of >=70% strong-scaling efficiency at four devices
for a documented 3D case. If the target is missed, publish the measured limit
and bottleneck without weakening the criterion.

## 9. Documentation, examples, and releases

Maintain four reader paths: first run in <5 minutes; trust/equations/units;
case construction and restart; development/testing/benchmarks/performance.

Curated examples cover Hartmann CLI, Hunt Python, custom field, restart,
manufactured/convergence verification, FreeMHD and experimental validation,
one independently checked differentiable design, and one end-to-end CPU/GPU
benchmark. Every example has a tested command, runtime tier, expected output,
and documentation link.

Before a public research release add `CITATION.cff`, changelog, contribution
and authorship policy, code of conduct, security/support policy, archival DOI,
and a checksummed acceptance summary.

## 10. Milestones and exit gates

### M0 — reproducible quality gate: complete

The gate, lockfile, CI lanes, strict documentation build, coverage policy, and
provenance machinery are established. The latest accepted audited baseline is
873 passed, 8 optional-data skips, 95.17% branch coverage, and 131.1 seconds on
Python 3.10. Any later numerical change must establish a new full-gate record;
it does not inherit this result.

**Exit:** from the final M1 source fingerprint, supported Python endpoint jobs
collect the full portable battery, pass in <=600 seconds (target <=360), achieve
>=95% branch coverage, and pass strict docs plus deterministic provenance.

### M1 — Benchmark A and matched FreeMHD: complete

Completed work includes canonical specifications and materialization; the
matched four-level Ha=20 FreeMHD ladder; analytical-reference-floor analysis;
conservative current and power diagnostics; checkpointed Table I campaigns;
exact insulating topology; robust high-Ha meshes; SOLVAX acceleration and line
solves; gauge-compatible tensor preconditioning; flexible GMRES; all eight
Table I rows; the refreshed Ha=20 ladder; and the combined, passing Benchmark A
acceptance record. The clean M0 quality gate, provenance, README, and benchmark
documentation are synchronized to that evidence.

**Exit:** all eight Table I rows and the Ha=20 FreeMHD studies have passing,
checksummed compact records from one final fingerprint; the combined Benchmark
A acceptance record separates analytical continuum, finite-grid cross-code,
conservation/power, and literature evidence; a clean locked CPU environment
reproduces it; M0 is green.

### M2 — slim verified core and user surface: complete

Characterize public behavior before moving code. Separate stable duct physics
from research-stage fringing/Q2D/blanket extensions; extract pure kernels and
diagnostics; merge duplicate campaigns and plotting paths; move advanced APIs
to submodules; provide one-release deprecations; curate examples and docs around
actual user journeys.

Final M2 baseline: 8,087 classified maintained-core/facade lines, 30 stable
root exports, 11 curated workflows with zero uncurated top-level workflows, a
16.1 ms median lightweight import, and a 6.46 MiB source checkout. The 65 large generated files are in the
checksummed `lmx-research-assets-v1` GitHub release; a freshly downloaded 24 MiB
archive passed membership, size, and SHA-256 verification before the local
copies were removed.
All internal Python workflows now import advanced APIs from their owning
submodules; the 255-name root compatibility path is covered by an explicit
one-release warning test. All 85 formerly uncurated workflows have completed
machine-checked dispositions: `examples/` contains only the 11 curated journeys,
73 research and evidence scripts are grouped under `campaigns/`, and reusable
duct/fringing TOML inputs are under `cases/`. No scientific workflow was deleted.
The first-run Hartmann command completes in 6.0 seconds, the compact Benchmark A
replay is byte-identical with SHA-256
`a8ab639141722cf2730dcc5aa1c4954610c82b5c18eec93256e32ac576dc0bb9`, and
the final locked full gate passes.

**Exit:** the checked-in inventory demonstrates <=30 stable root exports, <=20
curated examples, <=15,000 maintained core lines or a reviewed numeric
exception, and <=10 MB checkout; first run succeeds in <5 minutes; every M0/M1
gate and accepted behavior remains green.

### M3 — SOLVAX backend and validated differentiation: complete

SOLVAX 0.5.1 and its implicit PCG contract are released, runtime-pinned, and
promoted behind `auto`. CPU and RTX A4000 x64 solution, residual,
implicit-gradient, independent-transpose, compile/warm-time, and compiler-memory
records pass. The refreshed four-level Ha=20 FreeMHD ladder and all eight Table
I cases share the promoted solver fingerprint and pass. The final compact
replay, portable battery, strict documentation, and provenance are green.
Native `cg` remains explicit for comparison and a one-cycle compatibility
window; removal is a later reviewed change, not part of M3 closure.

**Exit:** forward solutions, iterations, balances, and the accepted gradient
contract pass on CPU and one GPU; Benchmark A stays green; median warm time and
peak memory regress by no more than 10% unless a reviewed tradeoff shows a
material accuracy or robustness gain; LMX pins a released SOLVAX revision.

### M4 — Benchmark B fringing-field validation: active

Freeze the published pipe and square-duct geometry, magnetic-field
reconstruction, stations, wall conductance, reference data, uncertainties, and
tolerances before production. Validate field admissibility, run three spatial
levels and tolerance/time independence, compare the pressure drop and secondary
profiles, and reproduce one matched FreeMHD case.

The pre-production freeze is complete. `alex-b1-pipe.toml` and
`alex-b2-square.toml`, their checksummed extracted ALEX anchors, and
`benchmark-b-specification.json` fix the distinct B1 axial and B2 transverse
pressure observables, pole-face coordinates, measured field, uncertainties,
wall model, three mesh levels, balance thresholds, and data rights. No
production result was used to choose a tolerance. M4 remains active until both
production cases and the matched FreeMHD case pass.

The first implementation slice is also complete: the checksummed anchors now
produce a bounded, cell-centred, monotone measured-field profile without
extrapolation; the pipe O-grid can include an explicit conducting annulus while
preserving the requested fluid resolution; and the pipe and duct lanes can
hold a prescribed mean flow. The fixed-flow pressure Lagrange multiplier and
direct B1 axial-pressure-loss/B2 transverse-pressure fields are implemented,
persisted, and covered by manufactured constraint/sign tests. B2 is the
side-minus-top wall pressure difference at the same axial station, with the
upstream plateau removed; it is not an axial pressure-drop surrogate. Frozen
nondimensional builders bind `Re=Ha^2/N`, the measured field, explicit
nominal/confirmation shells preserving `c_w`, and all three mesh minima; the
pipe grid explicitly resolves the `1/Ha` layer. The legacy current-scaled
pressure proxy is not admissible as ALEX evidence.

The nonuniform verification slice is accepted by the complete portable gate:

- rectangular gradients and Laplacians use physical centre/face spacing and
  pass linear exactness and smooth manufactured convergence;
- variable-conductivity potential, conservative current fluxes, charge
  divergence, and geometric face areas use nonuniform finite-volume metrics;
- masked diffusion places no-slip at the fluid/solid face;
- a compatible duct face-flux pressure projection closes its own discrete
  divergence; and
- mapped-pipe radial gradient, cylindrical Laplacian, and divergence operators
  use nonuniform radial metrics and pass manufactured checks.

Characterization proved that combining a finite-volume pressure Poisson matrix
with the old collocated cell-gradient correction does not close discrete
divergence, even when the Poisson residual is tiny. Both production branches
therefore use compatible face-flux divergence/gradient/projection pairs. B2
uses nonuniform rectangular metrics; B1 uses the stretched cylindrical O-grid
and explicit conducting annulus. Both include masked no-slip diffusion,
fixed-flow correction, conservative current, and volume-symmetrized SOLVAX
implicit PCG solves. Manufactured agreement, flow/charge closure, restart,
JIT, and finite-difference-checked coefficient-gradient tests passed the latest
accepted complete portable gate, so both ALEX implementation guards are
removed. Non-ALEX low-Ha behavior keeps its characterized legacy path until
equivalence tests justify migration.

Numerical independence is now the only M4 critical-path activity before the
three-level ladders. Exact B2 diagnostics first isolated loose electric and
pressure solves, then showed that a velocity-only stopping test could pass
while the published transverse-pressure observable still moved by more than
its uncertainty allowance. The active slice therefore uses implicit momentum,
independent electric/projection controls, warm-started gauge-fixed pressure,
SOLVAX acceleration, and persisted velocity, divergence, flow, charge, and
primary-pressure residual histories. The steady threshold is bounded by five
percent of the smaller combined reference uncertainty, and half-tolerance plus
doubled-iteration observable comparisons remain mandatory. This slice is not
accepted until those comparisons, the full portable gate, Benchmark A replay,
and the exact baseline all pass at one source fingerprint. Work proceeds in
this order:

1. make the exact B1 and B2 coarse baselines satisfy the frozen steady, mass,
   current, boundary-current, and power gates without changing tolerances;
2. demonstrate solver-tolerance, iteration-limit, and nominal-versus-
   confirmation thin-wall independence with checkpointed records;
3. run and accept the frozen three-level B1/B2 ladders; and
4. reproduce one exactly matched case in the pinned FreeMHD Docker image.

**Exit:** both published pressure-drop cases pass their frozen
uncertainty-aware gates; current, mass, power, and `div(B)` close; refinement is
credible; compact evidence and the supported fringing envelope are documented.

### M5 — CPU/GPU performance and strong scaling (active unblock)

Profile only accepted M1/M4 paths. No single acceptance variant may run for an
hour while a faster equivalent execution path remains available. Remove
synchronization and materialization
costs, batch independent cases with `vmap`, then introduce named spatial
sharding and justified halo communication. Keep compilation, warm execution,
memory, transfers, and iterations separate in every report.

**Exit:** CPU and GPU baselines are reproducible; physics and gradients are
equivalent across device counts; a representative 3D case has published
1/2/4-device results and >=70% four-device strong-scaling efficiency, or the
unchanged target is reported as missed with the measured bottleneck and no
performance claim.

### M6 — broader physics, one validated tier at a time

Release Benchmark C (Q2D turbulence), then D (3D turbulence/magnetic
obstacles), then E (energy/buoyancy). Each tier receives its own specification,
external evidence, balances, refinement, performance record, and documented
envelope. A later tier may not weaken an earlier claim.

## 11. Authoritative execution queue

Work on one numbered package at a time. A package is complete only when its
listed artifacts exist, focused tests pass, the complete portable gate passes
when portable LMX code changed, and documentation/provenance are synchronized.
The compact Benchmark A replay is mandatory after a solver, operator, material,
mesh, boundary, or normalization change. The performance unblock is the one
approved exception to numerical package order: expensive M4 campaigns pause
until production CPU/GPU sharding is physics-equivalent and their projected
wall time is acceptable.

1. **Complete — freeze M0/M1.** Preserve the eight Table I rows, four-level Ha=20
   FreeMHD ladder, analytical-reference-floor audit, conservative balances, and
   combined acceptance record. Never overwrite accepted records with a run from
   a different fingerprint; create a new record and compare it.

2. **Complete — finish the M2 disposition.** Use the tracked architecture inventory
   as the baseline. The 85 uncurated workflows were classified as: merge into one of the
   11 curated journeys, move to a non-importable `campaigns/` area, retain as a
   tested research-stage workflow, or delete as duplication. The eight-file
   autodiff slice, the remaining campaign/case/tutorial relocations, and the
   upload/retrieval/removal of all 65 large generated assets are complete. The
   machine-checked catalog reports 11 curated and zero uncurated examples; the
   uploaded asset manifest has durable URLs; the source checkout is 6.47 MiB.

3. **Complete — finish M2 core consolidation.** Consolidate duplicate campaign, validation,
   reporting, and plotting paths before changing numerical kernels. Then extract
   pure kernels only where characterization tests prove equivalence. Preserve the
   30-name stable root API; keep legacy root names as one-release warning
   shims. Deliverables: architecture audit <=15,000 maintained core lines, no
   new module >1,000 lines, import-time audit, migration guide, and unchanged
   M0/M1 evidence.

4. **Complete — finish the M2 user surface.** Make README and docs follow four tested paths:
   install/first result in <5 minutes; equations/units/trust; case construction,
   wall/field models and restart; developer tests/benchmarks/performance. Every
   curated example must declare stability, runtime tier, expected outputs, and
   its validation status. Add citation, contribution/authorship, conduct,
   security/support, changelog, and release-asset policy. Exit M2 only after the
   full gate and compact Benchmark A replay are green.

5. **Complete — release SOLVAX PCG without changing LMX defaults.** SOLVAX 0.5.1
   at commit `e348c0b4a1b9995c3e33ceb11c04f93e7aa48e63` provides the documented
   PCG and implicit-solve contracts, diagnostics, breakdown and transpose
   coverage, correctness/performance fixtures, 203 passing tests, 97.76% package
   branch coverage, and 100% PCG branch coverage. Strict docs, package build,
   macOS/Linux CI, PyPI, and the GitHub release are green. The release makes no
   GPU claim; GPU equivalence is the downstream M3 promotion gate.

6. **Complete — promote released SOLVAX PCG.** Version 0.5.1 is a pinned runtime
   dependency and `auto` selects it. Matching CPU/GPU records retain device,
   JAX/jaxlib, compile/warm time, compiler memory, iterations, primal and
   transpose residuals, and gradient evidence. The refreshed Ha=20 FreeMHD and
   eight-row Table I campaigns pass at one promoted solver fingerprint; the
   compact replay, portable battery, strict docs, and provenance are green.
   Native `cg` remains explicit through one compatibility cycle.

7. **Complete — freeze Benchmark B before production runs.** The versioned B1
   Ha=6600, N=10700, c=0.027 and B2 Ha=2900, N=540, c=0.07 specifications,
   checksummed extracted anchors, DOI/page/figure/station provenance,
   digitization/scatter uncertainty, rights, exactly divergence-free field
   reconstruction, wall model, three mesh levels, tolerances, and distinct
   pressure observables are frozen. Portable parsers and the deterministic
   specification index pass; the index contains no production results.

8. **Complete — accept the nonuniform operator slice.** Metric validation,
   degenerate-axis, half-cell Dirichlet wall, nonuniform Poisson dispatch and
   warm-start gauge, and invalid fluid-mask topology tests cover the new
   contracts. Formatting, lint, strict docs, deterministic provenance, and the
   complete locked portable battery pass. The current tree passes 899 tests
   with 8 optional-data skips, 95.09% branch coverage, and a 180.6-second local
   runtime, safely inside the 600-second CI ceiling. The compact Benchmark A acceptance
   file remains byte-identical at SHA-256
   `a8ab639141722cf2730dcc5aa1c4954610c82b5c18eec93256e32ac576dc0bb9`.

9. **Complete — integrate the B2 square-duct production path.** The ALEX B2
   branch uses actual nonuniform metrics, masked no-slip diffusion, conservative
   current, and one compatible face-flux projection. A stationwise pressure
   multiplier followed by an x-invariant divergence-free correction enforces
   prescribed flow without reopening continuity. Volume-symmetrized SOLVAX
   implicit PCG solves both pressure and electric potential. Manufactured
   agreement, fixed flow, charge/boundary balance, restart, JIT, RHS and
   coefficient gradients, strict docs, the unchanged Benchmark A hash, and the
   complete 870-pass portable gate succeed.

10. **Complete — integrate the B1 mapped-pipe production path.** The cylindrical
    finite-volume diffusion and compatible face-flux projection use the
    stretched O-grid, and the electric solve includes the explicit conducting
    annulus without a mean-spacing approximation. Manufactured, wall,
    fixed-flow, restart, JIT, coefficient-gradient, and reduced solve tests pass.
    The accepted integration fingerprint passed 873 portable tests with 8
    optional-data skips, 95.17% branch coverage, and a 131.1-second runtime; the
    compact Benchmark A hash remained unchanged.

11. **Active after accepted coarse B2 evidence — establish numerical independence.**
    Retain per-iteration velocity, divergence, flow, charge, and linear-solver
    histories so every failed gate identifies its responsible operator. The
    canonical mixed-dimensional shell, strict per-variant steady gate, and
    bounded SOLVAX Aitken continuation close the exact coarse B2 campaign. At
    source fingerprint `7de3d682...`, all four checkpointed variants pass
    steady, mass, current, and boundary-current gates. The half-tolerance
    observable shift is `0.08381` of experimental uncertainty, the
    doubled-iteration shift is zero, and the thin-wall relative difference is
    `1.03e-12`. This is the replay after decomposition-safe diagnostics and
    scaling-validation hardening, so coarse
    B2 is accepted at the current numerical fingerprint. Next close coarse B1.
    Do not start medium/fine production runs
    until both coarse cases pass, and do not launch another hour-scale variant
    until package 13 supplies a faster equivalent production path. A failed
    gate changes the discretization or solver, never the frozen experimental
    tolerance.

12. **Pending — execute and accept Benchmark B.** Run the frozen three-level
    B1/B2 ladders, validate their distinct pressure observables first and
    published profiles second, close mass/current/interface/power and `div(B)`
    balances, and reproduce one exactly matched case in the pinned FreeMHD
    Docker image. Publish compact acceptance and diagnostic records separately.
    Fringing becomes stable only when both experiments pass their frozen,
    uncertainty-aware gates and refinement is credible.

13. **Active — optimize and scale accepted paths.** Establish reproducible single-CPU and
   single-GPU cold/warm baselines for M1 and M4. Profile before changing code;
   remove host synchronization/materialization, batch ensembles with `vmap`,
   then add named spatial sharding and justified halos. Publish equivalent
   1/2/4-device physics, gradients, iterations, memory, transfers, and timing.
   Claim strong scaling only at >=70% four-device efficiency for the documented
   3D problem; otherwise publish the miss and bottleneck. Named axial sharding
   is now wired into the production ALEX B2 extruded solve and must pass shard-placement,
   one/two-device physics-equivalence, and end-to-end timing gates before this
   package is accepted. The first `24 x 24 x 24`, two-step CUDA checkpoint
   passes one/two-GPU field placement and L2-signature equivalence (largest
   relative difference `1.4e-6`) but misses scaling: `29.10 s` on one A4000 and
   `40.72 s` on two. Process-stable meshes and cached kernels now make repeated
   two-GPU solves deterministic, but warm scaling still misses: `8.79 s`
   versus `21.43 s` on the small case and `10.48 s` versus `38.28 s` on a
   `48 x 36 x 36` case. A `102 x 77 x 77` footprint fits one A4000 at about
   `15.7 GiB` and takes `34.08 s` warm for two outer steps. Next reduce global
   PCG synchronization and run independent variants concurrently across the
   GPUs; do not claim strong scaling yet. The Benchmark B campaign runner now
   supports `--gpu-devices 0,1`, assigning one variant per GPU in two
   restart-aware waves without duplicating the campaign implementation. It
   refuses to launch the dependent wave when prerequisite physics gates fail.
   SOLVAX 0.7.0 single-reduction PCG is now enabled for sharded B2 momentum,
   projection, and electric solves and reduces the compiled per-step reduction
   stages. A later conservative-residual audit supersedes the earlier apparent
   signature parity: on a matched `48 x 36 x 36`, two-step case, one GPU passes
   with maximum charge residual `6.66e-5`, whereas two GPUs report
   `1.91e-3`--`2.53e-3` and change velocity/current L2 by about 1.8%/0.24%.
   Standard PCG reproduces the failure, so it is not caused by the
   single-reduction recurrence. The scaling harness now rejects any row that
   fails charge, boundary-current, electric-local-residual, or linear-solve
   convergence gates. Halo/domain decomposition and the remaining combined
   reductions still block domain strong scaling. Forced
   two-device CPU sharding is likewise rejected (`5.03 s` versus `3.19 s`).
   A symmetric point-gauge prototype is also rejected: although it passes the
   single-device manufactured reconstruction, the real two-GPU B2 probe becomes
   unstable with charge residual near `1e9`. Keep the rank-one gauge until a
   decomposition-aware nullspace treatment passes the production gate.
   A granular follow-up clears the distributed stencil and linear solver: axial
   neighbor exchange is bitwise identical, manufactured cold/warm solves agree
   to `7.4e-15`, and the B2 conductivity-jump solve agrees within `1.3e-11`
   cold and `4.1e-12` warm relative. The remaining boundary-current failure was
   a diagnostic bug that mixed global end fluxes into each station; the
   finite-volume divergence integral now gives a decomposition-safe slab
   balance. A near-converged `102 x 77 x 77` restart gives one/two-GPU
   potential/current agreement of about `4.2e-9`/`6.2e-7`, charge below
   `2.8e-5`, and boundary flux below `2e-14`; velocity differs by `3.2e-6` on
   the two-step probe. A six-step two-GPU continuation closes the steady gate
   at `3.28e-5`, with `2.50e-5` charge, `1.76e-14` boundary flux, and
   gauge-invariant potential update near `1e-6`. Records store an explicit
   `2.5e-5` signature limit tied to half the frozen nonlinear tolerance and are
   grouped by actual executed update count. The rejected forced-continuation
   branch proves that convergence may not silently change the nonlinear map.
   Scaling workers now accept checksummed steady restarts and keep them in a
   separate comparison group from cold starts. Commit `3d5de4e` closes the
   source-bound correctness gate from restart `75097639...`: one and two GPUs
   both stop after four updates, pass every steady/conservation/electric gate,
   and agree in the main L2 signatures within `2.2e-15` relative. The fix
   suspends Aitken only after the primary state is already within tolerance,
   preventing decomposition-order roundoff from destabilizing the converged
   pressure tail. Reusing the in-loop conservative diagnostics also closes
   repeated sharded solves in one process and removes a duplicate evaluation.
   The initial matched warm times were `37.78 s` on one A4000 and `109.23 s`
   on two (speedup `0.346`, efficiency `0.173`). Commit `9e0d1dc` batches 14
   outer-step diagnostics into one host transfer, reducing the two-GPU row to
   `50.18 s`. Commit `036d26b` then removes axial line relaxation from sharded
   projection/electric preconditioners while retaining the dominant transverse
   y/z blocks and single-reduction PCG. Final source-matched three-repeat warm
   rows are `36.96 s` and `22.23 s`; speedup is `1.662`, two-device efficiency
   is `0.831`, and the two warm sharded repeats agree within 0.12%. Cold rows
   are `66.71 s` and `65.75 s`; main signatures agree within `1.8e-16`, and all
   gates pass. Preserve the new compact result in
   `benchmarks/results/gpu-strong-scaling-20260713.json`. M5 remains active only
   because its exit contract requires a measured four-device row, which the
   current two-GPU host cannot supply.
   A direct production-path Mac check confirms that normal one-device JAX is
   already threaded: `30.7` CPU-seconds over `12.5` wall-seconds with
   `OMP_NUM_THREADS=1`, while requested thread counts 1/4/8 give essentially
   identical `3.44/3.45/3.58 s` warm times. Local acceleration therefore uses
   normal JAX threading plus process-level case concurrency, not fake CPU
   devices. An HLO audit confirms that automatic named sharding already lowers
   the axial stencil to two one-plane `collective-permute` exchanges with no
   all-gather, so a manual `shard_map` rewrite is deferred unless a GPU trace
   contradicts that result. Prioritize a geometric multigrid preconditioner
   because the current exact coarse B2 electric solve takes about 722--723 PCG
   iterations—and global reductions—per outer update, then fuse larger solver
   regions and reprofile the remaining collectives. Accept either change
   only after production field, balance, gradient, iteration, and restart
   equivalence; then require an actual one-to-two-GPU speedup before attempting
   the frozen four-device >=70% efficiency gate.
   A conditional skip of electric iterative refinement is rejected: it reduced
   the reported solve from about 720 to 600 iterations without improving wall
   time and degraded maximum charge residual from `2.63e-5` to `7.76e-4`.
   Preserve the stronger refinement result and improve its preconditioner.
   A B2-wide cross-section line-block ablation succeeds where the earlier
   pressure-only experiment did not: exact baseline electric work falls from
   720 to 571--572 iterations and runtime from `63.9` to `52.1 s`, with charge
   residual near `2.7e-5`, major-field L2 changes around `1e-7`, transverse
   absolute changes below `5.8e-7`, and primary-observable change `8.38e-9`.
   Use cross-section line blocks for single-device B2 momentum, projection, and
   electric PCG; retain existing B1/generic preconditioners. The same choice on
   sharded projection/electric solves reduces collective-permute `17 -> 7`,
   all-reduce `25 -> 15`, and all-gather `12 -> 0`, but fails two-GPU
   velocity/current signature parity by about 1.8%/0.24% and takes `42.99 s`.
   Reject that branch. Restoring axial line blocks for sharded
   projection/electric does not restore conservative parity, so the current
   two-GPU path remains research diagnostic only while explicit-halo and
   decomposition work proceeds.
   The matched small Mac production solve also improves from `3.44` to
   `3.15 s` warm (8.5%) with velocity/current L2 changes near `1.1e-8` and
   `3.5e-7` relative.
   Clean reruns close the remaining line-block variants. Multiplicative y--z--y
   SOLVAX reduces electric PCG to 380--381 iterations with `2.48e-5` charge
   residual but takes `64.5 s`, 24% slower than the accepted additive y/z
   block; one direction exceeds 95 seconds without completing. The clean
   matched one-GPU `48 x 36 x 36` row is `9.71 s` warm versus the prior
   `10.23 s`. The matched two-GPU timings (`42.99`--`45.15 s`) are rejected
   because their conservative physics gates fail; they are bottleneck evidence,
   not scaling rows.
   A 123 MiB external warm trace remains outside git and reports 102 compile
   events (`7.10 s`), 932 cache misses (`1.30 s`), and six PCG while calls
   (`2.29 s`), with negligible host/device copy time. Stabilize compiled closure
   identities and fuse larger solver regions in parallel with multigrid.
   A fresh accepted-path two-GPU trace is likewise external (218 MiB). Its
   `166.5 s` solve duration is quarantined because an unrelated SPECTRAX process
   saturated both devices, but its phase structure remains actionable:
   projection occupies `103.6 s`, 580 `pjit` cache misses take `6.43 s`, 62
   backend compile/load events take `3.80 s`, and two pressure-response calls
   take `4.11 s`; collective launch spans are much smaller. Batch the output
   reductions and investigate reuse of the fixed pressure response before
   tuning collectives further.
   Commit `8a069fe` completes the first output-path batch: 102 station rows now
   use one stack and one host transfer. Isolated restart timing improves from
   `1.283 s` cold and `0.636`--`0.692 s` warm to `0.476 s` cold and
   `0.003`--`0.006 s` warm with identical diagnostics. Contended whole-worker
   timings are not accepted scaling evidence.
   An axial-invariant pressure-response prototype is rejected and removed.
   Isolated warm time falls from `0.0236 s` for 102 copies to `0.0011 s` for
   one copy, but the new shape adds roughly one second of compilation and the
   clean one-GPU restart regresses from `37.78 s` to `39.72 s` warm while its
   maximum charge residual rises 6.2%. Main L2 signatures remain exact. The
   profile's `4.11 s` pressure-response span is therefore tracing/compilation,
   not a persistent kernel bottleneck; prioritize stable larger compiled
   regions rather than a separate cross-section kernel.
   A vectorized three-component momentum PCG prototype is also rejected and
   removed. It deletes 36 source lines and batches reductions, but the batched
   while-loop waits for the slowest component. Its exclusive-A4000 one-GPU
   worker exceeded 135 seconds before completing two repeats, versus about 100
   seconds cold-plus-warm for the accepted row. Keep independent component
   stopping and pursue fusion only outside the iterative loops.
   The accepted outside-loop fusion stacks 14 scalar diagnostics into one host
   transfer per outer update. One-GPU warm time improves 2.9%, while two-GPU
   warm time improves 54% with exact physics, closing the dominant host
   synchronization penalty. The full portable gate passes 899 tests with 8
   expected skips and 95.06% branch coverage in `504.16 s`: under the 600-second
   limit, though above the 450-second warning target for this Mac run.
   The post-change accepted-source two-GPU trace remains external (207 MiB).
   Profiling inflates wall time to `91.41 s`; structurally, projection occupies
   `71.24 s` of the `73.49 s` public solve span, with 580 `pjit` misses
   (`6.20 s`) and 62 compile/load events (`4.11 s`). Labelled collective spans
   total only `0.091 s` for all-reduce, `0.110 s` for collective-permute, and
   `0.002 s` for all-gather, but do not capture full asynchronous device-loop
   cost. Retain SOLVAX single-reduction PCG and require either a stronger
   accepted preconditioner or a device-resolved timeline before changing its
   recurrence.
   A stable top-level JIT for the complete outer-step diagnostic vector is
   exact on Mac CPU and one/two A4000s but is rejected and removed. It adds 115
   net source/test lines while warm time is neutral: `36.82 s` versus `36.68 s`
   on one GPU and `50.26 s` versus `50.18 s` on two. The full-size Mac cold
   restart passes in `48.03 s`. Do not pursue further standalone diagnostic
   fusion; return the active experiment to solver-side work.
   The accepted solver-side experiment keeps line relaxation transverse to the
   sharded axis. The previous cross-section-only ablation predated corrected
   decomposition-safe current diagnostics; at the accepted fingerprint it now
   passes exact physics and improves two-GPU warm time from `50.18 s` to
   `22.23 s`. It adds three net package lines. The full portable gate passes
   899 tests with 8 expected skips and 95.06% branch coverage in `512.04 s`,
   below the 600-second hard limit. The one-to-two-GPU efficiency target is
   closed at 83.1%; acquire a four-GPU row before closing M5.
   A stride-four SOLVAX Galerkin prototype with exact-adjoint restriction
   reduced a manufactured solve from 39 to 27 PCG iterations, but its
   diagonal coarse action is rejected on production B2: one sweep stalls at
   1200 iterations with charge residual near 52, while four sweeps cause an
   immediate PCG preconditioner breakdown. The next multigrid implementation
   must build a provably stronger SPD coarse solve once per fixed conductivity,
   not rebuild an approximate hierarchy for every electric right-hand side.
   Exact tensor-cosine Galerkin spaces confirm the same cost constraint: 27
   modes increase B2 work from 573 to 586--588 iterations; 64 modes reduce it
   to 514--515 with exact physics but worsen one-GPU warm time from `37.78 s`
   to `57.98 s`. Reject dense spectral setup. A viable hierarchy must have
   near-linear transfer/setup cost and persistent reuse.
   Unsmoothed piecewise-constant aggregation is also rejected. Factor-four
   aggregation changes a manufactured solve from 39 to 41 iterations;
   factor-two improves that probe to 35 but drives both B2 electric stages to
   the full 1200 combined iterations. Proceed only with smoothed aggregation
   or a rediscretized hierarchy with a genuine coarse line solve.
   A factor-two rediscretized Galerkin coarse line solve is SPD but cannot be
   added directly to the fine line inverse: the manufactured solve changes
   from 39 to 54 iterations and B2 again consumes all 1200 combined iterations,
   with charge residual `6.7e-5`. Reject additive overlap; a future coarse line
   solve must live inside a properly balanced symmetric V-cycle.
   The balanced form `L + (I-LA) C (I-AL)` is stable but also rejected for
   piecewise-constant aggregates: the manufactured row changes 39 -> 41 and
   B2 changes 573 -> 852 iterations. This closes the unsmoothed aggregation
   family. Shift the active M5 experiment to reduction/fusion profiling while
   retaining smoothed interpolation as the only future multigrid candidate.
   Hoisting the fixed conductivity and mask into the compiled electric closure
   preserves exact one/two-GPU physics and gives `37.37 s` one-GPU warm time,
   but worsens two-GPU warm time from `109.23 s` to `118.66 s`; reject the
   additional captured-constant cache key and retain dynamic solver arguments.
   Long B1/B2 runs now bound retained Anderson states to the configured history
   depth, removing growth proportional to the total outer-iteration count.
   The obsolete public 2-D stencil microbenchmark is removed now that production
   and 3-D operator paths cover the required evidence; this deletes 237 net
   lines across implementation and tests without removing solver coverage.
   Two geometry-aware B1 sharding probes are rejected and removed. Axial
   sharding preserves reduced-case fields but leaves the second A4000 idle and
   exceeds two minutes, versus `26.94 s` cold and `7.37 s` warm on one GPU.
   Azimuthal sharding passes a forced two-CPU-device probe with `9.2e-15`
   relative velocity agreement and a small `6.48 -> 6.04 s` speedup, but also
   leaves the second A4000 idle and exceeds 70 seconds, versus `26.62 s` cold
   and `6.85 s` warm on one. The existing tridiagonal line solver cannot be
   reused as a theta block: its manufactured local residual is `4.58e-8`
   against the `1e-8` gate because theta is periodic. Keep B1 process-parallel
   across independent variants. A future spatial path requires a tested cyclic
   SPD/batched line solve that preserves partitioning; do not pad or alter the
   frozen 101-station mesh to manufacture divisibility.
   SOLVAX branch checkpoint `47831dd` implements the exact batched cyclic
   tridiagonal solve and passes 241 tests at 98.00% branch coverage. Draft
   [SOLVAX PR #12](https://github.com/uwplasma/SOLVAX/pull/12) publishes it with
   strict docs, minimum-stack, Ubuntu/macOS, and Codecov checks green. It also
   passes LMX's manufactured pipe gates, but direct use as a theta block is
   rejected and removed: the reduced one-A4000 worker exceeds 100 seconds,
   versus `26.62 s` cold and `6.85 s` warm for the accepted x/r block. The
   next periodic-line experiment must reuse persistent factors or otherwise
   prove lower apply cost before another production B1 run.
   Reusing SOLVAX's existing periodic-banded factors passes the manufactured
   gates but is also rejected: the reduced worker is CPU-bound with the A4000
   idle after 66 seconds. A viable periodic inverse must retain a fused
   accelerator apply; general scanned banded LU is not that path.
   The accepted next B1 step reuses one compiled momentum-PCG system while
   retaining eager cylindrical coefficient/wall preparation and the original
   pressure-response path. A paired ten-update `32 x 16 x 32` A4000 run changes
   `62.68 s -> 50.56 s` (`1.240x`); all main signatures agree within `6.9e-11`
   relative, divergence improves `3.37e-3 -> 3.16e-3`, and charge residual
   improves `2.05e-4 -> 2.02e-4`. The whole-operator JIT variant was faster warm
   but rejected because it amplified short-run divergence. Evidence is in
   `benchmarks/results/b1-momentum-jit-20260713.json`. Resume coarse B1 through
   process-parallel independent variants.
   The subsequent full coarse baseline/thin-wall wave is rejected on runtime:
   both one-A4000 variants exceeded 3600 seconds at full GPU utilization without
   producing a result or restart checkpoint. Do not launch the dependent wave
   until one coarse variant is bounded below one hour. The production outer
   loop now emits source-fingerprinted progress after every iteration and an
   atomic, restart-compatible partial state every eight iterations, on final
   iteration, and on convergence. `--resume` automatically selects that state
   only when its fingerprint and checksum match; explicit variant restarts
   retain priority.
   Both reduced production geometries exercise this path. The source-matched
   six-worker gate passes 902 tests with 8 expected skips and 95.10% branch
   coverage in 114.9 seconds. Checkpoint evidence is in
   `benchmarks/results/b1-checkpoint-resume-20260713.json`; runtime evidence is in
   `benchmarks/results/b1-coarse-runtime-cap-20260713.json`; independent-variant
   concurrency is throughput, not strong scaling of one solve. Rerun one coarse
   B1 variant with checkpoints before scheduling the two-variant wave.
   That bounded rerun now completes an uncontended first outer iteration in
   `93.01 s`, writes a checksummed 20 MiB state, and resumes it into a new
   checksummed state in `91.39 s`. At the unchanged 128-step allowance this is
   about `11,698 s`, so runtime—not observability—is the remaining gate. The
   resumed electric solve consumes 4,570 PCG iterations. SOLVAX single-reduction
   PCG is rejected (`62.46 s` versus `55.46 s`, worse divergence). An exact
   batched FFT theta-line preconditioner initially appeared 6--7% slower on the
   reduced proxy despite reducing its first electric solve from 609 to 341
   iterations. The source-matched full `101 x 64 x 128` result reverses that
   screen: five precursor updates plus a checksum-resumed 128-update segment
   complete in `2860.11 s`, below the `3600 s` gate and `4.30x` faster than the
   prior rate projection. Accept the axisymmetric B1 FFT line; keep it disabled
   for generic pipe coefficients. The terminal charge diagnostic passes at
   `1.44e-4`, but steady residual `1.97e-3` and divergence `1.27e-3` miss the
   frozen physics gates. Evidence is in
   `benchmarks/results/b1-pipe-pcg-screening-20260713.json`. Use the full coarse
   checkpoint to screen Anderson damping/history next, without loosening any
   physics tolerance; accept baseline/thin-wall physics before dependent
   variants. The independent thin-wall run also finishes below the cap in
   `2645.59 s`; its `1.23%` observable delta is provisionally inside the `2%`
   wall-realization limit, but neither unconverged field is admissible for final
   independence. Full-mesh 16-update restart screens reject Anderson damping
   `0.5` (minimum residual `1.18e-3`) and history depth `8` (`4.03e-4`). Stop
   scalar accelerator sweeps. Physical velocity normalization also reproduces
   the history-depth trace because potential updates are already negligible;
   reject it after nine full-mesh steps. Restart now restores the checkpointed
   axial pressure-loss gradient, eliminating an artificial first-step residual
   without changing fields. Diagnose the state-residual floor and its
   pressure/continuity coupling before the next full wave. Bounded Aitken is
   stable and monotone, with conservation inside the frozen gates, but ends 16
   full-mesh updates at `3.98e-4`; raising its inactive relaxation floor from
   `0.05` to `0.1` reproduces the history exactly. Reject it for steady closure.
   The requested `dt=0.01` is actually capped at `9.35e-8` by the B1
   electromagnetic scale. Raising that cap by factors of 500 and 1000 leaves
   16-step residuals above `1.8e-2`; Anderson variants remain above `6.1e-3`
   and worsen continuity. Condition-filtered Anderson is also rejected: a
   `1e4` condition limit diverges to `1.09e-2`, while `1e8` ends at `3.87e-4`
   after 32 updates but has a `4.48e-3` continuity excursion. Accelerator,
   scalar pseudo-time, and history controls are closed. The next bounded B1
   experiment was matrix-free FGMRES on the affine low-Re fixed-point equation.
   Its reusable SOLVAX implementation and condition-filtered Anderson support
   pass 239 tests at 98.77% branch coverage and all hosted checks in draft PR
   [#15](https://github.com/uwplasma/SOLVAX/pull/15). The method crosses the
   steady gate on a tiny B1 mesh (`2.61e-5`), but the exact prior
   `32 x 16 x 32` production screen stagnates after 64 Krylov iterations at
   `3.17e-4`; a local electromagnetic-plus-diffusive rate preconditioner
   regresses the tiny probe to `1.03e-2`. Reject composite-map FGMRES for B1.
   The next structural experiment must remove the explicit electromagnetic
   timestep from the operator: solve the steady coupled Stokes-electric block
   with a physics-preserving Schur preconditioner before another production
   campaign. The first compact foundation is merged at `cd5c76e`: the existing
   cylindrical diffusion solver now supports both shifted implicit updates and
   the exact steady SPD operator through one shared implementation. Its
   manufactured no-slip reconstruction passes without duplicating the test,
   the accepted reduced B1 path is unchanged, and the source-matched portable
   gate passes 903 tests with 8 expected skips and 95.08% branch coverage in
   `125.2 s`. Build the divergence/flow-constrained Stokes Schur response on
   this operator next; do not reintroduce a parallel solver stack. A paired
   `32 x 16 x 32` A4000 screen also rejects retaining projected face fluxes as
   the iteration state: it improves terminal divergence from `9.62e-4` to
   `3.97e-4`, but leaves the controlling residual unchanged at `2.03e-3`, is
   6.1% slower, and adds 122 net source/test lines. Preserve the projection
   idempotence diagnosis as evidence, not production complexity.
   A steady electromagnetic reaction split is retained as the next block
   preconditioner: adding `sigma |B|^2 / rho` to the shared steady diffusion
   operator and adding the same linear term to its right-hand side preserves
   the fixed point and changes the tiny B1 map from unstable to contractive.
   The first stationwise Schur prototype then closes the exact reduced flow
   span to `3.07e-5` in a warm `68.62 s`, but is rejected because it applies the
   transient `(1/rho)` pressure projection to a steady momentum response and
   raises divergence to `7.68e-2`. Truncating that Schur space to three or four
   vectors retains `5.88--6.35e1 s` warm runtimes and closes flow below the
   frozen gate, but divergence remains about `7e-2`. A direct coupled pressure
   prototype confirms that mixing the cell-centred gradient with the
   face-flux divergence is not a compatible discrete `G/D` pair; preserve it
   only on `agent/b1-steady-response` at `5d2d296`. Before another exact GPU
   campaign, extract the existing cylindrical face interpolation and
   divergence into one reusable operator set, add its matching pressure
   gradient and cell reconstruction, and prove constant-nullspace,
   `D G` consistency, weighted adjointness, projection closure, and JIT/AD
   behavior on a manufactured mesh. Build `D A^-1 G` only from that verified
   pair. This is a smaller and more decisive step than adding another solver
   or tuning more Krylov dimensions; evidence is in the existing B1 screening
   record. The compatible face operator set is now merged at `c3f7954`; its
   manufactured test proves `D G` agreement with the shared variable-
   coefficient diffusion operator, the constant nullspace, volume-weighted
   symmetry, and JIT differentiation, while the full fringing module passes.
   On `agent/b1-compatible-stokes`, `dfaeb09` closes manufactured steady
   Stokes divergence and flow below `1e-7` with both identity and actual
   viscous `A^-1`. The B1 saddle diagnosis at `fef95df` then separates the
   remaining blocks: an exact `7 x 7` axial flow-response matrix reduces its
   preconditioned flow residual to `3.83e-8`, but transient, unit-response,
   and point-diagonal pressure mobilities leave the pressure-block residual at
   `1.79` or become unstable. Reference JAX GMRES reproduces the SOLVAX result,
   so do not tune the Krylov implementation. Retain the compatible operators
   and exact axial block; next prototype a bounded approximation to the
   pressure `D A^-1 G` action itself, beginning with axial-mode deflation and
   independent cross-sectional block solves. Require monotone pressure-block
   residual reduction on the tiny case before reconnecting electromagnetics,
   and do not schedule another exact GPU campaign before that gate passes.
   The follow-up rank diagnosis at experimental commit `1750ff0` closes that
   candidate in its current form. Axisymmetric deflation reduces the first
   pressure-block residual from `1.7873` to `1.75e-6`, but the next
   Lorentz-forced map excites every azimuthal Fourier mode and diverges. More
   decisively, explicit assembly of the `1008 x 1008` tiny Schur operator finds
   numerical rank `1006` and condition number `1.46e19`: the collocated
   face-to-cell-to-face momentum response admits checkerboard pressure modes.
   A first momentum-weighted Rhie--Chow screen restores full rank and improves
   conditioning by about ten orders of magnitude, but `1.66e9` remains too
   ill-conditioned and the two-step divergence/flow gates still fail. This
   agrees with the established collocated-grid momentum-interpolation
   literature and supersedes the cross-sectional-block-only next step.
   Next derive the face mass flux from the shared steady momentum diagonal,
   express stationwise mean-free pressure in a volume-weighted orthonormal
   basis, and verify constant nullspace, full reduced rank, refinement-stable
   conditioning, adjoint consistency, and zero influence from relaxation or
   pseudo-time parameters. Dense assembly is a tiny diagnostic only; the
   accepted operator must remain matrix-free and must first pass identity,
   viscous, reaction-dominated, checkerboard, and B1 two-map gates. Only then
   add axial/coarse and Fourier-radial block preconditioning, followed by exact
   GPU correctness and sharding/scaling measurements.
   The bounded follow-up at experimental commit `6067ff1` passes the complete
   tiny B1 solve/restart test without relaxing a gate. A volume-weighted
   Householder pressure basis, momentum-consistent Rhie--Chow correction,
   exact stationwise flow refinement, and only the symmetry-required `m=0`
   and `m=2` azimuthal coarse spaces reduce the terminal two-map mean-free
   divergence to `1.93e-6` and flow error to `1.73e-13`; the restart terminates
   at `7.64e-7` and `5.42e-14`. A fixed-point-preserving reaction split factor
   of two closes the frozen restart-pressure gate, and the run takes about
   `19.15 s` on the reference Mac. The full fringing module and both base and
   weighted-modal manufactured Stokes variants pass. This is a passed physics
   prototype, not yet the production M5 exit: some Euclidean Krylov status
   flags remain false despite the physical component gates, and the exact
   coarse action is densely assembled on the tiny mesh. Next define stopping
   from the independent mean-free-divergence and normalized-flow residuals,
   replace dense coarse assembly with separable axial/Fourier-radial blocks,
   delete diagnostic-only branches, and prove refinement-independent
   iterations and memory before enabling the path by default or resuming exact
   GPU/sharding campaigns.
   Experimental commit `b34bc64` then replaces the dense modal action with
   independent per-axial-station `m=0,2` radial blocks followed by the existing
   exact flow response; outer flexible GMRES retains axial coupling. The full
   tiny solve/restart and fringing module still pass. Modal storage drops from
   `28,224` to `3,703` coefficients on the tiny mesh and, for the planned
   `32 x 16 x 32` screen, from about `2.36e6` to `70,688`; it is linear in
   axial stations rather than quadratic. Tiny runtime is about `22.0 s`, a
   modest setup tradeoff that must be checked at larger meshes. This closes the
   dense-coarse replacement subgate. Before promotion, remove the now-redundant
   dense fallback and environment-only plumbing, factor the modal transforms
   into concise tested helpers, align Krylov status with the physical residual
   decomposition, and demonstrate iteration/memory behavior across at least
   three tiny-to-medium refinements.
   That refinement gate now passes at experimental commit `cf33000`. The
   `7 x 9 x 16`, `9 x 13 x 24`, and `11 x 17 x 32` ladder reports total
   projection iterations `135`, `155`, and `169` while fluid unknowns grow
   from `896` to `5,632`. Warm runtime grows from `5.77 s` to `23.85 s`
   (`4.13x` for `6.29x` more unknowns), and three-band modal storage grows
   from `11,158` to `73,018` coefficients. Every projection reports physical
   convergence; the worst mean-free divergence is `5.00e-5`, below the frozen
   `1e-3` balance gate. Nearest-neighbor axial blocks use the pinned SOLVAX
   block-Thomas factorization and are reused by flexible GMRES. Compact
   evidence is tracked in
   `benchmarks/results/b1-compatible-projection-refinement-20260713.json`.
   This closes code consolidation, physical/Krylov status agreement, dense
   fallback removal, and the bounded refinement-cost gate. Experimental commit
   `be74706` then passes the exact two-step solve/restart gate on one RTX A4000
   with JAX 0.10.2, CUDA 13 wheels, and SOLVAX 0.7.0. The cold process takes
   `137.39 s`; the default 120-second test timeout interrupts compilation, so
   the hardware lane uses a bounded 600-second timeout while the portable CPU
   lane remains unchanged. Reusing immutable modal block factors removes their
   restart rebuild and passes the complete fringing module.
   Two compatible theta-sharding probes are rejected and removed. Keeping the
   periodic FFT line drives both GPUs initially but returns zero mean flow in
   `329.28 s`; two virtual CPU devices expose the underlying invalid distributed
   FFT layout. Disabling that line passes the virtual CPU gate, but CUDA still
   returns zero mean flow in `323.79 s` and leaves GPU 1 idle during the
   dominant Krylov phase. Placement alone is therefore not a multi-GPU solver.
   Compact evidence is tracked in
   `benchmarks/results/b1-compatible-gpu-correctness-20260713.json`. Next make
   the compatible momentum/Schur Krylov executable retain explicit sharding,
   require exact one/two-GPU physics, and only then record strong-scaling time.
   The first accepted mode-space block is now implemented at experimental
   commit `93331fd`. For the frozen axisymmetric conductivity, an orthonormal
   DCT-II in x and real FFT in theta reduce the electric finite-volume operator
   to batched radial tridiagonal systems solved by pinned SOLVAX. The zero mode
   pins only its redundant outer radial equation and then restores the
   volume-weighted gauge. Manufactured reconstruction, reverse-mode coefficient
   differentiation, CUDA execution, the compatible solve/restart gate, and the
   full fringing module all pass. On a `32 x 16 x 32` A4000 probe, warm time is
   `0.0506 s` versus `0.2811 s` for 48-step PCG (`5.55x`); local residual
   improves from `3.70e-5` to `2.18e-10`, and maximum discrete error improves
   from `1.69e-9` to `2.44e-15`. Evidence is in
   `benchmarks/results/b1-separable-electric-gpu-20260713.json`. This closes the
   electric block only. Next carry the compatible momentum/pressure Schur
   action into theta mode space so the dominant Krylov batch can be explicitly
   partitioned across devices without distributing an FFT or dense gauge basis.
   The first Fourier momentum implementation is rejected and fully removed.
   On a `32 x 16 x 32` reaction-shifted manufactured operator, complex
   Fourier-mode PCG is exact but takes `0.419--0.437 s` warm on one A4000 versus
   `0.231--0.239 s` for the existing 3-D PCG. Explicitly sharding the mode batch
   takes `0.768--0.800 s` on two GPUs and, more importantly, repeated
   gather/reshard calls corrupt the reconstructed field: maximum error grows
   from `3.95e-10` cold to `1.12e-3` and `6.47e-1` while the internal residual
   is unchanged. This independently reproduces the unsafe CUDA reshard boundary
   and forbids inverse-transform/gather inside each Krylov action. The only LMX
   change retained at `86da2b2` strengthens the shared steady/implicit
   reconstruction test with an x-r-varying reaction.
   The screen also found a reusable SOLVAX gap. Commit `0f65a5e` on
   `agent/complex-tridiagonal` promotes real bands and complex right-hand sides
   to a common dtype; 29 focused and 216 full tests pass, and fused CUDA
   tridiagonal residual is `2.67e-15`. It is pushed for review but is not yet
   part of pinned SOLVAX 0.7.0. Evidence is in
   `benchmarks/results/b1-fourier-momentum-rejection-20260713.json`. The next
   multi-GPU design must keep Fourier state persistently sharded and express
   gradient, divergence, Rhie--Chow, and momentum response directly in mode
   space, or prove component-task parallelism. Do not gather within the Schur
   loop and do not retain a slower one-GPU alternative merely to expose shards.
   A smaller component-task probe is also rejected: dispatching the three
   independent momentum responses to GPU `0/1/0` takes `0.123--0.127 s` versus
   `0.041--0.047 s` sequentially on GPU 0 and changes the checksum by 24.2% at
   the cross-device return. It confirms that `device_put` is not an admissible
   synchronization boundary for this CUDA stack.
   The same probe exposes and closes a more valuable single-device bottleneck.
   Experimental commit `21144b3` JITs and reuses the complete compatible
   momentum closure, keyed by backend, numerical controls, and a content
   fingerprint of viscosity and steady reaction. Three isolated warm solves
   improve from `0.856--0.984 s` to `0.0473--0.0482 s`, with relative field
   differences below `4.8e-11`. The exact two-step A4000 solve/restart gate
   improves from `137.39 s` to `100.39 s` (`1.369x`, 26.9% reduction), passes
   below the normal 120-second timeout, and reduces peak RSS by 3.4%. The full
   fringing module passes, and the refactor adds only seven net package lines.
   Evidence is in
   `benchmarks/results/b1-compatible-momentum-jit-20260713.json`. Profile this
   accepted source before the next decomposition; any multi-GPU design must
   keep its state device-resident and beat this stronger one-GPU baseline.
   That profile is now closed at experimental commit `004eb3b`. The exact
   compatible flag is recorded explicitly: an initial legacy-path trace that
   omitted it was rejected. On the isolated A4000, the compatible two-step
   restart falls from `14.31 s` to `8.12 s` (`1.76x`, 43.2% reduction) by
   materializing modal factors before caching the complete pressure GMRES.
   Reducing the Arnoldi basis from 64 to the already-tested 24 entries trims
   Krylov workspace by 62.5% with identical physics gates, though it adds only
   a further 2.4% warm speedup. A post-change trace shows the cached solve is
   now compute-bound at about 18 Arnoldi iterations. Evidence is in
   `benchmarks/results/b1-compatible-schur-jit-gpu-20260713.json`. The next
   accepted optimization must reduce Schur/preconditioner applications or
   keep that Krylov state persistently device-resident across devices; compile
   caching and restart-size tuning are no longer the dominant levers.
   A direct `jax.vmap` screen of the three component momentum inverses is also
   rejected and fully removed. It holds one A4000 at 100% for more than `213 s`
   without completing, already over `2.4x` the accepted complete solve/restart
   budget. Independently converging PCGs cannot be forced into one lockstep
   batch. A component decomposition must preserve per-component convergence
   and remain inside one compiled multi-device boundary.
   A cross-process persistent-cache screen is accepted as supporting policy,
   not as an algorithmic milestone: a `4.2 MiB` source-keyed cache reduces a
   fresh exact solve from `76.15 s` to `68.04 s` (10.6%). The existing campaign
   cache implementation is therefore retained, but it does not alter the Schur
   optimization or two-GPU acceptance gates.
   The complete accepted compatible series is integrated on `main` at
   `8061f40`; touched-file lint, all 81 consolidated fringing tests, and the
   exact compatible CPU solve/restart gate pass from the merged source. Further
   multi-GPU work branches from this fingerprint.
   The first new-branch two-GPU contract passes: 500 float64 updates across 20
   repeated `jit`/`shard_map` calls retain two shards and match a serial
   reference exactly in `0.421 s`. Inputs are explicitly placed once and all
   global scalars use `lax.psum`; there is no host gather inside the iteration.
   This confines the earlier CUDA corruption to host gather/reshard boundaries,
   not device-resident collectives. Evidence is in
   `benchmarks/results/b1-two-gpu-shard-map-contract-20260713.json`. Implement
   one exact B1 Schur action under this contract before wrapping the full GMRES.
   That exact action now passes on `agent/b1-multigpu` at `88fac50`. Three
   cylindrical momentum inverses are padded to four components, two per GPU;
   each device runs its local PCGs sequentially so convergence is not forced
   into lockstep. The complete 896-entry Schur action—including pressure force,
   Rhie--Chow faces, finite-volume divergence, weighted gauge reduction, and
   flow constraint—is bitwise identical to one GPU and improves from
   `12.10--12.49 ms` to `7.12--7.26 ms` warm (`~1.70x`). All 81 fringing tests
   and the exact compatible CPU solve/restart pass. Evidence is in
   `benchmarks/results/b1-two-gpu-schur-action-20260713.json`.
   Directly nesting this kernel beneath modal-factor construction and the full
   GMRES is not accepted: cold compilation exceeded `180 s` with idle GPUs and
   was terminated and removed. The next implementation must materialize modal
   factors and the preconditioner outside the multi-device trace, then wrap only
   the bounded runtime Krylov loop around the exact sharded Schur action.
   Commit `3d01ea7` closes the first half of that requirement: after modal
   factors are materialized, Schur and preconditioner are cached as non-inline
   JIT boundaries. On one A4000 the exact first solve improves from `76.42 s`
   to `70.27 s`, and restart from `8.12 s` to `7.13 s` (`1.14x`), with unchanged
   physics. The same boundaries do not make a monolithic two-GPU GMRES
   admissible: a second bounded screen again exceeds `180 s` in CPU compilation
   with idle GPUs and is removed. The next Krylov implementation must remain
   outside one monolithic multi-device XLA transform and call the exact compiled
   Schur/preconditioner kernels through bounded differentiable boundaries.
   SOLVAX commit `428d7d3` on `agent/staged-gmres` provides and documents an
   independently compilable `gmres_cycle`; 16 focused and all 232 SOLVAX tests
   pass in `84.77 s`, including implicit reverse-mode agreement with a dense
   solve. It is pushed but not released or pinned by LMX. A disposable LMX
   consumption screen is rejected and fully removed: even one staged Arnoldi
   cycle still exceeds `180 s` of CPU compilation because XLA traces through
   the nested B1 Schur/preconditioner calls. No unreleased SOLVAX dependency or
   public two-GPU path remains in LMX. The next boundary must be genuinely
   opaque to XLA—custom call/FFI with a registered implicit derivative—not
   another composition of nested `jit`.
   The accepted exact component kernel/hook and non-inline one-GPU pressure
   kernels are integrated on `main` at `dff2da9`; no public two-GPU solve or
   unreleased SOLVAX dependency is exposed. Touched-file lint, focused
   projection/diffusion tests, and the exact compatible CPU solve/restart pass
   from the merged source.

14. **Pending — prepare the research release.** At one source fingerprint, run the full
    supported-Python matrix, strict docs, provenance, Benchmark A, Benchmark B,
    SOLVAX CPU/GPU equivalence, and scaling lanes. Publish the acceptance index,
    environment manifests, DOI, and checksummed release assets. The portable
    suite must remain <=600 seconds per job, with <=360 seconds as the planning
    target and >=95% branch coverage. The bounded release-readiness gate now
    passes with no hard blockers: local generated files and exact paths verified
    in the uploaded `lmx-research-assets-v1` manifest are treated equivalently,
    while missing or unverified assets still fail closed. This does not close the
    three explicitly deferred research-grade physics lanes. Compact evidence is
    recorded in `benchmarks/results/bounded-release-readiness-20260713.json`;
    its source-matched portable gate passes 900 tests with 8 expected skips,
    95.09% branch coverage, and a 141.1-second wall time. The source-matched
    wheel/sdist build, Twine inspection, clean Python 3.10 install, imports, and
    CLI smoke check also pass; the universal wheel is 300 KiB. Checksums are in
    `benchmarks/results/package-smoke-20260713.json`. The current SOLVAX 0.7.0
    CPU and RTX A4000 equivalence records pass; the full 0.7.0 physics refresh
    must replace, rather than relabel, the historical 0.5.1 acceptance evidence.

15. **Deferred — expand physics only through new release tiers.** Benchmark C is Q2D
    turbulence, D is 3D turbulence/magnetic obstacles, and E is energy/buoyancy.
    Each starts with a frozen external evidence contract and cannot weaken or
    silently broaden an earlier validated claim.

## 12. Change control

The queue changes only when new evidence changes a dependency, scope boundary,
or acceptance gate. A change records the evidence, affected milestone, and
replacement criterion in the same pull request. Implementation discoveries may
change tactics without changing this plan. No benchmark result changes a
tolerance retrospectively; no performance result changes a physics gate; no
slimming target justifies deleting a validated capability without an explicit
scope decision.

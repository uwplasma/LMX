# LMX 1.0 Plan

## Goal

Ship a research-grade `1.0` inductionless MHD code with:

- a stable fully developed duct solver
- benchmark-quality Hartmann, Shercliff, and Hunt validation
- explicit restart, logging, plotting, and CLI workflows
- a documented differentiable core
- a clear roadmap to the 3D fringing-field solver family

## Scope for 1.0

### In scope

- `fully_developed_inductionless`
- Hartmann, Shercliff, Hunt
- `rect_duct` and `layered_duct`
- TOML and Python-driver workflows
- restartable `.npz` outputs
- validation, convergence, and benchmark scripts
- user and developer documentation

### Out of scope

- turbulence
- heat transfer
- free surfaces
- full induction
- production-ready 3D fringing-field solver

## Open items

1. Keep hardening the default fully developed solver family in the manual
   release-validation lane.
2. Expand benchmark and physics depth for larger comparison datasets.
3. Harden and extend the first true `extruded_inductionless` solver slice into
   a broader production 3D family.
4. Extend the differentiable lane beyond the shipped Hartmann example set.
5. Replace the current branch-sensitive straight-duct manuscript lane with a
   more faithful validation path if the remaining Shercliff/Hunt error plateau
   stays above the accepted `L2 <= 1.2e-2` release target.

## Evidence pass and closeout sequence

This pass re-checked the plan against the local git history, current docs,
tests, examples, FreeMHD source/case files, and the main literature anchors.
The project direction is still right, but the next work has to stay ordered:
do not expand capability demos faster than the verified solver surface.

### Evidence anchors

- Samper et al. remains the organizing V&V ladder:
  - A: fully developed laminar ducts
  - B: 3D laminar developing/fringing MHD
  - C: quasi-2D turbulent MHD
  - D: 3D turbulent or magnetic-obstacle MHD
  - E: heat-transfer / buoyant-convection MHD
- FreeMHD and its OpenFOAM source remain the main independent executable
  comparison surface. The relevant parity target is observable agreement, but
  the source confirms the numerical structure LMX should match where possible:
  conservative electric-potential source assembly, face-current reconstruction,
  gauge-handled potential comparisons, and Lorentz forcing from the same
  current convention.
- The local FreeMHD case files make the mesh policy explicit: high-Ha duct
  comparisons use boundary-layer-focused meshes with several cells inside the
  Hartmann and side layers and smooth expansion into the core. LMX high-Ha
  validation should therefore be judged on a mesh-convergence ladder, not on a
  single visually acceptable plot.
- The JAX documentation and current scaling data point to the same performance
  plan: separate compile time from warm runtime, profile before claiming strong
  scaling, keep memory estimates visible, and move the CPU scaling figure
  toward the real `extruded_inductionless` projection arithmetic rather than a
  presentation-only kernel.
- The PyPA publishing guidance points to a release workflow built around
  build artifacts, TestPyPI/PyPI Trusted Publishing, and a tagged-release gate
  after tests, docs, coverage, and validation artifacts pass.

### Current status classification

Paper-ready or close to paper-ready:

- low-cost numerical operator verification on smooth, clustered, and
  interface/coefficient-jump cases
- retained Hartmann/Shercliff/Hunt analytical profile overlays at the accepted
  `L2 <= 1.2e-2` release target for the current reader-facing cuts
- dense internal rectangular Benchmark B fringing conservation metrics
- layered Benchmark B closure on mirror-aware current/pressure observables
- Q2D decay, forced, and wall-bounded reduced validation examples
- magnetic-obstacle bounded response and regime-scan figures as Benchmark D
  precursors
- runtime logging, ETA/progress output, restart, plotting, and artifact-summary
  workflows

Demonstrated but not yet research-grade validation:

- FreeMHD paper-slice observable parity for fully developed ducts, especially
  the physical constant-`Q` / `flow_rate` path and `potE`, `J`, `J×B` cuts
- mapped-pipe external parity against the high-`Ha`, high-`Re` Bühler case;
  this is now explicitly future work, not a blocker for the current closeout
- full Benchmark C turbulent observables
- full Benchmark D turbulent or experimentally anchored magnetic-obstacle
  parity
- Benchmark E heat-transfer / buoyant-convection
- higher-inertia bent-pipe physics beyond the low-De baseline
- variable-field validation beyond bounded duct/layered/pipe/bent examples
- solver-faithful CPU strong scaling and memory-model documentation
- broad `95%` coverage and PyPI release automation

### Immediate technical sequence

1. Repair the fully developed constant-`Q` / `flow_rate` path.
   The next solver campaign should derive and test the constrained
   flow-rate/pressure-gradient update directly, then compare velocity,
   potential, current, Lorentz force, flow rate, and pressure-proxy observables
   against analytical ducts and FreeMHD paper slices. The acceptance target is
   not plot-level agreement; it is stable profile/integral errors under mesh
   refinement with the retained profile cuts at or below `1.2e-2` and the
   external observable lane trending toward `O(1e-2)`.
2. Make boundary-layer meshing first-class.
   Add a documented mesh ladder for Hartmann and side layers with at least
   8-10 cells across the thinnest layer in high-Ha validation runs, smooth
   expansion into the core, no repeated faces, and explicit convergence of
   `Q̃`, profile cuts, current closure, and pressure proxy.
3. Close conservative-current parity.
   Every fully developed and extruded solver family should expose diagnostics
   that can be compared with the FreeMHD-style conservative face-current
   reconstruction: `phi` gauge, `J`, `J×B`, `div J`, wall leakage, interface
   continuity, and net boundary-current residual.
4. Reach `95%` coverage by testing live behavior and deleting dead branches.
   The target is not more long solves. The next coverage pass should focus on
   `lmx/solvers.py` branch helpers, validation fallbacks, restart/initial
   condition paths, current/forcing/flow-rate modes, and cheap manufactured or
   monkeypatched tests.
5. Promote only literature-gated capability lanes.
   Bent pipe, variable-field, Benchmark C, Benchmark D, and Benchmark E should
   each require one numerical invariant, one physics/literature gate, one
   artifact-producing example, and one docs entry stating whether the case is
   verification, validation, or only a capability demo.
6. Finish performance and differentiability gates.
   The solver must report progress, ETA, memory-relevant grid sizes, compile
   versus warm runtime, and device placement. Differentiable workflows must
   carry finite-difference gradient checks, JIT/eager agreement, batched
   objective consistency, and at least one pressure-drop or field-design
   inverse problem.
7. Split large modules after behavior is locked.
   Refactor `fringing`, `autodiff`, `plotting`, `solvers`, and `validation`
   only behind passing tests and stable public import facades.
8. Add release automation after the gates are real.
   The publish lane should build sdist/wheel artifacts, run fast tests, docs,
   coverage, and selected validation artifact checks, publish to TestPyPI
   first, then PyPI through Trusted Publishing on tagged releases.

## Finish-line gates

LMX reaches the next research-grade milestone only when the following gates are
explicitly documented and passing:

### Physics gates

- Benchmark A
  - Samper Table I parameter sets are covered explicitly:
    - insulating square duct at `Ha = 500, 5000, 10000, 15000`
    - conducting-wall square duct at the same `Ha` values with Hartmann-wall
      conductance ratio `cw = 0.01`
  - Hartmann/Shercliff/Hunt profiles converge under mesh refinement
  - flow rate, current-density, Lorentz-force, and pressure-proxy trends are
    stable with respect to resolution and timestep
  - the dimensionless flow-rate integral `Q̃` agrees with the analytical
    values used in the literature ladder, not only the local profile cuts
  - charge-balance and interface-current residuals stay below configured
    thresholds on both routine and heavier manual datasets
- Benchmark B
  - Samper Table II parameter sets are staged explicitly:
    - B1 pipe at `Ha ≈ 6600`, `N ≈ 10700`, `cw ≈ 0.027`
    - B2 square duct at `Ha ≈ 2900`, `N ≈ 540`, `cw ≈ 0.07`
  - rectangular, layered, and mapped-pipe fringing cases satisfy the
    conservation thresholds
  - dimensionless pressure-drop comparison between the documented measurement
    taps is available for the B1/B2 parity table
  - throughput variation outside the field-ramp region remains bounded
  - pressure span rises through the magnetized region and relaxes downstream
  - field/mean-velocity correlation carries the expected negative sign

### Quality gates

- fast routine test lane under five minutes
- strict docs build
- restart continuation equivalence on CLI/TOML and Python workflows
- stable JSON/CSV/NPZ output schemas
- figure/movie examples regenerate the committed docs assets
- branch coverage keeps shrinking in live solver code rather than being hidden
  behind dead branches

### Executable validation exercise

The current top-level validation workflow that should be used before manuscript
drafting is:

```bash
python scripts/run_full_validation_exercise.py \
  --output artifacts/validation/full_validation_exercise \
  --ha-values 10,20 \
  --resolution 12 \
  --fringing-resolutions 8,12 \
  --skip-paraview \
  --write-plot
```

That run must produce:

- Benchmark A case artifacts and summaries
- Benchmark B fringing gate summaries
- one combined JSON summary
- one combined CSV table
- one combined Markdown report

### Validation ladder after the current duct/fringing set

Following the benchmark sequence summarized by Samper et al., the next
publication-grade additions after the current A/B ladder are:

- quasi-2D turbulent duct validation
- 3D turbulent magnetic-obstacle or duct validation
- heat-transfer / buoyant-convection validation
- closed-pipe fringing validation
- open-channel / free-surface validation
- current-driven channel validation

### Coverage and verification roadmap

The remaining test/coverage work is not "add more tests until the percentage
goes up." The goal is to lock the code against:

- literature-backed physics regressions
- numerical/discretization regressions
- branch-heavy helper/fallback regressions

The governing references for that workstream are:

- Samper et al., *An approach to verification and validation of MHD codes for
  fusion applications*
- the FreeMHD V&V paper and bundled case/reference data
- Sommeria-Moreau Q2D model literature
- magnetic-obstacle literature from Cuevas/Votyakov and related JFM work
- standard CFD/MHD verification practice through manufactured solutions and
  observed-order studies

That workstream is organized as follows.

#### Phase 1: branch-heavy core cleanup

Scope:

- `lmx/solvers.py`
- `lmx/validation.py`
- bounded helper/script surfaces that still drag branch coverage down

Rules:

- if a branch is part of the intended public behavior, test it directly
- if a branch is dead or historical, remove it rather than covering around it
- prefer cheap direct tests over slow integration runs

Current module priorities:

- `lmx/validation.py`: fallback paths, missing-field handling, acceptance
  branches, profile extraction branches, optional-reference branches
- `lmx/solvers.py`: helper/fallback branches, restart/initialization paths,
  forcing and inlet-mode branches, bounded update/diagnostic branches

#### Phase 2: numerical verification

Required additions:

- manufactured-solution tests for the electric-potential and diffusion-like
  operators
- observed-order tests on uniform and clustered meshes
- coefficient-jump/interface tests on layered media
- iterative-solver safety tests:
  - residual monotonicity on small SPD systems
  - finite-denominator/nonfinite guard behavior
  - backend equivalence on bounded toy problems
- discrete invariants:
  - net current closure
  - symmetry preservation under symmetric data
  - boundedness / no obvious overshoot on simple elliptic cases

Implementation targets:

- `tests/test_operators.py`
- `tests/test_linear.py`
- `tests/test_solver.py`
- a dedicated manufactured-solution test module if the operator coverage grows
  beyond the current files

#### Phase 3: literature-backed physics verification

Required additions:

- parameterized Hartmann validation ladder
- parameterized Shercliff validation ladder
- parameterized Hunt validation ladder
- explicit integral/trend checks, not only local profile overlays
- stronger benchmark-observable regression checks for Benchmark B/C/D figures
- if the present fully-developed pseudo-steady path remains branch-sensitive,
  add a dedicated straight-duct validation path for the paper figures:
  either a direct coupled fully-developed solve or a long-duct extruded solve
  sampled far downstream

Implementation targets:

- `tests/test_physics.py`
- `tests/test_fringing.py`
- `tests/test_q2d.py`
- additional benchmark-specific validation test modules where that keeps the
  assertions readable

#### Phase 4: artifact/regression protection

Every paper-facing example or docs-facing benchmark figure should regenerate a
JSON summary carrying:

- the input parameters
- the governing observables
- the pass/fail gates
- the artifact paths

Tests should validate the summary values and schema rather than image pixels.

Verification-heavy tests that demonstrate numerics or physics quality should
not stay test-only when they are manuscript-relevant. Each such family should
also have a companion artifact-producing example or script that emits:

- a publication-ready figure
- a summary JSON with the governing observables and references
- stable artifact paths for docs and future manuscript inclusion

#### Phase 5: software architecture and module split

The current codebase is functionally broad enough that the remaining quality
work is now partly architectural. The largest files should be split without
changing behavior, public API semantics, FreeMHD parity status, or existing
artifact paths.

Immediate module-size hotspots:

- `lmx/fringing.py` (~2900 lines)
- `lmx/autodiff.py` (~2100 lines)
- `lmx/plotting.py` (~2000 lines)
- `lmx/solvers.py` (~1400 lines)
- `lmx/validation.py` (~1100 lines)

Target split map:

- `lmx/fringing.py`
  - `lmx/fringing/problems.py`: problem builders and case assembly
  - `lmx/fringing/solve.py`: executable projection loop and stepping logic
  - `lmx/fringing/metrics.py`: conservative current/pressure/throughput metrics
  - `lmx/fringing/validation.py`: benchmark gates and literature-facing slices
  - `lmx/fringing/reference.py`: reference comparison helpers and normalizations
- `lmx/autodiff.py`
  - `lmx/autodiff/objectives.py`: differentiable loss/objective definitions
  - `lmx/autodiff/gradients.py`: gradient/JVP/VJP utilities and checks
  - `lmx/autodiff/design.py`: inverse-design loops and optimization wrappers
  - `lmx/autodiff/uq.py`: local uncertainty propagation / sensitivity utilities
- `lmx/plotting.py`
  - `lmx/plotting/profiles.py`
  - `lmx/plotting/benchmarks.py`
  - `lmx/plotting/media.py`
  - `lmx/plotting/fields.py`
- `lmx/validation.py`
  - `lmx/validation/profiles.py`
  - `lmx/validation/reference.py`
  - `lmx/validation/reports.py`
- `lmx/solvers.py`
  - keep a thin public façade
  - move helper-heavy internals into:
    - `lmx/solvers/fully_developed.py`
    - `lmx/solvers/potential.py`
    - `lmx/solvers/diagnostics.py`
    - `lmx/solvers/logging.py`

Refactor rules:

- preserve existing public import paths during the split cycle
- move tests with the code they exercise
- split only after the behavior is locked by direct tests
- do not mix refactors with numerical changes in the same patch when possible

#### Phase 6: autodiff, UQ, and optimization verification

LMX should not stop at "gradients exist." The differentiable lane needs its own
verification ladder so sensitivity analysis, inverse design, uncertainty
quantification, and mirror/okamak/stellarator-related optimization work are
defensible.

Required additions:

- gradient checks against finite differences for:
  - straight-duct profile objectives
  - fringing pressure-drop / current-response objectives
  - WHAM-like mirror-field coil-separation objectives
  - tabulated-field amplitude/placement objectives
- optimization regressions:
  - objective decreases monotonically on bounded inverse-design demos
  - final design outperforms the initial design on the tracked objective
- UQ baselines:
  - local linear uncertainty propagation versus finite-difference covariance
  - low-dimensional Monte Carlo or sparse-grid checks on reduced examples
- JAX-specific correctness checks:
  - `jit` and eager agreement
  - `grad` and finite-difference sign agreement
  - `vmap` / batched-objective consistency
  - dtype/device stability where the code claims support

Domain-facing optimization roadmap:

- mirror optimization:
  coil spacing, coil current scale, pipe placement, pressure-drop objectives
- tokamak/blanket optimization:
  tabulated 3D field loading, pressure-drop sensitivity, wall-conductance scans
- stellarator / non-axisymmetric field support:
  begin with tabulated-field sensitivity and shape/placement parameter studies,
  not full equilibrium optimization

#### Phase 7: comments, docstrings, and developer-facing clarity

The remaining maintainability gap is not only tests. The solver needs explicit
documentation at the code level so the numerical intent is reviewable.

Rules:

- every public function should have a docstring with:
  - purpose
  - expected units / normalization
  - array shape conventions where relevant
  - return values and failure modes
- module docstrings should state:
  - governing approximation
  - literature anchor where appropriate
  - what is verified versus what is only demonstrated
- comments should be sparse and technical:
  - explain conservative reconstructions, gauge choices, symmetry metrics, and
    non-obvious numerical safeguards
  - do not narrate obvious assignments or control flow

#### Example and benchmark validation matrix

Every example should be classified as one of:

- verification example
- validation example
- capability demo
- optimization/autodiff demo

Current target matrix:

- Benchmark A / straight-duct verification
  - `hartmann_example.py`
  - `shercliff_example.py`
  - `hunt_example.py`
  - `straight_duct_profile_comparison.py`
- Benchmark B / 3D fringing validation
  - `fringing_benchmark_demo.py`
  - `extruded_validation_campaign.py`
  - `pipe_reference_comparison_demo.py`
  - bent-pipe inductionless baseline
- Benchmark C / Q2D validation
  - `q2d_decay_validation.py`
  - `q2d_forced_validation.py`
  - `q2d_wall_bounded_validation.py`
- Benchmark D / magnetic-obstacle and mirror-field validation
  - `magnetic_obstacle_benchmark.py`
  - `magnetic_obstacle_regime_scan.py`
  - `wham_mirror_pipe_demo.py`
  - `autodiff_wham_pressure_sensitivity.py`
- Variable-field capability and tabulated-field validation
  - `variable_field_extruded_demo.py`
  - `variable_field_layered_demo.py`
  - `variable_field_bent_pipe_demo.py`
  - `variable_field_tabulated_demo.py`

Every item in that matrix should ultimately have:

- a validated JSON summary
- named physics/numerics gates
- a test that reads the summary and asserts the governing observables
- a clear statement in docs whether it is verification, validation, or only a
  capability demonstration

#### Manuscript figure plan

The paper should not be assembled ad hoc from whatever figures happen to exist.
Each major validation lane needs one or two canonical figures, generated from a
tested example, with its summary checked in CI.

Target manuscript figures:

- Straight-duct verification
  - Hartmann/Shercliff/Hunt analytical overlays on the same normalization
  - cross-sectional contour panel showing Hartmann and side/Hunt layers
  - optional mesh-inset figure showing boundary-layer clustering for the
    highest-Ha verification case
- Benchmark B: laminar developing flow in non-uniform magnetic field
  - rectangular and layered 3D fringing cross-sections plus streamwise history
  - one quantitative summary panel for pressure span, current closure, and
    throughput constancy
  - mapped-pipe comparison figure remains optional until external parity is
    reopened and closed
- Benchmark C: Q2D validation
  - decay/forced/wall-bounded amplitude-versus-time validation panel
  - one asymptotic Q2D figure showing stronger transverse field suppressing
    three-dimensionality in the expected direction
- Benchmark D: magnetic obstacle / localized-field interaction
  - wake-deficit / recovery / cross-cut distortion benchmark panel
  - one regime map or literature-slice comparison figure showing how response
    changes with obstacle strength
- Variable-field and mirror-field capability
  - WHAM-like mirror-field overview with coils, field contours, pipe location,
    executable response, and autodiff sensitivity
  - one tabulated-field validation figure for structured 3D fields
- Bent-pipe and curved-geometry capability
  - curved-pipe geometry/solution/response figure with layer zoom and symmetry
    or curvature-response diagnostics

Rules for manuscript figures:

- every paper-facing figure must come from a standalone example
- the example must write a summary JSON with governing observables and gates
- the README/docs may use lighter variants, but the manuscript figures should be
  traceable to the tested examples
- numerics-facing figures are allowed and encouraged when they demonstrate
  observed order, solver robustness, or verification quality in a way that is
  literature-consistent and manuscript-useful

#### Testing documentation plan

The testing surface is now large enough that it needs first-class
documentation, not only scattered references in the README.

Required docs additions:

- a dedicated testing architecture page describing:
  - test taxonomy: unit, regression, verification, validation, literature
  - fast lane versus heavy manual lane
  - artifact-summary testing strategy
  - coverage targets and how to interpret line versus branch coverage
- a verification and validation guide mapping LMX tests/examples to the
  Samper A/B/C/D/E ladder
- a numerical verification page documenting:
  - manufactured solutions
  - observed-order studies
  - solver-safety and invariant tests
- an autodiff verification page documenting:
  - gradient checks
  - inverse-design regressions
  - UQ/sensitivity tests
  - JIT/eager/batched consistency expectations
- a benchmark-artifact page listing every JSON summary used to support the
  manuscript/README/docs figures

#### Code-structure documentation plan

As the large modules are split, the code structure should be documented
explicitly so contributors can find the numerical and scientific logic quickly.

Required docs additions:

- one architecture page covering:
  - case specification and config flow
  - mesh generation
  - fully developed solver path
  - extruded/fringing solver path
  - field-model and tabulated-field path
  - validation/reporting path
  - plotting/media path
  - autodiff/design/UQ path
- per-module docstrings summarizing governing equations, conventions, and
  literature anchors
- developer notes explaining where numerical helpers end and benchmark logic
  begins, so benchmark-specific assertions do not leak back into solver code

#### Additional literature-backed tests worth adding

Beyond the current set, the following would materially strengthen a future LMX
 paper if implemented and validated correctly.

- High-Ha straight-duct asymptotics
  - very-high-Ha Hartmann/Shercliff/Hunt verification slices with boundary-layer
    focused meshes, aligned with the high-Ha cases emphasized in the FreeMHD
    paper and related fusion-blanket literature
- Q2D turbulent/shear benchmark extension
  - benchmark-C-style quasi-2D forced shear or vortex problem anchored in the
    Sommeria-Moreau / Pothérat literature
- Stronger magnetic-obstacle comparison
  - wake deficit, recovery distance, and cross-cut distortion against published
    obstacle datasets from Cuevas/Votyakov-type studies, not only internal
    response trends
- Heat-transfer / buoyant-convection benchmark
  - benchmark-E-style liquid-metal magnetoconvection case with wall-bounded
    temperature field and magnetically modified convection
- Curved/toroidal duct benchmark
  - low-Re or low-Ha toroidal/curved duct comparison, following the class of
    validation cases used by robust 3-D fusion MHD solver papers
- Current-driven and electrically forced shallow-layer flows
  - electrically driven vortex / localized-field interaction cases for stronger
    validation of localized Lorentz forcing and Q2D response
- Tabulated 3D fusion-field loading
  - mirror/tokamak/stellarator-like field interpolation, divergence audit, and
    response sensitivity against structured or sampled magnetic data

#### Final pre-implementation rule set

Before a new capability is declared "done", it should satisfy all of:

- one direct unit test for helper/fallback logic
- one numerical verification or invariant test
- one physics or literature-facing validation test
- one example that writes a checked summary JSON
- one docs entry that states whether the case is verification, validation, or a
  capability demonstration

This keeps parity claims with FreeMHD and the literature explicit and prevents
README-only features from drifting ahead of the tested solver surface.

#### Performance, differentiability, and accuracy priorities

Implementation work should continue to optimize for all three of:

- accuracy:
  numerical verification, literature agreement, conservative invariants, and
  physically meaningful benchmark observables
- end-to-end differentiability:
  avoid introducing non-differentiable helper paths into the public design/UQ
  workflows when a JAX-native alternative is practical
- performance:
  keep the routine lane bounded, keep artifact workflows reproducible, and
  prefer operator/solver changes that do not undermine the stronger-scaling and
  differentiable execution goals

When tradeoffs appear, the expected closeout path is:

- first make the physics and numerics correct
- then preserve or recover differentiability
- then optimize performance without silently changing validated behavior

#### Coverage targets

The practical targets are:

- broad routine line coverage stays at or above `95%`
- `lmx/` line coverage stays at or above `95%`
- module-level branch coverage keeps increasing in:
  - `lmx/solvers.py`
  - `lmx/validation.py`
  - `lmx/fringing.py`
  - `lmx/plotting.py`

Branch coverage is expected to lag line coverage. The correct closeout path is
continued helper-path testing and dead-path removal, not inflating the routine
validation lane with heavier solver runs.

### Remaining real gaps

- README/media QA
  - the landing-page startup movies still need a final high-resolution refresh
    with symmetric, layer-resolved Hartmann visuals
  - the movie path should use solver settings and gauge treatment that do not
    introduce visible left/right bias in closed symmetric cases
- Benchmark B quantitative closure
  - dense `rect_duct` fringing is now inside the quantitative internal gate
  - mapped-pipe external comparison is now quantitative but intentionally
    deferred; it compares against a high-`Ha`, high-`Re` Bühler reference while
    the current LMX mapped-pipe lane is still a lower-Re inductionless slice
  - the next literature-anchored B1/B2 closure still needs the Samper-style
    observables, not only the internal fringing metrics:
    - B1 pipe: pressure-drop comparison between the documented tap locations
      plus center and offset profile errors on a shared normalization
    - B2 square duct: pressure-drop comparison through the ramp plus matched
      cross-sectional velocity/potential cuts at reference axial stations
- CPU scaling
  - the current longer-run CPU artifact is honest, but it is still a surrogate
    benchmark rather than the final solver-faithful CPU scaling story
  - the next accepted CPU figure must be tied more directly to the executable
    `extruded_inductionless` operator path and backed by profiling
- Solver coverage
  - `lmx/solvers.py` still carries the main branch-heavy coverage debt
  - remaining coverage work should remove or test historical branches rather
    than hiding them behind low-value integration harnesses
- Validation runtime discipline
  - the combined A/B validation exercise is a manual lane rather than a
    routine sub-five-minute gate

### Capability roadmap beyond the current geometry set

#### Bent-pipe geometry

Required implementation work:

- centerline-following structured mesh for a constant-radius bend
- mapped wall metrics and face areas consistent with conservative current
  reconstruction
- inlet/outlet boundary handling for curved centerlines
- 3D electric-potential solve with the same compatibility and boundary-current
  audits used in the existing fringing lane

Required validation work:

- no-field Dean-flow baseline before any MHD coupling
- uniform-field bent-pipe inductionless case
- nonuniform/fringing-field bent-pipe case
- mesh-convergence study on curvature, pressure span, and current closure

Current status:

- public bent-pipe preprocessing support now exists through
  `generate_bent_pipe_mesh(...)`
- preview/QA examples now exist for the curved-centerline geometry lane
- the first executable curved-pipe lane now exists through
  `build_bent_pipe_extruded_problem(...)`,
  `validate_bent_pipe_low_de_baseline(...)`, and
  `examples/bent_pipe_inductionless_demo.py`
- the current bent-pipe lane is the low-De inductionless baseline, validated
  against the straight-pipe limit rather than a full Dean-vortex benchmark
- current bounded example (`Ha = 20`, `R = 0.45`, `R_c = 3.6`,
  `15 × 18 × 40`) gives:
  - `De ≈ 5.19e-7`
  - `cross_section_l2_error = 0`
  - `centerline_l2_error = 0`
  - `max_charge_balance_residual ≈ 2.15e-2`
  - `volumetric_flow_rate_span ≈ 1.14e-9`
- the remaining work is the higher-inertia curved-pipe solver path, not the
  geometry-construction-side

#### Spatially varying magnetic fields

Required implementation work:

- executable support for analytic and tabulated 3D magnetic-field profiles
- validation of interpolation, normalization, and field loading through both
  Python and TOML/CLI workflows
- conservative current/Lorentz assembly with fully spatially varying `B(x,y,z)`

Required validation work:

- manufactured divergence-free field checks
- recovery of the existing fringing benchmarks using the generic field path
- variable-field duct studies where pressure span, flow-rate distortion, and
  current closure are compared under mesh refinement

Current status:

- reusable analytic divergence-free cross-sectional field builders and
  finite-difference divergence checks are now in the public API
- example workflows now cover both geometry-plus-solve field usage and pure
  field QA before a solve
- executable rectangular `extruded_inductionless` support for analytic
  cross-sectional magnetic fields now exists through
  `build_variable_field_duct_extruded_problem(...)`,
  `validate_variable_field_extruded_solution(...)`, and
  `examples/variable_field_extruded_demo.py`
- layered and curved-pipe extensions now also exist through
  `build_variable_field_layered_extruded_problem(...)`,
  `build_variable_field_pipe_ogrid_extruded_problem(...)`,
  `build_variable_field_bent_pipe_extruded_problem(...)`, and the paired
  example drivers
- tabulated-field rectangular support now also exists through
  `examples/variable_field_tabulated_demo.py` and
  `examples/fringing_tabulated_case.toml`
- current bounded rectangular tabulated-field result:
  `field_velocity_correlation ≈ -7.52e-1`,
  `current_proxy_change ≈ 8.28e-1`,
  `max_charge_balance_residual ≈ 5.34e-6`
- the remaining work is broader parity-grade validation on those new
  geometries and generic 3D tabulated fields beyond the current structured
  duct baseline

### Literature-anchored benchmark expansion

Using the V&V sequence summarized by Samper et al., the benchmark ladder after
the current closed-duct and fringing set should be organized as:

1. Benchmark A
   - 2D fully developed laminar steady MHD duct cases
   - Hartmann, Shercliff, Hunt profiles, integrals, and conservation
2. Benchmark B
   - 3D laminar developing flow in nonuniform magnetic fields
   - rectangular duct, layered duct, mapped pipe, then bent pipe
3. Benchmark C
   - quasi-2D turbulent duct/channel MHD
4. Benchmark D
   - fully 3D turbulent or magnetic-obstacle MHD
5. Benchmark E
   - heat-transfer / buoyant-convection MHD

For LMX, each benchmark level should only be promoted once the following are
documented together:

- governing observables
- mesh and timestep refinement rules
- conservation thresholds
- external reference or experimental source
- executable driver and committed example inputs

Current Benchmark C baseline:

- `examples/q2d_decay_validation.py` now provides the first executable Q2D
  validation surface
- scope:
  periodic Hartmann-friction decay of a single 2D mode
- observables:
  final-state `L2/L∞` error and amplitude-decay error against the analytic
  exponential decay
- remaining work before any turbulent Benchmark C claim:
  Q2D duct forcing, wall/friction closures closer to Sommeria-Moreau, and
  literature-anchored turbulent observables

Current forced Benchmark C slice:

- `examples/q2d_forced_validation.py` now adds the first forced Q2D duct
  validation problem
- observables:
  steady-state `L2/L∞` error and amplitude error against the analytic forced
  solution
- remaining work before any turbulent Benchmark C claim:
  wall-bounded Q2D duct forcing, closure terms closer to the Sommeria-Moreau
  model, and literature-anchored turbulent observables
- current bounded result from `examples/q2d_forced_validation.py`:
  `l2_error ≈ 4.44e-4`, `linf_error ≈ 4.44e-4`,
  `steady_amplitude_rel_error ≈ 4.44e-4`

Current wall-bounded Benchmark C slice:

- `examples/q2d_wall_bounded_validation.py` now adds the first no-slip Q2D
  duct benchmark
- observables:
  final-state `L2/L∞` error and amplitude error against the exact Dirichlet
  transient solution
- remaining work before any turbulent Benchmark C claim:
  Sommeria-Moreau-style closures, literature-anchored wall-bounded duct
  observables, and Q2D turbulence
- current bounded result from `examples/q2d_wall_bounded_validation.py`:
  `l2_error ≈ 4.89e-4`, `linf_error ≈ 4.89e-4`,
  `amplitude_rel_error ≈ 4.89e-4`

Current first Benchmark D slice:

- `examples/magnetic_obstacle_benchmark.py` now provides the first stronger
  magnetic-obstacle benchmark on the rectangular extruded lane
- observables:
  normalized velocity-deficit ratio, pressure-excess response, centerline-cut
  distortion, current response, and conservation metrics
- remaining work before any full Benchmark D claim:
  stronger-inertia obstacle regimes, literature parity on magnetic-obstacle
  observables, a more forceful executable mirror-field pipe response, and
  truly turbulent 3D validation
- current bounded result from `examples/magnetic_obstacle_benchmark.py`:
  `peak_velocity_deficit_ratio ≈ 3.23e-2`,
  `peak_station_velocity_deficit_ratio ≈ 2.87e-2`,
  `integrated_velocity_deficit_ratio ≈ 2.78e-2`,
  `peak_centerline_deficit_ratio ≈ 3.76e-1`,
  `recovery_station ≈ 4.76`,
  `peak_pressure_excess ≈ 5.01e-1`,
  `pressure_excess_proxy ≈ 1.22e-1`,
  `current_proxy_peak ≈ 4.56`,
  `y_l2_distortion ≈ 2.31e-1`,
  `z_l2_distortion ≈ 2.08e-1`,
  `peak_crosscut_distortion ≈ 2.31e-1`,
  `divergence_to_field_ratio ≈ 1.69e-2`,
  `max_charge_balance_residual ≈ 3.98e-13`,
  `benchmark_pass = true`
- current bounded literature-facing slice on the same case:
  `peak_station ≈ 3.00`,
  `recovery_distance ≈ 1.76`,
  `normalized_recovery_distance ≈ 6.25e-1`,
  `literature_pass = true`
- `examples/magnetic_obstacle_regime_scan.py` now stages the next step beyond
  that bounded point by sweeping obstacle cases over field scale and forcing
  and recording how deficit, pressure response, current response, and
  distortion strengthen across the small regime map
- `examples/wham_mirror_pipe_demo.py` now adds a tabulated WHAM-like mirror
  field on the pipe lane using `magpylib_jax`
- `examples/autodiff_wham_pressure_sensitivity.py` now adds the first
  coil-separation sensitivity study for pressure-drop proxy on that mirror
  topology
- current bounded reduced sensitivity result:
  `pressure_drop_proxy ≈ 3.85`,
  `d(Δp)/ds ≈ 2.98e-1` at `s = 1.96 m`
- current bounded executable tabulated-pipe result:
  field loading and conservation are stable, but the nominal-WHAM low-Re
  response is still weak
  (`field_velocity_correlation ≈ -9.70e-1`,
  `max_charge_balance_residual ≈ 3.06e-5`,
  `pressure_drop_proxy ≈ 1.86e-11`)

## Parallelization work required before manuscript closeout

- CPU scaling
  - replace the current forced-logical-device host benchmark with a more
    representative CPU benchmark tied to the real `extruded_inductionless`
    arithmetic mix or another higher-intensity 3D operator
  - use JAX profiling traces to confirm whether the benchmark is limited by
    memory bandwidth, cross-device communication, or kernel launch overhead
  - document the accepted CPU scaling path explicitly rather than mixing
    host-device sharding and thread-count claims
- GPU scaling
  - keep the larger fixed-problem two-GPU benchmark as the main scaling figure
  - extend that benchmark toward the actual 3D solver path once the CPU lane is
    made consistent
- profiling and implementation checks
  - profile the strong-scaling kernel and at least one `extruded_inductionless`
    solve on CPU
  - compare single-device CPU execution, forced host-device sharding, and any
    revised higher-intensity benchmark path
  - if automatic sharding remains communication-limited, prototype an explicit
    `jax.shard_map` path for the most communication-heavy stencil/projection
    kernels and compare it against the current automatic-sharding baseline
  - only promote a CPU scaling figure once the chosen execution model is
    coherent with the solver implementation and the measured bottleneck

## Conservation hardening lane

The next research-grade solver work must keep conservation properties explicit,
not implicit. The main implementation targets are:

- conservative face-current construction in every solver family
- compatibility projection of potential right-hand sides onto a zero-net-source
  subspace before every Poisson-like electric solve
- interface-current continuity diagnostics across fluid/solid material jumps
- explicit boundary-current audits:
  - wall-normal current leakage on insulating or externally closed boundaries
  - inlet/outlet axial current imbalance
  - net boundary-flux residual over the full 3D control volume
- pressure-velocity updates that reduce both `div(u)` and charge/current
  imbalance together on the extruded 3D lane

The fully developed lane already carries compatibility projection and
charge-balance diagnostics. The fringing 3D slice now also carries axial
current and wall-leakage diagnostics, and the manual validation lane now
supports turning those metrics into hard pass/fail gates.

The current retained larger-dataset hard gate is:

- Hartmann / Shercliff / Hunt at `Ha = 10, 20`, resolution `10`
- fringing `rect_duct`, `layered_duct`, and `pipe_ogrid` at the same `Ha` values with
  `nx_stations = 5`
- hard thresholds:
  - `charge_balance <= 8e-1`
  - `interface_current <= 2.5e-1`
  - `wall_current_leakage <= 1e-1`
  - `boundary_current <= 1e-5`

That retained gate now passes for all three retained fringing geometries.

## Current status

- The default duct solver family is now `fully_developed_inductionless`.
- The historical reduced solver has been removed from the shipped codebase.
- Runtime diagnostics now include linear residuals, flow rate, current,
  Lorentz power, and conservation signals.
- The logging surface now has a documented boolean `verbose` alias and explicit
  `verbosity = quiet|normal|detailed|debug` controls in TOML, CLI, and Python
  driver usage.
- Python `3.10` support is now explicit through the `tomli` fallback and the
  broader JAX dependency acceptance in packaging.
- Time integration now uses bounded step-count logic so `t_final` is not
  rounded up spuriously on fractional `dt` ratios.
- Public docs and examples now present a clean `1.0` release surface.
- The documentation surface has now been expanded into a fuller standalone
  manual: landing page, getting-started guide, numerics page, geometry and
  mesh workflows, testing strategy, richer theory notes, and Python-native
  variable-field/custom-geometry examples are all part of the shipped tree.
- External executable comparisons are documented only as secondary benchmark
  evidence, not as implementation guidance.
- Public JSON/example/report outputs have been scrubbed to avoid leaking
  workstation-specific absolute paths.
- The remaining `1.0` gate is now dominated by solver-heavy physics and
  validation tests rather than benchmark/example/I/O harness overhead.
- Benchmark, I/O, and example tests have been rewritten to use synthetic or
  monkeypatched orchestration paths where full solves were unnecessary.
- Validation report tests now stub solver execution where they are asserting
  report/schema behavior rather than analytical acceptance.
- CI coverage no longer forces `JAX_DISABLE_JIT=1`, because that setting was
  inflating runtime on solver-heavy tests without improving release confidence.
- Default CI is now being narrowed to a fast ship gate, with benchmark and
  validation-artifact workflows moved to manual `workflow_dispatch` runs so
  routine pushes do not consume research-artifact runtime on every change.
- The default push/PR gate now excludes the heaviest `physics` marker tests;
  those remain available in a manual workflow-dispatch lane together with
  benchmark, artifact, and extended coverage runs.
- The default push/PR gate also excludes the heavier `regression` marker
  tests, which are now part of the manual release-validation lane together
  with physics.
- The latest local default push/PR lane
  `python -m pytest -m "unit or validation" -q` passes and remains within the
  hard five-minute guard, but the older sub-minute runtime notes are stale.
- Current combined coverage for `lmx/` and `scripts/` is `94.39%`.
- Budgeted CLI and restart smokes now pass on the shipped Hartmann TOML path;
- the executable `extruded_inductionless` path now also supports restart and a
  structured `system/fields/postProcessing/restart/logs` output tree through
  both Python and TOML/CLI workflows
- the differentiable lane now includes field-level inverse design over
  selected extruded `u`, `phi`, `J_y`, and `p` slices
- the heavier manual validation lane now has a larger bounded 3D campaign with
  JSON/CSV outputs and retained summary figures
  the release gate uses short-budget generated TOMLs rather than full
  long-horizon example runs so the interface is verified without violating the
  five-minute rule.
- CPU and GPU strong-scaling artifacts now exist for the dominant stencil kernel,
  with a committed summary figure under `docs/_static/generated/strong_scaling.png`.
- The differentiable Hartmann example now has a committed summary figure under
  `docs/_static/generated/autodiff_summary.png`, showing both Hartmann-number
  sensitivity and inverse recovery of a forcing parameter.
- A second autodiff example now validates Hartmann and forcing sensitivities
  against finite differences for a compact derivative-verification figure.
- A third autodiff example now performs full-profile inverse design over both
  forcing and Hartmann number, broadening the differentiable lane from scalar
  matching to small field-level inverse problems.
- A fourth autodiff example now performs fringing-history inverse design over
  peak Hartmann number and axial field-profile shape parameters, which is the
  first retained bridge from Hartmann-only autodiff into fringing-oriented
  research objectives.
- A fifth autodiff example now performs fringing multi-observable inverse
  design against both axial mean-velocity and current-proxy histories, which
  is the current lightweight bridge from scalar fringing objectives toward
  richer multi-observable fringing calibration.
- Rectangular fringing cases now have a first true low-Re
  `extruded_inductionless` 3D projection slice, exposing `u`, `v`, `w`, `p`,
  `phi`, current, Lorentz, and charge-balance fields through the explicit
  `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` Python API.
- Layered fringing ducts now use the same first low-Re 3D projection slice.
- Layered fringing ducts now also use the same conservative face-current audit
  and closed-current axial boundary treatment as the rectangular 3D slice.
- Mapped `pipe_ogrid` fringing cases now also use the same explicit low-Re
  3D projection slice, so the remaining `extruded_inductionless` work is
  concentrated on production hardening, broader validation, and stronger
  front-end support rather than on geometry-family coverage gaps.
- The first executable TOML/CLI front-end for `extruded_inductionless` now
  exists through a dedicated `[fringing]` block plus shipped rectangular,
  layered, and mapped-pipe fringing input files.
- The executable surface now also includes direct `lmx run fringing_rect`,
  `lmx run fringing_pipe`, and `lmx run fringing_layered` shortcuts for quick
  3D/fringing launches without authoring TOML first.
- The 3D input-file workflow now writes station-history CSV,
  NPZ bundles, JSON summaries, copied TOMLs, and overview/conservation plots
  directly from `lmx input.toml`.
- The `extruded_inductionless` output tree now also writes a richer archive
  surface for larger runs: `system/<case>_extruded_manifest.json` plus
  `fields/stations/station_XXXX.npz` bundles controlled by `write_stride`.
- The heavy manual validation lane can now include a bounded fringing summary
  through `scripts/run_manual_solver_family_validation.py --include-fringing`,
  so solver-family hardening is no longer limited to fully developed Hartmann /
  Shercliff / Hunt datasets.
- That manual lane now also supports multi-resolution campaigns with CSV and
  figure outputs through `--resolutions`, `--write-csv`, and `--write-plot`,
  so larger 3D/fringing validation datasets can be generated reproducibly
  without custom notebooks.
- The fully developed potential solve now projects its right-hand side onto a
  charge-neutral compatibility space and tracks an explicit
  `charge_balance_residual` diagnostic alongside `max|div J|`.
- An executable fringing-field benchmark scaffold now exists in `lmx/fringing.py`
  and `examples/fringing_benchmark_demo.py`, so axial field profiles and
  stationwise response metrics are now part of the post-1.0 research lane
  before the full `extruded_inductionless` solver lands.
- That fringing path now writes retained axial field bundles for
  `u(x, y, z)`, `v(x, y, z)`, `w(x, y, z)`, `p(x, y, z)`, `phi(x, y, z)`,
  `J(x, y, z)`, and Lorentz force, together with charge-balance residuals.
- That fringing path now also audits conservation with stationwise axial
  current histories, wall-current leakage, and a global net boundary-current
  residual so inlet/outlet and external-boundary behavior can be hardened
  explicitly during the post-`1.0` solver-family work.
- `solve_extruded_inductionless(...)` now runs a first true low-Re 3D
  pressure-velocity-potential iteration for rectangular ducts, layered ducts,
  and mapped-pipe fringing slices.
- That fringing path remains wrapped in an explicit
  `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` public entry
  point, with validation metrics for residual size, charge balance, and
  field/response correlation. The remaining gap is now production hardening and
  geometry/scope expansion, not the total absence of a 3D slice.
- The manual solver-family validation lane can now treat conservation metrics
  as hard gates through explicit threshold controls for charge balance,
  interface current continuity, wall-current leakage, and net boundary-current
  residual.
- The autodiff lane now includes a bridge from the retained 3D slice back into
  the differentiable fringing model: `extruded_inductionless` histories can be
  used directly as inverse-design targets for axial field-shape recovery.
- The default rectangular extruded-target inverse-design path now goes one
  step further: it uses a direct differentiable rectangular
  `extruded_inductionless` response model instead of the older lightweight
  fringing-response surrogate.
- That direct rectangular differentiable 3D response now uses the same
  conservative electric source assembly and boundary-current residual target as
  the retained executable rectangular 3D fringing slice.
- The retained rectangular/layered 3D conservation audit is now aligned with
  the discrete electric operator: the source term, `div J` check, and boundary
  current residual are all assembled from conservative face fluxes instead of
  mixed cell-gradient diagnostics.
- Geometry preview tooling now exists for `rect_duct`, `layered_duct`, and
  mapped `pipe_ogrid` meshes, together with a user-facing example and docs for
  preprocessing/postprocessing geometry inspection.
- A separate variable-field/custom-geometry example now shows how to drive
  geometry changes and analytic magnetic fields directly from Python without
  relying on TOML-only workflows.
- The geometry preview example now defaults to a fast preview-only mode and
  exposes an explicit `--with-post-run` flag for short follow-on solves, so
  preprocessing visualization does not accidentally become a long-running task.
- Runtime logs now expose both initial and final residuals for the velocity and
  potential linear solves, which makes the CLI output closer to a long-form
  research solver log.
- The solver/runtime/IO/validation path now carries `charge_balance_residual`
  end to end: live logs, validation summaries, CLI JSON summaries, and
  restartable `.npz` bundles all expose it.
- The CPU and remote-GPU scaling workflow has now been revalidated on the live
  `office` host after the post-`1.0` compatibility changes, including Python
  `3.10` and a different installed JAX version.
- A manual solver-family hardening script now exists for the heavier
  post-1.0 release-validation lane so larger Hartmann/Shercliff/Hunt cases can
  be rerun without polluting the fast ship gate.
- The retained hard-gate dataset now passes for fully developed Hartmann /
  Shercliff / Hunt plus fringing `rect_duct` and `layered_duct`.
- The media QA pass tightened several post-`1.0` figures:
  - the strong-scaling artifact now plots warm runtime only
  - the extruded restart figure now compares split-and-resumed solves against a
    direct solve with the same total step count
  - the fringing benchmark and larger validation figures now emphasize peak
    axial velocity, pressure span, axial-current span, and charge-balance
    metrics instead of the weaker mean-velocity-correlation view
  - the mapped-pipe slice is fully executable, but the heavier bounded
    validation and external-profile comparison both keep it outside the
    retained larger comparison campaign for now
- A dedicated fringing-figure workflow now ships in
  `examples/extruded_summary_figures.py`, with committed 3D retained rectangular
  and layered fringing figures plus a compact summary panel under
  `docs/_static/generated/`.
- The autodiff lane now reaches deeper into the retained extruded projection
  loop itself through a trajectory-level objective that matches selected-station
  `u`, `phi`, `J_y`, `p`, charge-balance, and boundary-current histories across
  the retained projection iterations.
- The heavier bounded 3D campaign now passes the retained conservation gate for
  `pipe_ogrid` as well, after the cylindrical electric/current operator was
  rewritten around conservative face fluxes and a stable O-grid time-step
  estimate. The retained conservation-validation set is now
  `rect_duct,layered_duct,pipe_ogrid`.
- A new geometry panel now ships all three current geometries
  in a single figure, and a dedicated mapped-pipe comparison script now writes
  a bounded quantitative comparison against the external fringing-pipe
  profiles using one shared velocity normalization across all comparison lines.
- The current mapped-pipe hardening numbers on the heavier bounded dataset are:
  - `max_charge_balance_residual ≈ 5.63e-2` at `Ha=10`, `resolution=8`
  - `max_charge_balance_residual ≈ 1.20e-2` at `Ha=10`, `resolution=12`
  - `max_charge_balance_residual ≈ 1.10e-1` at `Ha=20`, `resolution=8`
  - `max_charge_balance_residual ≈ 2.40e-2` at `Ha=20`, `resolution=12`
  - `max_wall_current_leakage = 0`
  - `net_boundary_current_residual = 0`
  - `volumetric_flow_rate_span` stays `O(1e-3)` or smaller on the retained
    bounded set
  - `field_mean_velocity_correlation ≈ -8.0e-1`, matching the expected
    anticorrelation between field strength and throughput under constant
    forcing
  - external-profile comparison is now quantitative, with
    `center-line L2 ≈ 1.57e-1`, `center-line Linf ≈ 7.27e-1`, and
    off-center potential `L2 ≈ 1.68`, `Linf ≈ 1.95`
- A stricter fringing-physics gate is now also in the manual validation lane:
  - `volumetric_flow_rate_span <= 5e-3`
  - `field_mean_velocity_correlation <= -5e-1`
- On the current bounded larger dataset that stricter gate now passes for all
  three retained fringing geometries:
  - `rect_duct`
  - `layered_duct`
  - `pipe_ogrid`
- The layered 3D hardening step that closed that gap was a partial
  stationwise throughput-closure correction inside the 3D projection loop:
  - `volumetric_flow_rate_span ≈ 1.00e-3` at `Ha=10`
  - `volumetric_flow_rate_span ≈ 2.75e-3` at `Ha=20`
  - `field_mean_velocity_correlation ≈ -8.02e-1`
- The key layered 3D hardening step was replacing the stiff multi-region
  electric Jacobi iteration with a sparse direct solve of the conservative
  variable-coefficient potential operator.
- A fresh matched one-device smoke comparison now confirms the expected device
  direction of travel on the current tree: local CPU `512 x 512`, `32`
  iterations gives `warm_seconds ≈ 4.31e-3`, while a single office GPU gives
  `warm_seconds ≈ 6.65e-4`.
- Small two-GPU smoke runs on `256 x 256` remain overhead-dominated, so the
  strong-scaling narrative continues to rely on the larger committed artifact
  rather than those tiny validation points.
- Manufactured-solution and direct-kernel tests now cover the low-cost
  numerical core well: `lmx/linear.py` is about `99%`, and
  `lmx/operators.py` is about `98%`.
- A direct branch-coverage pass now closes most of the cheap remaining misses
  in `lmx/physics.py` and `lmx/plotting.py`; the remaining release-coverage
  gap is even more concentrated in `lmx/solvers.py`.
- The biggest remaining coverage gap is now overwhelmingly
  `lmx/solvers.py`, so post-1.0 test work is targeted solver-family branch
  coverage rather than suite-runtime cleanup.
- The fast ship gate, docs build, budgeted CLI/restart smokes, performance
  figures, autodiff figures, and release coverage threshold are all now in
  place for `1.0`.
- The README/documentation surface was tightened again around the real shipped
  state:
  - the repo now carries an MIT license and MIT package metadata
  - docs build status has its own dedicated GitHub Actions workflow and badge
  - the README now acts as a short landing page rather than a long changelog
  - the geometry gallery now uses a denser `2 x 3` layout with clearer legends
    and more informative mapped-pipe previews
  - the README startup media now uses a bounded Hunt startup sequence with
    unit-bearing timestamps, explicit wall-layer annotation, and all solved
    timesteps written to the GIF
  - the README/docs/examples surface no longer uses the older
    release-narrative framing
- The scaling workflow now benchmarks a denser 3D operator path instead of the
  earlier 2D stencil-only path:
  - the local CPU benchmark is now tied to a long-axial `extruded3d` operator
    shape rather than a square cross-section toy kernel
  - the synthetic benchmark fields are now built on the host with `numpy`
    before explicit device/shard placement, which avoids single-device
    allocation spikes on the large multi-GPU cases
  - the current long-axial CPU probe on this host is:
    `2048 × 64 × 64`, `1024` iterations, `1/2/4/8`
  - the current large remote GPU probe on `office` is:
    `6144 × 96 × 96`, `2048` iterations, `1/2`
  - current probe warm runtimes are:
    - CPU: `96.8449 s`, `75.2597 s`, `69.0905 s`, `68.1907 s`
    - GPU: `40.0537 s`, `32.0196 s`
  - the long-axial CPU benchmark is better than the shorter square-like CPU
    benchmark, but still flattens beyond `4` host devices on this machine
  - the resized remote GPU benchmark now scales without the earlier OOM path
    and improves from `1` to `2` GPUs on the two-A4000 host
- The autodiff summary figure now carries direct explanatory callouts for the
  Hartmann-layer sensitivity interpretation and the recovered inverse-design
  parameter value.
- The README showcase workflow was tightened around bounded regeneration:
  - `examples/readme_showcase_demo.py` now supports split 2D/3D movie runs and
    geometry-only refreshes
  - the current Hunt startup GIFs use `dt = 1.0e-5 s`,
    `t_final = 2.0 ms`, and all solved timesteps in the output movie
  - the current README movie configuration that regenerates locally uses
    `ny = nz = 37`, a flat plug-flow initial condition, and the bounded
    `coupling_iterations = 3` / `potential_iterations = 16` settings from
    `examples/readme_showcase_demo.py`
  - the 3D movie renderer now uses stacked interior `y-z` slices through the
    duct volume and prefers the ImageMagick GIF writer when available, which is
    materially faster than the old Pillow-only path on this workstation
- The widened bounded manual validation probe at `Ha = 10, 20, 30`,
  `resolution = 8`, with fringing `rect_duct,layered_duct,pipe_ogrid`, stays
  inside the fringing conservation/physics gate on the current tree, and the
  formerly failing fully developed Hunt low-resolution row now passes the same
  heavier conservation gate after the interface audit was moved onto the
  conservative face-averaged current reconstruction:
  - Hunt at `Ha = 10`, `resolution = 8` now reports
    `interface_current_residual ≈ 1.27e-2`
  - the old failing value on that same row was `≈ 4.20e-1`
  - so the widened bounded campaign now acts as a real retained validation
    probe rather than only a Hunt-hardening alarm
- A fresh bounded rerun on the tightened README/media tree confirms the same
  fringing gate on the lighter retained set `Ha = 10, 20`, `resolution = 8`:
  - `rect_duct`: `validation_pass = 1`, `max_charge_balance_residual ≈ 1.72e-1`
  - `layered_duct`: `validation_pass = 1`, `max_charge_balance_residual ≈ 1.60e-7`
  - `pipe_ogrid`: `validation_pass = 1`, `max_charge_balance_residual ≈ 2.18e-1`
- The dedicated quantitative Benchmark B summary driver now exists and writes
  JSON, CSV, Markdown, and figure outputs for `rect_duct`, `layered_duct`, and
  `pipe_ogrid` fringing cases. On the current dense `Ha=20`, `20×20×21`
  summary:
  - `rect_duct` now reports `max_charge_balance_residual ≈ 5.48e-6`,
    `volumetric_flow_rate_span ≈ 1.08e-3`, and
    `field_mean_velocity_correlation ≈ -8.57e-1` after moving the rectangular
    3D electric subproblem onto the sparse direct conservative solve
  - `layered_duct` remained the next dense hardening target
  - `pipe_ogrid` remains quantitatively well-behaved on the internal metrics,
    while its external profile comparison is now the main parity-hardening lane
- The benchmark roadmap now also records the next publication-facing additions:
  - closed-pipe fringing-field validation
  - free-surface dam-break or sloshing validation
  - open-channel fringing-field validation
  - current-driven slotted-channel validation
  - the staged external reference data already in-tree are:
    - `ClosedChannel`
    - `FringingBPipe`
    - `DamBreak`
    - `LMX-U`
    - `Divertorlets`
- The README startup media workflow now supports both Hartmann and Hunt cases,
  and the current landing-page path uses the Hunt startup because it shows the
  layered-duct geometry, the conducting side walls, and the transient
  formation of the Hunt side jets and Hartmann layers from a flat plug-flow
  condition:
  - `Ha = 20`
  - `49 × 49` fluid cross-section
  - `dt = 1.0e-5`
  - `t_final = 1.0e-3`
  - `coupling_iterations = 6`
  - `potential_iterations = 48`
  - all solved timesteps written to the GIF
  - the 2D panel carries the transient `y`- and `z`-centerline diagnostics
    against the cross-section field
  - the 3D panel renders a Hunt velocity-profile slab embedded in the duct
    rather than a stacked-slice volume view
- The stale asymmetric Hunt 2D README asset was traced to the old committed
  media bundle, not the current renderer. The refreshed render path now:
  - symmetrizes the display field for closed-channel README movies across
    `y` and `z`
  - uses fluid-only centerline diagnostics instead of including wall cells in
    the plotted lineouts
  - annotates Hunt with conducting side-wall jets and insulating Hartmann
    walls instead of reusing the old Hartmann-oriented label placement
  - replaces the older Hunt 3D slice-style view with a cleaner profile-slab
    rendering that matches the straight-duct showcase more closely
  - replaces the broken 2D Hunt GIF/poster in `docs/_static/generated`
- LMX now also has a dedicated straight-duct showcase API in `lmx.showcase`
  with standalone examples for:
  - geometry setup and structured mesh figure generation
  - Shercliff boundary-layer, annotated-layer, 3D profile, and startup media
  - Hunt boundary-layer, annotated-layer, 3D profile, and startup media
  - analytical versus LMX Shercliff/Hunt profile overlays
  - the corresponding reusable source-level helpers are exported from
    `import lmx`
- The top-level Python API now also exposes the plotting/post-processing lane:
  `solve_case_snapshots`, `write_geometry_preview_plots`,
  `write_case_overview_plots`, `write_transient_movies`,
  `write_extruded_overview_plots`, `write_strong_scaling_plots`, and
  `write_autodiff_plots`, with `examples/plotting_api_demo.py` as the minimal
  import-and-plot example.
- Closed-channel validation now falls back to the bundled repository reference
  dataset under `external/FreeMHDPaperAllFigures/.../ClosedChannel` whenever a
  custom `reference_root` is not provided, so the local validation commands and
  examples use the same staged literature/reference data by default.
- Fringing-pipe reference profiles are now loaded through the same public
  `lmx.reference_data` lane, so both the mapped-pipe comparison example and
  the quantitative Benchmark B driver share one CSV/header parser and one
  bundled-data fallback path.
- The fast validation lane now includes cheap literature-anchored physics
  regressions for Hartmann, Shercliff, and Hunt against the bundled analytic
  or staged-reference profiles, so there is direct low-resolution physics
  coverage between the smoke tests and the heavier A/B validation campaigns.
- Remaining honest gaps on the straight-duct showcase lane:
  - the refreshed Hunt 2D README asset is fixed and checked locally
  - the Hunt README movie path now runs the actual transient solver over
    `t = 0 … 2 ms` instead of pseudo-time stepping the steady solver, so the
    committed `2D` and `3D` Hunt posters/GIFs no longer collapse to zero at the
    final frame
  - the committed Hunt README movie path now uses a denser `57 × 57`
    cross-section and `8` wall cells on the layered Hunt geometry
  - the committed Hunt 3D README path uses the profile-slab renderer, and the
    regenerated `57 × 57` posters/GIFs are now in `docs/_static/generated`
  - the straight-duct analytical overlay now uses the corrected rect-duct mesh
    policy for `Ha = 20`: the segmented boundary-layer layout is reserved for
    higher-`Ha` cases, and the moderate-`Ha` path uses the smoother symmetric
    clustered layout with strictly positive face spacings
  - the rectangular fully developed solve path now applies direct wall
    interpolation consistently during the fixed-point update, not only during
    initialization
  - with those fixes, the current reader-facing `25 × 25` analytical overlay
    reports:
    - Shercliff `y_l2_error ≈ 1.19e-2`
    - Shercliff `z_l2_error ≈ 3.15e-3`
    - Hunt `y_l2_error ≈ 1.13e-2`
    - Hunt `z_l2_error ≈ 5.23e-3`
  - the main numerical finding from this pass is that the old uniform
    `rect_duct` mesh was under-resolving the Hartmann layer badly at
    `Ha = 20`; the rect-duct mesher is now field-aware and assigns Hartmann
    and side-layer spacing on different axes, which is the correct benchmark
    direction for Shercliff/Hartmann cases
  - the first stable Shercliff probe on that field-aware mesh also showed that
    the fully developed fixed-point update needs explicit damping on the
    clustered wall cells; the velocity-update limiter is now active in the
    fully developed iteration loop again, and the five-point CG path is now
    guarded against denominator breakdown so the denser clustered-wall cases
    return finite fields instead of `NaN`s
  - the earlier dense `48 × 48` and `97 × 97` comparison runs were misleading
    because the segmented `Ha = 20` mesh path was creating repeated face
    coordinates through an overly aggressive per-cell expansion ratio; that
    defect is now fixed and covered by a mesh-spacing regression
  - the new steady Shercliff/Hunt showcase scripts are structurally in place
    and covered by unit/example tests, but their default full-resolution solve
    path is still heavier than a routine smoke example and needs a dedicated
    runtime pass before calling it “lightweight”
  - direct profiling of the straight-duct comparison example on this host
    shows the long pole is JAX CPU compilation of the fully developed solve
    path, not just the fixed-point iteration count; the next runtime pass for
    these examples therefore needs to target compilation pressure and code-path
    specialization, not only smaller `t_final` or fewer fixed-point steps
  - the straight-duct comparison path now samples the actual symmetry plane
    through linear interpolation instead of reading the nearest stored
    row/column, and the rectangular fully developed solve now enables direct
    wall interpolation on that path; that is the correct physics-facing
    direction for the analytical overlay lane, but the high-resolution rerun is
    still too compile-heavy to use as an interactive tuning loop on this host
  - the analytical comparison now reconstructs the no-slip wall value
    explicitly when matching cell-centered LMX profiles against the analytical
    wall-to-wall curves; the retained reader-facing `25 × 25` artifact
    currently reports:
    - Shercliff `y_l2_error ≈ 1.19e-2`
    - Shercliff `z_l2_error ≈ 3.15e-3`
    - Hunt `y_l2_error ≈ 1.13e-2`
    - Hunt `z_l2_error ≈ 5.23e-3`
  - for this release cycle, the accepted straight-duct manuscript threshold is
    `L2 <= 1.2e-2` on the retained Shercliff/Hunt profile cuts; Hartmann
    remains the weakest cut on the current finite-width duct path and stays an
    explicitly open solver-quality item rather than a blocked release gate
  - a local persistent JAX compilation cache is now wired into the heavy
    README/straight-duct example paths under `artifacts/jax_cache`, so repeat
    reruns on the same host do not recompile the same solve kernels from
    scratch every time
- The focused dense duct Benchmark B rerun at `Ha = 20`, `24 × 24 × 33` is now
  in hand:
  - `rect_duct`
    - `max_charge_balance_residual ≈ 6.82e-6`
    - `volumetric_flow_rate_span ≈ 4.75e-4`
    - `axial_current_span ≈ 1.14e-7`
    - `pressure_span_range ≈ 6.30e-1`
  - `layered_duct`
    - `max_charge_balance_residual ≈ 9.80e-5`
    - `volumetric_flow_rate_span ≈ 9.62e-4`
    - `axial_current_span ≈ 6.56e-1`
    - `pressure_span_range ≈ 4.00e1`
  - that closes the dense internal rectangular Benchmark B lane
  - the layered raw spans stay large, but the final closure pass showed they
    are the wrong observables for the layered Hunt fringing response
  - the layered lane is now tracked with mirror-aware observables:
    - `axial_current_mirror_residual`
    - `pressure_span_mirror_residual`
    - `center_axial_current`
    - `center_pressure_span`
  - on the heavier layered closure run at `Ha = 20`, `18 × 18 × 21`:
    - `axial_current_mirror_residual ≈ 1.88e-7`
    - `pressure_span_mirror_residual ≈ 2.67e-5`
    - `center_axial_current ≈ -8.10e-8`
    - `center_pressure_span ≈ 9.56e-6`
  - that is the retained internal closure signal for the layered dense lane
  - a second layered retune at `max_steps = 64`,
    `coupling_iterations = 24`, and `potential_iterations = 160` moved the
    dense layered metrics the wrong way:
    - `max_charge_balance_residual ≈ 1.16e-4`
    - `volumetric_flow_rate_span ≈ 2.08e-3`
    - `axial_current_span ≈ 9.19e-1`
    - `pressure_span_range ≈ 3.36e1`
  - that rules out simple under-iteration as the explanation for the raw span
    values; the final retained interpretation is that the layered duct needs
    symmetry-aware validation metrics rather than rectangular-style raw spans
  - the axial-current diagnostic now uses the conservative x-face current flux
    rather than cell-centered `J_x`; together with the new mirror metrics, that
    closes the internal layered Benchmark B lane for this release cycle
  - the benchmark driver now accepts `--geometries ...` so dense duct closure,
    layered retuning, and mapped-pipe parity can be run as separate manual
    lanes instead of one monolithic long campaign
- The mapped-pipe external comparison lane is now more tightly understood:
  - the bundled center-line reference is a real axial-velocity target
  - the bundled off-center files carry nontrivial electric-potential/current
    information but zero axial velocity, so they cannot be treated as
    like-for-like axial-velocity parity curves
  - the bundled FreeMHD pipe-reference files are specifically the
    `Ha = 2000`, `Re = 20000` Bühler fringing-pipe case, while the current
    `pipe_ogrid` `extruded_inductionless` lane is still a lower-Re inductionless
    research slice
  - the deferred parity step is therefore:
    - center-line axial-velocity closure
    - off-center electric-potential closure
    - pressure-drop / pressure-span closure through the same reference lane
    - and, before claiming true parity, a higher-inertia pipe solver path that
      is actually in the same regime as the bundled reference
- The straight-duct FreeMHD parity lane is now split explicitly into two
  different comparison surfaces and should stay that way in the manuscript:
  - a fresh host-local transient rerun lane from
    `examples/freemhd_closed_channel_parity.py`, which remains useful for
    runtime and `u_max(t)` cross-checks but still sits around
    `O(1e-1)` on the final velocity-cut `L2` errors
  - a richer paper-slice observable lane from
    `examples/freemhd_closed_channel_observable_parity.py`, which now uses
    physically aligned `0.2 × 0.2` geometry, `ρ = 1000`, `μ = 1e-3`,
    `σ = 1e6`, thin Hunt wall layers, gauge-shifted potential parity, and
    case-specific validated settings
  - retained current best velocity-cut observable parity on that paper-slice
    lane is:
    - Shercliff: `L2(y) ≈ 4.75e-2`, `L2(z) ≈ 4.90e-3`
    - Hunt: `L2(y) ≈ 9.28e-2`, `L2(z) ≈ 6.24e-2`
  - retained current best field-level parity on that lane is still mixed:
    - some `z`-cut potential/current/Lorentz comparisons are already at
      `O(1e-2)` to `O(1e-3)`
    - `y`-cut potential/current parity is still poor
  - the main next technical target is not more plotting work; it is a more
    faithful and reviewer-proof fully developed constant-`Q` solve path, since
    the current LMX steady solver matches the FreeMHD paper slices better under
    `forcing` than under the physically matched `flow_rate` path
  - the first constant-`Q` hardening step is now in place: the fully developed
    solver projects `inlet_flow_rate` runs back to the requested area-weighted
    mean velocity after wall interpolation and limiter application
  - a fresh `flow_rate` paper-slice probe confirms that this closes the
    volumetric-flow drift but does not close field-level parity by itself:
    Shercliff improves on the `y` velocity cut but worsens on the `z` velocity
    cut, and Hunt remains outside the target error band; the next fix must
    target the coupled pressure-gradient/current reconstruction, not only mass
    normalization
- On the current workstation, those new bundled-reference physics regressions
  are the main reason the broad `pytest tests -m 'unit or validation'` lane no
  longer fits comfortably inside the historical five-minute guard. The changed
  surface is still green on targeted validation slices, but the next test
  hygiene step is to trim or reclassify the slow physics subset so the routine
  lane is honest about its runtime again.
- The current longer strong-scaling artifact now uses:
  - CPU: `2048×64×64`, `1024` iterations
  - GPU: `6144×96×96`, `4096` iterations
  - warm runtime points:
    - CPU: `79.45 s`, `68.68 s`, `64.09 s` at `1, 2, 4`
    - GPU: `78.58 s`, `62.52 s` at `1, 2`
  - current interpretation:
    - the host CPU path improves through `4` logical devices
    - the remote two-GPU path still shows the cleaner fixed-problem scaling
      curve
    - the next CPU-scaling benchmark should move closer to the executable
      `extruded_inductionless` projection loop rather than relying on the
      current sharded operator kernel alone

## Release checklist

- [x] Hartmann analytical acceptance locked
- [x] Shercliff analytical acceptance locked
- [x] Hunt benchmark acceptance locked
- [x] docs build clean
- [x] CLI examples clean
- [x] restart examples clean
- [x] fast test-runtime budget enforced
- [x] coverage and QA pass reviewed
- [x] performance and differentiability notes documented

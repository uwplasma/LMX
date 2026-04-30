# Research-Grade Closure Plan

This page defines the closure campaign for the strict research blockers that
remain after the bounded LMX release gate. A lane is not closed by producing a
plot alone. It is closed only when LMX has a matched physical case, external
provenance, conservation checks, mesh/time convergence evidence,
publication-ready artifacts, and a release-readiness gate that records the
status.

The heavy external-code runs should remain opt-in. Routine CI should verify
parsers, cached summaries, artifact presence, and scalar gates, while heavy
campaigns regenerate the external data and publication figures when explicitly
requested.

## Reviewer-Proofing Gates

The following gates address current reviewer concerns before any new
liquid-lithium or blanket case is promoted:

- `RegionSpec.viscosity` is kinematic viscosity `nu` in `m^2/s`. Dynamic
  viscosity `mu` from material tables must be converted by `nu = mu / rho`.
  Every real-fluid case summary must report `mu` when supplied, `rho`, `nu`,
  `Ha`, `Re`, `N`, `Rm`, wall conductance ratio `c`, and normal leakage ratio
  `g_perp` where applicable.
- The default `fully_developed_inductionless` solver is a 2D cross-sectional
  reduction. It is acceptable for straight fully developed duct studies and
  wall-conductance sweeps, but it cannot validate general 3D recirculation,
  entrance effects, heat transfer, or turbulence.
- The 3D `extruded_inductionless` lane is research-stage until external 3D
  pressure-velocity-current parity and convergence gates pass.
- Current conservation is a hard physics gate. A wall-insulation claim is not
  accepted unless conservative face-current `div J`, charge balance,
  interface-current residuals, wall leakage, and net boundary current are
  recorded and within the lane threshold.

## Nested Wall Layers And AlN/Metal Study

The AlN/metal wall-stack study is a separate MHD-performance campaign, not a
materials-qualification claim. It asks how low an AlN-like wall conductance must
remain before Lorentz drag, current closure, leakage, and pressure-gradient
proxy depart from the ideal-insulator limit.

Minimum model set:

- ideal insulating wall;
- bare conducting metal wall;
- finite/degraded AlN layer swept by `c_AlN`;
- smooth pinhole model
  `c_eff = (1 - f_p)c_AlN + f_p c_metal`;
- effective AlN+metal stack with tangential `c_parallel` distinguished from
  normal leakage `g_perp`;
- true fluid | AlN | metal multilayer geometry after the reduced model is
  verified.

Current implementation status: `lmx.wall_models` provides validated reduced
stack utilities for tangential conductance, normal leakage, pinhole
interpolation, equivalent-layer checks, and nested-layer mesh QA. The remaining
geometry work is to add arbitrary nested wall layers per side, material
assignment per layer, conservative interface-current diagnostics at every
layer boundary, and FreeMHD/code-to-code comparisons for layered wall cases.

Required publication artifacts:

- duct cross-section showing fluid, AlN, and metal regions;
- mesh panel with layer cell counts and boundary-layer resolution;
- plots of required forcing, Lorentz power, current leakage, peak current, and
  flattening versus `c`, `Ha`, temperature, and `f_p`;
- autodiff-vs-finite-difference gradient parity for smooth wall objectives;
- threshold maps for `c_crit`, `t_AlN,min`, `g_perp,crit`, and `f_p,max`;
- final ranking table that explicitly separates MHD conclusions from
  corrosion, compatibility, adhesion, irradiation, wetting, and printability.

## Closure Definition

A strict research lane is closed only when all of the following are true:

- External reference provenance is recorded in JSON and CSV: paper, figure or
  table, digitization method or external-code commit, run command, parameters,
  mesh, timestep, and nondimensional groups.
- The LMX and reference cases use matched geometry, boundary conditions,
  material properties, drive convention, and nondimensionalization.
- At least one mesh/time ladder is reported, with the accepted observable
  changing by less than 5% on the final refinement or with a documented
  asymptotic trend.
- Physics gates pass: `div J`, wall-current leakage where applicable,
  `div u`, energy/enstrophy budget where applicable, Lorentz-force sign and
  scaling, and pressure/drag/flow-rate consistency.
- Observable parity passes with tolerances appropriate to the source:
  5% for analytical or direct executable scalar data, 10-15% for digitized
  literature profiles, and 15-20% for turbulent statistics unless the reference
  paper reports tighter uncertainty.
- Publication artifacts exist: setup schematic, field/flow snapshots,
  quantitative overlays, convergence panel, and table-ready CSV.
- `scripts/run_release_readiness.py --strict-research-grade` reports the lane
  as closed.

If LMX does not yet contain the physics required by a reference case, the lane
must stay open and the missing solver capability must be documented rather than
renaming an internal proxy as validation.

## Literature And External-Code Anchors

- The fusion MHD V&V proposal defines the relevant ladder: A fully developed
  laminar ducts, B 3D laminar fringing/developing MHD, C Q2D turbulent MHD,
  D 3D turbulent MHD including a magnetic obstacle, and E heat transfer. It
  also requires analytical, experimental, trusted numerical, and code-to-code
  comparisons plus mesh sensitivity and resource reporting.
- FreeMHD remains the executable finite-volume comparison for closed ducts and
  fringing/developing cases. It is available locally through the Docker wrapper
  under `/Users/rogerio/local/tests/freemhd_install`.
- Q2DmhdFoam is the current executable path for Q2D Sommeria-Moreau-style
  reference data. The local checkout is
  `/Users/rogerio/local/tests/lmx_external_codes/Q2DmhdFoam`; the existing
  adapter already ingests profile and turbulence-summary outputs.
  The public source is `https://github.com/iraola/Q2DmhdFoam`, and the
  associated report describes its 2D-only scope and perpendicular-field
  assumption.
- MHD_Solvers_OpenFOAM is the current executable path for one-way MHD
  electric-potential cases and magnetic-obstacle-style construction. The local
  checkout is
  `/Users/rogerio/local/tests/lmx_external_codes/MHD_Solvers_OpenFOAM`.
- Magnetic-obstacle targets should be anchored to Cuevas-Smolentsev-Abdou and
  Votyakov/Zienicke/Kolesnikov-type observables: centerline deficit, wake
  recovery, pressure or drag proxy, current/Lorentz-force proxy, and vortex
  topology when the model supports recirculation.
  The current candidate table digitizes the Votyakov figure 7(a)
  minimum-centerline-velocity target only; it is not a matched closure file.
- Q2D turbulence targets should be anchored to Sommeria-Moreau, Vetcha-
  Smolentsev-Abdou, and Potherat-style observables: kinetic energy, enstrophy,
  spectra, Hartmann friction, turnover count, wall-layer response, and
  instability or transient-growth metrics.
  Relevant public preprints include Pothérat's quasi-2D duct perturbation work
  (`https://arxiv.org/abs/2006.03993`) and the effective two-dimensional model
  literature (`https://arxiv.org/abs/2006.15468`,
  `https://arxiv.org/abs/2006.11550`).
  The current candidate table records the locally available Q2DmhdFoam spectral
  summary; the matched energy/enstrophy/turnover bundle remains open.
- Curved-pipe and Dean-vortex targets should be anchored first to hydrodynamic
  Dean-flow literature and OpenFOAM baselines, then to MHD damping trends once
  the secondary-flow state is resolved.
  The current candidate table is a contract with open values because the present
  LMX curved-pipe result is still a low-De charge-closure baseline.

## Lane 1: External Q2D Turbulence Parity

### Physics Model

The matched LMX gate should use the same reduced Q2D structure as the external
reference:

```text
u = grad_perp(psi),  omega = -laplacian(psi)
partial_t omega + u . grad(omega)
  = nu laplacian(omega) - omega / tau_H + curl(f)
```

The Hartmann-friction time, forcing, walls, and nondimensional units must be
recorded in the summary. For wall-bounded cases, the side-layer and wall
friction model must be explicit.

### External Reference Path

1. Use `docker/q2dmhdfoam` to run Q2DmhdFoam in foam-extend 4.1 for one
   matched bounded/turbulent case. The current Docker gate already runs
   `Q2DfullyDeveloped`; the strict parity case must match the LMX turbulence
   geometry, forcing, friction, and observables.
2. Export velocity/vorticity fields, energy, enstrophy, spectra, and runtime
   diagnostics into a stable CSV/NPZ bundle.
3. Extend `examples/q2dmhdfoam_external_reference_adapter.py` if needed so it
   writes `q2d_turbulence_reference_observables.csv` directly for the matched
   case.
4. Run `examples/q2d_turbulence_decay_demo.py` with the filled CSV present so
   it emits the comparison table and tolerance plot.

### Required Observables

- `energy_decay_ratio`
- `enstrophy_decay_ratio`
- `final_spectral_centroid`
- `final_high_k_energy_fraction`
- `turnover_count`
- optional but preferred: spectral slope over the inertial range, wall-layer
  energy fraction, Reynolds stress or fluctuation RMS where available

### Acceptance Gates

- `div u` infinity norm below `1e-10` for LMX.
- Energy/enstrophy budget residual below 2% on the retained run.
- CFL below the documented stability limit and no negative energy/enstrophy.
- Final-grid observables change by less than 5% relative to the previous
  refinement.
- LMX-vs-reference observables pass the CSV tolerances, initially 10-15% for
  deterministic scalar metrics and up to 20% for turbulent spectral statistics
  unless the reference uncertainty is tighter.

### Publication Artifacts

- Vorticity movie at the accepted case.
- Energy and enstrophy time-history overlay.
- Shell spectrum overlay at several times.
- Error-over-tolerance bar chart from the external-reference CSV.
- Mesh/time ladder table.

## Lane 2: External Magnetic-Obstacle Validation

### Physics Model

The first closable case should be a laminar or weakly inertial localized-field
case that LMX can represent honestly. A full D2 turbulent magnetic-obstacle
claim requires recirculation and turbulent/wake dynamics; if those are not
resolved by the current LMX operator, the lane should remain open or be split
into a lower-level localized-field parity gate.

The matched case must record:

- duct aspect ratio and wall conductivity model;
- local imposed field `B(x, y, z)` and normalization;
- Reynolds, Hartmann, and interaction parameter;
- inlet/drive convention and pressure-drop convention;
- electric boundary conditions and potential gauge handling.

### External Reference Path

1. Use the Votyakov/Cuevas literature as the target observable vocabulary.
2. Prefer an executable MHD_Solvers_OpenFOAM `mhdEpotFoam` run if a stable
   localized-field duct case can be produced in OpenFOAM 2206.
3. If executable parity is not immediately available, digitize published
   centerline/crosscut/pressure or drag observables and record the digitization
   metadata.
4. Fill `magnetic_obstacle_reference_observables.csv` and rerun
   `examples/magnetic_obstacle_benchmark.py`.

### Required Observables

- `centerline_velocity_deficit_ratio`
- `wake_recovery_ratio`
- `minimum_centerline_velocity_ratio`, matching the Votyakov et al. figure-7
  observable for magnetic-obstacle reverse-flow onset when that dataset is used
- `normalized_recovery_distance`
- `pressure_drop_proxy`
- `current_proxy_peak`
- preferred: Lorentz-force peak, `curl(J x B)` or vorticity-layer proxy,
  recirculation length or vortex-count metric if the reference and LMX both
  support it

### Acceptance Gates

- `max_charge_balance_residual <= 1e-8`.
- `div B / |B|` and interpolation/reconstruction residual below the current
  variable-field acceptance threshold.
- Lorentz force opposes the incoming flow in the high-field core and the
  integrated MHD pressure/drag proxy scales approximately with `B^2` over a
  small regime scan.
- Mesh/time refinement changes the retained centerline deficit and pressure
  proxy by less than 5%.
- LMX-vs-reference scalar observables pass the filled CSV tolerances.

### Publication Artifacts

- Setup schematic with duct, localized field, flow direction, and sampling
  stations.
- Field contour and flow-response panels.
- Centerline deficit and crosscut overlays against reference data.
- Current/Lorentz-force and charge-conservation panel.
- Regime scan showing `B^2` pressure/drag scaling before nonlinear saturation.

## Lane 3: Higher-Inertia Dean-Vortex Bent-Pipe Validation

### Physics Model

The current bent-pipe gate is a low-De inductionless current-closure baseline.
It cannot close higher-inertia Dean-vortex validation by itself. The next model
must resolve or approximate secondary cross-section velocity, not only the axial
profile and reduced bend skew.

The first research-grade step should be hydrodynamic Dean-flow validation
without magnetic field. After that passes, add uniform and then variable-field
MHD damping.

### External Reference Path

1. Use published Dean-flow data for curved pipes or curved rectangular
   channels as the primary reference for vortex topology and velocity skew.
2. Build an OpenFOAM hydrodynamic curved-pipe baseline as an executable
   reference if digitized observables are insufficient.
3. Fill `dean_vortex_reference_observables.csv` for the selected case.
4. Extend LMX with a resolved secondary-flow state or a documented
   asymptotic/reduced Dean model before marking this lane closed.
5. Rerun `examples/bent_pipe_inductionless_demo.py` with the filled CSV and
   promote the resulting comparison to release readiness only after convergence.

### Required Observables

- `secondary_flow_rms_ratio`
- `secondary_flow_peak_ratio`
- `normalized_velocity_centroid_shift`
- `inner_outer_velocity_ratio`
- `pressure_loss_proxy`
- preferred: vortex-center location, vortex-pair symmetry metric, and
  outboard-wall axial-velocity peak location

### Acceptance Gates

- No-field Dean-flow topology matches the reference regime: no false vortices
  below the onset range, paired Dean vortices in the moderate-De regime, and
  correct outboard axial-velocity shift.
- Secondary-flow and axial-skew observables pass the filled CSV tolerances.
- Mesh refinement changes secondary-flow RMS and pressure-loss proxy by less
  than 5%.
- With magnetic field enabled, secondary-flow strength decreases monotonically
  over the retained Hartmann-number sweep unless the chosen reference shows a
  documented nonmonotonic regime.
- Inductionless current closure remains at the current low-De quality level
  when MHD coupling is enabled.

### Publication Artifacts

- Curved-pipe geometry and mesh QA panel.
- Cross-section axial velocity and secondary-flow streamlines.
- Inboard/outboard axial velocity ratio versus station and time.
- Pressure-loss coefficient versus Dean number.
- MHD damping sweep versus Hartmann number once the no-field Dean gate passes.

## Implementation Sequence

### Phase 0: Reproducibility Audit

- Re-run the existing external-code audit and record exact commands, container
  images, commits, local patches, and output directories.
- Add a `docs/_static/generated/research_grade_closure_status.json` file that
  summarizes each lane as `open`, `data_acquired`, `matched_run_available`,
  `converged`, or `closed`.
- Keep this phase lightweight in CI: validate schema and artifact presence, not
  heavy external-code execution.

Current status: the closure-status artifact is generated by
`examples/research_grade_closure_status.py` and copied to
`docs/_static/generated/research_grade_closure_status.json`. The latest bounded
run reports `0/3` strict lanes closed:

- Q2D turbulence: external adapter ready; matched LMX-vs-Q2DmhdFoam parity open.
- Magnetic obstacle: external observables open.
- Dean vortex: resolved secondary-flow physics open.

The local external-code/data inputs are audited by
`examples/research_grade_external_data_audit.py`. That artifact checks which
external checkouts, processed reference files, and matched reference CSVs are
actually present before a lane can be promoted from open to closed.
`examples/research_grade_external_target_figures.py` now writes
`research_grade_external_targets.png` plus three `*_candidate.csv` files:
`magnetic_obstacle_reference_observables_candidate.csv`,
`q2d_turbulence_reference_observables_candidate.csv`, and
`dean_vortex_reference_observables_candidate.csv`. These are target-acquisition
artifacts for the paper and next runs, not strict validation inputs.

The latest strict closure attempt is also generated and archived as
`research_grade_strict_blocker_attempt.*`. It is a release guard, not a solver
shortcut:

![Strict blocker closure attempt](_static/generated/research_grade_strict_blocker_attempt.png)

- Magnetic obstacle: a low-resolution scan near `base_bz = 105` produced a
  Votyakov-scale reverse-flow candidate (`min u/u_ref ≈ -0.14`), but the
  current `40 × 40 × 25` rerun at the same field scale gave
  `min u/u_ref ≈ 0.997`. The candidate is therefore not converged and cannot be
  used as external validation.
- Q2D turbulence: local Q2DmhdFoam evidence exists, but the available lid-
  driven/Vetcha/cylinder outputs are not matched to the current LMX periodic
  SM82-style turbulence example. The strict
  `q2d_turbulence_reference_observables.csv` remains absent.
- Dean vortex: the current bent-pipe result is a low-De current-closure gate
  (`De ≈ 5.19e-7`) with negligible secondary flow. It cannot validate
  higher-inertia Dean vortices until LMX has a resolved secondary-flow model
  and a matched curved-pipe reference.

Because these three checks remain open, a research-grade release tag is blocked
even when bounded release readiness is green.

Follow-up support work now adds:

- richer Q2DmhdFoam ingestion of force coefficients and probe histories, so the
  external Q2D artifact has profile, spectral, force, and time-history
  observables ready for a matched LMX run;
- a Dean-flow literature validation artifact based on the Bayat-Rezai
  `V_De = 0.031 (nu / s) De^1.63` correlation and a reduced two-cell
  secondary-flow field for plotting and design QA.

These artifacts close data/model-preparation gaps, not the strict solved-case
parity requirements.

### Phase 1: External Data Acquisition

- Q2D: produce one matched Q2DmhdFoam turbulent reference bundle and convert it
  into `q2d_turbulence_reference_observables.csv`.
  The local Q2DmhdFoam source is a foam-extend 4.1 solver, not a drop-in
  OpenFOAM-v2206 solver. A direct `wmake` attempt inside the existing
  OpenFOAM-v2206 container fails at link time. The reproducible runner now
  lives in `docker/q2dmhdfoam` and has already produced a steady
  `Q2DfullyDeveloped` VTK/profile bundle; the remaining work is a matched
  turbulent case, not basic solver execution. The runner also executes
  non-default checked-out cases through `CASE_RELATIVE_PATH`; `run/lidDriven`
  now runs in MPI, exports VTK, and feeds
  `examples/q2dmhdfoam_lid_driven_vtk_artifact.py` for field-level velocity and
  vorticity observables. The matching LMX side-wall comparison now exists in
  `examples/q2d_lmx_q2dmhdfoam_lid_driven_parity.py`; it uses an isothermal
  `run/lidDriven` rerun with `ZERO_THERMAL=1`. The current strict table reads
  cell-centered OpenFOAM fields and applies graded-cell area weights. Speed RMS
  and peak vorticity pass, while the area-weighted mean-speed mismatch remains
  the current strict offender. This is still not a turbulent parity claim until
  the side-wall offender and a matched nonlinear case are closed.
- Magnetic obstacle: either build a reproducible MHD_Solvers_OpenFOAM localized
  field case or digitize one Votyakov/Cuevas observable set with provenance.
  The first digitized Votyakov Fig. 7(a) centerline target is now wired through
  `examples/magnetic_obstacle_votyakov_strict_attempt.py`; it compares the
  current positive LMX minimum centerline velocity ratio (`≈ 0.998`) with the
  recirculating target (`≈ -0.13`) and leaves the lane open as an explicit
  physics mismatch rather than a missing-data placeholder.
- Dean: digitize one Dean-flow reference set or generate one OpenFOAM curved-
  pipe baseline before adding magnetic damping. The first Bayat-Rezai strict
  attempt is now wired through
  `examples/dean_vortex_bayat_rezai_strict_attempt.py`; it compares the current
  low-De LMX secondary-flow ratios (`O(1e-17)`) against a moderate-De
  literature target (`secondary_flow_rms_ratio ≈ 4.13e-2`) and leaves the lane
  open as an explicit higher-inertia secondary-flow mismatch.

### Phase 2: LMX Matched Runs

- Add one example per lane that runs the exact matched LMX case from top-level
  constants, writes all raw fields needed for audit, and writes a summary JSON.
- Do not hide case parameters behind CLI parsing in examples; put them at the
  top of the script.
- Add reusable source helpers for observables, mesh ladders, and comparison
  plots so examples remain readable.

### Phase 3: Gates And Tests

- Add unit tests for each new parser and observable calculation.
- Add regression tests against small cached reference snippets.
- Add physics tests for conservation, monotonic scaling, and budget residuals.
- Add validation tests that read cached summaries and assert the documented
  tolerances.
- Update `scripts/run_release_readiness.py` so `--strict-research-grade` closes
  only when all three lanes have external-reference parity and convergence.

### Phase 4: Documentation And Publication Figures

- Add a docs page per closed lane with equations, assumptions, boundary
  conditions, nondimensional groups, mesh/time ladder, external-reference
  provenance, and generated figures.
- Update `examples/publication_figure_campaign.py` so the manifest reports
  each lane as paper-ready only after the strict gates pass.
- Copy final PNG/PDF and CSV artifacts to `docs/_static/generated`.
- Include the final status in `docs/validation_report.md` and the README
  validation table.

## Initial Work Order

1. Close magnetic-obstacle data acquisition first, because the current LMX
   comparison plumbing is complete and the blocker is external observable
   provenance.
2. Close Q2D turbulence next, because Q2DmhdFoam already runs locally and the
   remaining gap is a matched turbulent case rather than parser
   infrastructure.
3. Close Dean-vortex last, because it likely requires a real LMX solver
   extension for secondary-flow physics before a research-grade claim is
   honest.

## Stop Conditions

- If external reference data are unavailable but digitization is possible,
  proceed with digitization and record uncertainty.
- If neither executable nor digitized data can be obtained, leave the lane open
  and record the data request or paper-contact path.
- If the data exist but LMX lacks the governing physics, do not relax the
  validation threshold. Add the missing physics capability, or reclassify the
  current artifact as an internal gate.

# Benchmark Matrix

This page defines the benchmark ladder for the first LMX paper and `1.0`
release.

## Mandatory now: Benchmark A

### A1. Hartmann / insulating-duct style validation

- solver family: `fully_developed_inductionless`
- geometry: `rect_duct`
- literature target:
  - Samper et al. Table I insulating square-duct cases at `Ha = 500, 5000,
    10000, 15000`
  - compare the dimensionless flow-rate integral `Q̃`, not only local cuts
- observables:
  - velocity profiles
  - potential profiles
  - flow-rate and pressure-gradient surrogates
  - dimensionless flow-rate integral under mesh refinement

### A2. Shercliff and Hunt conducting/insulating wall validation

- solver family: `fully_developed_inductionless`
- geometries:
  - `rect_duct`
  - `layered_duct`
- literature target:
  - Samper et al. Table I conducting-wall square-duct cases at `Ha = 500,
    5000, 10000, 15000` with Hartmann-wall conductance ratio `cw = 0.01`
- observables:
  - matched `y` and `z` profiles
  - current-density profiles
  - Lorentz-force profiles
  - integral flow-rate and conservation diagnostics
  - side-layer / jet structure in the conducting-wall case

## Mandatory next: Benchmark B

These are the first nontrivial 3D inductionless targets from the benchmark
ladder summarized by [Samper et al.](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf).

### B1. Conducting pipe in a fringing magnetic field

- solver family: `extruded_inductionless`
- required geometry support: mapped pipe O-grid
- literature target:
  - Samper et al. Table II pipe case: `Ha ≈ 6600`, `N ≈ 10700`, `cw ≈ 0.027`
- required observables:
  - dimensionless pressure drop between the documented upstream/downstream taps
  - axial velocity distortion through the magnetic-field ramp
  - electric-potential redistribution on the wall and across the pipe section
  - mesh convergence of pressure drop and current-closure metrics

### B2. Conducting square duct in a fringing magnetic field

- solver family: `extruded_inductionless`
- literature target:
  - Samper et al. Table II square-duct case: `Ha ≈ 2900`, `N ≈ 540`, `cw ≈ 0.07`
- required observables:
  - dimensionless pressure drop between the documented upstream/downstream taps
  - cross-sectional velocity distortion through the fringing region
  - current-density redistribution and Lorentz-force localization
  - mesh convergence of pressure drop, current closure, and throughput recovery

## Validation gates for Benchmarks A and B

The current codebase should be judged against a fixed set of physics and
quality gates rather than only against visual agreement:

- mesh resolution
  - high-Ha Hartmann and side layers use focused meshes with several cells
    inside the thinnest boundary layer and smooth expansion into the core
  - mesh convergence is demonstrated on profiles, integral flow rate, current
    closure, and pressure/forcing observables
- profile agreement
  - normalized velocity/potential/profile errors on matched cuts
- integral agreement
  - flow rate, pressure-span surrogate, axial-current span, and Lorentz-power
    trends under mesh refinement
- literature observables
  - Benchmark A: dimensionless flow-rate integral `Q̃` against the analytical
    values tabulated by Samper et al.
  - Benchmark B: dimensionless pressure drop between the documented taps and
    matched velocity/potential cuts at the reference axial stations
- conservation
  - `div J`
  - charge-balance residual
  - interface-current residual
  - wall-current leakage
  - net boundary-current residual
- fringing-response physics
  - throughput constancy outside the field ramp
  - negative field/mean-velocity correlation through the ramp
  - pressure growth in the magnetized zone and recovery downstream
- quality gates
  - restart continuation equivalence
  - stable CLI/TOML and Python-driver workflows
  - machine-readable JSON/CSV outputs
  - strict docs build
  - fast routine test lane under five minutes

## Combined validation exercise

The current executable path for the full Benchmark A/B validation sweep is:

```bash
python scripts/run_full_validation_exercise.py \
  --output artifacts/validation/full_validation_exercise \
  --ha-values 10,20 \
  --resolution 12 \
  --fringing-resolutions 8,12 \
  --skip-paraview \
  --reference-root ./references/ClosedChannel \
  --write-plot
```

That workflow produces:

- Benchmark A case directories with field/profile artifacts
- Benchmark B fringing summaries and optional fringing-resolution plot
- one combined JSON summary
- one combined CSV table
- one combined Markdown gate report

When run from the source tree, `--reference-root` can be omitted and the
workflow will use the bundled closed-channel reference dataset under
`external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel`.

For the fringing-only quantitative summary, use:

```bash
python scripts/run_benchmark_b_quantitative.py \
  --output artifacts/validation/benchmark_b_quantitative \
  --ha-peak 20 \
  --duct-ny 20 \
  --duct-nz 20 \
  --pipe-nr 20 \
  --pipe-ntheta 80 \
  --nx-stations 21 \
  --max-steps 24 \
  --coupling-iterations 12 \
  --potential-iterations 80
```

That workflow writes one JSON summary, one CSV table, one Markdown table, and
one four-panel figure over charge balance, throughput span, axial-current
span, and pressure-span range.

That internal summary is the low-cost quantitative gate. The literature-anchored
Benchmark B closure still needs a second layer:

- B1 pipe:
  - dimensionless pressure-drop comparison against the documented tap spacing
  - shared-normalization profile comparisons on center and offset cuts
- B2 square duct:
  - dimensionless pressure-drop comparison through the field ramp
  - matched cross-sectional velocity and potential cuts at the same axial
    stations used in the reference data

Current dense-slice duct summary at `Ha = 20`, `24×24×33`:

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

For the layered Hunt-style duct, those raw span metrics are not the right
closure measure by themselves. The dense layered response is mirror-structured:
the axial current is odd about the magnet midplane and the cross-sectional
pressure span is even, with both quantities approaching zero at the center
station. On the heavier layered run used for the final closure pass
(`Ha = 20`, `18×18×21`):

- `axial_current_mirror_residual ≈ 1.88e-7`
- `pressure_span_mirror_residual ≈ 2.67e-5`
- `center_axial_current ≈ -8.10e-8`
- `center_pressure_span ≈ 9.56e-6`

That means the layered internal Benchmark B lane is now best interpreted as a
symmetry/closure problem rather than a raw-span problem. The rectangular dense
slice is closed on its original raw metrics; the layered dense slice is closed
on the new mirror-aware metrics that match the expected odd/even fringing
response.

A follow-on layered retune with `max_steps = 64`, `coupling_iterations = 24`,
and `potential_iterations = 160` did not improve the dense layered metrics.
It moved them in the wrong direction:

- `max_charge_balance_residual ≈ 1.16e-4`
- `volumetric_flow_rate_span ≈ 2.08e-3`
- `axial_current_span ≈ 9.19e-1`
- `pressure_span_range ≈ 3.36e1`

That means the dense layered issue was not a simple “run it longer” issue, but
it also was not purely an operator failure. The axial-current diagnostic now
uses the conservative x-face current flux rather than cell-centered `J_x`, and
the layered closure gate now tracks mirror residuals and center-station closure
instead of penalizing the expected odd/even fringing response with a raw span.

The current mapped-pipe external comparison remains the main Benchmark B
external gap. On the latest bounded pipe-reference comparison:

- center cut: `L2 ≈ 1.57e-1`, `L∞ ≈ 7.27e-1`
- negative offset cut: `L2 ≈ 1.68`, `L∞ ≈ 1.96`
- positive offset cut: `L2 ≈ 1.68`, `L∞ ≈ 1.95`

That is enough to say the comparison is quantitative, but not enough to call it
parity closure.

The underlying reason is now clearer from the bundled FreeMHD reference set:
the mapped-pipe comparison files correspond to the Bühler fringing-pipe case
at `Ha = 2000`, `Re = 20000`, while the current LMX `pipe_ogrid`
`extruded_inductionless` lane is still a low-Re inductionless research slice.
That means the present pipe comparison is still useful for qualitative shape
and sign checks, but quantitative parity will require a higher-inertia pipe
solver path rather than only denser sampling or plot cleanup.

Mapped-pipe external parity is therefore deferred from the current closeout.
The retained Benchmark B work should continue to use the internal conservation
and pressure-response gates for rectangular, layered, and mapped-pipe cases,
while any future parity claim against the Bühler/FreeMHD pipe dataset must move
the LMX pipe solver into a matching high-`Ha`, high-`Re` regime first.

## Staged but deferred

- Q2D turbulent duct flow
- turbulent duct flow / externally validated magnetic obstacle
- natural convection / heat transfer
- sudden expansion
- blanket mock-up / coupled-duct effects

These remain part of the research roadmap, but not the `1.0` solver promise.

Current executable Q2D Hartmann-friction decay lane:

- `examples/q2d_decay_validation.py`
  - quasi-2D Hartmann-friction decay of a single periodic mode
  - observables:
    final-state `L2/L∞` error and amplitude-decay error against the analytic
    exponential decay, plus the modal Q2D energy-budget residual
  - role:
    first Q2D validation surface before adding turbulent closures

Current forced Q2D Hartmann-friction lane:

- `examples/q2d_forced_validation.py`
  - forced periodic Q2D Hartmann-friction duct mode
  - observables:
    steady-state `L2/L∞` error and steady-amplitude error against the analytic
    forced solution, plus the modal production-dissipation residual
  - role:
    first forced Q2D duct slice before turbulent closures

Current wall-bounded Q2D Hartmann-friction lane:

- `examples/q2d_wall_bounded_validation.py`
  - forced wall-bounded Q2D Hartmann-friction duct mode
  - observables:
    final-state `L2/L∞` error and amplitude error against the exact transient
    Dirichlet solution, plus the wall-bounded modal energy-budget residual
  - turbulence-facing observables:
    scalar kinetic energy, fluctuation energy, enstrophy proxy, Hartmann
    friction dissipation proxy, viscous dissipation proxy, shell energy
    spectrum, spectral peak, log-spectrum slope, and high-wavenumber energy
    fraction
  - figure:
    `write_q2d_turbulence_observable_plots(...)` writes the wall-bounded field,
    shell spectrum, and energy/dissipation proxy panel used in the docs
  - role:
    first wall-bounded Q2D duct slice before Sommeria-Moreau-style closures
    and literature/experiment turbulent datasets

Current bounded result:

- `96 × 96`, `ν = 0.01`, Hartmann-friction `= 2.0`
- `l2_error ≈ 4.44e-4`
- `linf_error ≈ 4.44e-4`
- `steady_amplitude_rel_error ≈ 4.44e-4`
- `relative_budget_l2 ≈ 5.60e-4`

Current bounded wall-bounded result:

- `96 × 96`, `ν = 0.01`, Hartmann-friction `= 2.0`
- `l2_error ≈ 4.15e-4`
- `linf_error ≈ 4.15e-4`
- `amplitude_rel_error ≈ 1.42e-4`
- `relative_budget_l2 ≈ 5.12e-4`

Current localized magnetic-obstacle response lane:

- `examples/magnetic_obstacle_benchmark.py`
  - localized-field magnetic-obstacle response on the rectangular extruded
    inductionless lane, compared directly against a matched no-field LMX
    reference
  - observables:
    normalized velocity-deficit ratio, pressure-excess response, current
    response, centerline-cut distortion, and conservation metrics
  - role:
    internal response and conservation gate before any external
    magnetic-obstacle validation claim

Current bounded result:

- `24 × 24 × 17` rectangular duct with localized analytic obstacle field
  against a matched no-field reference
- `peak_velocity_deficit_ratio ≈ 3.23e-2`
- `peak_station_velocity_deficit_ratio ≈ 2.87e-2`
- `integrated_velocity_deficit_ratio ≈ 2.78e-2`
- `peak_centerline_deficit_ratio ≈ 3.76e-1`
- `peak_centerline_station_deficit_ratio ≈ 3.49e-1`
- `recovery_station ≈ 4.76`
- `peak_pressure_excess ≈ 5.01e-1`
- `pressure_excess_proxy ≈ 1.22e-1`
- `current_proxy_peak ≈ 4.56`
- `y_l2_distortion ≈ 2.31e-1`
- `z_l2_distortion ≈ 2.08e-1`
- `peak_crosscut_distortion ≈ 2.31e-1`
- `divergence_to_field_ratio ≈ 1.69e-2`
- `max_charge_balance_residual ≈ 3.98e-13`
- `internal_response_pass = true`
- `research_grade_validation_pass = false`

These observables are intentionally chosen to prepare for magnetic-obstacle
literature comparisons: streamwise wake deficit, centerline recovery location,
pressure response, and cross-cut distortion. The current reference is still a
matched no-field LMX run, so it is not an external validation.

The code now exposes `magnetic_obstacle_literature_reference_cases()` and
`validate_magnetic_obstacle_external_readiness(...)`. These register the
Cuevas-Smolentsev-Abdou quasi-2D target, the Votyakov-Zienicke-Kolesnikov
constrained-flow target, and the Andreev-Kolesnikov-Thess experimental target,
then reports the LMX observables in that vocabulary. This is a readiness gate,
not a pass/fail parity gate, until digitized reference profiles or executable
external cases are added.

The external parity data contract is now explicit:

- `examples/magnetic_obstacle_external_reference_template.py`
  writes the scalar-observable CSV template for digitized references
- `load_magnetic_obstacle_reference_observables(...)`
  loads filled CSV files with observable values, units, sources, and tolerances
- `compare_magnetic_obstacle_reference_observables(...)`
  compares LMX readiness observables against those reference rows
- `write_magnetic_obstacle_reference_comparison_table(...)`
  writes the resulting publication-table-ready CSV
- `write_magnetic_obstacle_reference_comparison_plots(...)`
  writes the paired PNG/PDF observable comparison and error/tolerance gate
- `examples/magnetic_obstacle_benchmark.py`
  automatically emits those external-reference artifacts when a filled
  `magnetic_obstacle_reference_observables.csv` is present, and otherwise
  emits the template while keeping the lane marked open

This closes the bookkeeping gap for external parity. The remaining scientific
gap is still the actual digitization or generation of a matched external
reference case.

The choice of observables follows the wake/recovery framing used in magnetic-
obstacle studies such as Cuevas et al., *On the flow past a magnetic obstacle*
and Votyakov et al., *Constrained flow around a magnetic obstacle*, even
though the current LMX slice is still a bounded low-inertia inductionless case
rather than a full turbulent or experimental parity benchmark.

The magnetic-obstacle example now also emits
`magnetic_obstacle_schematic.png/pdf`. That figure is deliberately setup-first:
it shows the rectangular duct, flow direction, localized field sheet, peak-field
velocity slice, cross-sectional deficit, and streamwise response before the
matched no-field comparison panel. This keeps the README and docs from showing
only response curves without the physical configuration.

Current bounded shape summary:

- `peak_station ≈ 3.00`
- `recovery_distance ≈ 1.76`
- `normalized_recovery_distance ≈ 6.25e-1`
- `literature_shape_gate = true`
- `literature_pass = false`

Current bounded regime scan:

- `examples/magnetic_obstacle_regime_scan.py`
  - sweeps the same localized-field rectangular obstacle case over
    `Bz` scale and forcing
  - writes a four-panel response map over velocity deficit, pressure response,
    current response, and mean cross-cut distortion
- bounded `3 × 3` scan:
  - low response at `Bz = 20`
  - clearly stronger obstacle signature and passing bounded gate at `Bz = 40`
  - strong distortion and pressure/current response at `Bz = 60`
- strongest bounded point in the checked scan:
  - `Bz = 60`, `forcing = 2`
  - `peak_velocity_deficit_ratio ≈ 7.01e-2`
  - `pressure_excess_proxy ≈ 6.74e-1`
  - `current_proxy_peak ≈ 9.40e1`
  - `mean cross-cut distortion ≈ 6.22e-1`

Current tabulated-field / mirror-field extension:

- `examples/wham_mirror_pipe_demo.py`
  - writes a tabulated WHAM-like mirror field with `magpylib_jax`
  - solves the current low-Re inductionless pipe baseline through that field
  - writes the table in solver streamwise coordinates, `x ∈ [0, L]`, while
    recording the offset back to the centered coil frame; this prevents
    unintended tabulated-field extrapolation in the downstream half of the pipe

The rectangular tabulated-field example now has two separate gates:

- table self-consistency from `tabulated_field_quality_metrics(...)`
- solver-point reconstruction from
  `tabulated_cross_section_reconstruction_metrics(...)`

The current manufactured-field run has solver-point
`relative_l2_error ≈ 1.92e-5` and `relative_linf_error ≈ 4.18e-5`, so the
visible tabulated-field response differences in the README are not an
interpolation mismatch. The remaining tabulated-field research gap is external
3D field-response validation, not the structured rectangular table.
  - observables:
    field/velocity anticorrelation, pressure-drop proxy, current response, and
    conservation metrics
- `examples/autodiff_wham_pressure_sensitivity.py`
  - treats the same mirror topology as a differentiable stationwise profile
  - observables:
    pressure-drop proxy and `d(Δp)/ds` with respect to coil separation

Current bounded reduced sensitivity result:

- reference separation `s = 1.96 m`
- `pressure_drop_proxy ≈ 3.85`
- `d(Δp)/ds ≈ 2.98e-1`
- sweep trend:
  monotone pressure-drop growth over `s = 1.5 … 2.2 m`, then flattening near
  the largest tested separation

Current bounded executable tabulated-pipe result:

- the field-loading and conservation path is stable
- the WHAM field table is now aligned with the solver coordinate frame and the
  summary records `coil_frame_x_offset` for reproducibility
- the structured 3D field-quality gate passes with zero table-node
  interpolation error and `divergence_to_field_ratio ≈ 3.01e-2`
- `field_velocity_correlation ≈ -9.70e-1`
- `max_charge_balance_residual ≈ 2.45e-2`
- `pressure_drop_proxy ≈ 1.66e-5`
- the nominal-WHAM low-Re response is therefore still too weak to count as a
  forceful localized-field executable comparison case

## Additional benchmark targets for the next publication cycle

The broader validation ladder used in recent inductionless liquid-metal MHD
solver papers suggests the following next additions after the current duct and
fringing set:

- closed pipe in a fringing magnetic field
  - observables:
    wall potential, pressure redistribution, and distorted axial velocity
- free-surface dam-break or sloshing benchmark
  - observables:
    front position, free-surface shape, and magnetic damping of the transient
- open-channel fringing-field benchmark
  - observables:
    free-surface deformation, recirculation, and current closure near the field
    ramp
- current-driven slotted-channel benchmark
  - observables:
    wall-current closure, jet structure, and electric-potential redistribution

Those cases extend the current validation ladder in the same direction as the
existing duct and fringing workflows: from fully developed 2D ducts to 3D
fringing response, then to free-surface and current-driven configurations.

## External reference datasets already staged in-tree

The repository already carries several external figure/data bundles that can be
used to stage the next validation campaigns:

- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel`
  - closed-channel Hartmann, Shercliff, and Hunt profile references for the
    Benchmark A family
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/FringingBPipe`
  - conducting-pipe fringing profiles for the Benchmark B1 family
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/DamBreak`
  - free-surface transient references for a post-Benchmark-B liquid-metal
    validation lane
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/LMX-U`
  - open-channel liquid-metal datasets relevant to free-surface and outlet
    modeling
- `external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/Divertorlets`
  - current-driven / complex-geometry reference data for later engineering
    validation

## Planned geometry and field extensions

Two capability lanes should be treated as explicit benchmark projects rather
than as ad hoc feature work:

- bent-pipe geometry
  - baseline: low-De inductionless straight-pipe-equivalence verification
    using `examples/bent_pipe_inductionless_demo.py`
  - inductionless extension: uniform-field bent pipe
  - nonuniform-field extension: fringing-field bent pipe
  - required observables:
    low-De profile equivalence, pressure span, secondary-flow structure,
    current closure, and mesh convergence with curvature

Current bounded low-De baseline:

- `Ha = 20`, `R = 0.45`, `R_c = 3.6`, `15 × 18 × 40`
- `De ≈ 5.19e-7`
- `cross_section_l2_error = 0`
- `centerline_l2_error = 0`
- `max_charge_balance_residual ≈ 2.15e-2`
- `volumetric_flow_rate_span ≈ 1.14e-9`
- newly reported Dean/curvature observables:
  `secondary_flow_rms_ratio ≈ 6.16e-18`,
  `secondary_flow_peak_ratio ≈ 1.71e-17`,
  `normalized_velocity_centroid_shift = 0`, and
  `inner_outer_velocity_ratio = 1.0`
- full Dean-vortex research validation remains open until the secondary-flow
  structure and curvature-shift observables are compared against a curved-duct
  reference dataset
- spatially varying magnetic fields
  - baseline: manufactured divergence-free field verification
    plus executable rectangular `extruded_inductionless` validation through
    `examples/variable_field_extruded_demo.py`
  - layered extension:
    `examples/variable_field_layered_demo.py`
  - curved-pipe extension:
    `examples/variable_field_bent_pipe_demo.py`
  - tabulated-field extension:
    `examples/variable_field_tabulated_demo.py` and
    `examples/fringing_tabulated_case.toml`
  - recovery test: reproduce the current fringing benchmarks through the
    generic field-loading path
  - extension: tabulated or analytic 3D fields for ducts and pipes
  - required observables:
    pressure redistribution, Lorentz-force localization, throughput change, and
    charge/current closure under mesh refinement

Current bounded results:

- layered variable-field duct:
  `field_velocity_correlation ≈ -9.98e-1`,
  `current_proxy_change ≈ 2.69e2`,
  `max_charge_balance_residual ≈ 1.36e-1`
- tabulated rectangular variable-field duct:
  bounded validation pass on the same conservation and divergence metrics as
  the analytic rectangular lane, currently
  `field_velocity_correlation ≈ -7.52e-1`,
  `current_proxy_change ≈ 8.28e-1`,
  `max_charge_balance_residual ≈ 5.34e-6`;
  the table-quality gate reports zero table-node interpolation error and
  `divergence_to_field_ratio ≈ 4.81e-4`
- bent-pipe variable-field low-De comparison:
  straight/bent equivalence still satisfies
  `cross_section_l2_error ≈ 8.12e-6`,
  `centerline_l2_error ≈ 8.20e-6`,
  while the field-response gate passes on the normalized divergence metric

## Key references

These are the main benchmark and comparison references currently driving the
LMX validation ladder and the planned manuscript figures.

- [Samper et al., *An approach to verification and validation of MHD codes for fusion applications*](https://www.sciencedirect.com/science/article/pii/S0920379614003263)
- [FreeMHD V&V paper, arXiv:2409.08950](https://arxiv.org/abs/2409.08950)
- [Quasi-two dimensional perturbations in duct flows under transverse magnetic field](https://arxiv.org/abs/2006.03993)
- [On the flow past a magnetic obstacle](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/on-the-flow-past-a-magnetic-obstacle/F4185BE5315273DBA9D1C53DD49990AA)
- [Constrained flow around a magnetic obstacle](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/constrained-flow-around-a-magnetic-obstacle/DFD706B066E0B0C7E8598544E1783BC0)
- [Validation and verification of a robust 3-D MHD code](https://www.sciencedirect.com/science/article/pii/S0920379618300358)

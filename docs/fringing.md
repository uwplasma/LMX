# Fringing-Field Research Slice

LMX now ships the first executable `extruded_inductionless` solver-family
front-end together with the retained 3D research slice that the next paper
phase can build on.

The intended model distinction is:

- `fully_developed_inductionless`
  - 2D cross-sectional solves for `u(y,z)` and `\phi(y,z)` with streamwise
    forcing and conservative transverse currents
- `extruded_inductionless`
  - 3D low-Re slices in `x-y-z` with `u(x,y,z)`, `v(x,y,z)`, `w(x,y,z)`,
    `p(x,y,z)`, and `\phi(x,y,z)` under a prescribed axial magnetic-field
    profile

That distinction matters physically. The 2D family is the right model for
fully developed benchmark ducts, while the 3D family is the right model once
axial field variation, inlet/outlet current closure, or fringing-region
pressure redistribution become important.

The current retained 3D slice includes:

- a smooth axial fringing-field profile generator in `lmx/fringing.py`
- a stationwise sweep driver that reuses the fully developed solver as a cheap
  research scaffold
- an `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` workflow
  that now runs a true low-Re rectangular-duct, layered-duct, or mapped-pipe
  `u, v, w, p, phi` projection slice
- a stacked axial field-bundle builder that exposes `u(x, y, z)`,
  `v(x, y, z)`, `w(x, y, z)`, `p(x, y, z)`, `phi(x, y, z)`, current,
  Lorentz, axial-current, wall-current leakage, and charge-balance histories
- an explicit validation summary for that extruded slice
- a publication-style example in `examples/fringing_benchmark_demo.py`

This is explicit by design. The current slice is a real 3D
pressure-velocity-potential iteration and is now available through both Python
and TOML/CLI workflows, including direct `lmx run fringing_*` entry points for
quick rectangular, layered, and mapped-pipe launches. It is still a research
slice rather than the final
production family, but it is now a real executable part of LMX rather than a
Python-only staging path.

## Run the scaffold

```bash
lmx examples/fringing_rect_case.toml
lmx examples/fringing_layered_case.toml
lmx examples/fringing_pipe_case.toml
lmx run fringing_rect --ha 20 --nx-stations 21 --output out/fringing_rect
lmx run fringing_layered --ha 20 --nx-stations 21 --wall-cells 1 --insulator-cells 1 --output out/fringing_layered
lmx run fringing_pipe --ha 20 --radius 0.5 --nr 24 --ntheta 48 --output out/fringing_pipe
python examples/fringing_benchmark_demo.py \
  --geometry-kind rect_duct \
  --ha-peak 20 \
  --ny 12 \
  --nz 12 \
  --nx-stations 11 \
  --max-steps 18 \
  --coupling-iterations 10 \
  --potential-iterations 60 \
  --output artifacts/examples/fringing_benchmark
python examples/fringing_benchmark_demo.py \
  --geometry-kind layered_duct \
  --output artifacts/examples/fringing_benchmark_layered
python examples/fringing_benchmark_demo.py \
  --geometry-kind pipe_ogrid \
  --output artifacts/examples/fringing_benchmark_pipe_exploratory
python examples/extruded_paper_figures.py \
  --output artifacts/examples/extruded_paper_figures
```

The example writes:

- `fringing_benchmark_summary.json`
- `fringing_benchmark.png`
- `fringing_benchmark.pdf`
- an `extruded_bundle` section in the JSON summary with axial field-bundle shape
  and charge-balance histories
- a `validation` section with residual and field/response consistency metrics
- for TOML/CLI runs, `*_station_history.csv`, `*_extruded_results.npz`,
  `overview.png`, `overview.pdf`, a station-archive manifest, per-station
  `station_XXXX.npz` field bundles, and a JSON summary with conservation metrics

The shipped input files `examples/fringing_rect_case.toml`,
`examples/fringing_layered_case.toml`, and
`examples/fringing_pipe_case.toml` are now the recommended publication-scale
starting points for 3D fringing studies. They enable figure writing directly
through `[output].write_plots = true` and keep the full solver setup in the
input file rather than hiding it in Python glue.

The paired restart template is `examples/fringing_layered_restart_case.toml`.
Use it after a base layered run has written its extruded restart bundle under
`artifacts/examples/toml_fringing_layered/restart/`.

Current retained fringing publication artifact:

![LMX fringing rectangular-duct slice](_static/generated/fringing_benchmark_rect.png)

The mapped-pipe example remains useful for exploratory research and solver
development, but on the heavier retained validation campaign it is currently
kept outside the locked publication set.

Current exploratory mapped-pipe comparison artifact:

![LMX mapped-pipe exploratory comparison](_static/generated/pipe_reference_comparison.png)

Restart / resume reproducibility artifact:

![LMX extruded restart reproducibility](_static/generated/extruded_restart_demo.png)

Larger 3D validation campaign artifact:

![LMX extruded validation campaign](_static/generated/extruded_validation_campaign.png)

The larger retained campaign is intentionally a resolution study, not a claim
that every coarse layered case is already publication-grade. In particular, the
coarsest layered high-field point remains visibly underresolved, so the figure
is used to show convergence behavior and screening logic rather than as a final
table of locked publication values.

The heavier exploratory campaign that reintroduces mapped pipes shows that
`pipe_ogrid` still falls outside the larger retained gate on the current
post-`1.0` tree. Its charge-balance residual exceeds the retained threshold on
the heavier `Ha=20` and finer-mesh runs, so the mapped-pipe slice remains a
research/development lane rather than a locked publication benchmark.

## Governing equations for the retained 3D slice

The current 3D slice is still incompressible and inductionless. It solves

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\rho \frac{\partial \mathbf{u}}{\partial t}
= -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{J}\times\mathbf{B}
$$

$$
\mathbf{J} = \sigma\left(-\nabla \phi + \mathbf{u}\times\mathbf{B}\right)
$$

$$
\nabla\cdot\mathbf{J} = 0
$$

using a simple low-Re projection loop for the momentum-pressure part and a
Poisson-like solve for `\phi`. The current field is then audited through both
local and integral constraints.

For the mapped-pipe slice, the same equations are evaluated in the local
`(x,r,\theta)` frame, with the prescribed transverse magnetic field projected
onto local `r` and `\theta` directions before the electric and Lorentz terms
are assembled.

## Conservation gates

The fringing lane now treats charge conservation as a hard validation target,
not just a descriptive metric. Each retained extruded bundle now reports:

- `charge_balance_residual(x)`
  - a stationwise compatibility and `\nabla\cdot J` residual
- `axial_current(x)`
  - the integrated streamwise current crossing each `x = const` plane
- `wall_current_leakage(x)`
  - the integrated external-wall leakage current on the non-periodic radial or
    `y/z` boundaries
- `net_boundary_current_residual`
  - the full control-volume boundary-flux imbalance

These are the physically relevant hardening metrics for inlet/outlet treatment.
In a closed inductionless control volume, the integrated current must satisfy

$$
\int_{\partial \Omega} \mathbf{J}\cdot\mathbf{n}\, dS = 0,
$$

and the extruded validation lane now checks that condition directly instead of
relying only on local `\nabla\cdot J` norms.

For rectangular and layered extruded cases, the retained audit now assembles
that boundary-flux check from conservative face currents with closed-current
axial boundary treatment. This aligns the reported `div J` and boundary-current
metrics with the actual discrete operator used in the electric solve instead of
mixing a face-based solve with a cell-gradient diagnostic at conductivity
jumps.

The heavier validation driver
`scripts/run_manual_solver_family_validation.py` can now turn these metrics
into pass/fail gates with:

- `--max-charge-balance`
- `--max-interface-current`
- `--max-fringing-wall-current-leakage`
- `--max-fringing-boundary-current`
- `--fail-on-threshold`

The repository now also ships a bounded larger-dataset wrapper:

```bash
python examples/extruded_validation_campaign.py \
  --output artifacts/examples/extruded_validation_campaign
```

That example runs the hard-gate fringing campaign on a retained larger
resolution set, writes JSON/CSV summaries, and produces a publication-style
summary figure.

For reviewer-facing figures, `examples/extruded_paper_figures.py` now writes a
bounded retained set of 3D visualizations:

- a rectangular peak-field cross-section rendered in 3D
- a layered peak-field cross-section rendered in 3D
- a compact summary panel over mean velocity, charge-balance residual,
  axial current, and pressure span

Current reviewer-facing retained paper figures:

![LMX rectangular fringing 3D paper figure](_static/generated/paper_rect_3d.png)

![LMX layered fringing 3D paper figure](_static/generated/paper_layered_3d.png)

![LMX fringing reviewer summary](_static/generated/paper_reviewer_summary.png)

Its default larger retained conservation dataset is now
`rect_duct,layered_duct,pipe_ogrid`. The mapped-pipe slice is inside that hard
gate on the bounded larger dataset, but its external profile comparison is still best
treated as qualitative because the shipped reference dataset is a high-`Ha`,
high-`Re` benchmark while the retained LMX pipe slice is still a low-`Re`
research model.

Current retained hard-gate dataset:

- fully developed Hartmann / Shercliff / Hunt at `Ha = 10, 20`, resolution `10`
- fringing `rect_duct`, `layered_duct`, and `pipe_ogrid` at the same `Ha`
  values with
  `nx_stations = 5`
- hard thresholds:
  - `max_charge_balance <= 8e-1`
  - `max_interface_current <= 2.5e-1`
  - `max_fringing_wall_current_leakage <= 1e-1`
  - `max_fringing_boundary_current <= 1e-5`
  - `volumetric_flow_rate_span <= 5e-3`
  - `field_mean_velocity_correlation <= -5e-1`

That retained gate now passes for `rect_duct`, `layered_duct`, and
`pipe_ogrid`. The layered case joined the retained gate after the multi-region
electric subproblem was switched to a sparse direct solve of the conservative
variable-coefficient potential operator, and the mapped-pipe slice joined after
the cylindrical electric/current operator was rewritten around the conservative
face-flux form and a stable O-grid time-step estimate.

The last two gates are explicitly physics-facing rather than solver-facing:

- `volumetric_flow_rate_span`
  enforces near-constant axial throughput across the fringing region for these
  incompressible low-`Re` benchmark slices
- `field_mean_velocity_correlation`
  enforces the expected anti-correlation between local field strength and mean
  streamwise velocity under constant forcing

On the current bounded larger dataset, those added physics gates now pass for
all three retained fringing geometries:

- `rect_duct` passes
- `layered_duct` passes after adding a partial stationwise throughput-closure
  step inside the 3D projection loop
- `pipe_ogrid` passes

Current retained layered hardening signals on that dataset:

- `volumetric_flow_rate_span ≈ 1.00e-3` at `Ha=10`
- `volumetric_flow_rate_span ≈ 2.75e-3` at `Ha=20`
- `field_mean_velocity_correlation ≈ -8.02e-1`
- `max_charge_balance_residual <= 1.37e-5`

So the current reviewer-proof statement is:

- the retained conservation gate covers `rect_duct`, `layered_duct`, and
  `pipe_ogrid`
- the stricter fringing physics gate now also covers `rect_duct`,
  `layered_duct`, and `pipe_ogrid`
- mapped pipe still remains qualitative on the external profile-comparison side,
  even though it is now inside the retained internal validation gate

Current retained mapped-pipe hardening signals on the heavier bounded
dataset:

- `max_charge_balance_residual ≈ 5.63e-2` at `Ha=10`, `resolution=8`
- `max_charge_balance_residual ≈ 1.20e-2` at `Ha=10`, `resolution=12`
- `max_charge_balance_residual ≈ 1.10e-1` at `Ha=20`, `resolution=8`
- `max_charge_balance_residual ≈ 2.40e-2` at `Ha=20`, `resolution=12`
- `max_wall_current_leakage = 0`
- `net_boundary_current_residual = 0`

The shipped qualitative comparison against the external high-`Ha`, high-`Re`
fringing-pipe profile data gives:

- center line normalized axial-velocity shape error: `≈ 1.66e-1`
- negative-offset absolute line error: `≈ 9.89e-1`
- positive-offset absolute line error: `≈ 9.89e-1`

So the reviewer-facing conclusion is still deliberately conservative: mapped
pipe is now part of the retained conservation/solver-validation set, but the
current external pipe-profile comparison remains qualitative rather than a
publication parity gate.

## What the example shows

- a smooth entrance/exit fringing profile along the duct axis
- the stationwise peak axial velocity response, which is a better proxy for the
  M-shaped redistribution described in the fringing-field literature than the
  nearly conserved cross-sectional mean flow rate
- the stationwise pressure span `max(p)-min(p)`, which is a better publication
  observable than the earlier current-weighted proxy when reviewing 3D fringing
  behavior
- contour views of the stacked velocity bundle in `x-y` and `x-z`
- the stationwise axial current together with charge-balance residuals, which
  are the key current-closure diagnostics for inlet/outlet hardening
- a first true 3D pressure field `p(x, y, z)` inside the research slice
- layered conducting/insulating wall fringing responses through the same API
- the first mapped-pipe fringing slice through the same public API

## QA note on the retained figures

During the publication QA pass, the fringing figures were tightened to avoid
misleading observables. In a constant-forcing incompressible slice, the
cross-sectional mean flow can remain nearly unchanged even when the velocity
profile redistributes strongly. The literature instead emphasizes profile
deformation, pressure losses, and axial-current closure in non-uniform fields.
That is why the retained figures now highlight peak axial velocity, pressure
span, and axial-current/conservation metrics rather than the earlier
mean-velocity correlation view.

## Publication context

The duct benchmark ladder and fringing-region validation targets documented
here are aligned with the standard low-magnetic-Reynolds-number literature and
with fusion liquid-metal V&V practice:

- [Samper et al., verification and validation benchmark ladder for MHD codes](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf)
- [Hunt, *Magnetohydrodynamic flow in rectangular ducts*](https://doi.org/10.1017/S0022112065000344)
- [Recent low-Rm structured-grid duct validation study](https://doi.org/10.1016/j.fusengdes.2013.01.092)
- [Fringing-field rectangular-duct benchmark study](https://www.sciencedirect.com/science/article/abs/pii/S0920379611003188)

## Source map

- `lmx/fringing.py`
  - fringing-profile construction, stationwise sweep utilities, the extruded
    slice solver entry point, validation helpers, and stacked axial field bundles
- `examples/fringing_benchmark_demo.py`
  - user-facing fringing benchmark slice with publication-style plots
- `docs/benchmark_matrix.md`
  - benchmark targets that this scaffold is preparing

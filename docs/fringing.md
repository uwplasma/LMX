# Fringing-Field Workflows

LMX includes an executable `extruded_inductionless` solver-family front-end for
3D fringing-field workflows.

In plain terms, a fringing-field problem is a duct or pipe flow where the
magnetic field does not stay uniform from inlet to outlet. Instead, the fluid
enters a magnetized region, passes through the strongest field, and then exits
again. That spatial field variation drives pressure redistribution, current
closure, and velocity-profile changes that do not appear in a strictly fully
developed 2D benchmark.

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

The three fringing geometries are:

- `rect_duct`
  a plain rectangular liquid-metal channel
- `layered_duct`
  the same channel with explicit conducting and insulating wall regions
- `pipe_ogrid`
  a circular pipe represented on an O-grid so radial and azimuthal resolution
  can be controlled separately

The current 3D solver lane includes:

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
- a detailed example in `examples/fringing_benchmark_demo.py`

This is explicit by design. The current lane is a real 3D
pressure-velocity-potential iteration and is now available through both Python
and TOML/CLI workflows, including direct `lmx run fringing_*` entry points for
quick rectangular, layered, and mapped-pipe launches.

## Run the scaffold

```bash
lmx examples/cases/fringing/fringing_rect_case.toml
lmx examples/cases/fringing/fringing_layered_case.toml
lmx examples/cases/fringing/fringing_pipe_case.toml
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
  --output artifacts/examples/fringing_benchmark_pipe
python campaigns/fringing/extruded_summary_figures.py \
  --output artifacts/examples/extruded_summary_figures
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

The input files `examples/cases/fringing/fringing_rect_case.toml`,
`examples/cases/fringing/fringing_layered_case.toml`, and
`examples/cases/fringing/fringing_pipe_case.toml` are now the recommended starting points for
3D fringing studies. They enable figure writing directly
through `[output].write_plots = true` and keep the full solver setup in the
input file rather than hiding it in Python glue.

The paired restart template is `examples/cases/fringing/fringing_layered_restart_case.toml`.
Use it after a base layered run has written its extruded restart bundle under
`artifacts/examples/toml_fringing_layered/restart/`.

Fringing overview artifact:

![LMX fringing rectangular-duct slice](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/fringing_benchmark_rect.png)

Mapped-pipe comparison artifact:

![LMX mapped-pipe comparison](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/pipe_reference_comparison.png)

Restart / resume reproducibility artifact:

![LMX extruded restart reproducibility](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/extruded_restart_demo.png)

Larger 3D validation campaign artifact:

![LMX extruded validation campaign](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/extruded_validation_campaign.png)

The larger campaign is intentionally a resolution study, not a claim
that every coarse layered case is already fully converged. In particular, the
coarsest layered high-field point remains visibly underresolved, so the figure
is used to show convergence behavior and screening logic rather than as a final
reference table.

## Governing equations for the 3D solver lane

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

The fringing lane treats charge conservation as a hard validation target. Each
extruded bundle now reports:

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

For rectangular and layered extruded cases, the audit assembles
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
python campaigns/fringing/extruded_validation_campaign.py \
  --output artifacts/examples/extruded_validation_campaign
```

That example runs the hard-gate fringing campaign on a larger
resolution set, writes JSON/CSV summaries, and produces a compact comparison
figure.

Its default larger conservation dataset is now
`rect_duct,layered_duct,pipe_ogrid`. The mapped-pipe slice is inside that hard
gate on the bounded larger dataset, and its external profile comparison is now
quantitative. The remaining gap is physical rather than bookkeeping: the
reference dataset is a high-`Ha`, high-`Re` benchmark while the current LMX
pipe slice is still a laminar inductionless model.

Current hard-gate dataset:

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

That gate now passes for `rect_duct`, `layered_duct`, and
`pipe_ogrid`. The layered case joined the gate after the multi-region
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
all three fringing geometries:

- `rect_duct` passes
- `layered_duct` passes after adding a partial stationwise throughput-closure
  step inside the 3D projection loop
- `pipe_ogrid` passes

Current layered hardening signals on that dataset:

- `volumetric_flow_rate_span ≈ 1.00e-3` at `Ha=10`
- `volumetric_flow_rate_span ≈ 2.75e-3` at `Ha=20`
- `field_mean_velocity_correlation ≈ -8.02e-1`
- `max_charge_balance_residual <= 1.37e-5`

So the current bounded-validation statement is:

- the conservation gate covers `rect_duct`, `layered_duct`, and
  `pipe_ogrid`
- the stricter fringing physics gate also covers `rect_duct`,
  `layered_duct`, and `pipe_ogrid`
- mapped-pipe external profile comparison is now quantitative and shows a
  real high-`Ha`, high-`Re` parity gap against the current laminar pipe slice

The widened bounded manual campaign at `Ha = 10, 20, 30`, `resolution = 8`
keeps those three fringing geometries inside the combined conservation and
fringing-physics gate on the current tree. The same bounded probe now also
keeps the previously failing fully developed Hunt row inside the heavier
conservation gate after the interface audit was moved onto the conservative
face-averaged current reconstruction. On the repaired `Ha = 10`,
`resolution = 8` Hunt run, `interface_current_residual ≈ 1.27e-2`.

Current mapped-pipe hardening signals on the heavier bounded
dataset:

- `max_charge_balance_residual ≈ 5.63e-2` at `Ha=10`, `resolution=8`
- `max_charge_balance_residual ≈ 1.20e-2` at `Ha=10`, `resolution=12`
- `max_charge_balance_residual ≈ 1.10e-1` at `Ha=20`, `resolution=8`
- `max_charge_balance_residual ≈ 2.40e-2` at `Ha=20`, `resolution=12`
- `max_wall_current_leakage = 0`
- `net_boundary_current_residual = 0`

The current quantitative comparison against the external high-`Ha`, high-`Re`
fringing-pipe profile data gives:

- center-line `L_2 ≈ 1.64e-1`, `L_\infty ≈ 9.82e-1`
- negative-offset `L_2 ≈ 9.94e-1`, `L_\infty ≈ 1.00`
- positive-offset `L_2 ≈ 9.94e-1`, `L_\infty ≈ 1.00`

Those numbers are now computed on one shared velocity normalization across all
three comparison lines, so they describe a real parity gap rather than a
per-line normalization artifact. The physical interpretation is also direct:
the external dataset is a high-`Ha`, high-`Re` fringing-pipe case, while the
current `pipe_ogrid` slice is still a laminar inductionless model.

The focused dense duct Benchmark B summary from
`scripts/run_benchmark_b_quantitative.py` at `Ha = 20`, `24×24×33` currently
reports:

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
That summary separates the current dense-slice status clearly:

- rectangular dense slices are now quantitatively clean on the current
  Benchmark B settings
- layered dense slices need symmetry-aware closure metrics rather than raw
  spans. On the heavier `Ha = 20`, `18×18×21` layered closure run:
  - `axial_current_mirror_residual ≈ 1.88e-7`
  - `pressure_span_mirror_residual ≈ 2.67e-5`
  - `center_axial_current ≈ -8.10e-8`
  - `center_pressure_span ≈ 9.56e-6`
  The raw axial-current span and pressure-span range remain large because the
  layered Hunt fringing response is odd/even about the magnet midplane.
- mapped pipe remains a separate quantitative external-comparison lane, and its
  parity gap is still dominated by the high-`Ha`, high-`Re` reference regime
  mismatch described above

![Benchmark B quantitative summary](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/benchmark_b_quantitative_summary.png)

## What the example shows

- a smooth entrance/exit fringing profile along the duct axis
- the stationwise peak axial velocity response, which is a better proxy for the
  M-shaped redistribution described in the fringing-field literature than the
  nearly conserved cross-sectional mean flow rate
- the stationwise pressure span `max(p)-min(p)`, which is a more stable
  observable than the earlier current-weighted proxy when reviewing 3D fringing
  behavior
- for prescribed-flow problems, the stationwise pressure Lagrange multiplier
  reported as positive pressure-loss gradient `-dp/dx`, and the signed
  transverse pressure difference between adjacent wall midpoints,
  `p(side,+z)-p(top,+y)`
- contour views of the stacked velocity bundle in `x-y` and `x-z`
- the stationwise axial current together with charge-balance residuals, which
  are the key current-closure diagnostics for inlet/outlet hardening
- a first true 3D pressure field `p(x, y, z)`
- layered conducting/insulating wall fringing responses through the same API
- the first mapped-pipe fringing slice through the same public API

The pressure span and current-weighted quantity remain exploratory diagnostics.
They are not accepted substitutes for the direct ALEX observables. The frozen
B1 comparison uses `axial_pressure_loss_gradient` with its downstream plateau
removed; B2 uses `transverse_pressure_difference` with the uniform-field/no-field
plateau removed.

### ALEX B2 numerical path

The frozen ALEX B2 square-duct builder now selects a dedicated nonuniform
finite-volume path. It differs deliberately from the characterized low-`Ha`
demonstration path:

- masked diffusion places no-slip at the fluid/wall face instead of forcing the
  adjacent fluid cell centre to zero;
- one face-flux divergence/gradient pair is used for both the pressure equation
  and velocity correction, so a small Poisson residual cannot conceal a
  collocated continuity error;
- the stationwise pressure multiplier is applied before projection, followed by
  one divergence-free global response that restores the exact prescribed mean
  flow without reopening continuity;
- pressure and variable-conductivity electric-potential equations use the
  released SOLVAX implicit PCG backend with cell-volume symmetrization and an
  explicit rank-one Neumann gauge; and
- stretched-grid gradients, conservative face currents, wall areas, and charge
  diagnostics use the actual face and centre spacing.

Manufactured agreement, fixed-flow preservation, current closure, restart,
`jax.jit`, and finite-difference-checked implicit gradients are portable tests.
This establishes the B2 solver implementation; it does **not** establish the
experimental benchmark. Steady/tolerance independence, the frozen thin-wall
comparison, the three production meshes, and matched FreeMHD evidence remain
mandatory before B2 is called validated. ALEX B1 remains guarded until the
accepted mapped-pipe evidence is rerun at the final release fingerprint.

For the explicit B2 shell, fluid-to-wall face transmissibility uses the
mixed-dimensional thin-wall limit: tangential conductance remains
`sigma_wall * thickness`, while artificial wall-center half-cell resistance is
excluded from normal coupling. Shell cell widths and conductivity are mapped to
the frozen nominal numerical thickness while preserving that sheet
conductance; the confirmation thickness therefore tests the collapsed surface
operator instead of introducing a second volumetric wall. Sustained exact
coarse independence remains required before acceptance. In particular, every
independence variant must satisfy its own requested coupling tolerance. A
tight-tolerance run that merely remains below the looser baseline steady limit
is not accepted, even when its observable difference is small.

B2 uses SOLVAX's vector Aitken relaxation with a maximum factor of two.
Checkpoint-matched exact-grid probes found this cap monotone; factors three and
four developed late oscillations and are rejected. The setting accelerates the
same fixed-point map and does not alter the governing equations or acceptance
tolerances.

## Literature context

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
  - user-facing fringing benchmark slice with detailed plots
- `docs/benchmark_matrix.md`
  - benchmark targets that this scaffold is preparing

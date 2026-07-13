# Validation Closure Notes

This page records how the current public validation lanes were closed, what
equations and boundary conditions were used, and which assumptions are still
part of the bounded release scope. It is intended to make the README figures,
JSON summaries, and release gates auditable without reading the full git
history.

The wording here is deliberately conservative. "Closed" means the lane passes
the documented bounded-release gate with reproducible artifacts in
`docs/_static/generated`. It does not mean every related research problem is
complete.

## Governing Model Used By The Closed Lanes

The closed liquid-metal duct and fringing examples use the low magnetic
Reynolds number inductionless model with prescribed magnetic field:

$$
\nabla \cdot \mathbf{u} = 0,
$$

$$
\rho\left(\partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}\right)
= -\nabla p + \mu\nabla^2\mathbf{u} + \mathbf{J}\times\mathbf{B},
$$

$$
\mathbf{J} = \sigma\left(-\nabla\phi + \mathbf{u}\times\mathbf{B}\right),
\qquad
\nabla\cdot\mathbf{J} = 0.
$$

For the straight-duct validation figures, the solver uses the fully developed
reduction

$$
\mathbf{u} = \left(u(y,z), 0, 0\right),
\qquad
\mathbf{B} = \left(0, B_y, B_z\right).
$$

The electric current components in the cross-section are

$$
J_y = \sigma\left(-\partial_y\phi - uB_z\right),
\qquad
J_z = \sigma\left(-\partial_z\phi + uB_y\right),
$$

and charge conservation gives

$$
\nabla_\perp\cdot\left(\sigma\nabla_\perp\phi\right)
= \nabla_\perp\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})_\perp\right).
$$

The streamwise momentum equation solved in the straight-duct lane is

$$
\rho \partial_t u
= -\partial_x p + \mu\nabla_\perp^2 u + J_yB_z - J_zB_y.
$$

The mapped-pipe and bent-pipe examples use the same inductionless current law
and charge equation, but assemble the electric fluxes in local
$(x,r,\theta)$ coordinates.

## Closure Ledger

| Lane | Current status | Acceptance evidence |
| --- | --- | --- |
| Hartmann/Shercliff/Hunt reader-facing profiles | Closed for bounded release | `https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/analytic_velocity_profiles.png`; all retained cuts are below `L2 <= 1.2e-2` |
| Hunt `Ha = 100` side-layer analytical cut | Closed for the public ladder | `docs/_static/generated/straight_duct_validation_ladder_summary.json`; Hunt `Ha=100` has `y_l2 = 4.42e-3`, `z_l2 = 2.89e-3` |
| Dense rectangular internal fringing slice | Closed as an internal conservation/response gate | `benchmark_b_quantitative_summary.png`; charge and throughput metrics are inside the documented bounded thresholds |
| Dense layered internal fringing slice | Closed with mirror-aware observables | Odd/even axial-current and pressure-span residuals replace misleading raw spans |
| Bent-pipe low-De current closure | Closed | `bent_pipe_inductionless_summary.json`; `max_charge_balance_residual = 2.16e-12`, `max_wall_current_leakage = 0`, `net_boundary_current_residual = 0` |
| Tabulated rectangular variable-field reconstruction | Closed | `variable_field_tabulated_summary.json`; table-node and solver-point reconstruction gates pass |
| Release readiness | Closed at bounded-release level | `scripts/run_release_readiness.py` reports no hard blockers |

Deferred strict research lanes remain documented separately:

| Lane | Why it remains deferred |
| --- | --- |
| External magnetic-obstacle validation | The current figure is an internal response/conservation gate until external observables are filled |
| External Q2D turbulent parity | Current Q2D diagnostics verify SM82-style behavior and nonlinear boundedness, but not an external turbulent dataset |
| Higher-inertia Dean-vortex bent pipe | The low-De straight-pipe-equivalence gate is closed; a real Dean-vortex reference is still required |
| Mapped-pipe external parity | The bundled reference corresponds to a higher-`Ha`, higher-`Re` pipe case than the current low-Re mapped-pipe slice |

## Hunt Side-Layer Closure

### Model And Boundary Conditions

The Hunt validation case is a square duct with a uniform magnetic field normal
to the conducting Hartmann walls. In LMX coordinates for this example:

- the imposed field is along `y`
- the conducting Hartmann walls are the `y = +/- a` wall layers
- the insulating side walls are the `z = +/- a` wall layers
- the velocity is no-slip at the fluid wall
- the electric potential is solved in both fluid and explicit solid layers
- insulating external wall segments enforce zero normal current

The multi-region current equation uses conductivity-weighted conservative face
conductances at fluid-solid interfaces. No Dirichlet electric potential is
imposed at the wall for the closed-channel Hunt validation.

### Thin-Wall Conductance Match

The bundled FreeMHD/Ni analytical Hunt files use a thin conducting-wall model:

$$
c = \frac{\sigma_w t_w}{\sigma a}.
$$

For the retained `Ha = 100` validation:

$$
t_w = 0.001,\qquad \frac{\sigma_w}{\sigma} = 5,\qquad a = 0.1,
\qquad c = 0.05.
$$

The earlier failed validation preserved only the conductance ratio by using a
much thicker explicit wall (`t_w = 0.02`) and a lower wall conductivity. That
is not equivalent for the finite-thickness multi-region potential solve: the
solid region changes where the potential and current redistribute, even when
the scalar conductance ratio is the same. The accepted validation therefore
uses the physical thin-wall thickness and conductivity ratio from the reference
files instead of a thick-wall surrogate.

### Numerical Choices

The retained public ladder uses:

- zero initial velocity and zero initial electric potential
- `face_averaged` current reconstruction
- no-slip wall reconstruction when comparing cell-centered LMX profiles to
  wall-to-wall analytical curves
- `45 x 45` fluid cells for Shercliff
- `49 x 49` fluid cells plus two explicit wall cells per side for Hunt
- normalized profile errors, with the reference file path saved in the JSON
  summary

The key design decision is that profile error is the acceptance metric, not
nominal layer-cell count alone. The high-Ha Hunt side layer is sensitive to the
wall model, current closure, and side-jet placement. Blind mesh-only increases
to `65 x 65`, `81 x 81`, and a previous `97 x 97` probe did not improve
monotonically. Those failed probes are kept in the plan so reviewers can see
that the final gate was not obtained by hiding a mesh-ladder failure.

### Current Acceptance Numbers

The regenerated artifact
`docs/_static/generated/straight_duct_validation_ladder_summary.json` reports:

| Case | `Ha` | `y_l2` | `z_l2` |
| --- | ---: | ---: | ---: |
| Shercliff | 20 | `5.63e-3` | `5.41e-3` |
| Shercliff | 100 | `4.89e-3` | `7.93e-3` |
| Hunt | 20 | `5.31e-3` | `3.16e-3` |
| Hunt | 100 | `4.42e-3` | `2.89e-3` |

The reader-facing profile comparison in
`docs/_static/generated/straight_duct_profile_comparison_summary.json` is also
generated from zero initial velocity. Its retained Hunt `Ha = 20` cuts are
`8.54e-3` and `4.86e-3`.

## Bent-Pipe Charge Closure

### Model And Assumptions

The bent-pipe example is a low-De, low-Re inductionless mapped-pipe baseline.
It is not a high-inertia Dean-vortex validation. The purpose of this lane is to
verify that the curved-centerline mapped pipe preserves the straight-pipe
limit and closes current locally and globally before higher-inertia physics is
claimed.

The pipe solve is assembled in local cylindrical coordinates:

$$
\mathbf{J} = \sigma\left(-\nabla\phi + \mathbf{u}\times\mathbf{B}\right).
$$

The conservative local charge residual is evaluated as

$$
\nabla\cdot\mathbf{J}
= \frac{F^x_{i+1/2}-F^x_{i-1/2}}{\Delta x}
  + \frac{r_{j+1/2}F^r_{j+1/2}-r_{j-1/2}F^r_{j-1/2}}
         {r_j\Delta r_j}
  + \frac{F^\theta_{k+1/2}-F^\theta_{k-1/2}}
         {r_j\Delta\theta}.
$$

Here `F` denotes the face current flux assembled by the same conservative
current helper used for the post-solve diagnostic.

### Boundary And Metric Treatment

The current low-De mapped-pipe implementation uses:

- symmetry-style treatment at `r = 0`
- zero radial velocity at the center and outer wall
- a bounded wall interpolation for axial and azimuthal velocity in the current
  research slice
- stationwise flow-rate stabilization for restart/baseline consistency
- no external wall current leakage in the conservative electric-flux audit
- zero net boundary-current residual over the integrated control volume

The wall interpolation is a bounded research-slice treatment, not the final
high-inertia no-slip Dean solver. Because the bent and straight reference
solutions use the same mapped-pipe operator and boundary treatment, the low-De
equivalence gate remains valid.

### Sign Fix In The Conservative Potential Solve

The mapped-pipe sparse potential matrix represents the operator

$$
-\nabla\cdot\left(\sigma\nabla\phi\right).
$$

The EMF source helper returns

$$
\nabla\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})\right).
$$

Since

$$
\nabla\cdot\mathbf{J}
= -\nabla\cdot\left(\sigma\nabla\phi\right)
  + \nabla\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})\right)
= 0,
$$

the sparse mapped-pipe potential solve must use the negative EMF divergence as
its right-hand side:

$$
-\nabla\cdot\left(\sigma\nabla\phi\right)
= -\nabla\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})\right).
$$

Before the fix, the pipe branch solved the opposite sign. That produced a
local conservative `div J` residual of order `1e-2` even though global leakage
was zero. The fix is implemented in `lmx/_fringing.py`, and the regression
test `test_pipe_sparse_potential_cancels_conservative_emf_divergence` verifies
that the same conservative EMF and current-flux operators cancel to machine
precision.

### Current Acceptance Numbers

The regenerated bent-pipe summary reports:

| Observable | Value |
| --- | ---: |
| `cross_section_l2_error` | `0` |
| `centerline_l2_error` | `0` |
| `max_charge_balance_residual` | `2.16e-12` |
| `max_wall_current_leakage` | `0` |
| `net_boundary_current_residual` | `0` |
| `research_grade_charge_balance_pass` | `true` |
| `volumetric_flow_rate_span` | `3.30e-11` |

The remaining bent-pipe research lane is the higher-inertia Dean-vortex
comparison, which requires secondary-flow structure and curvature-response
observables against a curved-duct reference dataset.

## Internal Fringing And Variable-Field Closure

The rectangular and layered fringing lanes use the same inductionless current
law, but on an extruded 3D mesh. The electric source, post-solve `div J`
diagnostic, and boundary-current audits are all assembled from conservative
face fluxes. This is the important design decision: at conductivity jumps, a
cell-centered gradient diagnostic can show spurious current imbalance even when
the conservative face fluxes are balanced.

The rectangular dense slice is judged by raw charge/throughput/pressure
metrics. The layered Hunt-style fringing slice is judged by mirror-aware
current and pressure observables because its response is odd/even about the
field midplane. Penalizing raw axial-current span would incorrectly mark the
expected antisymmetric response as a failure.

The tabulated variable-field lane is closed by three checks:

- the tabulated field has a small divergence-to-field ratio
- interpolation back to solver points has low relative `L2` and `Linf` errors
- the extruded inductionless solve retains local charge closure

## Release Readiness Interpretation

`scripts/run_release_readiness.py` is the current machine-readable release
gate. After the closures above, it reports:

- no bounded-release hard blockers
- straight-duct reader-facing profiles below the `1.2e-2` target
- bent-pipe low-De current closure below the `1e-3` target
- every required public artifact present locally or verified in the uploaded,
  checksummed release-asset manifest

It still reports a bounded release class rather than full research-grade
completion because external Q2D turbulence parity, external magnetic-obstacle
reference data, and higher-inertia Dean-vortex validation remain deferred.

## Reproduction Commands

Regenerate the main artifacts discussed here:

```bash
python campaigns/ducts/straight_duct_profile_comparison.py
python campaigns/ducts/straight_duct_validation_ladder.py
python campaigns/fringing/bent_pipe_inductionless_demo.py
```

Copy regenerated artifacts into the docs static directory if the examples are
run manually:

```bash
cp artifacts/examples/straight_duct_profile_comparison/analytic_velocity_profiles.* docs/_static/generated/
cp artifacts/examples/straight_duct_profile_comparison/straight_duct_profile_comparison_summary.json docs/_static/generated/
cp artifacts/examples/straight_duct_validation_ladder/closed_channel_validation_ladder.* docs/_static/generated/
cp artifacts/examples/straight_duct_validation_ladder/straight_duct_validation_ladder_summary.json docs/_static/generated/
cp artifacts/examples/bent_pipe_inductionless/bent_pipe_overview.* docs/_static/generated/
cp artifacts/examples/bent_pipe_inductionless/bent_pipe_inductionless_summary.json docs/_static/generated/
```

Then run the release and docs gates:

```bash
PYTHONPATH=. python scripts/run_release_readiness.py --output artifacts/release/release_readiness.json
PYTHONPATH=. sphinx-build -W -b html docs docs/_build/html
```

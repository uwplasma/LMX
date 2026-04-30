# Publication Figure Plan

This page tracks the figure campaign needed for an LMX methods and validation
paper. It is intentionally separate from the README: the README should remain a
landing page, while this page records literature anchors, required observables,
code features, generator scripts, and validation status.

## Literature And Code Anchors

The current figure plan is organized around the validation pattern used in the
liquid-metal MHD literature:

- **Closed ducts and wall conductance.** FreeMHD validates against analytical
  Shercliff and Hunt velocity profiles over Hartmann numbers and wall
  conductance ratios, then extends to 3D and experimental cases
  ([FreeMHD V&V paper](https://arxiv.org/abs/2409.08950)).
- **Fusion blanket V&V hierarchy.** The Samper V&V proposal separates basic
  analytical tests, 3D benchmark cases, and application-facing fusion blanket
  simulations; LMX should keep those categories explicit in tables and plots
  ([Samper V&V paper](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf)).
- **Blanket pressure-drop design.** Blanket design studies prioritize pressure
  drop, flow distribution, local current closure, and field/geometry scaling
  ([blanket MHD review](https://www.osti.gov/pages/biblio/1977147),
  [Rhodes et al. manifold study](https://bpb-us-w2.wpmucdn.com/research.seas.ucla.edu/dist/d/39/files/2019/08/PoFv30_Tyler-Rhodes-et-al_Magnetohydrodynamic-pressure-drop-and-flow-balancing-of-liquid-metal-flow-in-a-prototypic-fusion-blanket-manifold_-Physics-of-Fluids_-Vol-30-No-5.pdf)).
- **Curved-pipe MHD.** Curvature produces Dean secondary flow and shifts axial
  velocity toward the outboard wall; transverse magnetic fields suppress and
  reshape that response. The resulting observables are axial-velocity skew,
  secondary-flow strength, and pressure drop versus Dean number and Hartmann
  number ([curved-pipe MHD formulation](https://doi.org/10.1016/j.compfluid.2015.05.025),
  [OSTI curved-pipe thesis record](https://www.osti.gov/biblio/6336594),
  [OSTI curved-pipe paper record](https://www.osti.gov/biblio/5585279)).
- **Quasi-2D turbulence.** Q2D duct work uses Hartmann friction, energy/enstrophy
  budgets, spectra, transient growth, and perturbation growth rates rather than
  only movies ([Pothérat Q2D duct study](https://arxiv.org/abs/2006.03993)).
- **External codes.** FreeMHD remains the executable OpenFOAM comparison for
  closed-channel and free-surface families. Q2DmhdFoam-style and HIMAG-style
  runs are the natural external references for Q2D turbulence, magnetic
  obstacle, and blanket-manifold pressure-drop figures.

## Figure Suite

| Figure family | What the plot should show | Current LMX generator | Needed additions |
|---|---|---|---|
| Closed duct analytical validation | Shercliff/Hunt profiles, layer zooms, `L2/L∞` convergence, pressure-gradient parity versus `Ha` and wall conductance | `examples/straight_duct_profile_comparison.py`, `examples/hartmann_validation_ladder.py` | Add a single publication table that combines mesh, `Ha`, wall conductance, and FreeMHD observable errors |
| FreeMHD observable parity | `u`, `φ`, `J`, `J×B`, pressure-gradient, transient `u_max(t)`, runtime and memory on matched cases | `examples/freemhd_closed_channel_observable_parity.py`, `examples/freemhd_closed_channel_flow_rate_parity.py` | Add electric-current streamline or vector plots and a ranked offender table with rerun commands |
| WHAM blanket geometry and field | Pipe route, coil position, centerline `B`, transverse `B_\perp`, mapped-mesh QA, current closure residual | WHAM geometry, mesh, field, and current-closure examples | Add one combined overview panel with geometry, field contours, mesh slice, and current-closure summary |
| WHAM blanket pressure drop | Cumulative `Δp(s)`, pressure-budget components, `Δp` versus `B`, `U`, coil separation, and pipe radius | `examples/wham_blanket_flow_demo.py`, `examples/wham_blanket_autodiff_research_demo.py` | Add velocity and radius sweeps, then a nondimensional pressure plot versus `Ha`, `N`, and Dean number |
| WHAM curved-bend velocity skew | Inboard/outboard axial velocity during startup and steady state, with the inboard side slower than the outboard side in the bend | `examples/wham_blanket_flow_demo.py` | Replace the reduced Dean-skew proxy with a resolved secondary-flow solve and validate against curved-pipe MHD literature |
| Magnetic obstacle | Setup schematic, centerline velocity deficit, cross-cut distortion, pressure/drop drag proxy, `curl(J×B)` layer indicator | `examples/magnetic_obstacle_benchmark.py` | Import external reference observables from Cuevas/Smolentsev/Abdou or a reproducible external-code run |
| Q2D turbulence | Vorticity movie, kinetic energy, enstrophy, spectra, Hartmann-friction decay, transient-growth comparison | Q2D decay/forced examples | Add external Q2D reference run and acceptance gates on energy decay and spectra |
| Bent pipe Dean-vortex validation | Axial velocity contours, outboard shift, secondary-flow streamfunction/vorticity, pressure drop versus `De` and `Ha` | `examples/bent_pipe_inductionless_demo.py`, WHAM blanket flow preview | Add a resolved cross-section secondary-flow state and mesh ladder |
| Variable/tabulated fields | Interpolation error, `∇·B`, field-line/contour plots, pressure response, autodiff sensitivities to coil parameters | variable-field and WHAM field examples | Add divergence-cleaning comparisons and full sensitivity plots for tabulated fields |
| Performance and differentiability | CPU/GPU strong scaling, memory, compile time, gradient runtime, optimization trajectory | strong-scaling and autodiff examples | Tie all performance plots to the real operator paths used by the validation figures |
| Strict closure dashboard | Closed support gates, external strict mismatches, and research-grade closure ledger | `examples/research_grade_closure_dashboard.py` | Replace failed panels with passed solved-physics comparisons as each lane closes |

## Required Code Features

- `lmx.publication` or `lmx.validation_figures`: thin orchestration helpers that
  collect already-existing plotting functions into a reproducible manuscript
  campaign without making examples too large.
- `examples/publication_figure_campaign.py`: a bounded default campaign that
  records all tracked manuscript figures, optionally refreshes the fast WHAM
  figure family, and writes one summary JSON plus a CSV table.
- `examples/publication_heavy_campaign.py`: opt-in heavy campaign for external
  FreeMHD/Q2D/HIMAG reruns, high-Ha mesh ladders, and scaling figures.
- A common figure-summary schema containing case name, equations, boundary
  conditions, mesh, timestep, solver settings, reference source, error metrics,
  artifact paths, and pass/fail gates.
- A current/field diagnostic bundle for every 3D MHD case: `φ`, `J`, `J×B`,
  `∇·J`, wall-current leakage, pressure gradient, and cumulative pressure.
- A curved-pipe cross-section state that can store axial velocity, secondary
  velocity, inboard/outboard masks, Dean number, Hartmann number, and local
  bend pressure loss.
- External-data adapters for FreeMHD, Q2DmhdFoam/HIMAG-style runs, and
  literature digitized data, with provenance recorded in JSON.

## WHAM Bend-Skew Interpretation

For a hydrodynamic curved pipe, centrifugal inertia drives Dean vortices and
pushes the axial-velocity maximum toward the outer wall. The inboard side is
therefore expected to be slightly slower than the outboard side once the bend
response has developed. In a strong transverse magnetic field this skew is
damped and can change shape because Lorentz forces suppress secondary motion.

The current WHAM movie uses a reduced centerline pressure/velocity model and a
bounded Dean-skew diagnostic. It is useful for showing the expected sign and
time development of the inboard/outboard imbalance, but it is not yet a
research-grade curved-pipe MHD secondary-flow validation. The research-grade
version must solve the cross-section secondary flow and compare `U_out/U_in`,
pressure drop, and secondary-flow strength against curved-pipe MHD references
over a mesh/`De`/`Ha` ladder.

## Immediate Implementation Order

1. Keep the current WHAM pressure and movie artifacts with the bend
   inboard/outboard velocity diagnostic now present in the movie, transient
   panel, CSV, and JSON summary.
2. Keep `examples/publication_figure_campaign.py` as the bounded manifest gate
   for all currently available manuscript artifacts.
3. Keep `examples/research_grade_closure_dashboard.py` as the reviewer-facing
   closure ledger for strict blockers and support gates.
4. Add the combined WHAM overview panel and nondimensional pressure-sweep plots
   versus `Ha`, `N`, `De`, and `B^2`.
5. Add external-data adapters and heavy rerun scripts for magnetic obstacle,
   Q2D turbulence, and curved-pipe Dean validation.
6. Promote each closed figure family into CI/release gates only after the
   runtime remains bounded and the external-reference provenance is recorded.

## Current Bounded Manifest

The bounded manifest currently tracks 11 manuscript-facing figure families.
All tracked artifacts and summary files are present, so the manifest is not a
release blocker. The paper-ready flag remains false because three strict
research lanes still require external or resolved validation: magnetic-obstacle
external observables, Q2D turbulent parity, and higher-inertia Dean-vortex
curved-pipe validation.

The executable closure plan for those lanes is maintained in
[](research_grade_closure_plan.md). Publication figures should be promoted from
the bounded manifest to paper-ready status only after that plan's external
reference, convergence, and physics gates pass.

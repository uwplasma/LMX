# Validation

LMX separates numerical verification, physics validation, and external-code
comparison. A test passes only its stated claim; finite output alone is never a
validation result.

| Model | Required evidence | Current claim |
|---|---|---|
| Hartmann duct | analytical profile, charge closure, power balance, refinement | validated within documented mesh/tolerance gates |
| Shercliff and Hunt ducts | packaged benchmark values, symmetry, wall/interface current, mesh trends | validated within documented mesh/tolerance gates |
| High-$Ha$ fully developed flow | layer resolution, Richardson trend, integral balances | bounded accepted campaign cases |
| Rectangular 3-D fringe | manufactured operators, projection, restart, refinement, FreeMHD B2 | active validation; each artifact states the gates it passes |
| Straight-pipe 3-D fringe | production-field and derivative parity, mapped operators, fixed flow, annular current, Benchmark B1 data | differentiable generic core accepted; external production validation requires the complete matched gate |
| Periodic Q2D | analytical decay, energy identity, spectral incompressibility, spatial refinement, CPU/GPU parity | verified for the documented SM82 model and numerical gates |
| Magnetic obstacle | divergence-free field sampling, conservation, symmetry, and bounded-response observables | development application, not an externally validated benchmark |

For 2-D cases, `validation_summary` reports convergence, current continuity,
gauge, interface, flow, and profile metrics. `hartmann_validation` compares
the computed profile with the analytical Hartmann solution.

For 3-D cases, `ExtrudedInductionlessValidation` reports maximum update,
charge-balance and divergence residuals, boundary-current closure, stationwise
flow variation, pressure variation, and response correlations. Mesh and solver
independence remain separate required checks.

The benchmark specifications and reference arrays shipped under `src/lmx/data/benchmarks`
are versioned package data. `benchmarks/provenance.json` records bibliographic
sources and executable test/workflow links.

## Quantitative evidence

The CI-executed `examples/hartmann_example.py` case uses $Ha=20$ on a
$24\times24$ cross-section. Its analytical errors are 0.02276 in $L_2$ and
0.06204 in $L_\infty$, with charge-balance residual
$4.24\times10^{-19}$ and final velocity update $9.48\times10^{-9}$. The
documented profile-error limits are 0.05 and 0.10.

The pinned two-update B2 Docker comparison executes LMX and FreeMHD from the
same observed contract. It passes execution, artifact identity, contract,
native-output observation, and comparison gates. The normalized transverse
pressure difference has RMS error 0.004518 and maximum error 0.01092 against
FreeMHD on the harness mesh, below its frozen 0.16 and 0.32 bounds. This is an
executable integration check, not a production-mesh validation result.

Production Benchmark B records keep numerical and external evidence separate.
Each baseline stores the exact coordinate hashes, full and fluid mesh shapes,
physical cell count, and a three-dimensional characteristic spacing. Combining
coarse, medium, and fine campaigns reports solver/wall independence,
literature-weighted errors, unequal-ratio observed order, and the fine-grid
Grid Convergence Index (GCI), using a 1.25 safety factor only when both
three-dimensional refinement ratios are at least 1.3. The machine-readable
record also retains the B2 post-map momentum defect and requires it to be below
the frozen electromagnetic-force-normalized balance limit. Its status is:

- `mesh_incomplete` until all frozen grids are present;
- `numerical_rejected` when a conservation, independence, literature, or
  refinement gate fails;
- `external_validation_open` when every numerical gate passes but the required
  matched independent solve is absent or fails; or
- `accepted` only when numerical and matched external gates both pass.

Thus a converged mesh campaign can be reported without being mislabeled as an
externally validated result. Final `pass` remains fail-closed.

The Q2D Taylor--Green case matches its exact viscous/Hartmann decay, and a
nonlinear three-grid test compares $12^2$ and $24^2$ solutions with a $48^2$
reference. On a $256^2$, 80-step float32 workload run with JAX 0.6.2, the final
CPU and RTX A4000 fields agree to relative $L_2=2.38\times10^{-6}$. This is a
backend-parity result, not external physics validation. Its GPU speed-up is not
quoted. The JAX 0.10.2 float32 timings in `benchmarks/results` put the $256^2$
ratio at 14.15x on a different, 20-step workload, and none of those reports
records the matmul precision a float32 GPU number must carry (ADR 0005, D15).
Plan step 2.1 re-measures them.

## Smolentsev et al. 2015 Table I on the staggered core (in progress, plan step 1.13)

Flow rate $\tilde Q=\int_{-1}^{1}\int_{-1}^{1}\tilde U\,dy\,dz$ for unit
$-dP/dx$ with half-width, density, viscosity and conductivity one, which is the
normalisation of `lmx.duct_problem`; $\tilde Q$ is four times the mean velocity
the steady tests compare. `wall_conductance` in `duct_problem` sets the two
walls normal to the field, so A2 (Hartmann walls $c=0.01$, insulating side
walls) is `duct_problem(..., wall_conductance=0.01)`. An exact series (Fourier
in the side-wall direction, closed form along the field) reproduces the
analytic column to all four printed digits and the spectral reference of
`validation/shercliff.py` to $5\times10^{-9}$.

Measured so far, float64, default tolerance, meshes `Ny:layer_y:Nz:layer_z`
(fitted geometric stretching, `layer` cells inside $1/Ha$ along the field and
$1/\sqrt{Ha}$ across it); relative error against the analytic column:

| Row | Ha | Mesh | Relative error | CG iterations |
|---|---|---|---|---|
| A1 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +2.20 % / +0.98 % / +0.43 % | — |
| A1 | 500 | 96:12:48:6 / 48:6:96:12 | +0.27 % / +0.95 % | — |
| A2 | 500 | 64:8:32:4 / 96:12:48:6 | +0.38 % / +0.18 % | 66 / 95 |
| A1 | 15,000 | 64:8:32:4 / 64:8:64:8 / 128:16:32:4 / 128:16:64:8 | +1.10 % / +1.08 % / +3.15 % / +3.16 % | 122 / 138 / 127 / 142 |
| A2 | 15,000 | 64:8:32:4 / 128:16:64:8 | +1.25 % / +0.33 % | 500 / 632 |

At Ha 500 the A1 series is second order (observed order 2.00 over ratio 1.5)
and its Richardson value is within $1\times10^{-5}$ of the analytic one; the
error lives in the Hartmann layer, not the side layer. At Ha 15,000 refining
the Hartmann direction makes A1 worse: the fast-diagonal potential solve loses
float64 accuracy on these meshes. The along-field eigenvalue of the constant
mode comes out $5.5\times10^{-6}$ instead of zero on 128:16 (smallest cell
$6.3\times10^{-7}$, largest eigenvalue $5.4\times10^{12}$), one direct solve
leaves a relative residual of $5.3\times10^{-3}$, and the core current is a
$1/Ha$ cancellation of $u\times B$ and $\nabla\phi$, so the error reaches the
flow rate amplified. The gate of validation row 27 is not yet assessed.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.

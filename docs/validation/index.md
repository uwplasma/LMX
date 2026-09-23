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

## Smolentsev et al. 2015 Table I on the staggered core (plan step 1.13)

Flow rate $\tilde Q=\int_{-1}^{1}\int_{-1}^{1}\tilde U\,dy\,dz$ for unit
$-dP/dx$ with half-width, density, viscosity and conductivity one, which is the
normalisation of `lmx.duct_problem`; $\tilde Q$ is four times the mean velocity
the steady tests compare. `wall_conductance` in `duct_problem` sets the two
walls normal to the field, so A2 (Hartmann walls $c=0.01$, insulating side
walls) is `duct_problem(..., wall_conductance=0.01)`. The reference is Table I's
analytic column, which has four digits: its rounding is $6\times10^{-5}$ to
$3.6\times10^{-4}$ relative, depending on the row.

All runs are float64 at the default tolerance $10^{-9}$, on JAX 0.6.2 (the CI
floor stack). Meshes are `Ny:layer_y:Nz:layer_z`, built with the fitted
geometric `wall_resolving_faces`: `layer` cells inside $1/Ha$ along the field
($y$) and inside $1/\sqrt{Ha}$ across it. Each series scales all four numbers
by 1.5, so the three meshes are one mapping at three spacings. The order is the
observed order of the three flow rates, and "Richardson" is the extrapolated
flow rate relative to the analytic one.

| Row | Ha | Meshes | Relative error, coarse / medium / fine | Order | Richardson | CG iterations | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +2.20 % / +0.98 % / +0.43 % | 2.00 | $-1.5\times10^{-5}$ | 49 / 62 / 66 | gated |
| A2 | 500 | 32:4:32:4 / 48:6:48:6 / 72:9:72:9 | +0.77 % / +0.36 % / +0.18 % | 1.96 | $+2.2\times10^{-4}$ | 53 / 69 / 79 | gated |
| A1 | 5,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +0.91 % / +0.41 % / +0.22 % | 2.44 | $+1.1\times10^{-3}$ | 124 / 180 / 307 | reported |
| A2 | 5,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +0.74 % / +0.34 % / +0.16 % | 2.01 | $+2.0\times10^{-4}$ | 301 / 370 / 432 | reported |
| A1 | 10,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.04 % / +0.47 % / +0.23 % | 2.16 | $+6.3\times10^{-4}$ | 132 / 210 / 426 | reported |
| A2 | 10,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.03 % / +0.46 % / +0.20 % | 1.88 | $-3.5\times10^{-4}$ | 393 / 574 / 704 | reported |
| A1 | 15,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.10 % / +0.49 % / −0.65 % | none | none | 147 / 277 / 522 | **not converging** |
| A2 | 15,000 | 64:8:32:4 / 96:12:48:6 / 144:18:72:9 | +1.25 % / +0.56 % / +0.24 % | 1.90 | $-3.3\times10^{-4}$ | 511 / 784 / 1,006 | reported |

**Gated.** `tests/test_table_i.py` (channel shard) solves both Ha 500 rows on
their three meshes and requires a monotone error, an observed order between 1.8
and 2.2, and a Richardson value within $10^{-3}$ of Table I (validation row 27).
Both measured Richardson values lie inside the table's own rounding. The error
is in the Hartmann layer: for A1, refining only the field direction
(96:12:48:6) gives +0.27 %, refining only the side direction (48:6:96:12)
gives +0.95 %.

**Reported, not gated.** The Ha 5,000 and 10,000 rows fall monotonically at
close to second order, but A1 at Ha 5,000 has not reached the asymptotic range
(order 2.44), and its Richardson value is just outside row 27's $10^{-3}$. These
rows cost 30–75 s per solve on a loaded laptop, too much for a pull-request
shard. A2 at Ha 15,000 converges on this series, too.

**A1 at Ha 15,000 does not converge with refinement.** On the series above, the
fine-mesh step moves the flow rate by 1.1 %, twice as far as the medium step
did, and it moves it past the analytic value. Refining further along the field
makes it worse: 128:16:32:4 gives +3.15 % and 128:16:64:8 gives +3.16 %, while
64:8:32:4 gives +1.10 %. The evidence points to float64 conditioning of the
fast-diagonal potential solve rather than to layer resolution:

- On 128:16 the smallest cell is $6.3\times10^{-7}$ and the largest
  along-field eigenvalue is $5.4\times10^{12}$.
- The constant (Neumann null) mode's along-field eigenvalue comes out
  $5.5\times10^{-6}$ instead of zero. On 64:8 it is $7\times10^{-11}$.
- One direct solve leaves a relative residual of $5.3\times10^{-3}$. At Ha 500
  on 48:6, the figure is $1.4\times10^{-7}$.
- The core current is a $1/Ha$ cancellation between $u\times B$ and $\nabla\phi$,
  so the potential error reaches the flow rate amplified.

The pressure solve is protected by the double projection in `steady_residual`,
but the potential solve has no such correction. Two float64 refinement sweeps of
the direct solve leave the 64:8:32:4 results unchanged (A1 +1.1003 %, A2
+1.2526 %), so the coarse-mesh error is discretisation. The same sweeps were not
completed on the fine meshes. Why A2 converges at the same Hartmann number while
A1 does not has not been established. At tolerance $10^{-11}$, the A1
128:16:64:8 solve does not converge at all.

The confirming run, two refinement sweeps on 128:16:64:8, was not finished.
The fix is plan step 2b.3: refine the potential solve against the exact face
stencil. Once it lands, re-measure A1 at Ha 10,000 and 15,000 and gate the
remaining rows. The A1 Ha 15,000 row of validation row 27 stays open until then.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.

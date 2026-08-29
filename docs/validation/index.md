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
reference. On the documented $256^2$, 80-step float32 workload, one RTX A4000
is 25.78x faster than CPU after compilation; final fields agree to relative
$L_2=2.38\times10^{-6}$. This is a measured backend-parity and performance
result, not external physics validation.

## Test gates

The portable suite includes analytical, manufactured, regression, physics, and
validation markers. Combined line/branch coverage must exceed 95%. Structural
3-D changes additionally run the reduced pinned FreeMHD Docker case before
acceptance.

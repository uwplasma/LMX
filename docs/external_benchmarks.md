# External Benchmark Comparisons

LMX keeps external solver comparisons as benchmark evidence, not as the source
of truth for the governing equations.

## Current external references

- [FreeMHD validation paper](https://doi.org/10.1063/5.0230242)
- [OpenFOAM pressure-velocity algorithm documentation](https://www.openfoam.com/documentation/guides/latest/doc/guide-applications-solvers-pressure-velocity-intro.html)
- [Fusion Engineering and Design paper on multi-region MHD solvers built on OpenFOAM](https://doi.org/10.1016/j.fusengdes.2024.114216)

## What should be compared

For the fully developed duct solver, the meaningful comparison targets are:

- velocity slices and line cuts
- electric-potential slices and line cuts
- current-density observables
- Lorentz-force observables
- integral flow-rate and pressure-drop surrogates
- matched runtime on the same host when the comparison is used for
  performance context rather than only for physics parity

## What should not be primary acceptance criteria

- backend-specific pressure-correction loop traces
- implementation-specific residual histories
- source-code-level solver choices in external codes

## Rationale

LMX is a clean-room inductionless MHD implementation. External executables are
useful because they provide an independent finite-volume baseline, but the LMX
solver should be accepted or rejected based on physically matched observables.

## Fresh FreeMHD closed-channel parity

LMX now includes a fresh host-local FreeMHD cross-check for the supported
straight-duct transient lane:

```bash
python examples/freemhd_closed_channel_parity.py
```

That example reuses fresh Shercliff and Hunt reruns from
`/Users/rogerio/local/tests/freemhd_install/freemhd_output`, reconstructs the
matching LMX transient cases from the copied OpenFOAM fields and control files,
and writes a parity panel plus checked summary JSON. The panel now includes:

- final transverse and vertical velocity cuts
- transient `u_max(t)` histories from FreeMHD `fieldMinMax.dat` and LMX
  diagnostics
- same-host runtime comparison
- ranked profile-error and runtime offender tables in the summary JSON, so
  reviewer-facing parity gaps are explicit rather than inferred from the plot

Current bounded result from the fresh reruns on this host:

- Shercliff:
  - FreeMHD wall time: `35.32 s`
  - LMX wall time: `22.78 s`
  - `u_max` mismatch: `≈ 3.04e-2`
  - profile errors: `L2(y) ≈ 6.68e-2`, `L2(z) ≈ 1.04e-1`
- Hunt:
  - FreeMHD wall time: `35.28 s`
  - LMX wall time: `23.81 s`
  - `u_max` mismatch: `≈ 6.49e-2`
  - profile errors: `L2(y) ≈ 7.32e-2`, `L2(z) ≈ 7.88e-2`

This is useful as an implementation cross-check and host-runtime comparison,
but it is not currently the manuscript acceptance gate for the straight-duct
lane. The literature-backed analytical ladder remains the primary acceptance
surface because it is tighter and better conditioned. The FreeMHD transient
parity figure is still worth keeping in the docs and future paper as an
independent executable baseline.

![LMX vs FreeMHD straight-duct parity](_static/generated/freemhd_closed_channel_parity.png)

## FreeMHD paper-slice observable parity

LMX also includes a richer observable comparison workflow against the bundled
FreeMHD paper slices for the closed-channel family:

```bash
python examples/freemhd_closed_channel_observable_parity.py
```

That example compares normalized midplane profiles of:

- axial velocity `u`
- gauge-shifted electric potential `potE - potE(center)`
- cut-aligned current components `J_y` and `J_z`
- streamwise Lorentz force `J×B_x`

for Shercliff and Hunt against the processed slice CSV files in
`/Users/rogerio/local/tests/freemhd_test_cases/FreeMHDPaperAllFigures/ClosedChannel`.
It now runs on case-specific settings rather than a single showcase default:

- Shercliff:
  - analytical pressure gradient `2512.1961 Pa/m`
  - `49 × 37`, `64` steady steps, `face_averaged` current reconstruction
- Hunt:
  - analytical pressure gradient `514.2123 Pa/m`
  - `49 × 37`, `64` steady steps, `face_averaged` current reconstruction

Current retained result on that richer parity lane:

- Shercliff:
  - velocity: `L2(y) ≈ 8.49e-3`, `L2(z) ≈ 5.17e-3`
  - potential: `L2(y) ≈ 8.14e-1`, `L2(z) ≈ 3.51e-4`
  - current: `L2(y) ≈ 5.25e-1`, `L2(z) ≈ 1.07e-2`
  - Lorentz: `L2(y) ≈ 1.25e-2`, `L2(z) ≈ 1.07e-2`
- Hunt:
  - velocity: `L2(y) ≈ 1.26e-2`, `L2(z) ≈ 7.70e-3`
  - potential: `L2(y) ≈ 2.89e-1`, `L2(z) ≈ 1.68e-3`
  - current: `L2(y) ≈ 5.52e-1`, `L2(z) ≈ 9.80e-3`
  - Lorentz: `L2(y) ≈ 1.31e-2`, `L2(z) ≈ 9.80e-3`

This is now a useful manuscript-grade external comparison figure because it
shows much more than one velocity cut, but it also makes the remaining parity
gap explicit. The processed-slice extractor now interpolates between symmetric
near-center planes instead of interleaving duplicate cuts when the CSV has no
exact centerline. Near-degenerate y-cuts for `potE` and `J_y` are labelled as
low-signal in the summary JSON, so the physically significant blockers are now
the Shercliff/Hunt Lorentz-y response and the remaining y-velocity distortion.
Switching the Hunt comparison to the conservative face-current
reconstruction removed the previous dominant `J_z` / `J×B_z` artifact. Driving
the cases with the analytical pressure gradients also brings the absolute
velocity, current, and Lorentz peak scales into `O(1-5%)` agreement for the
dominant cuts. The retained `49 × 37` anisotropic mesh gives ten Hartmann-layer
cells and six side-layer cells, closing the main velocity/current/Lorentz cuts
to about `5e-3` to `1.3e-2`. The low-signal `potE(y)` / `J_y` cuts are tracked
separately rather than treated as leading physics offenders.

![LMX vs FreeMHD observable parity](_static/generated/freemhd_closed_channel_observable_parity.png)

## Next external-parity gate

The next FreeMHD-facing work should focus on the straight-duct fully developed
path before adding more comparison plots. The target is a physically matched
constant-flow-rate solve with:

- boundary-layer-focused mesh convergence
- matched velocity, gauge-shifted potential, current-density, and Lorentz-force
  cuts
- integral flow-rate and pressure/forcing observables
- explicit comparison of compile time, warm runtime, and memory-relevant grid
  sizes when the same host is used

The current code now enforces the requested area-weighted mean velocity after
wall interpolation and limiter application for `inlet_flow_rate` runs, and
`processed_slice_area_mean` derives case-specific constant-`Q` targets directly
from the nonuniform processed FreeMHD slice. This matters because the Ha=20
paper slices imply different mean speeds for the two wall models
(`0.97017` for Shercliff and `0.11741` for Hunt). On the retained `49 x 37`
mesh, the constrained-flow solve now matches the requested mean exactly and
gives velocity-cut errors of `8.49e-3` / `5.20e-3` for Shercliff and
`1.26e-2` / `7.70e-3` for Hunt. A follow-up Hunt `57 x 43` run improved the
side cut but worsened the wall-normal cut, so the remaining `O(1.3e-2)` Hunt
shape mismatch should be treated as a solver/observable-parity issue rather
than a basic layer-cell-count issue.

The latest targeted solver probes ruled out two tempting but weak fixes. The
Hunt limiter has no measurable effect on the retained velocity shape, and the
`hybrid_face_lorentz` path is numerically identical to `face_averaged` for the
straight-duct velocity profile. A face-current-diagonal implicit reaction
prototype was also rejected: it was neutral on the cheap Hunt flow-rate probe
and regressed the small Hunt validation gate. The remaining parity work should
therefore focus on a fully implicit face-current momentum operator, including
off-diagonal EMF terms, or a direct OpenFOAM-mesh reproduction, not on limiter
tuning or diagonal-only reaction splitting.

`examples/reference_slice_mesh_diagnostic.py` is the mesh-isolation driver for
that next step. On the bundled Hunt Ha=20 processed slice, the external point
grid carries about ten Hartmann-layer intervals and about forty-two side-layer
intervals, while the retained generated LMX mesh carries about ten and six.
That makes side-layer mesh parity a concrete diagnostic to run before changing
the fully developed momentum operator again.

The observable-parity summary JSON now ranks offenders across velocity,
gauge-shifted potential, current, and Lorentz-force cuts while demoting
low-signal normalized cuts. That ranking is the triage surface for the next
solver work: fix the last `O(1.3e-2)` Hunt/Shercliff wall-normal
velocity/Lorentz offenders before adding more FreeMHD comparison figures.

Mapped-pipe external parity remains documented but deferred. The bundled pipe
reference corresponds to a high-`Ha`, high-`Re` fringing-pipe case, while the
current mapped-pipe LMX path is still a lower-Re inductionless slice. It should
not be promoted to a parity claim until the pipe solver is in the matching
regime.

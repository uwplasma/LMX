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
  - `17 × 17`, `48` steady steps, `face_averaged` current reconstruction
- Hunt:
  - `13 × 13`, `48` steady steps, `cell_centered` current reconstruction

Current retained result on that richer parity lane:

- Shercliff:
  - velocity: `L2(y) ≈ 6.30e-2`, `L2(z) ≈ 1.01e-1`
  - potential: `L2(y) ≈ 3.54e-1`, `L2(z) ≈ 2.31e-2`
  - current: `L2(y) ≈ 6.46e-1`, `L2(z) ≈ 4.12e-2`
  - Lorentz: `L2(y) ≈ 7.86e-2`, `L2(z) ≈ 4.12e-2`
- Hunt:
  - velocity: `L2(y) ≈ 2.39e-1`, `L2(z) ≈ 3.73e-2`
  - potential: `L2(y) ≈ 1.10e0`, `L2(z) ≈ 3.49e-2`
  - current: `L2(y) ≈ 1.02e0`, `L2(z) ≈ 5.52e-1`
  - Lorentz: `L2(y) ≈ 2.71e-1`, `L2(z) ≈ 5.52e-1`

This is now a useful manuscript-grade external comparison figure because it
shows much more than one velocity cut, but it also makes the remaining parity
gap explicit: the fully developed constant-`Q`/flow-rate lane is still the
main blocker, and field-level parity is not yet closed on `potE`, `J`, and
`J×B`.

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
wall interpolation and limiter application for `inlet_flow_rate` runs. A fresh
`flow_rate` probe shows that this removes one numerical ambiguity but does not
close observable parity by itself: the remaining gap is in the coupled
pressure-gradient/current reconstruction and field-level matching, not only in
mass-flow normalization.

The observable-parity summary JSON now also ranks offenders across velocity,
gauge-shifted potential, current, and Lorentz-force cuts. That ranking is the
triage surface for the next solver work: fix the largest `potE`, `J`, and
`J×B` offenders before adding more FreeMHD comparison figures.

Mapped-pipe external parity remains documented but deferred. The bundled pipe
reference corresponds to a high-`Ha`, high-`Re` fringing-pipe case, while the
current mapped-pipe LMX path is still a lower-Re inductionless slice. It should
not be promoted to a parity claim until the pipe solver is in the matching
regime.

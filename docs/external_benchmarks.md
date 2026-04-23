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
and writes a parity panel plus checked summary JSON.

Current bounded result from the fresh reruns on this host:

- Shercliff:
  - FreeMHD wall time: `35.32 s`
  - LMX wall time: `15.06 s`
  - `u_max` mismatch: `≈ 5.57e-2`
  - profile errors: `L2(y) ≈ 6.66e-2`, `L2(z) ≈ 1.03e-1`
- Hunt:
  - FreeMHD wall time: `35.28 s`
  - LMX wall time: `27.66 s`
  - `u_max` mismatch: `≈ 5.08e-2`
  - profile errors: `L2(y) ≈ 7.21e-2`, `L2(z) ≈ 8.14e-2`

This is useful as an implementation cross-check and host-runtime comparison,
but it is not currently the manuscript acceptance gate for the straight-duct
lane. The literature-backed analytical ladder remains the primary acceptance
surface because it is tighter and better conditioned. The FreeMHD transient
parity figure is still worth keeping in the docs and future paper as an
independent executable baseline.

![LMX vs FreeMHD straight-duct parity](_static/generated/freemhd_closed_channel_parity.png)

# Validation Report

## Implemented now

- Structured duct meshes with optional conducting wall layers.
- Laminar inductionless field solve with `u`, `phi`, `J`, and `JxB`.
- Hartmann analytical profile helper.
- Hartmann validation JSON generation from the CLI and the validation suite runner.
- Shercliff and Hunt analytical comparison JSON generation from the CLI and the validation suite runner.
- Shercliff and Hunt processed-slice `XSlice` comparison JSON generation when Zenodo CSV exports are available.
- CSV and ParaView outputs for centerline and field inspection.
- FreeMHD comparison harness metadata, container bundle generation, and a local container execution helper script.
- FreeMHD environment/setup inspection scripts that report Docker CLI vs daemon availability and recommend the smallest current target case.
- Partial `StartingFiles.zip` inspection and selective case materialization utilities for recovering early laminar paper cases before the full archive is complete.
- Recovered Shercliff `Ha0` and `Ha20` starting-case materialization from the partial archive.
- FreeMHD setup inspection with `--extra-case-root` so recovered `/tmp` cases can be reported in the same setup JSON as the checked-in assets.
- FreeMHD run-helper fail-fast classification for missing local Docker image tags.
- FreeMHD container preflight reporting for local image presence and bounded-time base-image registry resolution.
- FreeMHD container preflight reporting for timed base-image pull checks on the local Docker daemon.
- Darwin-specific FreeMHD header-patch helper for reproducible local `wmake` experiments on macOS.
- FreeMHD run-control overrides for short smoke/parity runs (`deltaT`, `endTime`, `writeInterval`).
- FreeMHD multi-region case inspection of `0/`, `processors*`, and `fieldMinMax.dat` artifacts.
- Coarse `LMX`-vs-`FreeMHD` transient comparison based on the latest FreeMHD `mag(U)` max entry.
- FreeMHD sampled line-cut extraction through `postProcessing/lmxSampleDict/...` and profile-based `LMX`-vs-`FreeMHD` parity metrics.
- Nonzero transient initialization in LMX through `CaseSpec.initial_velocity`.
- Automatic line-cut geometry inference from the latest FreeMHD `fieldMinMax.dat` location data.
- A checked-in parity report runner that wraps LMX case construction plus `compare_with_freemhd` into one JSON artifact.
- GitHub Actions validation artifacts for Hartmann, Shercliff, and Hunt runs.
- GitHub Actions benchmark artifacts for the current Hartmann timing path.
- Zenodo closed-channel analytical text ingestion for Shercliff and Hunt.
- Zenodo processed closed-channel slice ingestion for future parity checks against the paper figures.

## Planned next

- Improve Hunt accuracy now that the default path is bounded and emits finite analytical and processed-slice metrics.
- Extend the current sampled line-cut parity path beyond the recovered Shercliff `Ha20` smoke case.
- Promote the current sampled `y/z` profile comparison into a checked-in CI artifact/report job.
- Generalize the current geometry inference beyond `fieldMinMax`-driven duct cases when more FreeMHD geometries are added.
- Fringing-field mapped-pipe operators.
- Stronger acceptance thresholds that can fail CI on parity regressions rather than only emitting artifacts.

## Current limitation

- Hartmann `Ha=20` is now stable on the default finer mesh and shows a low normalized error in the current analytical comparison.
- Shercliff `Ha=20` no longer clips on the default finer mesh, but its normalized error is still too large to count as parity-complete.
- Hunt no longer clips at the default pseudo-time settings after adding the adaptive velocity-update limiter, and the validation reports now emit finite Hunt comparison metrics.
- Hunt still requires solver-fidelity work because the bounded default solution is not yet close enough to the ingested reference data to count as parity-complete, but the tuned default pseudo-step now improves the `Ha=20` Hunt errors materially relative to the earlier bounded baseline.
- Semi-implicit treatment of the linear Lorentz damping term improved Hartmann and Shercliff robustness further, but Hunt needed the additional adaptive update limiter on top of that.
- The Docker daemon is now reachable on the local machine, and recovered Shercliff `Ha20` case directories can be surfaced through `inspect_freemhd_setup.py --extra-case-root ...`.
- The Docker/image blocker has now been reduced further: the image builds and runs locally as `lmx-freemhd-smoke` when forced to `linux/amd64` on this machine.
- The recovered Shercliff `Ha20` case now executes end to end in the container for short smoke runs and writes reconstructed `0.0001/` output plus `fieldMinMax.dat`.
- The first coarse transient parity check is now available:
  - FreeMHD latest `max |U| = 0.973457584` at `t = 1e-4`
  - LMX short transient with matched `initial_velocity = 0.9725` gives `max |U| = 0.9721652865`
  - absolute difference is about `1.29e-3`
- The sampled-line profile comparison is now also available on the same recovered Shercliff `Ha20` smoke case:
  - `freemhd_sample_y_l2_error ≈ 5.83e-4`
  - `freemhd_sample_z_l2_error ≈ 2.67e-4`
  - these are based on sampled `centerlineY` and `centerlineZ` cuts from reconstructed FreeMHD `0.0001/liquid/U`
- The sampling runner now infers the current Shercliff `Ha20` cut geometry automatically from the latest FreeMHD `fieldMinMax.dat` record:
  - `x_position = 0.015`
  - `y = [-0.1, 0.1]`
  - `z = [-0.099995, 0.099995]`
- The current local Darwin `wmake` path now moves past the libc++ header-shadowing failure when the patch helper is applied, but it then fails on a new `fvMesh.H` include regression. That means the repo now has a reproducible two-step local build diagnosis instead of a single opaque `wmake` failure.

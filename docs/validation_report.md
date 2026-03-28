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
- GitHub Actions validation artifacts for Hartmann, Shercliff, and Hunt runs.
- GitHub Actions benchmark artifacts for the current Hartmann timing path.
- Zenodo closed-channel analytical text ingestion for Shercliff and Hunt.
- Zenodo processed closed-channel slice ingestion for future parity checks against the paper figures.

## Planned next

- Improve Hunt accuracy now that the default path is bounded and emits finite analytical and processed-slice metrics.
- Fringing-field mapped-pipe operators.
- Tighten the FreeMHD execution helper into solver-specific parity runners that extract matching LMX/FreeMHD metrics.
- Acquire or reconstruct a standalone `epotMultiRegion*` laminar paper case locally, since the current local assets only include the solver sources and processed figures.
- Resume the larger `StartingFiles.zip` archive far enough to recover `shercliff_Ha20_ConstantQ_OutletZeroGradientInletCodedUxBpotE` after the now-working partial-archive tooling.
- Stronger acceptance thresholds that can fail CI on parity regressions rather than only emitting artifacts.

## Current limitation

- Hartmann `Ha=20` is now stable on the default finer mesh and shows a low normalized error in the current analytical comparison.
- Shercliff `Ha=20` no longer clips on the default finer mesh, but its normalized error is still too large to count as parity-complete.
- Hunt no longer clips at the default pseudo-time settings after adding the adaptive velocity-update limiter, and the validation reports now emit finite Hunt comparison metrics.
- Hunt still requires solver-fidelity work because the bounded default solution is not yet close enough to the ingested reference data to count as parity-complete, but the tuned default pseudo-step now improves the `Ha=20` Hunt errors materially relative to the earlier bounded baseline.
- Semi-implicit treatment of the linear Lorentz damping term improved Hartmann and Shercliff robustness further, but Hunt needed the additional adaptive update limiter on top of that.
- The local machine currently has the Docker CLI installed but the Docker daemon is not reachable from the active environment.
- The current local asset set does not contain any standalone `epotMultiRegionFoam` or `epotMultiRegionInterFoam` case directories, so the bundled OpenFOAM Hartmann tutorial is only an environment smoke target, not a FreeMHD parity case.

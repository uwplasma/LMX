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
- Darwin-specific FreeMHD header-patch helper for reproducible local `wmake` experiments on macOS.
- GitHub Actions validation artifacts for Hartmann, Shercliff, and Hunt runs.
- GitHub Actions benchmark artifacts for the current Hartmann timing path.
- Zenodo closed-channel analytical text ingestion for Shercliff and Hunt.
- Zenodo processed closed-channel slice ingestion for future parity checks against the paper figures.

## Planned next

- Improve Hunt accuracy now that the default path is bounded and emits finite analytical and processed-slice metrics.
- Fringing-field mapped-pipe operators.
- Tighten the FreeMHD execution helper into solver-specific parity runners that extract matching LMX/FreeMHD metrics.
- Prove the Docker image build path end to end and run the first recovered Shercliff `Ha20` case smoke test inside the container.
- Extend the current setup and run reports into solver-specific parity runners that extract matching LMX/FreeMHD metrics once container execution is live.
- Stronger acceptance thresholds that can fail CI on parity regressions rather than only emitting artifacts.

## Current limitation

- Hartmann `Ha=20` is now stable on the default finer mesh and shows a low normalized error in the current analytical comparison.
- Shercliff `Ha=20` no longer clips on the default finer mesh, but its normalized error is still too large to count as parity-complete.
- Hunt no longer clips at the default pseudo-time settings after adding the adaptive velocity-update limiter, and the validation reports now emit finite Hunt comparison metrics.
- Hunt still requires solver-fidelity work because the bounded default solution is not yet close enough to the ingested reference data to count as parity-complete, but the tuned default pseudo-step now improves the `Ha=20` Hunt errors materially relative to the earlier bounded baseline.
- Semi-implicit treatment of the linear Lorentz damping term improved Hartmann and Shercliff robustness further, but Hunt needed the additional adaptive update limiter on top of that.
- The Docker daemon is now reachable on the local machine, and recovered Shercliff `Ha20` case directories can be surfaced through `inspect_freemhd_setup.py --extra-case-root ...`.
- The current Docker/image blocker has narrowed to image availability and base-image resolution: the helper now reports `docker-image-unavailable` cleanly when the requested tag does not exist locally.
- The current container preflight on this machine shows `openfoam/openfoam2206-paraview:latest` is not present locally and `docker manifest inspect` for that base image times out, so the unresolved part is now explicitly the base-image pull/resolve path rather than recovered-case discovery.
- The current local Darwin `wmake` path now moves past the libc++ header-shadowing failure when the patch helper is applied, but it then fails on a new `fvMesh.H` include regression. That means the repo now has a reproducible two-step local build diagnosis instead of a single opaque `wmake` failure.
- The recovered Shercliff `Ha20` case is structurally inspectable, but the container image path is still not proven end to end in this session, so no actual FreeMHD run has completed yet.

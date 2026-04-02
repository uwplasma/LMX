# LMX

LMX is a Python/JAX solver for inductionless liquid-metal magnetohydrodynamics.
It is designed to be self-consistent, differentiable, and fast on CPU and GPU while
staying focused on structured duct and simple pipe geometries.

## What LMX provides

- Structured and mapped-structured mesh generation for ducts and pipes.
- JAX-native finite-volume style operators and time stepping.
- Laminar inductionless MHD for Hartmann, Shercliff, and Hunt-style cases.
- Explicit fluid/solid conductivity regions for conducting and insulating wall layers.
- Hunt defaults expressed through wall conductance ratio, with optional direct
  wall-conductivity override when a case is specified that way.
- ParaView output, CSV/profile extraction, and benchmark reporting.
- Analytical validation helpers and regression tests for reproducibility.
- Optional external validation tooling for recovered FreeMHD/OpenFOAM cases.
- Validation summaries now include a normalized electric-potential equation
  residual, so solver/control issues in the `phi` solve are visible in normal
  artifacts instead of only through profile errors.
- Closed-channel validation artifacts now also include a combined profile error,
  so Hunt/Shercliff tradeoffs are not judged from `y` and `z` cuts separately.
- The electric-potential solve now supports `auto`, weighted-Jacobi, and CG
  control paths. The current default is a geometry-aware `auto` policy:
  single-region ducts use CG, while multi-region layered ducts use
  `cg_volume`, the cell-metric-scaled CG form of the layered `phi` system.
- Layered conducting-wall cases also expose `potential_solver="cg_volume"`,
  which solves the same layered `phi` equation after cell-metric scaling into a
  symmetric CG system. It is now the retained layered default because it
  improves the full recovered Hunt parity path at both `Ha20` and `Ha100`.
- Hunt diagnostics also expose `current_reconstruction`, with the retained
  default `cell_centered` and an experimental `face_averaged` mode for tracing
  current-distribution effects without changing the public case API.
- Hunt diagnostic replays also expose `velocity_update_limiter`. The retained
  default is `global_scale`; the experimental `local_clip` path is available
  for targeted layered-update diagnosis, but it is currently worse on the
  corrected `Ha20`, `t <= 6e-05` Hunt replay.

## Quick start

```bash
cd /Users/rogerio/local/tests/LMX
/Users/rogerio/base_env/bin/python3 -m pytest
/Users/rogerio/base_env/bin/python3 -m pytest -m unit
/Users/rogerio/base_env/bin/python3 -m pytest -m regression
/Users/rogerio/base_env/bin/python3 -m pytest -m physics
/Users/rogerio/base_env/bin/python3 -m pytest -m validation
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
/Users/rogerio/base_env/bin/python3 -m lmx.cli run shercliff --ha 20 --output ./out/shercliff
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hartmann --ha 20 --output ./out/validation/hartmann
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate shercliff --ha 20 --output ./out/validation/shercliff --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_convergence_suite.py --output ./artifacts/convergence --cases hartmann,shercliff,hunt --ha 20 --resolutions 16,32,48 --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_time_convergence_suite.py --output ./artifacts/time_convergence --cases hartmann,shercliff,hunt --ha 20 --resolution 32 --dts 0.002,0.001,0.0005 --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep --case hunt --ha 20 --resolution 48 --wall-cells 5 --parameter outer_iterations --values 2,4,6,8,10 --value-type int --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_hartmann --case hartmann --ha 20 --resolution 32 --parameter potential_iterations --values 50,100,200,400,800 --value-type int
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_phi_tol --case hartmann --ha 20 --resolution 32 --parameter potential_tolerance --values 1e-2,1e-3,1e-4 --value-type float
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_phi_backend --case shercliff --ha 20 --resolution 32 --parameter potential_solver --values auto,jacobi,cg --value-type str --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_hunt_phi_backend --case hunt --ha 100 --resolution 32 --parameter potential_solver --values jacobi,cg_volume --value-type str --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_hunt_current_reconstruction --case hunt --ha 20 --resolution 32 --parameter current_reconstruction --values cell_centered,face_averaged --value-type str --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_velocity_limit --case hunt --ha 20 --resolution 32 --parameter velocity_update_limit --values 5e-4,1e-3,2e-3,4e-3 --value-type float --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_grid_sweep.py --output ./artifacts/control_grid_hunt --case hunt --ha 20 --resolution 32 --parameter-a outer_iterations --values-a 4,6 --type-a int --parameter-b potential_relaxation --values-b 1.0,0.5 --type-b float --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/compare_hunt_trace_histories.py --freemhd-diag-json /tmp/lmx_hunt20_live_diag.json --lmx-report-json /tmp/lmx_hunt_auto_short_trace_force.json --output ./artifacts/hunt_trace_alignment.json
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

## Documentation

- Online docs: `https://lmx.readthedocs.io`
- Local docs entrypoint: [`docs/index.md`](docs/index.md)
- Theory: [`docs/theory.md`](docs/theory.md)
- Developer guide: [`docs/developer_guide.md`](docs/developer_guide.md)
- Case cookbook: [`docs/case_cookbook.md`](docs/case_cookbook.md)
- Validation report: [`docs/validation_report.md`](docs/validation_report.md)

## Validation backends

External validation backends and archived case directories are optional. They are
useful for regression, historical comparison, and solver cross-checks, but they do
not define the public interface or the governing formulation implemented by LMX.

Useful optional-backend commands:

```bash
/Users/rogerio/base_env/bin/python3 scripts/fetch_freemhd_assets.py --dest ./external
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/build_freemhd_container.py --image lmx-freemhd-localdiag --local-freemhd-root ./external/FreeMHD
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd-localdiag-density --case-dir /tmp/lmx_hunt_case_extract/StartingFiles/Hunt/hunt_exactBL_Ha20 --local-freemhd-root ./external/FreeMHD --patch-local-freemhd-logging --cores 4 --end-time 3e-05 --write-interval 1e-05 --output ./artifacts/freemhd_hunt_density_run.json
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_parity_suite.py --output ./artifacts/freemhd_parity
/Users/rogerio/base_env/bin/python3 scripts/run_hunt_solver_diagnostic_report.py --freemhd-run-dir ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel/Hunt/ha20 --ha 20 --output ./artifacts/hunt_solver_diagnostics.json
/Users/rogerio/base_env/bin/python3 scripts/run_hunt_solver_diagnostic_report.py --freemhd-run-dir /tmp/lmx_hunt_case_extract/StartingFiles/Hunt/hunt_exactBL_Ha20 --ha 20 --dt 1e-5 --t-final 3e-5 --max-steps 3 --current-reconstruction face_averaged --output ./artifacts/hunt_solver_diagnostics_faceavg.json
/Users/rogerio/base_env/bin/python3 scripts/patch_freemhd_coupled_logging.py --root ./external/FreeMHD
/Users/rogerio/base_env/bin/python3 scripts/extract_freemhd_coupled_log.py ./artifacts/freemhd_hunt.log --output ./artifacts/freemhd_hunt_diag.json
```

`build_freemhd_container.py` now uses a loadable `docker buildx build --load`
path so locally patched validation-backend images are available to subsequent
`docker run` smoke and parity scripts instead of staying only in builder output.
The parity loaders also infer `BtStartTime` and `BtDuration` from
`system/controlDict`, so short transient LMX replays now use the same magnetic
field ramp settings as the recovered validation cases when those controls are
present.
`run_freemhd_case.py` can now auto-build a missing local diagnostic image from a
patched checkout when `--local-freemhd-root` is provided, and it can patch that
checkout in place with `--patch-local-freemhd-logging` before the build. That is
the recommended path for reproducing Hunt trace diagnostics on this machine.
`run_hunt_solver_diagnostic_report.py` also mirrors recovered Hunt startup
cases more faithfully now: when the recovered run has a nonzero startup
velocity in `0/liquid/U`, the reduced replay automatically adds the matching
`inlet_velocity` boundary instead of treating the case as an unforced
`forcing = 0` transient.
In the reduced duct solver itself, only `inlet_flow_rate` now activates the
mean-flow drive closure. `inlet_velocity` remains available for recovered-case
metadata and startup-state parity, but it no longer imposes a global target
mean velocity across the reduced cross-section.

## Scope

LMX does not yet include:

- Free-surface VoF.
- Temperature coupling.
- Turbulence models.
- General unstructured polyhedral numerics.

## Repository Layout

- `lmx/`: package code.
- `tests/`: unit, regression, physics, validation, and benchmark tests.
- `docs/`: Read the Docs source pages.
- `.github/workflows/`: CI and benchmark workflows.
- `scripts/`: validation and benchmarking helpers.
- `docker/`: optional validation container bundle.

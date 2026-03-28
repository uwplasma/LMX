# LMX

LMX is a Python/JAX codebase for laminar inductionless MHD, structured around the FreeMHD laminar parity plan.

This first implementation includes:

- Tensor-product and mapped-grid mesh builders for ducts and pipes.
- JAX-native finite-volume style operators on structured cross-sections.
- A laminar fully developed inductionless MHD duct solver with explicit fluid/solid conductivity regions.
- Mesh-safe pseudo-transient defaults for Hartmann and Shercliff duct runs on finer meshes.
- ParaView XML writers for structured and mapped grids.
- Validation helpers, analytical Hartmann reference profiles, and a FreeMHD asset/container harness.
- Explicit unit, regression, physics, validation, and benchmark entrypoints with GitHub Actions workflows.
- Zenodo closed-channel analytical and processed-slice loaders for Shercliff and Hunt reference ingestion.
- Recovered `StartingFiles` case materialization for Shercliff laminar paper cases, plus FreeMHD environment probes and fail-fast container harness checks.
- Platform-aware FreeMHD container execution on Apple silicon via `linux/amd64`, plus short-run `controlDict` overrides for smoke/parity jobs.
- A first coarse `LMX`-vs-`FreeMHD` comparison path based on FreeMHD `fieldMinMax.dat` output and matched nonzero LMX initial velocity.
- Profile-based FreeMHD parity extraction plus a parity-suite artifact path in GitHub Actions that emits either a real report or a structured `skipped` report when no recovered case is available.
- A shared `LMX_FREEMHD_CASE_DIR` convention so the same parity-suite runner can produce real FreeMHD artifacts on machines that have recovered paper cases.

This repository does **not** yet include:

- Free-surface VoF.
- Temperature coupling.
- Turbulence models.
- Full 3D OpenFOAM-equivalent polyhedral numerics.

## Quick start

```bash
cd /Users/rogerio/local/tests/LMX
/Users/rogerio/base_env/bin/python3 -m pytest
/Users/rogerio/base_env/bin/python3 -m pytest -m unit
/Users/rogerio/base_env/bin/python3 -m pytest -m regression
/Users/rogerio/base_env/bin/python3 -m pytest -m physics
/Users/rogerio/base_env/bin/python3 -m pytest -m validation
/Users/rogerio/base_env/bin/python3 -m lmx.cli run shercliff --ha 20 --output ./out
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hartmann --ha 20 --output ./out_validation
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate shercliff --ha 20 --output ./out_validation/shercliff --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel --x-slice 1m
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
/Users/rogerio/base_env/bin/python3 scripts/fetch_freemhd_assets.py --dest ./external
/Users/rogerio/base_env/bin/python3 scripts/write_freemhd_container_files.py
/Users/rogerio/base_env/bin/python3 scripts/probe_freemhd_environment.py --output ./artifacts/freemhd_probe.json
/Users/rogerio/base_env/bin/python3 scripts/patch_freemhd_darwin_headers.py --output ./artifacts/freemhd_darwin_patch.json
/Users/rogerio/base_env/bin/python3 scripts/probe_freemhd_container.py --image lmx-freemhd --output ./artifacts/freemhd_container.json
/Users/rogerio/base_env/bin/python3 scripts/probe_freemhd_container.py --image lmx-freemhd --check-pull --pull-timeout-seconds 20 --output ./artifacts/freemhd_container_pull.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_starting_files_archive.py --pattern shercliff_Ha0_refinedMesh
/Users/rogerio/base_env/bin/python3 scripts/inspect_starting_files_archive.py --pattern shercliff_Ha0_refinedMesh --extract --output-dir /tmp/startingfiles_ha0
/Users/rogerio/base_env/bin/python3 scripts/materialize_starting_case.py --case-dir /tmp/startingfiles_ha0/StartingFiles/Shercliff/shercliff_Ha0_refinedMesh
/Users/rogerio/base_env/bin/python3 scripts/inspect_starting_files_archive.py --pattern shercliff_Ha20_ConstantQ_OutletZeroGradientInletCodedUxBpotE --extract --output-dir /tmp/startingfiles_ha20
/Users/rogerio/base_env/bin/python3 scripts/materialize_starting_case.py --case-dir /tmp/startingfiles_ha20/StartingFiles/Shercliff/shercliff_Ha20_ConstantQ_OutletZeroGradientInletCodedUxBpotE
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --extra-case-root /tmp/startingfiles_ha20/StartingFiles/Shercliff --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_case.py --case-dir ./external/FreeMHD/OpenFOAM-v2206/tutorials/electromagnetics/mhdFoam/hartmann --output ./artifacts/freemhd_case_hartmann.json
/Users/rogerio/base_env/bin/python3 scripts/build_freemhd_container.py --image lmx-freemhd-smoke --platform linux/amd64
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd-smoke --case-dir /absolute/path/to/freemhd_case --platform linux/amd64 --output ./artifacts/freemhd/run.json
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd-smoke --case-dir /absolute/path/to/freemhd_case --platform linux/amd64 --cores 8 --delta-t 1e-5 --end-time 1e-4 --write-interval 1e-4 --output ./artifacts/freemhd/run_smoke.json
/Users/rogerio/base_env/bin/python3 scripts/sample_freemhd_profiles.py --case-dir /absolute/path/to/freemhd_case --image microfluidica/openfoam:2206 --time 0.0001 --dict-name lmxSampleDict --output ./artifacts/freemhd/sample_profiles.json
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_parity_report.py --case-kind shercliff --ha 20 --freemhd-run-dir /absolute/path/to/freemhd_case --output ./artifacts/freemhd/parity_report.json
```

## Layout

- `lmx/`: package code.
- `tests/`: unit, regression, physics, and validation tests.
- `.github/workflows/`: CI and benchmark workflows.
- `docs/`: theory, developer, cookbook, and validation notes.
- `scripts/`: FreeMHD data, reference ingestion, and container helpers.
- `docker/`: checked-in FreeMHD container bundle synchronized with the generator script.

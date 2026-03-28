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
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate shercliff --ha 20 --output ./out_validation/shercliff --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
/Users/rogerio/base_env/bin/python3 scripts/fetch_freemhd_assets.py --dest ./external
```

## Layout

- `lmx/`: package code.
- `tests/`: unit, regression, physics, and validation tests.
- `.github/workflows/`: CI and benchmark workflows.
- `docs/`: theory, developer, cookbook, and validation notes.
- `scripts/`: FreeMHD data, reference ingestion, and container helpers.

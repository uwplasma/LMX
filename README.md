# LMX

LMX is a Python/JAX codebase for laminar inductionless MHD, structured around the FreeMHD laminar parity plan.

This first implementation includes:

- Tensor-product and mapped-grid mesh builders for ducts and pipes.
- JAX-native finite-volume style operators on structured cross-sections.
- A laminar fully developed inductionless MHD duct solver with explicit fluid/solid conductivity regions.
- ParaView XML writers for structured and mapped grids.
- Validation helpers, analytical Hartmann reference profiles, and a FreeMHD asset/container harness.
- Explicit unit, regression, physics, validation, and benchmark entrypoints with GitHub Actions workflows.

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
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

## Layout

- `lmx/`: package code.
- `tests/`: unit and smoke tests.
- `.github/workflows/`: CI and benchmark workflows.
- `docs/`: theory, developer, cookbook, and validation notes.
- `scripts/`: FreeMHD data and container helpers.

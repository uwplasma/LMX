# Case Cookbook

## Hartmann

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
```

Use this as the simplest solver smoke test and as the default analytical-profile
validation case.

## Shercliff

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run shercliff --ha 100 --output ./out/shercliff
```

This is the insulating-wall duct case. It exercises the same self-consistent solver
path as the other native cases while remaining a strong analytical comparison target.

## Hunt

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hunt --ha 100 --output ./out/hunt
```

This is the conducting-wall duct case. It is useful for checking how the solver
handles explicit solid conductivity layers and sharper boundary-layer physics.

## Optional external validation backends

Recovered external cases are optional. They are useful when you want to compare LMX
against archived paper cases or regenerated backend outputs, but they are not required
to run the native solver.

```bash
python3 scripts/fetch_freemhd_assets.py --dest ./external
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_parity_suite.py --output ./artifacts/freemhd_parity
```

## Validation checks

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hartmann --ha 20 --output ./out_validation/hartmann
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate shercliff --ha 20 --output ./out_validation/shercliff --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hunt --ha 20 --output ./out_validation/hunt --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
```

These commands produce the analytical and sampled comparison reports that back the
current regression and validation tests.

## Benchmarking

```bash
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

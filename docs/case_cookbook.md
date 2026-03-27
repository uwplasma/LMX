# Case Cookbook

## Hartmann

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
```

Use this for solver smoke tests and analytical profile comparisons.

## Shercliff

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run shercliff --ha 100 --output ./out/shercliff
```

This is the insulating-wall duct parity case. The present implementation supports the field solve and output generation; high-fidelity analytical parity ingestion remains in `lmx.validation`.

## Hunt

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hunt --ha 100 --output ./out/hunt
```

This creates a duct with explicit conducting Hartmann-wall layers.

## FreeMHD assets

```bash
python3 scripts/fetch_freemhd_assets.py --dest ./external
python3 scripts/fetch_freemhd_assets.py --dest ./external --include-starting-files
python3 scripts/write_freemhd_container_files.py
```

The first command fetches the FreeMHD repository and the processed-figures archive. The `--include-starting-files` variant is optional and downloads the much larger case-input archive.

## Hartmann validation

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hartmann --ha 20 --output ./out_validation/hartmann
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation
```

This writes a JSON summary with the normalized Hartmann-profile error against the current analytical helper. The validation suite runner also emits Shercliff and Hunt artifacts for CI.

## Shercliff and Hunt reference comparison

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate shercliff --ha 20 --output ./artifacts/reference_compare/shercliff --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hunt --ha 20 --output ./artifacts/reference_compare/hunt --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
```

These commands write normalized midplane error reports against the ingested Zenodo analytical files for the corresponding closed-channel cases.

## Benchmark

```bash
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

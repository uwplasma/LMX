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
/Users/rogerio/base_env/bin/python3 scripts/inspect_starting_files_archive.py --pattern shercliff_Ha0_refinedMesh
/Users/rogerio/base_env/bin/python3 scripts/inspect_starting_files_archive.py --pattern shercliff_Ha0_refinedMesh --extract --output-dir /tmp/startingfiles_ha0
/Users/rogerio/base_env/bin/python3 scripts/materialize_starting_case.py --case-dir /tmp/startingfiles_ha0/StartingFiles/Shercliff/shercliff_Ha0_refinedMesh
```

The first command fetches the FreeMHD repository and the processed-figures archive. The `--include-starting-files` variant is optional and downloads the much larger case-input archive. If that archive is incomplete, `inspect_starting_files_archive.py` can still read recoverable entries directly from local ZIP headers and selectively extract early cases. `materialize_starting_case.py` then expands the recovered `0.tar.gz`, `constant.tar.gz`, and `system.tar.gz` files into a normal OpenFOAM case directory.

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
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel --x-slice 1m
```

These commands write normalized midplane error reports against the ingested Zenodo analytical files for the corresponding closed-channel cases. When the processed slice CSV exists, they also emit figure-level `*_slice.json` reports against the paper-exported `XSlice` cuts.

## FreeMHD container harness

```bash
/Users/rogerio/base_env/bin/python3 scripts/write_freemhd_container_files.py
/Users/rogerio/base_env/bin/python3 scripts/probe_freemhd_environment.py --output ./artifacts/freemhd_probe.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_case.py --case-dir ./external/FreeMHD/OpenFOAM-v2206/tutorials/electromagnetics/mhdFoam/hartmann --output ./artifacts/freemhd_case_hartmann.json
docker build -t lmx-freemhd ./docker
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd --case-dir /absolute/path/to/freemhd_case --output ./artifacts/freemhd/run.json
```

This path generates a local Docker build context with OpenFOAM v2206, FreeMHD, and a case runner script. The probe command records whether the local OpenFOAM tree is sourceable, whether `wmkdepend` exists for local `wmake` builds, whether the Docker daemon is reachable, and classifies a few common `wmake` failure modes such as missing `wmkdepend` or the macOS libc++ header conflict. The setup inspection command reports whether runnable FreeMHD cases exist locally and recommends the smallest smoke target when they do not. The case inspection command records the actual case structure that a parity run would consume. A partially recovered Shercliff `Ha0` paper case can now be materialized locally from the truncated `StartingFiles.zip`; the next more relevant parity target remains the `Ha20` Shercliff case once the archive download is resumed far enough. The execution helper mounts the case directory, runs either serial or decomposed cases, and stores a machine-readable run report or a structured daemon-unavailable status.

## Benchmark

```bash
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

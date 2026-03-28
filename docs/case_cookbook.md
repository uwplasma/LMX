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

Current recovery status: the Shercliff `Ha0` and `Ha20` starting cases can now be materialized locally from the partial archive, including the `epotMultiRegionInterFoam` `Ha20` paper case shell.

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
/Users/rogerio/base_env/bin/python3 scripts/patch_freemhd_darwin_headers.py --output ./artifacts/freemhd_darwin_patch.json
/Users/rogerio/base_env/bin/python3 scripts/probe_freemhd_container.py --image lmx-freemhd --output ./artifacts/freemhd_container.json
/Users/rogerio/base_env/bin/python3 scripts/probe_freemhd_container.py --image lmx-freemhd --check-pull --pull-timeout-seconds 20 --output ./artifacts/freemhd_container_pull.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --extra-case-root /tmp/startingfiles_ha20/StartingFiles/Shercliff --output ./artifacts/freemhd_setup_recovered.json
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_case.py --case-dir ./external/FreeMHD/OpenFOAM-v2206/tutorials/electromagnetics/mhdFoam/hartmann --output ./artifacts/freemhd_case_hartmann.json
docker build -t lmx-freemhd ./docker
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd --case-dir /absolute/path/to/freemhd_case --output ./artifacts/freemhd/run.json
```

This path generates a local Docker build context with OpenFOAM v2206, FreeMHD, and a case runner script. The environment probe records whether the local OpenFOAM tree is sourceable, whether `wmkdepend` exists for local `wmake` builds, whether the Docker daemon is reachable, and classifies a few common `wmake` failure modes such as missing `wmkdepend`, the macOS libc++ header conflict, and the current post-patch include regression. The Darwin patch helper applies a reproducible local workaround to the vendored `external/FreeMHD` checkout by demoting the two problematic `lnInclude` directories from `-I` to `-idirafter` in the Darwin `wmake` rules. The container probe records whether the requested runtime image exists locally, whether the Dockerfile base image tag exists on Docker Hub, and, when `--check-pull` is enabled, whether a timed `docker pull` of that base image stalls locally. The setup inspection command reports whether runnable FreeMHD cases exist locally and recommends the smallest smoke target when they do not. The case inspection command records the actual case structure that a parity run would consume. The partial `StartingFiles.zip` recovery flow can now materialize Shercliff `Ha0` and `Ha20` paper cases into normal OpenFOAM directories. The execution helper mounts the case directory, runs either serial or decomposed cases, and stores a machine-readable run report or a structured `docker-image-unavailable` or daemon-unavailable status.
The setup inspector now accepts `--extra-case-root`, which lets it report recovered `/tmp` materializations alongside the checked-in `external/FreeMHD` tree. The execution helper now fails fast with `docker-image-unavailable` when the requested image tag is missing locally, instead of hanging in `docker run`.

Current blocker: the Docker daemon is reachable on this machine, and the recovered `Ha20` case is now discoverable, but the OpenFOAM container path is still not proven end to end because `lmx-freemhd` has not yet been built successfully in this session. The current container preflight report now shows a valid base image tag, `microfluidica/openfoam:2206`, but also shows that a timed `docker pull` of that image stalls locally, so the remaining Docker-side blocker is pull/build execution rather than tag discovery. On the non-Docker path, the Darwin patch helper removes the libc++ header collision and advances the local `wmake` failure to a new `fvMesh.H` include regression, which is now also machine-readable.

## Benchmark

```bash
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

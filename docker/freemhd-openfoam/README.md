# FreeMHD / OpenFOAM Docker

This image builds the FreeMHD `epotMultiRegionInterFoam` solver on top of
OpenFOAM `v2206` and provides a small runner for mounted Shercliff/Hunt-style
cases.

## Build

```bash
docker build \
  -f docker/freemhd-openfoam/Dockerfile \
  -t lmx-freemhd-openfoam:2206 \
  .
```

## Run a mounted case

Run from the repository root, or replace the paths with your own case/output
directories:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e FREEMHD_END_TIME=1e-5 \
  -e FREEMHD_WRITE_INTERVAL=1e-5 \
  -e FREEMHD_MAX_DELTA_T=5e-6 \
  -v /absolute/path/to/case:/workspace/case:ro \
  -v /absolute/path/to/output:/workspace/output \
  lmx-freemhd-openfoam:2206 \
  run-freemhd-case /workspace/case /workspace/output 2
```

The runner:
- updates `decomposeParDict` to the requested MPI rank count
- rebuilds a structured mesh if `system/blockMeshDict` is present and no active
  mesh is found
- runs `topoSet`, `splitMeshRegions`, `changeDictionary`, and `setExprFields`
  when region meshes need to be regenerated
- launches `epotMultiRegionInterFoam` with `mpirun`
- reconstructs the latest time
- writes VTK output under `output/VTK`

## Notes

- `--user "$(id -u):$(id -g)"` is required because FreeMHD uses OpenFOAM
  `dynamicCode` on some boundary conditions, and OpenFOAM blocks that path for
  root users.
- On Apple Silicon, this image runs through Docker's `linux/amd64` emulation,
  since the `v2206` image used here is x86-64.

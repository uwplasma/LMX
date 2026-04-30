# Q2DmhdFoam External Validation Container

This container builds Q2DmhdFoam against foam-extend 4.1 and runs the
`Q2DfullyDeveloped` tutorial as a reproducible external-code gate for LMX.
It exports ParaView-ready VTK files, a line-profile CSV, and a JSON summary.

```bash
docker build --platform linux/amd64 -t lmx-q2dmhdfoam:fe41 docker/q2dmhdfoam
mkdir -p artifacts/external/q2dmhdfoam_reference
docker run --rm --platform linux/amd64 \
  -e RANKS=2 \
  -v "$PWD/artifacts/external/q2dmhdfoam_reference:/output" \
  lmx-q2dmhdfoam:fe41
python examples/q2dmhdfoam_docker_reference_validation.py
```

The runner can also execute other checked-out Q2DmhdFoam cases for manual
validation work. For legacy cases that use `theta` instead of `T`, the script
copies and patches the field dictionary before running:

```bash
mkdir -p artifacts/external/q2dmhdfoam_lid_driven
docker run --rm --platform linux/amd64 \
  -e CASE_RELATIVE_PATH=run/lidDriven \
  -e RANKS=2 \
  -e FORCE_END_TIME=1 \
  -e END_TIME=100 \
  -e WRITE_INTERVAL=50 \
  -e EXTRACT_PROFILE=0 \
  -v "$PWD/artifacts/external/q2dmhdfoam_lid_driven:/output" \
  lmx-q2dmhdfoam:fe41
```

Generic cases export VTK and logs but do not claim LMX parity by themselves.
The strict Q2D turbulence lane still requires a matched LMX case and a filled
`q2d_turbulence_reference_observables.csv`.

Outputs:

- `artifacts/external/q2dmhdfoam_reference/VTK/`: VTK fields for ParaView.
- `artifacts/external/q2dmhdfoam_reference/profile.csv`: extracted
  cross-channel streamwise profile.
- `artifacts/external/q2dmhdfoam_reference/summary.json`: runtime and
  physics observables including Hartmann number, flow-rate error, and status.
- `docs/_static/generated/q2dmhdfoam_docker_reference.png`: LMX documentation
  panel generated from the Docker rerun.

The Dockerfile pins the upstream Q2DmhdFoam commit and applies only the
foam-extend 4.1 API compatibility patch needed by the local container build.
Routine LMX tests do not require Docker or Q2DmhdFoam; this path is an external
validation artifact generator.

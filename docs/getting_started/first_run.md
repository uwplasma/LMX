# Run a first duct case

Create one physical case, solve it, and inspect the explicit termination state:

```python
import lmx

case = lmx.make_hartmann_case(ha=20.0, ny=48, nz=48)
result = lmx.solve_steady(case)

assert result.converged, result.status
print(f"steps={result.steps} residual={result.state.residual:.3e}")
```

`CaseSpec` groups geometry, regions, magnetic field, boundary conditions,
drive, solver controls, and output policy. All viscosities in a case are
kinematic viscosities in m²/s; the LMX helpers convert and form the
dimensionless groups.

The command line uses the same schema:

```console
lmx examples/hartmann_case.toml --plots
```

LMX writes compact NPZ and JSON by default. VTK and plots are controlled by the
case output settings. A steady command exits nonzero if its convergence gates
are not met.

For an editable complete workflow, run:

```console
python examples/hartmann_example.py
```

The script compares the numerical centerline with the analytical Hartmann
profile and writes the errors beside the solved fields.

# Fully developed duct flows

Fully developed cases solve the axial velocity $u(y,z)$ and electric potential
$\phi(y,z)$ on a structured cross-section. Start with the named builders:

```python
import lmx

hartmann = lmx.make_hartmann_case(ha=20, ny=64, nz=64)
shercliff = lmx.make_shercliff_case(ha=20, ny=64, nz=64)
hunt = lmx.make_hunt_case(ha=20, ny=64, nz=64, wall_cells=4)

for case in (hartmann, shercliff, hunt):
    result = lmx.solve(case)
    print(case.name, result.converged, result.residual)
```

Hartmann applies the field normal to insulating Hartmann walls. Shercliff
orients it so side layers control the profile. Hunt resolves conducting side
walls and insulating Hartmann walls. Use `dataclasses.replace` to change a
visible part of a frozen case:

```python
from dataclasses import replace

case = replace(
    hartmann,
    forcing=2.0,
    time_stepper=replace(hartmann.time_stepper, steady_tolerance=1e-9),
)
```

After solving, check more than the update norm:

```python
from lmx.validation import hartmann_validation, validation_summary

comparison = hartmann_validation(result, ha=20)
metrics = validation_summary(result, case.name, ha=20)
print(comparison.l2_error, metrics["charge_balance_relative"])
```

`lmx.solvers.fully_developed_power_balance(case, result)` reports applied, viscous,
Lorentz, and residual power using the same discrete operators as the solve.
Increase wall and fluid resolution together for high Hartmann number cases;
the mesh-quality helpers report cells across Hartmann and side layers.

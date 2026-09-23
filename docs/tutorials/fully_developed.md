# Fully developed duct flows

Fully developed cases solve the axial velocity $u(y,z)$ and electric potential
$\phi(y,z)$ on a structured cross-section. Start with the named builders:

```python
import lmhdx

hartmann = lmhdx.make_hartmann_case(ha=20, ny=32, nz=32)
shercliff = lmhdx.make_shercliff_case(ha=20, ny=32, nz=32)
hunt = lmhdx.make_hunt_case(ha=20, ny=32, nz=32, wall_cells=4)

for case in (hartmann, shercliff, hunt):
    result = lmhdx.solve(case)
    print(case.name, result.converged, result.residual)
```

Hartmann applies the field normal to insulating Hartmann walls. Shercliff
orients it so side layers control the profile. Hunt resolves conducting Hartmann
walls and insulating side walls.

The discrete fully developed problem is linear, so a steady solve is one
affine fixed-point GMRES solve, not a march in pseudo-time. Each iteration
applies one potential solve and one momentum solve. `result.residual` is the
relative fixed-point residual `||G(u) - u|| / ||G(0)||`. `result.converged` is
true only when that residual meets `steady_tolerance` and the final potential
and momentum solves meet their gates. `result.steps` counts GMRES iterations.
Set the solver mode to `"transient"` for a time history.

Use `dataclasses.replace` to change a visible part of a frozen case:

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
from lmhdx.validation import hartmann_validation, validation_summary

comparison = hartmann_validation(result, ha=20)
metrics = validation_summary(result, case.name, ha=20)
print(comparison.l2_error, metrics["charge_balance_relative"])
```

`lmhdx.solvers.fully_developed_power_balance(case, result)` reports applied, viscous,
Lorentz, and residual power using the same discrete operators as the solve.
Increase wall and fluid resolution together for high Hartmann number cases;
the mesh-quality helpers report cells across Hartmann and side layers.

## Prescribe throughput and measure pumping work

Run `python examples/hunt_example.py` for a conducting-wall duct with a
requested flow rate. Edit `TARGET_FLOW_RATE` and `DUCT_LENGTH` near the top;
set the target to `None` to prescribe `FORCING` instead. The example solves at
unit drive to measure the flow per unit drive `G`, eliminates the required
drive analytically, and solves again from rest at that drive. Both solves must
report `converged`, and the second must reproduce `G*drive` to a `1e-8`
relative flow check. This checks the linear drive-to-flow relation, not an
independent physical reference. Its JSON summary records flow, drive,
`d(drive)/dQ = 1/G`, and hydraulic power `drive*L*Q`.
This is fully developed segment work, excluding entry/exit, manifolds and
thermal effects. It is not a complete blanket pumping budget.

## Fit a measured velocity profile

Use this bounded inverse problem to infer a positive pressure-gradient drive
and magnetic-field multiplier. Replace the synthetic `target` with axial
velocity samples on the same cell-centered mesh; keep SI units and geometry
consistent. Solid cells have zero weight. The normalized, area-weighted loss
uses the production field solve and its implicit SOLVAX derivatives, not an
iteration-history tape. SciPy controls the two design variables on the host;
the objective and gradient are compiled together. The bounded optimizer is
[SciPy L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

```python
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import lmhdx
from lmhdx.design import fluid_cell_areas

jax.config.update("jax_enable_x64", True)
case = lmhdx.make_shercliff_case(ha=5, ny=12, nz=12)
areas = fluid_cell_areas(case)

def velocity(parameters):
    return lmhdx.solve_fully_developed_fields(
        case, forcing=parameters[0], magnetic_field_scale=parameters[1]
    )[0]

truth = jnp.array([1.4, 1.25])
target = velocity(truth)
normalization = jnp.sum(areas * target**2)
if not float(normalization) > 0:
    raise ValueError("A nonzero fluid velocity target is needed to identify the field.")

def loss(parameters):
    return jnp.sum(areas * (velocity(parameters) - target)**2) / normalization

value_and_gradient = jax.jit(jax.value_and_grad(loss))

def evaluate(parameters):
    value, gradient = value_and_gradient(jnp.asarray(parameters))
    return float(value), np.asarray(gradient, dtype=float)

fit = minimize(evaluate, [1.0, 1.0], jac=True, method="L-BFGS-B",
               bounds=[(0.1, 3.0), (0.5, 2.0)],
               options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 60})
if not fit.success:
    raise RuntimeError(f"Profile fit did not converge: {fit.message}")
print("drive, field multiplier:", fit.x)
print("relative squared profile error:", fit.fun)
```

This synthetic example checks parameter recovery, not experimental validity.
The sign of the field is not identifiable from velocity alone: magnetic drag
depends on its squared magnitude, hence the positive field bound. Weak fields,
only a bulk-flow observation, or uncertain viscosity/geometry can also make
drive and field poorly distinguishable. Keep the full profile, inspect the
residual and parameter sensitivity, and report an irreducible residual for
targets outside this two-parameter model; optimizer success is not a fit-quality
certificate. Validate inferred parameters with the reporting solver and a
held-out finer mesh before using them in a design study.

For a prescribed flow rate rather than a profile, avoid optimizing drive:
`lmhdx.design.linear_flow_response(case).drive_for(target_flow_rate)` eliminates
it exactly using the linear response. Neither fit is a thermal blanket design;
wall/geometry optimization and heat-transfer validation have separate gates.

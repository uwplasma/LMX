# Differentiate a Hartmann solve

LMX exposes a focused differentiable Hartmann problem. The function returns JAX
arrays and composes the same discrete coefficient and SOLVAX iteration path as
the physical model.

```python
import jax
from lmx.autodiff import (
    build_hartmann_autodiff_problem,
    hartmann_mean_velocity,
)

problem = build_hartmann_autodiff_problem(ny=24, nz=24)


def response(ha):
    return hartmann_mean_velocity(problem, forcing=1.0, hartmann_number=ha)


value, derivative = jax.value_and_grad(response)(20.0)
print(value, derivative)
```

Mesh dimensions and fixed iteration counts are static compilation parameters.
Inputs such as forcing and Hartmann number remain differentiable arrays. The
accepted gradients are checked against finite differences and with JVP/VJP
identities.

Run `python examples/autodiff_design_demo.py` for a sensitivity scan and a
small inverse problem that recovers the forcing from a target profile.

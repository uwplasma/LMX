# Automatic differentiation and inverse design

LMX uses JAX to differentiate selected liquid-metal flow observables through
case construction and linear solves. A successful `jax.grad` call is only the
first gate: gradients must also be accurate, converged, and affordable.

![Checked sensitivities and a bounded inverse-design trace](_static/readme-autodiff.webp)

## Run the bounded example

```bash
python examples/autodiff_design_demo.py
```

Edit the mesh, scan, and design inputs at the top of the file. The example
constructs its public problem and objectives explicitly, evaluates
Hartmann-number sensitivities with `jax.grad`, and performs bounded inverse
design. Independent finite-difference checks remain in the test suite.

## Solver backends

The default `auto` path requires a compatible SOLVAX release in the supported
dependency range; `cg` is only a compatibility name for the same SOLVAX PCG
path. CI tests dependency-range endpoints instead of committing a lockfile.

SOLVAX integration is accepted only when the following agree:

- primal solution and residual;
- implicit gradient and centered finite difference;
- transpose solve and independently formed reference;
- CPU and GPU results within documented precision tolerances;
- end-to-end objective derivatives, not only a toy matrix solve.

## Differentiability contract

For an objective `F(q(alpha), alpha)` constrained by `R(q, alpha) = 0`, the
implicit derivative uses

```text
(dR/dq)^T lambda = (dF/dq)^T
dF/dalpha = partial(F)/partial(alpha) - lambda^T partial(R)/partial(alpha)
```

The primal and transpose tolerances must be tight enough that algebraic error is
small compared with the derivative acceptance tolerance. Report both residuals.

## What is supported

- analytical Hartmann sensitivities;
- selected fully developed duct objectives;
- bounded extruded field/pressure response objectives;
- finite-difference and transpose verification helpers;
- simple gradient-based design traces.

Research-stage extruded gradients should not be interpreted as validated design
predictions until the underlying primal benchmark passes. Differentiability
does not repair model-form or discretization error.

## Practical guidance

- Enable 64-bit JAX for quantitative validation.
- Scale design variables and objectives to avoid ill-conditioned derivatives.
- Keep topology and array shapes fixed inside compiled objectives.
- Use implicit differentiation for converged linear solves rather than storing
  every Krylov iteration.
- Recheck gradients after changing mesh, tolerance, backend, or precision.
- Record compilation separately from warm optimization iterations.

Public helpers are exposed through `lmx.autodiff`; implementation details remain
private so the user import path can stay stable while kernels are simplified.

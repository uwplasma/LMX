# Automatic differentiation and inverse design

LMX uses JAX to differentiate selected liquid-metal flow observables through
case construction and linear solves. A successful `jax.grad` call is only the
first gate: gradients must also be accurate, converged, and affordable.

## Run the bounded example

```bash
python examples/autodiff_design_demo.py --help
```

The example evaluates a pressure/flow objective, compares the automatic
derivative with a centered finite difference, and performs a short bounded
design update. Outputs go under `artifacts/`.

## Solver backends

The default `auto` path uses a compatible SOLVAX `0.8.2+` release below `1.0` when
available. The retained native CG path is available for independent comparison.
The lockfile records the exact CI environment; package metadata deliberately
does not pin one SOLVAX patch release.

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

# LMX

LMX solves inductionless liquid-metal MHD with JAX. It provides
analytical-reference fully developed flows, three-dimensional extruded duct
and pipe models for spatially varying magnetic fields, and periodic Q2D flow
with Hartmann-layer damping. LMX builds the physics;
[SOLVAX](https://github.com/uwplasma/SOLVAX) supplies reusable numerical solvers.

```{image} _static/fringing_solver_family.webp
:alt: Rectangular-duct fringing-field solution
:align: center
```

## Start here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Install and run
:link: getting_started/install
:link-type: doc
Install LMX, select a JAX backend, and check the command line.
:::

:::{grid-item-card} First duct solve
:link: getting_started/first_run
:link-type: doc
Solve and validate a Hartmann duct from Python or TOML.
:::

:::{grid-item-card} Three-dimensional fringing
:link: tutorials/fringing
:link-type: doc
Build a spatially varying field and inspect charge and flow diagnostics.
:::

:::{grid-item-card} Q2D vortex dynamics
:link: tutorials/q2d
:link-type: doc
Evolve a depth-averaged strong-field model and reproduce its poster and movie.
:::

:::{grid-item-card} Validation
:link: validation/index
:link-type: doc
See the analytical, numerical, and FreeMHD evidence for every claim.
:::

::::

```{toctree}
:hidden:
:caption: Get started

getting_started/install
getting_started/first_run
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/fully_developed
tutorials/fringing
tutorials/walls_and_fields
tutorials/differentiation
tutorials/q2d
```

```{toctree}
:hidden:
:caption: How-to guides

how_to/restart_and_output
```

```{toctree}
:hidden:
:caption: Physics and numerics

physics/equations
physics/numerics
```

```{toctree}
:hidden:
:caption: Validation

validation/index
validation/freemhd
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/cli
reference/bibliography
```

```{toctree}
:hidden:
:caption: Development

develop/architecture
develop/contributing
```

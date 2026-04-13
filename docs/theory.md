# Theory

LMX solves low-magnetic-Reynolds-number liquid-metal magnetohydrodynamics in the
inductionless limit on structured meshes. This page documents the governing
equations, the reductions used by the currently shipped solver families, and
where those equations are assembled in the source tree.

## Scope

The current public solver scope is:

- incompressible single-phase liquid-metal flow
- prescribed magnetic fields
- electrically conducting multi-region cross-sections
- laminar duct benchmarks
- differentiable benchmark/inverse-design workflows on the JAX core

The current public scope does **not** yet include:

- turbulence
- thermal coupling
- free surfaces
- full magnetic induction
- the final 3D `extruded_inductionless` pressure-velocity-potential solver

## Governing equations

In the inductionless limit, the current solver family uses:

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\rho\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right)
= -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{J}\times\mathbf{B}
$$

$$
\mathbf{J} = \sigma\left(-\nabla \phi + \mathbf{u}\times\mathbf{B}\right)
$$

$$
\nabla \cdot \mathbf{J} = 0
$$

where:

- $\mathbf{u}$ is velocity
- $p$ is pressure
- $\phi$ is electric potential
- $\mathbf{J}$ is current density
- $\mathbf{B}$ is the prescribed magnetic field
- $\rho$, $\mu$, and $\sigma$ are density, dynamic viscosity, and electrical conductivity

These equations are represented in the current source tree by:

- `lmx/physics.py`
  - material fields and prescribed magnetic-field components
- `lmx/operators.py`
  - structured differential operators
- `lmx/solvers.py`
  - assembled momentum and potential solves

## Fully developed duct reduction

The default shipped solver family is
`solver.kind = "fully_developed_inductionless"`.

For this family, LMX assumes a streamwise velocity field
$\mathbf{u} = (u(y,z,t), 0, 0)$ and a cross-sectional electric potential
$\phi(y,z,t)$. The imposed magnetic field may contain transverse components
$B_y(y,z,t)$ and $B_z(y,z,t)$.

Under this reduction:

$$
\mathbf{u}\times\mathbf{B} = \left(0,\,-uB_z,\,uB_y\right)
$$

and the cross-sectional current becomes

$$
J_y = \sigma\left(-\frac{\partial \phi}{\partial y} - u B_z\right), \qquad
J_z = \sigma\left(-\frac{\partial \phi}{\partial z} + u B_y\right)
$$

Charge conservation gives the potential equation

$$
\frac{\partial J_y}{\partial y} + \frac{\partial J_z}{\partial z} = 0
$$

or, equivalently,

$$
\nabla_{\perp}\cdot\left(\sigma \nabla_{\perp}\phi\right)
= \nabla_{\perp}\cdot\left(\sigma(\mathbf{u}\times\mathbf{B})_{\perp}\right)
$$

The streamwise momentum equation becomes

$$
\rho\frac{\partial u}{\partial t}
= -\frac{\partial p}{\partial x}
  + \mu \nabla_{\perp}^2 u
  + (\mathbf{J}\times\mathbf{B})_x
$$

with

$$
(\mathbf{J}\times\mathbf{B})_x = J_y B_z - J_z B_y
$$

These are the two fields actually advanced by the current default solver.

## Boundary and interface conditions

The currently shipped boundary-condition vocabulary is declared in
`lmx/specs.py` and interpreted by `lmx/physics.py` and `lmx/solvers.py`.

The main physical conditions are:

- `no_slip`
  - streamwise velocity vanishes at the wall
- `insulating`
  - no normal current leaves the domain
- `conducting_wall`
  - the wall region carries its own conductivity and is solved as part of the
    multi-region potential problem
- `imposed_current_density`
  - reserved for current-driven studies
- `inlet_velocity`, `inlet_flow_rate`, `outlet_pressure`
  - currently interpreted in the reduced/fully-developed sense used by the
    present solver family

Across fluid-solid interfaces, the multi-region formulation enforces
conductivity-weighted continuity through conservative face conductances. That
is why LMX tracks:

- `div_current_max_history`
- `charge_balance_residual_history`
- `interface_current_residual_history`

as first-class diagnostics.

## Charge conservation and compatibility projection

The discrete potential equation must satisfy a zero-net-source compatibility
condition. In practice, this means the right-hand side of the Poisson-like
potential solve cannot contain a net source over the connected domain.

LMX enforces this through a compatibility projection before the solve and then
measures the remaining charge/current mismatch through diagnostics.

This treatment lives in `lmx/solvers.py` and is exposed in the runtime outputs
by:

- `lmx/runtime_logging.py`
- `lmx/io.py`
- `lmx/validation.py`

## Magnetic-field models

LMX currently supports:

- constant magnetic fields
- analytic magnetic fields specified as a Python callback
- affine startup ramps applied to either of the above

The magnetic-field specification is declared in `lmx/specs.py` and evaluated in
`lmx/physics.py`.

The ramp law is:

$$
\alpha_B(t) = \mathrm{clip}\left(
\frac{t - t_{\mathrm{start}}}{t_{\mathrm{duration}} + 10^{-6}},
0, 1\right)
$$

so the actual field used by the solver is $\alpha_B(t)\mathbf{B}(y,z)$.

## Geometry models

The current geometry dataclasses are declared in `lmx/specs.py` and meshed by
`lmx/mesh.py`.

Implemented today:

- `rect_duct`
- `layered_duct`
- `pipe_ogrid` mesh generation and preview utilities

The current production solver families operate on rectangular and layered
cross-sections. The mapped pipe mesh is presently available for geometry and
workflow staging while the full 3D solver family is being developed.

## Solver-family interpretation

### `fully_developed_inductionless`

This is the main research solver. It solves the coupled cross-sectional
streamwise velocity/potential problem directly and exposes steady and transient
modes.

### `reduced_inductionless`

This is the reduced-model alternative retained for reduced-model studies and
comparative testing. It is still useful, but it is not the primary path for
new benchmark-quality results.

### `extruded_inductionless`

This name is already part of the public API, but the final 3D pressure,
velocity, and potential solver is not yet implemented. The current fringing
workflow is an explicit vertical slice toward that family, not the final
algorithm.

## JAX and differentiability

LMX uses JAX in three layers:

1. vectorized PDE operators and array programs
2. the differentiable Hartmann lane in `lmx/autodiff.py`
3. strong-scaling experiments on the dominant stencil kernels in `lmx/scaling.py`

The current differentiable lane uses fixed-iteration solves so the reverse-mode
path remains explicit and stable for benchmark-sized inverse problems.

## Literature anchors

LMX’s documentation and validation ladder follow the standard low-magnetic-
Reynolds-number duct benchmark literature and the benchmark hierarchy used in
fusion liquid-metal verification and validation studies.

Useful references:

- [Samper et al., verification and validation benchmark ladder for MHD codes](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf)
- [arXiv:2409.08950, multi-region electrically conductive flow solver validation study](https://arxiv.org/abs/2409.08950)
- [JAX advanced autodiff](https://docs.jax.dev/en/latest/advanced-autodiff.html)
- [Lineax solvers](https://docs.kidger.site/lineax/api/solvers/)
- [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/)

## Source map

- `lmx/specs.py`
  - public dataclasses and enumerations
- `lmx/mesh.py`
  - geometry discretization
- `lmx/physics.py`
  - magnetic fields and material properties
- `lmx/operators.py`
  - discrete operators
- `lmx/linear.py`
  - iterative solves and residual norms
- `lmx/solvers.py`
  - solver-family implementations
- `lmx/autodiff.py`
  - differentiable lane
- `lmx/fringing.py`
  - fringing-field research slice

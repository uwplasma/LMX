# Theory

LMX solves low-magnetic-Reynolds-number liquid-metal magnetohydrodynamics in the
inductionless limit on structured meshes. This page documents the governing
equations, the reductions used by the current solver families, and
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
- the final production-hardening pass for the 3D `extruded_inductionless`
  pressure-velocity-potential solver family

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

The default solver family is
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

The current boundary-condition vocabulary is declared in
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

For the 3D fringing workflows, the same conservation view is extended to
domain boundaries and axial control volumes. LMX also tracks:

- stationwise integrated axial current
- stationwise wall-current leakage at the external `y` and `z` boundaries
- a net boundary-current residual over the full extruded control volume

Those are the concrete diagnostics needed to harden inlet/outlet handling and
external-wall current closure, which are the places where inductionless MHD
solvers most often lose charge conservation in practice.

## Charge conservation and compatibility projection

The discrete potential equation must satisfy a zero-net-source compatibility
condition. In practice, this means the right-hand side of the Poisson-like
potential solve cannot contain a net source over the connected domain:

$$
\int_{\Omega} \nabla\cdot\mathbf{J}\, dV
= \int_{\partial\Omega}\mathbf{J}\cdot\mathbf{n}\, dS
= 0.
$$

For the fully developed cross-sectional solve, this reduces to a zero-net
source condition on the 2D electric problem. For the extruded 3D slice, the
same requirement becomes a full boundary-flux balance across inlet, outlet,
and wall surfaces.

LMX enforces this through a compatibility projection before the solve and then
measures the remaining charge/current mismatch through diagnostics. The
implementation sequence is:

1. assemble `\nabla\cdot(\sigma \nabla \phi)` with conductivity-weighted face
   coefficients
2. project the right-hand side onto the compatible zero-net-source subspace
3. solve the Poisson-like electric problem
4. reconstruct face currents conservatively
5. audit the resulting field through local and integral conservation metrics

This treatment lives in `lmx/solvers.py` and is exposed in the runtime outputs
by:

- `lmx/runtime_logging.py`
- `lmx/io.py`
- `lmx/validation.py`

The 3D fringing slice in `lmx/fringing.py` uses the same principle for its
variable-coefficient electric solve: the right-hand side is projected onto the
conductivity-weighted compatibility space before the electric solve, and the
result is then audited through both `div J` and boundary-current integrals.
For layered ducts, the multi-region potential equation is solved with a sparse
direct solve of the conservative variable-coefficient operator rather than the
bounded Jacobi path used in the cheaper rectangular slice.
For rectangular and layered extruded cases, the face-flux audit also
enforces a closed-current condition on the axial inlet/outlet faces in the
conservative current reconstruction used for validation. This is the discrete
statement of

$$
\mathbf{J}\cdot\mathbf{n} = 0 \quad \text{on the electrically closed external boundary,}
$$

which is the boundary treatment needed to prevent spurious charge leakage
through the fringing section ends.

The conservation outputs are:

- `div_current_max_history`
- `charge_balance_residual_history`
- `interface_current_residual_history`
- extruded `axial_current(x)`
- extruded `wall_current_leakage(x)`
- extruded `net_boundary_current_residual`

Those are the metrics that matter most for inlet/outlet and wall treatment in
low-Rm multi-region MHD, where local residuals can look acceptable while the
integrated boundary flux still drifts.

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
- `pipe_ogrid` mesh generation, preview utilities, and the first
  `extruded_inductionless` fringing slice

The current production fully developed solver families operate on rectangular
and layered cross-sections. The mapped pipe mesh is now also used by the first
3D fringing slice in `lmx/fringing.py`, where the solver is expressed in the
local pipe frame but still follows the same inductionless current-closure
principles as the duct cases.

## Solver-family interpretation

### `fully_developed_inductionless`

This is the main research solver. It solves the coupled cross-sectional
streamwise velocity/potential problem directly and exposes steady and transient
modes.

### `extruded_inductionless`

This solver family now provides low-Re 3D
pressure-velocity-potential slices in `lmx/fringing.py`. It advances `u`, `v`,
`w`, and `p` with a projection loop, solves a 3D electric potential problem
for `phi`, and reports current, Lorentz, and charge-balance fields. The
currently exposed geometries are:

- rectangular ducts
- layered ducts
- mapped `pipe_ogrid` pipe slices

This is a real 3D pressure-velocity-potential workflow. Broader validation
sets, stronger mesh studies, and more geometry/material coverage remain future
work.

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
- [Hunt, *Magnetohydrodynamic flow in rectangular ducts*](https://doi.org/10.1017/S0022112065000344)
- [Shercliff, *The Theory of Electromagnetic Flow-Measurement*](https://assets.cambridge.org/97805213/35546/excerpt/9780521335546_excerpt.pdf)
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

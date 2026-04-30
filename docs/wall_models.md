# Wall Models And Viscosity Units

This page documents the wall-electrical reductions and unit conventions used
by LMX case setup and validation scripts.

## Viscosity Convention

LMX stores `RegionSpec.viscosity` as kinematic viscosity:

$$
\nu = \frac{\mu}{\rho}.
$$

The solver uses the density-divided momentum equation, so this value is the
diffusion coefficient multiplying the velocity Laplacian. Dynamic viscosity
`mu` in `Pa s` should be converted at the input boundary:

```python
from lmx import dynamic_to_kinematic_viscosity

nu = dynamic_to_kinematic_viscosity(dynamic_viscosity=2.0e-3, density=500.0)
```

The nondimensional helpers use the same convention:

$$
Ha = B_0 a \sqrt{\frac{\sigma}{\rho\nu}}, \qquad
Re = \frac{Ua}{\nu}, \qquad
N = \frac{\sigma B_0^2 a}{\rho U}, \qquad
Rm = \mu_0\sigma Ua.
$$

## Thin-Wall Conductance

For Hunt/Shercliff-style thin-wall studies, tangential wall-current closure is
usually represented by

$$
c = \frac{\sigma_w t_w}{\sigma_f a}.
$$

Use `lmx.wall_conductance_ratio(...)` or
`lmx.wall_layer_from_conductance_ratio(...)` to make that assumption explicit.
This is not the same quantity as normal leakage through a coating.

## Normal Leakage

For an insulating coating over metal, the normal shunt path is better described
by

$$
g_\perp
= \frac{\sigma_\mathrm{coat}/t_\mathrm{coat}}{\sigma_f/a}
= \frac{\sigma_\mathrm{coat}a}{\sigma_f t_\mathrm{coat}}.
$$

LMX exposes `normal_leakage_ratio(...)` for single coatings and
`normal_stack_leakage_ratio(...)` for nested layers in series. These helpers are
used to keep AlN-like wall-stack studies from silently mixing tangential and
normal conductance definitions.

## Nested Wall Layers

The first supported nested-wall API is a reduced electrical model:

```python
from lmx import WallLayer, tangential_stack_conductance_ratio

layers = (
    WallLayer("aln", conductivity=1.0e-8, thickness=2.0e-4, cells=4),
    WallLayer("metal", conductivity=1.0e6, thickness=1.0e-3, cells=8),
)
c_parallel = tangential_stack_conductance_ratio(
    layers,
    fluid_conductivity=1.0e6,
    length_scale=0.01,
)
```

For tangential thin-wall closure, layers conduct in parallel and their surface
conductances add. For normal leakage, layers conduct in series and the most
insulating layer dominates. `nested_wall_layer_resolution_summary(...)` reports
the cell counts and thicknesses that a true geometry implementation must honor.

True fluid | AlN | metal multilayer geometry remains a planned solver extension:
the reduced model gives reproducible conductance and sensitivity studies now,
while the full geometry lane will add arbitrary layers per side, interface
current-continuity diagnostics, layer-focused meshes, and FreeMHD/code-to-code
comparison cases.

## Literature Anchors

- Shercliff and Hunt rectangular-duct solutions define the insulating and
  conducting-wall fully developed benchmark families used by LMX.
- Ni/Tao-style analytical extensions and the FreeMHD validation cases provide
  wall-conductance and high-Hartmann-number comparison data.
- Fusion blanket studies commonly use the thin-wall conductance ratio `c` for
  tangential wall-current closure; coated-wall studies must additionally track
  normal leakage `g_perp` when an insulating layer can shunt into a metal
  substrate.
- The AlN/lithium wall-stack campaign must report MHD performance only and
  leave compatibility, corrosion, adhesion, wetting, irradiation, and
  manufacturability to separate materials tests.

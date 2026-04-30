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

## Li/AlN Phase 0-2 Reduced Study

The first Li/AlN wall-stack lane is executable through:

```bash
python examples/li_aln_wall_stack_phase0_2.py
```

The example writes a unit audit, nested-wall mesh QA, conductance/pinhole CSV
tables, and a report-ready figure under
`studies/li_aln_wall_mhd/results/processed/phase0_2`. It also copies the public
figure and summary into `docs/_static/generated`.

![Li/AlN wall-stack Phase 0-2 reduced study](_static/generated/li_aln_wall_stack_phase0_2.png)

The default case records liquid-lithium `mu`, `rho`, computed `nu`, `Ha`, `Re`,
`N`, and `Rm`, and it uses an inductionless velocity scale with
`Rm < 1e-2`. The response scalar is a reduced current-closure/Lorentz-drag
proxy that increases with effective wall conductance. It is a design-screening
quantity for electrical wall performance, not a substitute for a solved
multilayer velocity-current field.

The Phase 0-2 acceptance gates are:

- the viscosity convention is explicit and converted to kinematic viscosity at
  the input boundary;
- `Rm` is reported and the inductionless assumption is checked;
- each nested wall layer carries thickness, conductivity, and cell count;
- the reduced model separates tangential conductance `c_parallel` from normal
  leakage `g_perp`;
- `f_p = 0` returns the intact-coating limit and increasing `f_p` moves toward
  the metal-shunt limit;
- the summary explicitly states that it makes no material-compatibility claim.

## Li/AlN Phase 3-6 Reduced Parametric Study

The bounded Phase 3-6 lane is executable through:

```bash
python examples/li_aln_wall_stack_phase3_6.py
```

It writes an operating matrix over magnetic field and velocity, substrate
comparisons for `316L`, `IN625`, and molybdenum-like conductivities, AlN
degradation sweeps, and pinhole thresholds for a prescribed reduced
current-closure deviation. The artifact is deliberately conservative: it keeps
tangential wall conductance and normal through-layer leakage as separate
electrical paths, because increasing AlN thickness raises the former but lowers
the latter.

![Li/AlN wall-stack Phase 3-6 reduced parametric assessment](_static/generated/li_aln_wall_stack_phase3_6.png)

The Phase 3-6 gates are:

- every operating point reports `Ha`, `Re`, `N`, `Rm`, and inductionless status;
- substrate ranking is based on effective conductance ratio and pinhole
  fraction, not a hidden material-performance score;
- degradation thresholds report the maximum effective conductance for a chosen
  current-closure deviation;
- normal leakage and tangential conductance bounds are both tabulated;
- the summary keeps `material_compatibility_claim = false` until separate
  corrosion, wetting, adhesion, irradiation, and manufacturing evidence exists;
- true multilayer finite-volume geometry remains the next solver-extension
  lane, with reduced results serving as limiting-case design checks.

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

# Wall models and viscosity units

Liquid-metal wall effects depend strongly on conductivity, thickness, and the
direction of current closure. LMX therefore keeps material properties
dimensional at case construction and reports the resulting nondimensional wall
ratios explicitly.

## Viscosity convention

LMX material inputs use density `rho` and kinematic viscosity `nu`. Dynamic
viscosity is

```text
mu = rho * nu
```

For characteristic length `a`, conductivity `sigma`, and magnetic field `B`,

```text
Ha = B * a * sqrt(sigma / (rho * nu))
```

Always verify whether a literature source reports `mu` or `nu`, and whether its
length scale is a half-width, hydraulic diameter, or full width.

## Thin-wall conductance

For a wall of thickness `t_w` and conductivity `sigma_w` adjacent to fluid
scale `a` and conductivity `sigma_f`, a common conductance ratio is

```text
c = sigma_w * t_w / (sigma_f * a)
```

The exact boundary condition and length convention must accompany every quoted
`c`. Thin-wall reduction is appropriate only when through-thickness variation
can be neglected.

## Layered walls

`lmx.wall_models.WallLayer` and its helper functions provide:

- tangential stack conductance;
- normal stack leakage;
- pinhole/effective conductance estimates;
- equivalent single-layer reductions;
- nested-layer resolution summaries.

Example:

```python
from lmx.wall_models import WallLayer, tangential_stack_conductance_ratio

layers = [
    WallLayer(name="coating", thickness=1e-3, conductivity=20.0),
    WallLayer(name="metal", thickness=4e-3, conductivity=1.0e6),
]
c = tangential_stack_conductance_ratio(
    layers, fluid_conductivity=3.0e6, length_scale=0.05
)
```

For explicit layers, the mesh must resolve every thickness and the interface
current must be continuous. Report cells per layer, minimum spacing, conductivity
contrast, interface-current residual, and observable change on refinement.

## Promotion rules

A reduced wall model is accepted only after comparison with an explicit-layer
solution over its intended parameter range. A multilayer result additionally
requires charge conservation, power balance, and a mesh ladder. Material sweeps
and large wall-study artifacts belong in releases; compact reusable wall inputs
belong in `examples/cases/`.

Shercliff and Hunt cases provide the stable wall-current verification surface.
More complex Li/AlN and blanket stacks remain research workflows until their
material data and external reference are frozen.

Run `python examples/li_aln_wall_stack_example.py` for an editable, top-to-bottom
Li/AlN workflow. The script constructs each layer and `CaseSpec` explicitly,
runs intact-coating and bare-metal limits, checks solver diagnostics, and writes
one compact comparison figure under `artifacts/`.

![Research-stage Li/AlN multilayer convergence](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/readme-li-aln-multilayer-convergence.webp)

At `Ha = 220`, the current Li/AlN ladder changes pressure and current by less
than 10% per mesh step. It is mesh-step evidence, not experimental validation.

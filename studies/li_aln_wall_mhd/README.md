# Li/AlN Wall MHD Study

This directory stages the parametric LMX study for AlN-like insulating wall
concepts in flowing liquid lithium. The full collaborator plan is retained in
`plan.md`; the top-level repository `plan.md` contains the active integration
steps.

Scope:

- quantify MHD performance of ideal, conducting, degraded, pinholed, effective,
  and multilayer wall models;
- report required forcing, Lorentz power, current leakage, current closure,
  pressure-gradient proxies, and autodiff sensitivities;
- keep all conclusions limited to MHD performance.

Phase 0-2 is now executable from the repository root:

```bash
python campaigns/walls/li_aln_wall_stack_phase0_2.py
```

That command writes the reduced unit audit, nested wall-layer QA,
conductance/pinhole sweep tables, and first report figure under
`results/processed/phase0_2`, copies the figure into `figures/`, and publishes
the README/docs artifacts under `docs/_static/generated`. The current artifact
is a reduced electrical-performance gate; true `fluid | AlN | metal`
multilayer geometry and layer-interface current continuity are still later
study phases.

Phase 3-6 reduced operating, threshold, substrate, and pinhole sweeps are also
executable:

```bash
python campaigns/walls/li_aln_wall_stack_phase3_6.py
```

That command writes the bounded operating matrix, substrate-conductivity
comparison, AlN degradation sweep, and current-closure threshold tables under
`results/processed/phase3_6`, copies the figure into `figures/`, and publishes
the docs artifacts under `docs/_static/generated`. The output remains a reduced
MHD electrical-performance study; it is not a resolved multilayer wall solve.

The first explicit multilayer geometry gate is:

```bash
python campaigns/walls/li_aln_multilayer_mesh_qa.py
```

It writes a `fluid | AlN | metal` rectangular mesh with faces aligned at every
material interface, plus region IDs, conductivity fields, layer tables, and
interface tables under `results/processed/multilayer_mesh`. This closes the
geometry placeholder for rectangular wall stacks.

The first solved multilayer limiting-case gate is:

```bash
python campaigns/walls/li_aln_multilayer_solve.py
```

It runs ideal-insulator, intact-AlN, degraded-AlN, and bare-metal electrical
wall models on the explicit mesh with a prescribed flow rate. The artifact
writes pressure proxy, current magnitude, dimensional charge residuals for
audit, normalized global charge balance, normalized local current-divergence,
and normalized interface-current residuals under
`results/processed/multilayer_solve`. External-code comparisons and heavier
high-Hartmann-number mesh ladders are the next lane before claiming broader
validation.

The representative solved mesh ladder is:

```bash
python campaigns/walls/li_aln_multilayer_convergence.py
```

It refines the intact-AlN and bare-metal electrical wall limits and writes the
pressure/current convergence table and figure under
`results/processed/multilayer_convergence`. This closes the bounded internal
mesh-ladder gate for the wall-stack study; a matching FreeMHD/OpenFOAM
limiting-case comparison remains separate.

Out of scope:

- lithium compatibility;
- corrosion or dissolution;
- coating adhesion;
- wetting;
- thermal-cycling or irradiation survival;
- additive-manufacturing feasibility.

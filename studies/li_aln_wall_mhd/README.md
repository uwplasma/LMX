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
python examples/li_aln_wall_stack_phase0_2.py
```

That command writes the reduced unit audit, nested wall-layer QA,
conductance/pinhole sweep tables, and first report figure under
`results/processed/phase0_2`, copies the figure into `figures/`, and publishes
the README/docs artifacts under `docs/_static/generated`. The current artifact
is a reduced electrical-performance gate; true `fluid | AlN | metal`
multilayer geometry and layer-interface current continuity are still later
study phases.

Out of scope:

- lithium compatibility;
- corrosion or dissolution;
- coating adhesion;
- wetting;
- thermal-cycling or irradiation survival;
- additive-manufacturing feasibility.

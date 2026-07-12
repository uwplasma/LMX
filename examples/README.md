# LMX examples

`examples/` is the small, supported first-run surface. It contains 11 curated
workflows; reusable TOML inputs live in `cases/`, and production-size validation,
figure, and research studies live in `campaigns/`.

Every curated workflow has a tested command, an expected artifact contract, and
an explicit stability label in [`catalog.toml`](catalog.toml). A successful run
is evidence that the workflow works—not automatically that its physics is
externally validated.

## Curated workflows

| Journey | Command | Status | What it demonstrates |
|---|---|---|---|
| Hartmann CLI | `lmx examples/hartmann_case.toml` | stable | TOML/CLI solve, compact output, diagnostics |
| Hartmann Python | `python examples/hartmann_example.py` | stable | case construction and steady solve |
| Hunt Python | `python examples/hunt_example.py` | stable | mixed insulating/conducting walls |
| Operator verification | `python examples/operator_verification_demo.py` | stable | manufactured gradients/Laplacians and observed order |
| FreeMHD validation | `python -m examples.freemhd_closed_channel_observable_parity` | external data | matched observable comparison; requires reference data |
| Fringing benchmark | `python examples/fringing_benchmark_demo.py` | research-stage | bounded extruded solver workflow; Benchmark B remains open |
| Restart | `python examples/extruded_restart_demo.py` | research-stage | restart and compact extruded output |
| Custom imposed field | `python examples/variable_field_extruded_demo.py` | research-stage | analytic/tabulated field integration |
| Differentiable design | `python examples/autodiff_design_demo.py` | research-stage | smooth sensitivity and inverse design |
| CPU/GPU benchmark | `python examples/strong_scaling_demo.py` | research-stage | cold/warm timing harness; no general scaling claim yet |
| Experimental reference | `python examples/pipe_reference_comparison_demo.py` | external data | published-reference adapter; Benchmark B remains open |

Most commands accept `--help` and an output directory. Generated results belong
under `artifacts/`, which is ignored by Git.

## Reusable case files

Fully developed ducts:

```bash
lmx examples/hartmann_case.toml
lmx cases/ducts/hartmann_restart_case.toml
lmx cases/ducts/shercliff_case.toml
lmx cases/ducts/hunt_case.toml
```

Research-stage extruded/fringing cases:

```bash
lmx cases/fringing/fringing_rect_case.toml
lmx cases/fringing/fringing_layered_case.toml
lmx cases/fringing/fringing_layered_restart_case.toml
lmx cases/fringing/fringing_tabulated_case.toml
lmx cases/fringing/fringing_pipe_case.toml
```

The configuration reference is in
[`docs/input_reference.md`](../docs/input_reference.md); the cookbook covers
wall models, custom fields, outputs, and restart in
[`docs/case_cookbook.md`](../docs/case_cookbook.md).

## Research and evidence campaigns

Campaigns are preserved and tested, but intentionally kept out of the first-run
surface:

| Directory | Purpose |
|---|---|
| `campaigns/ducts/` | Benchmark A ladders, profiles, and showcases |
| `campaigns/freemhd/` | supporting FreeMHD parity and mesh diagnostics |
| `campaigns/fringing/` | developing-flow, bent-pipe, and Dean studies |
| `campaigns/autodiff/` | specialized inverse-design and trajectory studies |
| `campaigns/fields/` | representation-specific imposed-field checks |
| `campaigns/interfaces/` | discontinuous-conductivity verification |
| `campaigns/walls/` | Li/AlN multilayer and degradation studies |
| `campaigns/q2d/` | reduced-model and Q2DmhdFoam work toward Benchmark C |
| `campaigns/magnetic_obstacle/` | localized-field work toward Benchmark D |
| `campaigns/blanket/` | explicitly research-stage WHAM blanket studies |
| `campaigns/publication/` | manuscript figure generation |
| `campaigns/status/` | validation closure and blocker reports |
| `campaigns/tutorials/` | verbose geometry, plotting, and theory walkthroughs |

Run a campaign from the repository root, for example:

```bash
python campaigns/ducts/hartmann_validation_ladder.py
python campaigns/freemhd/freemhd_observable_mesh_ladder.py
python campaigns/walls/li_aln_multilayer_convergence.py
```

The relevant documentation states required external data, runtime tier, and the
claim the campaign is allowed to support. Templates and qualitative plots are not
accepted benchmark evidence.

## Artifact and media policy

Compact JSON/CSV observables used by tests may be tracked. Generated fields,
movies, meshes, and large figures belong in versioned release assets. The current
65-file bundle is indexed by
[`provenance/release-assets.json`](../provenance/release-assets.json) and verified
by `scripts/manage_release_assets.py`.

## Adding a workflow

Prefer extending an existing curated journey. A new curated example needs:

1. one clear user question not already answered;
2. a deterministic command and bounded default runtime;
3. expected outputs and a documentation link;
4. unit, physical-verification, and public-workflow tests;
5. an explicit stable, research-stage, or external-data status in `catalog.toml`.

Production sweeps and manuscript tooling belong in `campaigns/`, not in
`examples/`.

# LMX examples

`examples/` is the supported first-run surface. Each workflow is bounded,
tested, and labels its validation status honestly. Generated files go under
`artifacts/`, which is ignored by Git.

| Journey | Command | Status |
|---|---|---|
| Hartmann CLI | `lmx examples/hartmann_case.toml` | stable |
| Hartmann Python | `python examples/hartmann_example.py` | stable |
| Hunt duct | `python examples/hunt_example.py` | stable |
| Operator verification | `python examples/operator_verification_demo.py` | stable |
| FreeMHD comparison | `python -m examples.freemhd_closed_channel_observable_parity` | external data |
| Fringing field | `python examples/fringing_benchmark_demo.py` | research-stage |
| Restart | `python examples/extruded_restart_demo.py` | research-stage |
| Custom field | `python examples/variable_field_extruded_demo.py` | research-stage |
| Differentiable design | `python examples/autodiff_design_demo.py` | research-stage |
| CPU/GPU scaling | `python examples/strong_scaling_demo.py` | research-stage |
| Pipe reference | `python examples/pipe_reference_comparison_demo.py` | external data |

Reusable TOML inputs are grouped under `examples/cases/`:

```bash
lmx examples/cases/ducts/shercliff_case.toml
lmx examples/cases/ducts/hunt_case.toml
lmx examples/cases/fringing/fringing_rect_case.toml
```

Most Python examples accept `--help` and an output directory. The
[getting-started guide](../docs/getting_started.md) explains the stable path;
the [case cookbook](../docs/case_cookbook.md) covers restarts, wall models,
custom fields, and output.

Large parameter sweeps, raw external-solver outputs, movies, and manuscript
campaigns are release artifacts, not source-tree examples. Add a new example
only when it answers a distinct user question, runs within the portable test
budget, and has an explicit stable, research-stage, or external-data label.

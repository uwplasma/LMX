# Examples

The examples are explicit templates for research workflows. They show how to:

- build cases directly in Python
- run from TOML input files
- generate plots and movies
- save `.npz` state bundles
- resume runs from restart files
- benchmark strong scaling
- run autodiff sensitivity and inverse-design studies

## Quick examples

```bash
python examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
python examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
python examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
python examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
python examples/strong_scaling_demo.py --output ./artifacts/examples/strong_scaling_cpu
python examples/autodiff_design_demo.py --output ./artifacts/examples/autodiff_design
```

## Input-file examples

```bash
lmx examples/hartmann_case.toml
lmx examples/hartmann_restart_case.toml
lmx examples/shercliff_case.toml
lmx examples/hunt_case.toml
lmx run hartmann --ha 20 --verbose
lmx run hunt --ha 20 --verbosity debug
```

The example TOML files expose both `verbose = true|false` and
`verbosity = "quiet" | "normal" | "detailed" | "debug"` in their `[logging]`
block, so they double as templates for both batch and interactive runs.

## Replot saved NPZ outputs

```bash
python examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_ha20_results.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot
```

## Publication-facing demos

Strong scaling:

```bash
python examples/strong_scaling_demo.py --output ./artifacts/examples/strong_scaling_cpu
python examples/strong_scaling_demo.py --remote-host <your_gpu_host> --output ./artifacts/examples/strong_scaling_full
```

Autodiff:

```bash
python examples/autodiff_design_demo.py --output ./artifacts/examples/autodiff_design
```

The default autodiff demo now combines:

- a Hartmann-number sensitivity scan of mean velocity
- inverse recovery of a synthetic forcing parameter from a target profile
- publication-style `PNG`/`PDF` summary figures

## Teaching goal

The examples are intentionally written with explicit helper functions and
configuration blocks so that new users can adapt them into custom research
drivers.

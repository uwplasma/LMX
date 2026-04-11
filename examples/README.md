# Examples

The examples are explicit templates for research workflows. They show how to:

- build cases directly in Python
- run from TOML input files
- generate plots and movies
- save `.npz` state bundles
- resume runs from restart files

## Quick examples

```bash
python examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
python examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
python examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
python examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
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

## Teaching goal

The examples are intentionally written with explicit helper functions and
configuration blocks so that new users can adapt them into custom research
drivers.

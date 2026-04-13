# Examples

The examples are explicit templates for research workflows. They show how to:

- build cases directly in Python
- run from TOML input files
- generate plots and movies
- save `.npz` state bundles
- resume runs from restart files
- benchmark strong scaling
- run autodiff sensitivity and inverse-design studies
- preview geometries and meshes before launching a solve
- stage fringing-field benchmark workflows for the next solver family

## Quick examples

```bash
python examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
python examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
python examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
python examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
python examples/strong_scaling_demo.py --output ./artifacts/examples/strong_scaling_cpu
python examples/autodiff_design_demo.py --output ./artifacts/examples/autodiff_design
python examples/autodiff_sensitivity_demo.py --output ./artifacts/examples/autodiff_sensitivity
python examples/fringing_benchmark_demo.py --output ./artifacts/examples/fringing_benchmark
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output ./artifacts/examples/geometry_preview_full
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

Standard CLI runs can also select the execution backend directly:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
```

Autodiff:

```bash
python examples/autodiff_design_demo.py --output ./artifacts/examples/autodiff_design
python examples/autodiff_sensitivity_demo.py --output ./artifacts/examples/autodiff_sensitivity
```

The shipped autodiff examples now cover:

- a Hartmann-number sensitivity scan of mean velocity
- a finite-difference cross-check of autodiff gradients
- inverse recovery of a synthetic forcing parameter from a target profile
- publication-style `PNG`/`PDF` summary figures

Fringing-field scaffold:

```bash
python examples/fringing_benchmark_demo.py --output ./artifacts/examples/fringing_benchmark
```

That example stages a stationwise fringing benchmark on top of the current
fully developed solver family. It is explicit about being a research scaffold,
not the final `extruded_inductionless` solver.

Geometry and mesh preview:

```bash
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output ./artifacts/examples/geometry_preview_full
```

The default path is intentionally preview-only so it stays fast. Add
`--with-post-run` to append a short steady solve and matching postprocessing
figures in the same output tree.

## Teaching goal

The examples are intentionally written with explicit helper functions and
configuration blocks so that new users can adapt them into custom research
drivers.

# Case Cookbook

## Hartmann

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
/Users/rogerio/base_env/bin/python3 examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
/Users/rogerio/base_env/bin/lmx /Users/rogerio/local/tests/LMX/examples/hartmann_case.toml
```

Use this as the simplest solver smoke test and as the default analytical-profile
validation case.

## Shercliff

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run shercliff --ha 100 --output ./out/shercliff
/Users/rogerio/base_env/bin/python3 examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
/Users/rogerio/base_env/bin/lmx /Users/rogerio/local/tests/LMX/examples/shercliff_case.toml
```

This is the insulating-wall duct case. It exercises the same self-consistent solver
path as the other native cases while remaining a strong analytical comparison target.

## Hunt

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hunt --ha 100 --output ./out/hunt
/Users/rogerio/base_env/bin/python3 examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
/Users/rogerio/base_env/bin/lmx /Users/rogerio/local/tests/LMX/examples/hunt_case.toml
```

This is the conducting-wall duct case. It is useful for checking how the solver
handles explicit solid conductivity layers and sharper boundary-layer physics.
By default, the native Hunt factory uses wall conductance ratio `c = 0.05` and
derives wall conductivity from the fluid conductivity, Hartmann-wall spacing, and
wall thickness. Provide an explicit wall conductivity only when a case is defined
that way.

## Publication-ready plots

Each example script writes:

- ParaView output for the solved case
- midplane CSV profiles
- `example_report.json`
- `overview.png/.pdf` with field maps and profile comparisons
- `diagnostics.png/.pdf` with solver histories

Shercliff and Hunt examples automatically use the default closed-channel
analytical reference root under `./external/FreeMHDPaperAllFigures/...` when it
exists, so in the common local setup you can run them without passing extra
paths.

For a single meeting-ready run that prints verbose solver logs, writes `.npz`
result files, and produces multiple figures and movies:

```bash
/Users/rogerio/base_env/bin/python3 examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
```

That example writes:

- steady Hartmann, Shercliff, and Hunt overview/diagnostics plots
- steady result dumps like `hartmann/hartmann_ha20_results.npz`
- `meeting_demo_report.json`
- startup 2D/3D movies and poster frames for the selected `--movie-case`
- `shercliff/shercliff_startup_snapshots.npz`
- `shercliff/movie/shercliff_startup_2d.gif`
- `shercliff/movie/shercliff_startup_3d.gif`
- `shercliff/movie/shercliff_startup_2d_poster.png`
- `shercliff/movie/shercliff_startup_3d_poster.png`

The retained default movie case is Shercliff because it gives the clearest
current meeting visual at modest runtime. If you switch to `--movie-case hunt`
or `--movie-case hartmann`, the movie view follows that case instead. The Hunt
movie path automatically uses `u - <u>_fluid` because the early Hunt bulk flow
is nearly uniform and the deviation field is the clearest way to show the
boundary-layer development.

This example is also meant to be read, not just executed. It defines the local
workflow functions directly in `examples/theory_meeting_demo.py`, so users can
copy and adapt the case setup, solver controls, output rules, and logging
format for their own studies.

For the text-input equivalent, use the shipped TOML examples and the executable
path documented in [`docs/input_reference.md`](input_reference.md).

The NPZ files are intentionally first-class outputs. Replot a steady result or
a transient movie without rerunning the solver:

```bash
/Users/rogerio/base_env/bin/python3 examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_ha20_results.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot
/Users/rogerio/base_env/bin/python3 examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_startup_snapshots.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot_movie --movies --stem shercliff_replot
```

## Optional external validation backends

Recovered external cases are optional. They are useful when you want to compare LMX
against archived paper cases or regenerated backend outputs, but they are not required
to run the native solver.

```bash
python3 scripts/fetch_freemhd_assets.py --dest ./external
/Users/rogerio/base_env/bin/python3 scripts/inspect_freemhd_setup.py --output ./artifacts/freemhd_setup.json
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_parity_suite.py --output ./artifacts/freemhd_parity
```

## Validation checks

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hartmann --ha 20 --output ./out_validation/hartmann
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate shercliff --ha 20 --output ./out_validation/shercliff --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 -m lmx.cli validate hunt --ha 20 --output ./out_validation/hunt --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output ./artifacts/validation --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_convergence_suite.py --output ./artifacts/convergence --cases hartmann,shercliff,hunt --ha 20 --resolutions 16,32,48 --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_time_convergence_suite.py --output ./artifacts/time_convergence --cases hartmann,shercliff,hunt --ha 20 --resolution 32 --dts 0.002,0.001,0.0005 --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep --case hunt --ha 20 --resolution 48 --wall-cells 5 --parameter outer_iterations --values 2,4,6,8,10 --value-type int --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel
/Users/rogerio/base_env/bin/python3 scripts/run_solver_control_sweep.py --output ./artifacts/control_sweep_hartmann --case hartmann --ha 20 --resolution 32 --parameter potential_iterations --values 50,100,200,400,800 --value-type int
```

These commands produce the analytical and sampled comparison reports that back the
current regression and validation tests. The convergence summary now also records
estimated Hartmann-layer and side-layer cell counts for the duct cases so mesh
adequacy is visible in the artifact rather than inferred indirectly from error
trends. The time-convergence summary complements that by showing how much of the
remaining error changes with pseudo-time refinement at fixed mesh resolution. The
control sweep is useful when a remaining error appears to depend on coupling
controls rather than on mesh or pseudo-time alone. The Hartmann
`potential_iterations` sweep at `Ha20`, `32^2` is now a particularly useful
diagnostic because it exposes the current refinement blocker directly.

## Benchmarking

```bash
/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output ./artifacts/benchmarks/benchmark.json
```

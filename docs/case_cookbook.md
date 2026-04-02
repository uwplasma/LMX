# Case Cookbook

## Hartmann

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
/Users/rogerio/base_env/bin/python3 examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
```

Use this as the simplest solver smoke test and as the default analytical-profile
validation case.

## Shercliff

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run shercliff --ha 100 --output ./out/shercliff
/Users/rogerio/base_env/bin/python3 examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
```

This is the insulating-wall duct case. It exercises the same self-consistent solver
path as the other native cases while remaining a strong analytical comparison target.

## Hunt

```bash
/Users/rogerio/base_env/bin/python3 -m lmx.cli run hunt --ha 100 --output ./out/hunt
/Users/rogerio/base_env/bin/python3 examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
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

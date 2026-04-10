# Case Cookbook

## Hartmann

### CLI

```bash
python -m lmx.cli run hartmann --ha 20 --output ./out/hartmann
lmx examples/hartmann_case.toml
python -m lmx examples/hartmann_case.toml
```

### Python

```python
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
solution = solve_steady(case)
```

## Shercliff

### CLI

```bash
python -m lmx.cli run shercliff --ha 20 --output ./out/shercliff
lmx examples/shercliff_case.toml
```

## Hunt

### CLI

```bash
python -m lmx.cli run hunt --ha 20 --output ./out/hunt
lmx examples/hunt_case.toml
```

## Validation and convergence

```bash
python -m lmx.cli validate hartmann --ha 20 --output ./out/validation/hartmann
python scripts/run_validation_suite.py --output ./artifacts/validation
python scripts/run_convergence_suite.py --output ./artifacts/convergence --cases hartmann,shercliff,hunt --ha 20 --resolutions 16,32,48
python scripts/run_time_convergence_suite.py --output ./artifacts/time_convergence --cases hartmann,shercliff,hunt --ha 20 --resolution 32 --dts 0.002,0.001,0.0005
```

## Plotting and movies

```bash
python examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
python examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_ha20_results.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot
```

## Restart

```bash
lmx examples/hartmann_case.toml
lmx examples/hartmann_restart_case.toml
```

The second run resumes from the first run’s `.npz` state and extends the
simulation while appending diagnostics.

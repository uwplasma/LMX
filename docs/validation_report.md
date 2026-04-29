# Validation Report

## Validation tiers

LMX uses a layered validation strategy.

### Tier 1: analytical and semi-analytical benchmarks

- Hartmann
- Shercliff
- Hunt

Primary observables:

- velocity profiles
- electric-potential profiles
- current-density profiles
- integral flow-rate and pressure-gradient surrogates

### Tier 2: convergence and conservation

- mesh refinement
- time-step refinement
- linear residual histories
- current-divergence and interface-current residuals

### Tier 3: external reference-output comparisons

External executable results can be compared through exported fields, slices,
and profile files. These are benchmark cross-checks, not the definition of the
governing model.

## Current acceptance focus

The `1.0` release is aiming for:

- stable Hartmann analytical acceptance across representative meshes
- stable Shercliff analytical acceptance across representative meshes
- Hunt validation that is judged from literature-matched wall modeling,
  profile errors, and integral observables, not from ad hoc scalar trace tuning
- low-De bent-pipe current closure that is locally conservative, not only
  globally balanced

The latest closure details are collected in [](closure_notes.md). That page is
the audit trail for the two recent blockers: Hunt `Ha = 100` side-layer
agreement and bent-pipe local `div J`.

## Recently Closed Release Lanes

| Lane | Closure evidence |
| --- | --- |
| Hunt `Ha = 100` side-layer | Thin-wall reference model `t_w=0.001`, `sigma_w/sigma=5`, `c=0.05`; retained `z_l2 = 2.89e-3` |
| Bent-pipe low-De charge closure | Conservative mapped-pipe potential sign fixed; retained `max_charge_balance_residual = 2.16e-12` |
| Reader-facing straight-duct profiles | Hartmann, Shercliff, and Hunt retained cuts below `L2 <= 1.2e-2` |
| Bounded release readiness | `scripts/run_release_readiness.py` reports no hard blockers |

The strict research-grade deferred lanes remain external Q2D turbulence
parity, external magnetic-obstacle reference data, and higher-inertia
Dean-vortex bent-pipe validation.

The executable external-code audit now has a generated map that separates
available solver/data paths from completed observable-level parity:

![LMX executable external-code validation map](_static/generated/external_validation_readiness.png)

## Combined validation workflow

The top-level executable validation driver is:

```bash
python scripts/run_full_validation_exercise.py \
  --output artifacts/validation/full_validation_exercise \
  --ha-values 10,20 \
  --resolution 12 \
  --fringing-resolutions 8,12 \
  --skip-paraview \
  --write-plot
```

This combines Benchmark A artifact generation with Benchmark B fringing gate
checks and writes JSON, CSV, and Markdown summaries for the current thresholds.

## Runtime diagnostics now exposed

The default solver writes or reports:

- `linear_residual_history`
- `linear_iterations_history`
- `volumetric_flow_rate_history`
- `mean_current_magnitude_history`
- `lorentz_power_history`
- `div_current_max_history`
- `charge_balance_residual_history`
- `gauge_residual_history`
- `interface_current_residual_history`

These quantities are available through the solver log, JSON summaries, and NPZ
state bundles.

## External benchmark policy

Comparisons against external executables should use:

- matched field slices
- matched line cuts
- current-density and Lorentz-force observables
- flow-rate and pressure-drop surrogates

They should avoid relying on backend-specific pressure-correction traces as the
primary acceptance signal for the reduced fully developed solver.

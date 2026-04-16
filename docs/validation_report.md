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
- Hunt validation that is judged from profile and integral observables, not from
  ad hoc scalar trace tuning

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

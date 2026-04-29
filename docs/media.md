# Media and Figure Artifacts

LMX keeps the README and documentation media reproducible without making the
repository heavy. Large landing-page animations are hosted as GitHub release
assets. Compact figures, posters, CSV summaries, and the WHAM blanket flow GIF
are tracked under `docs/_static/generated`.

The machine-readable contract is
`docs/_static/generated/readme_media_manifest.json`. The release-readiness gate
checks that the release-hosted movies have valid release URLs, that local media
and posters exist, and that the core publication-facing figures are present.

## Landing-Page Movies

| Media | Storage | Generator | Status |
|---|---|---|---|
| 2D Hunt startup GIF | GitHub release asset | `examples/readme_showcase_demo.py --movie-view 2d` | Boundary-layer startup from flat plug flow |
| 3D Hunt startup GIF | GitHub release asset | `examples/readme_showcase_demo.py --movie-view 3d` | 3D profile-slab view of the same transient |
| Q2D turbulence GIF | GitHub release asset | `examples/q2d_turbulence_decay_demo.py` | Nonlinear SM82-style movie gate, external turbulent parity still open |
| WHAM blanket flow GIF | Tracked local docs asset | `examples/wham_blanket_flow_demo.py` | Reduced fixed-flow blanket route movie |

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_2d.gif" alt="LMX 2D Hunt startup movie" width="45%">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_3d.gif" alt="LMX 3D Hunt startup movie" width="45%">
</p>

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/q2d_turbulence_decay.gif" alt="LMX Q2D turbulence movie" width="45%">
  <img src="_static/generated/wham_blanket_flow.gif" alt="LMX WHAM blanket reduced-flow movie" width="45%">
</p>

## Publication-Facing Figures

| Artifact | Generator | What It Shows |
|---|---|---|
| `analytic_velocity_profiles.png` | `examples/straight_duct_profile_comparison.py` | Analytical Hunt/Shercliff overlays |
| `closed_channel_validation_ladder.png` | `examples/hartmann_validation_ladder.py` | Hartmann/Shercliff/Hunt mesh and Ha ladder |
| `q2d_turbulence_observables.png` | `examples/q2d_wall_bounded_validation.py` | Q2D energy, spectrum, and observable readiness |
| `magnetic_obstacle_schematic.png` | `examples/magnetic_obstacle_benchmark.py` | Localized-field geometry and response setup |
| `magnetic_obstacle_benchmark.png` | `examples/magnetic_obstacle_benchmark.py` | Internal matched no-field response gate |
| `bent_pipe_overview.png` | `examples/bent_pipe_inductionless_demo.py` | Curved-pipe geometry and low-De response |
| `variable_field_tabulated_reconstruction.png` | `examples/variable_field_tabulated_demo.py` | Tabulated-field interpolation/reconstruction gate |
| `wham_blanket_flow.png` | `examples/wham_blanket_flow_demo.py` | WHAM blanket pressure budget and local sections |
| `wham_blanket_autodiff_research.png` | `examples/wham_blanket_autodiff_research_demo.py` | WHAM pressure sensitivity and field-scale inverse design |
| `strong_scaling.png` | `scripts/run_strong_scaling_worker.py` | Solver-facing CPU/GPU scaling summary |

![LMX straight-duct analytical profile overlay](_static/generated/analytic_velocity_profiles.png)

![LMX WHAM blanket differentiable pressure-drop study](_static/generated/wham_blanket_autodiff_research.png)

## Regeneration Commands

Use the bounded local commands below before a release or manuscript-figure pass:

```bash
python examples/readme_showcase_demo.py --output docs/_static/generated --skip-geometry --movie-view 2d
python examples/readme_showcase_demo.py --output docs/_static/generated --skip-geometry --movie-view 3d
python examples/q2d_turbulence_decay_demo.py
python examples/wham_blanket_flow_demo.py
python examples/wham_blanket_autodiff_research_demo.py
python scripts/run_release_readiness.py --output artifacts/release/release_readiness.json
```

The first three commands may produce large GIFs. The repository policy is to
upload the large landing-page GIFs to a GitHub release and keep only posters
and compact local media in git. The release-readiness report enforces that
policy through `readme_media_manifest.json`.

# Media and Figure Artifacts

LMX keeps the README and documentation media reproducible without making the
repository heavy. Large landing-page animations are hosted as GitHub release
assets. Compact figures, posters, CSV summaries, and bounded comparison GIFs
are tracked under `docs/_static/generated`.

The machine-readable contract is
`docs/_static/generated/readme_media_manifest.json`. The release-readiness gate
checks that the release-hosted movies have valid release URLs, that local media
and posters exist, and that the core publication-facing figures are present.

## Landing-Page Movies

| Media | Storage | Generator | Status |
|---|---|---|---|
| 2D Hunt startup GIF | GitHub release asset | `campaigns/tutorials/readme_showcase_demo.py --movie-view 2d` | Boundary-layer startup from flat plug flow |
| 3D Hunt startup GIF | GitHub release asset | `campaigns/tutorials/readme_showcase_demo.py --movie-view 3d` | 3D profile-slab view of the same transient |
| Q2D turbulence GIF | GitHub release asset | `campaigns/q2d/q2d_turbulence_decay_demo.py` | Nonlinear SM82-style movie gate, external turbulent parity still open |
| LMX/Q2DmhdFoam Q2D comparison GIF | Tracked local docs asset | `campaigns/q2d/q2d_lmx_q2dmhdfoam_turbulence_comparison.py` | LMX Q2D movie with Q2DmhdFoam spectral-summary context; matched parity open |
| WHAM blanket flow GIF | Tracked local docs asset | `campaigns/blanket/wham_blanket_flow_demo.py` | Centerline pressure-velocity transient to steady flow |

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_2d.gif" alt="LMX 2D Hunt startup movie" width="45%">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_3d.gif" alt="LMX 3D Hunt startup movie" width="45%">
</p>

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/q2d_turbulence_decay.gif" alt="LMX Q2D turbulence movie" width="45%">
  <img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/wham_blanket_flow.gif" alt="LMX WHAM blanket reduced-flow movie" width="45%">
</p>

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/q2d_lmx_q2dmhdfoam_turbulence_comparison.gif" alt="LMX/Q2DmhdFoam Q2D comparison movie" width="54%">
</p>

## Publication-Facing Figures

| Artifact | Generator | What It Shows |
|---|---|---|
| `analytic_velocity_profiles.png` | `campaigns/ducts/straight_duct_profile_comparison.py` | Analytical Hunt/Shercliff overlays |
| `closed_channel_validation_ladder.png` | `campaigns/ducts/hartmann_validation_ladder.py` | Hartmann/Shercliff/Hunt mesh and Ha ladder |
| `q2d_turbulence_observables.png` | `campaigns/q2d/q2d_wall_bounded_validation.py` | Q2D energy, spectrum, and observable readiness |
| `q2d_lmx_q2dmhdfoam_turbulence_comparison.png` | `campaigns/q2d/q2d_lmx_q2dmhdfoam_turbulence_comparison.py` | LMX nonlinear Q2D observables alongside Q2DmhdFoam lid-driven spectral-summary evidence |
| `q2dmhdfoam_lmx_turbulence_match_audit.png` | `campaigns/q2d/q2dmhdfoam_lmx_turbulence_match_audit.py` | Case-dictionary audit preventing unmatched Q2DmhdFoam runs from being promoted to strict nonlinear Q2D parity |
| `q2dmhdfoam_docker_reference.png` | `campaigns/q2d/q2dmhdfoam_docker_reference_validation.py` | Docker-rerun Q2DmhdFoam external executable gate |
| `q2dmhdfoam_lid_driven_vtk.png` | `campaigns/q2d/q2dmhdfoam_lid_driven_vtk_artifact.py` | Generic Q2DmhdFoam Docker rerun VTK ingestion and field observables |
| `q2d_lmx_q2dmhdfoam_lid_driven_parity.png` | `campaigns/q2d/q2d_lmx_q2dmhdfoam_lid_driven_parity.py` | Matched isothermal side-wall LMX/Q2DmhdFoam field-observable comparison |
| `magnetic_obstacle_schematic.png` | `campaigns/magnetic_obstacle/magnetic_obstacle_benchmark.py` | Localized-field geometry and response setup |
| `magnetic_obstacle_benchmark.png` | `campaigns/magnetic_obstacle/magnetic_obstacle_benchmark.py` | Internal matched no-field response gate |
| `magnetic_obstacle_reference_comparison.png` | `campaigns/magnetic_obstacle/magnetic_obstacle_votyakov_strict_attempt.py` | Digitized Votyakov centerline target comparison; currently a strict mismatch |
| `magnetic_obstacle_votyakov_curve_comparison.png` | `campaigns/magnetic_obstacle/magnetic_obstacle_votyakov_curve_validation.py` | Full digitized Votyakov reverse-flow curve and current LMX localized-field mismatch |
| `bent_pipe_overview.png` | `campaigns/fringing/bent_pipe_inductionless_demo.py` | Curved-pipe geometry and low-De response |
| `dean_vortex_reference_comparison.png` | `campaigns/fringing/dean_vortex_bayat_rezai_strict_attempt.py` | Bayat-Rezai moderate-De secondary-flow target comparison; currently a strict mismatch |
| `variable_field_tabulated_reconstruction.png` | `campaigns/fields/variable_field_tabulated_demo.py` | Tabulated-field interpolation/reconstruction gate |
| `wham_blanket_flow.png` | `campaigns/blanket/wham_blanket_flow_demo.py` | WHAM blanket pressure budget and local sections |
| `wham_blanket_transient_flow.png` | `campaigns/blanket/wham_blanket_flow_demo.py` | WHAM blanket centerline pressure-velocity transient to steady state |
| `wham_blanket_pressure_sweep.png` | `campaigns/blanket/wham_blanket_flow_demo.py` | WHAM blanket cumulative pressure drop and field-strength scaling |
| `wham_blanket_autodiff_research.png` | `campaigns/blanket/wham_blanket_autodiff_research_demo.py` | WHAM pressure sensitivity and field-scale inverse design |
| `li_aln_wall_stack_phase0_2.png` | `campaigns/walls/li_aln_wall_stack_phase0_2.py` | Li/AlN unit audit, nested-wall QA, conductance sweep, and pinhole sensitivity |
| `li_aln_wall_stack_phase3_6.png` | `campaigns/walls/li_aln_wall_stack_phase3_6.py` | Li/AlN operating matrix, substrate comparison, degradation thresholds, and pinhole limits |
| `li_aln_multilayer_mesh_qa.png` | `campaigns/walls/li_aln_multilayer_mesh_qa.py` | Explicit `fluid | AlN | metal` mesh, aligned interfaces, region IDs, and conductivity field |
| `li_aln_multilayer_solve.png` | `campaigns/walls/li_aln_multilayer_solve.py` | Explicit multilayer wall-stack solved limiting-case response and conservative-current diagnostics |
| `li_aln_multilayer_convergence.png` | `campaigns/walls/li_aln_multilayer_convergence.py` | Representative explicit multilayer mesh ladder for pressure/current observables |
| `strong_scaling.png` | `scripts/run_strong_scaling_worker.py` | Archived operator-scaling diagnostic; production claims require `extruded_solve` evidence |
| `publication_figure_campaign_summary.json` | `campaigns/publication/publication_figure_campaign.py` | Manuscript figure manifest, status table, and remaining figure gaps |
| `research_grade_closure_status.json` | `campaigns/status/research_grade_closure_status.py` | Strict research-blocker closure status, physics gates, and next artifacts |
| `research_grade_external_data_audit.json` | `campaigns/status/research_grade_external_data_audit.py` | Local external-code/data inputs available for strict blocker closure |
| `research_grade_external_targets.png` | `campaigns/status/research_grade_external_target_figures.py` | Literature/external-code target panel for magnetic-obstacle, Q2D turbulence, and Dean-vortex strict lanes |
| `research_grade_closure_dashboard.png` | `campaigns/status/research_grade_closure_dashboard.py` | Reviewer-facing strict closure dashboard for closed support gates and open blockers |
| `research_grade_final_disposition.png` | `campaigns/status/research_grade_final_lane_disposition.py` | Last-push strict-lane disposition with measured offenders and required next physics |

![LMX straight-duct analytical profile overlay](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/analytic_velocity_profiles.png)

![LMX WHAM blanket differentiable pressure-drop study](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/wham_blanket_autodiff_research.png)

![LMX Li/AlN wall-stack Phase 0-2 reduced study](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/li_aln_wall_stack_phase0_2.png)

![LMX Li/AlN wall-stack Phase 3-6 reduced parametric assessment](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/li_aln_wall_stack_phase3_6.png)

![LMX Li/AlN explicit multilayer wall-stack mesh QA](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/li_aln_multilayer_mesh_qa.png)

![LMX Li/AlN explicit multilayer solved wall-stack gate](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/li_aln_multilayer_solve.png)

![LMX Li/AlN explicit multilayer mesh ladder](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/li_aln_multilayer_convergence.png)

![LMX strict external validation targets](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/research_grade_external_targets.png)

![LMX strict research-grade validation closure dashboard](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/research_grade_closure_dashboard.png)

![LMX final strict research-lane disposition](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/research_grade_final_disposition.png)

## Regeneration Commands

Use the bounded local commands below before a release or manuscript-figure pass:

```bash
python campaigns/tutorials/readme_showcase_demo.py --output docs/_static/generated --skip-geometry --movie-view 2d
python campaigns/tutorials/readme_showcase_demo.py --output docs/_static/generated --skip-geometry --movie-view 3d
python campaigns/q2d/q2d_turbulence_decay_demo.py
python campaigns/q2d/q2d_lmx_q2dmhdfoam_turbulence_comparison.py
python campaigns/q2d/q2dmhdfoam_lmx_turbulence_match_audit.py
python campaigns/q2d/q2dmhdfoam_docker_reference_validation.py
python campaigns/q2d/q2dmhdfoam_lid_driven_vtk_artifact.py
python campaigns/q2d/q2d_lmx_q2dmhdfoam_lid_driven_parity.py
python campaigns/magnetic_obstacle/magnetic_obstacle_votyakov_curve_validation.py
python campaigns/blanket/wham_blanket_flow_demo.py
python campaigns/blanket/wham_blanket_autodiff_research_demo.py
python campaigns/walls/li_aln_wall_stack_phase0_2.py
python campaigns/walls/li_aln_wall_stack_phase3_6.py
python campaigns/walls/li_aln_multilayer_mesh_qa.py
python campaigns/walls/li_aln_multilayer_solve.py
python campaigns/walls/li_aln_multilayer_convergence.py
python campaigns/publication/publication_figure_campaign.py
python campaigns/status/research_grade_closure_status.py
python campaigns/status/research_grade_external_data_audit.py
python campaigns/status/research_grade_external_target_figures.py
python campaigns/status/research_grade_closure_dashboard.py
python campaigns/status/research_grade_final_lane_disposition.py
python scripts/run_release_readiness.py --output artifacts/release/release_readiness.json
```

The first three commands may produce large GIFs. The repository policy is to
upload the large landing-page GIFs to a GitHub release and keep only posters
and compact local media in git. The release-readiness report enforces that
policy through `readme_media_manifest.json`.

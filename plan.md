# LMX plan

Status: active · Revised: 2026-09-06 · Audited source: `d18fefc` (main) plus open PR #63

This is the single roadmap and work log. It replaces the 2026-09-05 research
roadmap, which remains readable at
[`d18fefc:plan.md`](https://github.com/uwplasma/LMX/blob/d18fefce2eb241b4c8d8bc733e914e530e8ce4d4/plan.md).
That roadmap was a ten-milestone research program (thermal blankets, live
VMEX/ESSOS derivatives, device design, three papers). This revision narrows
the destination to what the code can credibly deliver next, retires the
experimental lanes that consumed most of August, and orders the work so that
GPU speed, showcase results and B1/B2 validation stop waiting on each other.

## 1. Destination

LMX is a small, differentiable JAX code for inductionless liquid-metal MHD in
ducts and pipes. Version 2.0 must deliver, with evidence:

1. **Fully developed ducts** (Hartmann, Shercliff, Hunt, layered walls) with
   analytical validation and implicit adjoints. Stable today.
2. **One generic 3-D extruded model** (rect/layered duct and straight pipe) in
   prescribed spatially varying fields, including tabulated coil fields, with
   convective momentum transport, a single conservative residual, restart,
   VTK output and checked derivatives.
3. **Q2D vortex dynamics** (Sommeria–Moreau) with forced turbulence, energy
   and enstrophy budgets, movies, and a Dedalus cross-check.
4. **CPU/GPU execution where one GPU is much faster than one CPU** for the
   3-D and Q2D models, and measured one-to-two-GPU strong scaling for both.
5. **B1/B2 validation** against properly digitised ALEX data and a matched
   FreeMHD run, executed through the generic 3-D model, not a private lane.
6. **A slim repository**: normal clone below 10 MB (3.5 MB today), package
   at or below 15 modules and roughly 9,000 source lines, seven examples that
   each produce a figure or movie, docs under 20 pages.

Parked until 2.0 ships: thermal/conjugate physics, buoyancy, curved or mapped
channels, live VMEX/ESSOS coil and equilibrium derivatives, blanket design
studies, B1 production convergence at Ha = 6600, and any paper beyond a
methods/software paper. They stay out of the package and out of user docs.

## 2. State on 2026-09-06

| Item | Measured | Meaning |
|---|---|---|
| Portable suite (local M3 Max, Python 3.11.14, JAX 0.10.2, SOLVAX 0.20) | 508 passed, 199 s wall, 884 test-seconds | Regression contracts hold; coverage is enforced by the hosted combine step (95.2 %) |
| Hosted CI on main | support 4.4 min, physics 3.2 min, fringing 5.1 min, coverage combine, FreeMHD B2 smoke 5.4 min | Green; cost is acceptable |
| Ruff, format, architecture audit, Sphinx `-W` | pass; 15 modules, 6,385 core / 14,835 total source lines, 28 root exports | Budgets hold |
| Fresh clone | 3.47 MB (Git 1.70 MB, mostly 4,000-line historical `plan.md` blobs) | No history rewrite needed |
| Open PRs | #58–#62 stacked, superseded by the green integration batch #63 (+698/−206) | Merge #63, close #58–#62 |
| Lane inventory | ≈11,100 of 32,000 lines (35 %) exist only for the ALEX B1/B2 specialised solvers, the Benchmark B independence runner, the FreeMHD parity harness and their tests | Primary simplification target |
| Generic 3-D recurrence | omits `u·∇u`; clips velocity and equalises stationwise flow; dispatches on case-name prefix (fixed in #63) | Not yet a physical 3-D model |
| Primal solve loops | Python `for` loops with per-step `float()`/`bool()`/`device_get` syncs; `lax.scan` unused; only the design (`design_parameters`) path is traced | Explains GPU being slower than CPU on the small duct and two GPUs slower than one |
| Global `jax_enable_x64` at import (`mesh.py`) | hidden; forces float64 3-D kernels | Blocks a float32 GPU fast path |
| Q2D | jitted `checkpointed_fori_loop`, FFT-based; 256² is 5.9× (primal) / 8.6× (gradient) faster on an A4000 than office CPU | The one model already GPU-ready |
| ALEX reference tables | 16–18 hand-anchored points per case with round values; no public dataset exists (Smolentsev 2015 Table II and Figs 3–4; ANL/FPP/TM-228) | Must be re-digitised with stated uncertainty before any acceptance claim |
| B2 steady state | momentum defect 0.138 versus the 1e-3 target after 60 commits on 2026-08-29 | The specialised lane did not converge; do not continue it |
| Showcase media | three WebP posters (152 KB); Q2D demo is a 64², 160-step laminar decay; no movie in the repo or docs | Does not yet meet the "engaging plots and movies" goal |
| Office GPU host | unreachable on 2026-09-06 (SSH timeout); last measured 2026-09-04 | GPU campaign needs the host back |

## 3. Decisions

1. **Retire the specialised B1/B2 solver lane from the package.** Delete the
   B2 finite-volume momentum/coupling machinery, the B1 Schur/modal pipe
   path, Anderson/Aitken restart state, the Benchmark B independence runner
   and their tests. Keep `src/lmx/data/benchmarks` (specs, references),
   `benchmarks/provenance.json`, the FreeMHD input/observation contract and
   the Docker comparator. The lane and its evidence stay reachable at tag
   `lmx-research-assets-v1` and commit `d18fefc`.
2. **B1/B2 become targets of the generic model.** Acceptance requires the
   generic residual with convection (item 5 below), three meshes, and matched
   FreeMHD and digitised experimental curves with uncertainty. B2 first; B1
   only after B2 is accepted and only if Ha = 6600 layers can be resolved.
3. **One traced recurrence for primal and derivative solves.** The existing
   `checkpointed_fori_loop` design path becomes the only 3-D solve path;
   host-side loops remain only around it for logging, restart and acceptance.
4. **Explicit precision.** Remove the import-time x64 switch. Cases declare
   `dtype`; fully developed and validation runs default to float64, 3-D and
   Q2D production runs may use float32 with documented tolerances.
5. **No new physics before 2.0** beyond convective transport in the generic
   3-D model and forcing in Q2D. Thermal, mapped geometry and live coil AD
   are parked, with their specifications kept in the previous plan.
6. **Figures and movies are release deliverables**, generated by the same
   seven examples at larger settings on the office GPU, stored as release
   assets, with the posters kept under the 500 KB media budget in Git.

## 4. Work order

Each item is one PR unless noted. Every PR reports net lines, files, and the
timing effect on the suite; “implemented” closes nothing without its gate.

| # | Work | Gate |
|---|---|---|
| P0 | Merge #63; close #58–#62 with the merge link; land this plan and the concise README | Hosted CI green on main |
| P1 | Lane retirement (decision 1): remove B1/B2 solver branches from `fringing.py`, `_fringing_duct.py`, `_fringing_pipe.py`, `_fringing_common.py`, `io.py`, `specs.py`; delete `scripts/run_benchmark_b_independence.py`, `tests/test_run_benchmark_b_independence.py`, most of `tests/test_benchmarks.py`; fold `validation/freemhd.py` + `scripts/run_freemhd_parity_suite.py` into one ≤600-line `lmx/freemhd.py` comparator; remove the test-only helpers listed in the audit | Suite green, coverage ≥95 %, source ≤10,000 lines, no public API for retired paths, docs and provenance updated |
| P2 | One traced 3-D recurrence: move duct/pipe primal loops onto the `checkpointed_fori_loop` path with `lax.scan` diagnostics; drop velocity clipping and stationwise equalisation or justify them as tested limiters; explicit dtype (decision 4); jitted fully developed coupling loop | Bitwise-identical results on the existing regression tests at float64; per-step host syncs zero; warm one-A4000 generic duct 64×32×32 faster than local CPU |
| P3 | Convective momentum transport in the generic model with a conservative, limited discretisation; B = 0 Poiseuille and manufactured-solution order checks; energy budget with body-drive, viscous and Joule work on one control volume (closes F2) | Observed second-order spatial convergence at B = 0 and Ha = 20; energy defect below 1e-6 on the manufactured case |
| P4 | Performance and scaling campaign on the office A4000s (extend `lmx benchmark`): cold/warm, primal/gradient, peak device memory via `jax.profiler`, one- and two-GPU strong scaling at three sizes for the 3-D duct and Q2D 512²/1024²; publish JSON plus one figure | ≥2× one-GPU speedup over office CPU and ≥60 % two-GPU efficiency on at least one production-size case each for 3-D and Q2D, or a documented reason |
| P5 | Showcase examples and movies: forced Q2D turbulence at 512² with energy/enstrophy spectra and an MP4; 3-D fringing duct with current streamlines and axial pressure; a tabulated stellarator-coil field duct (field sampled from ESSOS offline, stored as NPZ under 200 KB, no ESSOS dependency); regenerate the three posters with consistent styling | Each example runs portably in <60 s at demo settings, and produces the release figure at production settings on GPU |
| P6 | B2 validation through the generic model: re-digitise Smolentsev 2015 Figs 3–4 with WebPlotDigitizer and record uncertainty; build the matched FreeMHD B2 case in `freemhd_install` (blockMesh, tanh field, thin conducting wall) and add Shercliff/Hunt L2 assertions there; three meshes; compare axial pressure and side-wall potential | Accepted only when numerical (refinement, conservation) and external (FreeMHD, experiment within uncertainty) gates pass; otherwise `external_validation_open` with the numbers shown |
| P7 | Q2D cross-check with a 30-line Dedalus SM82 script (forced and decaying) and the Camobreco–Pothérat–Sheard subcritical cases as regression targets | Relative L2 below 1e-3 at matched resolution |
| P8 | Release 2.0: version, CITATION, docs pass, release assets (movies, JSON, figures), PyPI upload | All gates above; clean-install smoke on Python 3.10 and 3.13 |

P1 and P2 can proceed in parallel branches; P3 depends on P2; P4 and P5 need
the office host; P6 depends on P3. Documentation for each item ships in the
same PR.

## 5. Validation matrix for 2.0

| Layer | Cases | Evidence |
|---|---|---|
| Analytical | Poiseuille (B = 0), Hartmann, Shercliff, Hunt across Ha and conductance | Full profiles, flow/pressure/current identities, three meshes |
| Manufactured | Velocity/pressure/potential MMS with convection and nonuniform spacing; pipe metrics | Observed spatial order, not forcing built from the tested operator |
| Q2D | Taylor–Green decay, forced statistically steady state, Dedalus parity | Energy/enstrophy budgets, time-step refinement, spectra |
| External | FreeMHD Shercliff/Hunt (image demos with L2 assertions), FreeMHD B2 matched case | Pinned image, source hashes, tolerances declared before running |
| Experimental | ALEX B2 (Ha = 2900, N = 540, c = 0.07); B1 deferred | Digitised curves with uncertainty; numerical gates first |
| Derivatives | Field scale, wall conductivity, geometry scale, forcing, Q2D initial state | Finite-difference sweeps, JVP/VJP duality, Taylor remainder |
| Performance | 3-D duct 64×32×32 and 128×64×64; Q2D 512² and 1024² | Warm timings, peak memory, one/two-GPU scaling, CPU/GPU parity |

## 6. Budgets and CI

- Source ≤10,000 lines after P1, ≤15 modules, ≤30 root exports, tests below
  1.2× source lines, clone below 10 MB, Git media below 500 KB.
- Portable suite below 5 minutes of compute in three balanced shards; the
  FreeMHD comparator runs weekly and on PRs that touch `lmx/freemhd.py`,
  the 3-D modules or benchmark data; GPU campaigns are manual workflows that
  upload JSON evidence.
- No experimental public lanes: a feature is either supported with tests and
  docs or absent from the package.

## 7. References that shape 2.0

- Smolentsev et al., *Fusion Eng. Des.* 100 (2015) 65, DOI 10.1016/j.fusengdes.2014.04.049 (ALEX B1/B2; data by request only); Hua, Walker, Picologlou, Reed, ANL/FPP/TM-228 (1988), OSTI 6789158.
- Ni et al., *J. Comput. Phys.* 227 (2007) 174 (consistent current scheme); Urgorri et al., *PPCF* 66 (2024) 095005 (charge-conservation errors at high Ha; GridapMHD).
- Wynne et al., *Phys. Plasmas* 32 (2025) 013907 (FreeMHD V&V); Jung et al., arXiv 2606.18745 (FreeMHD induction); `rogeriojorge/freemhd_install` (pinned Docker image).
- Jiang and Smolentsev, *Fusion Eng. Des.* 2024 (3-D pressure-drop definition; COMSOL/HIMAG B2 comparison).
- Sommeria and Moreau, *JFM* 118 (1982) 507; Pothérat et al., *JFM* 424 (2000) 75; Camobreco, Pothérat, Sheard, *PRF* 10 (2025) 023905 (Q2D transition targets).
- Eardley-Brunt, Dubas, Davis, *PPCF* 66 (2024) 015015 (open OpenFOAM MHD accuracy and Δt ∝ Ha⁻²).
- Mistrangelo et al., *Nucl. Fusion* 65 (2025) 116006 (multi-code benchmark template).
- JAX-Fluids 2.0, *Comput. Phys. Commun.* 308 (2025); Exponax/APEBench, NeurIPS 2024 (differentiable multi-GPU solver patterns).
- JAX benchmarking, profiling, device-memory and `shard_map` documentation.

## 8. Work log

Keep at most ten entries; older evidence lives in PRs and tags.

| Date | Work and evidence | Next |
|---|---|---|
| 2026-09-06 | Full review of code, tests, PRs #1–#63, literature and comparators; local 508/508 in 199 s; Ruff/architecture/Sphinx pass; fresh clone 3.47 MB; README 317→209 lines; audit found the B1/B2 lane at 35 % of lines, host-synced primal loops and hidden x64 | P0: merge #63; then P1 and P2 in parallel |

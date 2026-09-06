# LMX plan: a differentiable, GPU-fast liquid-metal MHD code with validated results

Revised 2026-09-06 · Audited main `d18fefc` · Competing proposals: #64, #65 · Integration candidate: #63

This is one of three candidate roadmaps (#64, #65, this PR). It is written so that
an independent reviewer can merge the three into a single authoritative plan.
It differs from the other two in that it is built on an explicit literature and
software survey (Section 3) and on fresh runs of the current code at showcase
scale (Section 2), and it commits to a specific numerical architecture instead of
a list of gates. Everything here is a proposal until the reconciled plan lands.

---

## 0. How to read this document

- **Section 1** restates the goals as given by the project owner and turns them
  into measurable targets.
- **Section 2** is the evidence: what the code does today, measured this week.
- **Section 3** is what the literature and comparable codes say we should do
  differently. Each item names its source and how deeply it was read.
- **Section 4** lists the decisions, ADR-style: context, decision, consequences,
  and where #64 and #65 disagree.
- **Section 5** is the target architecture.
- **Section 6** is the phased work plan with concrete steps, effort, and exit
  gates. Phases 1, 2, 3 run in parallel tracks.
- **Section 7** is the validation ladder with numbers.
- **Section 8** lists the showcase figures and movies and how each is produced.
- **Section 9** is the working process (PR size, CI, nightly GPU, planning).
- **Section 10** lists risks and fallbacks; **Section 11** the references.

Terminology: *fully developed* = 2-D cross-section solves (Hartmann, Shercliff,
Hunt); *3-D core* = the extruded duct/pipe solver in spatially varying fields;
*Q2D* = the Sommeria–Moreau depth-averaged vorticity model; *B1/B2* = the ALEX
benchmarks of Smolentsev et al. (2015); *lane* = a code path that exists only to
serve one benchmark or experiment.

---

## 1. Goals and measurable targets

The owner's goals, restated as targets that a reviewer can check:

| # | Goal (owner's words) | Target for the next release |
|---|---|---|
| G1 | Physics of 3-D fields, Q2D and others, with engaging plots and movies | Seven examples, each producing a publication-quality figure; three movies (Q2D forced turbulence, 3-D fringing duct current loops, stellarator-coil-field channel); every figure regenerable by one script |
| G2 | Slim, lightweight repo below 10 MB | Ordinary clone ≤ 5 MB (3.5 MB today); Git media ≤ 1 MB (compressed WebP); movies as release assets |
| G3 | Minimise lines and files, maximise figures and performance | Package ≤ 9,000 source lines and ≤ 14 modules (14,835 / 15 today); tests ≤ 1.2× source lines; no lane that serves one benchmark |
| G4 | CPU and GPU, GPU much faster | One RTX A4000 ≥ 10× an office CPU core-set on the 3-D core at 128³ and ≥ 20× on Q2D 1024²; float32 and float64 both supported and tested |
| G5 | Parallelised with strong scaling (sharding OK) | Two A4000s ≥ 60 % strong-scaling efficiency at 256³ (3-D) and 2048² (Q2D), measured and published as JSON + figure |
| G6 | FreeMHD, OpenFOAM and other codes as inspiration | Adopt the Ni et al. consistent-and-conservative scheme used by FreeMHD, HIMAG, epotFoam and MHD-UCLA; adopt jax-cfd's solver architecture and JAX-Fluids' sharding pattern |
| G7 | B1/B2 validation | B2 (Ha 2900) accepted against digitised experiment and a matched FreeMHD run within stated uncertainty; B1 (Ha 6600) feasibility documented with a cost estimate, attempted only if affordable |
| G8 | Retire experimental lanes | The specialised B1/B2 solver lane (≈11,100 lines) deleted from `main` after the new core passes the same gates; history kept at a tag |

Non-goals until the release after next: buoyancy, curved/mapped channels, live
coil or equilibrium derivatives, neutronics or structural coupling, free surfaces,
magnetic induction (Rm ≪ 1 stays). Thermal transport enters as a *passive scalar*
on the new core (Phase 5) because it is cheap once the core is right and because
the blanket question is ultimately thermal; buoyancy and property variation do not.

---

## 2. Where the code is on 2026-09-06 (measured)

All numbers below were produced this week on an Apple M3 Max (Python 3.11.14,
JAX 0.10.2, SOLVAX 0.20) unless stated. The office A4000 host was unreachable
on 2026-09-06; GPU numbers are from the 2026-09-04 audit.

### 2.1 What works

| Check | Result |
|---|---|
| Portable suite at `d18fefc` | 508 passed in 199 s wall (884 test-seconds over 6 workers) |
| Ruff, format, architecture audit, Sphinx `-W` | pass |
| Seven examples | all run; 7–32 s each |
| Hosted CI on `main` and on #58–#65 | green (docs-only PRs skip numerics by design) |
| FreeMHD pinned Docker image, built locally under x86 emulation | 15 min build; B2 smoke passes in 17 s, pressure RMS 0.0031 / max 0.0068 against bounds 0.16 / 0.32; acceptance withheld by role |
| Fresh clone | 3.47 MB (Git 1.70 MB, dominated by ~290 KB historical `plan.md` blobs) |
| Hunt duct, default 72² fluid + 8 wall cells (CLI) | Ha 20 / 100 / 500 / 1000 converge in 20 / 28 / 63 / 34 s; side-layer jet position scales as Ha^-1/2 between Ha 500 and 1000 (ratio 1.41) |
| Q2D 256², 3000 steps, float32, CPU | 19.5 s; energy-budget defect 5.8e-6; clear vortex merging and a k^-3 enstrophy range |

The last two rows are new showcase material and are included in this PR
(Section 8): a Hunt Hartmann-number sweep figure and a 256² Q2D turbulence
animation.

### 2.2 What does not

| Finding | Evidence | Consequence |
|---|---|---|
| CLI `--ny/--nz` are ignored for fully developed cases (`cli.py:55–62`); `make_hunt_case(ny=96)` hits `step_limit` at 500 steps while the 72² default converges | fresh runs | Control surface needs one contract |
| The 3-D core has no steady solver and no working step control | `outer_steps = min(max_steps, max(6, 2·coupling_iterations))` in `fringing.py:293/925`; a request for 600 steps ran 16 and returned `step_limit` | Users cannot converge a 3-D case; every 3-D figure in the repo is a short transient |
| Generic 3-D steps omit convection, clip velocity and rescale stationwise flow | `_fringing_common.py:899–970`, `_fringing_pipe.py:546–607`; velocity limit `max(5, 2√Ha)` | Not a physical model at finite N; cannot reach B2 (Re = 15,574) |
| Primal loops are Python loops with per-step host syncs | `float()`/`bool()`/`device_get` at `fringing.py:602/627/641/1231–1243`, `cases.py:411/566–570/853`; `lax.scan` unused; only the `design_parameters` path is traced | One A4000 4× *slower* than a laptop CPU on the small duct; two GPUs slower than one (2026-09-04 audit) |
| `jax_enable_x64` forced at import (`mesh.py:13`) | pulled in by every module | No float32 path; A4000 fp64 is 1/64 of fp32 throughput |
| Specialised B1/B2 lane | ≈11,100 of 32,000 lines (35 %): `_fringing_duct.py:220–936, 1083–1554`, `_fringing_pipe.py:239–1292`, 18 conditional blocks in `fringing.py`, `validation/freemhd.py`, two scripts, three test files; B2 momentum defect 0.138 vs target 1e-3 after ~60 commits on 2026-08-29 | Maintenance and CI cost with no accepted result |
| Reverse-mode NaNs in the #63 residual | reported by #65: 81 non-finite VJP entries from a zero-gradient division in `_limited_linear_vector_face_weights_duct` (forward Jacobian finite) | Must be fixed before #63 merges; classic `jnp.where` double-branch issue |
| ALEX reference tables | 16–18 hand-anchored points with round values; no public dataset exists (Smolentsev 2015 Table II and Figs 3–4; ANL/FPP/TM-228 Figs 11–13) | No acceptance claim is possible until re-digitised with uncertainty |
| FreeMHD comparator demos | bundled Shercliff/Hunt cases run ten steps (endTime 1e-5 s) and return the initial profile | Install checks only; a validator needs physically long runs |
| Showcase media | three static WebP posters; Q2D demo is a 64² laminar decay; fringing demo is a 6×6 cross-section with a five-point triangular field | Does not meet G1 |
| Process | 64 PRs in 17 days, median 169 lines (already small); per-PR evidence template, ≥95 % combined-coverage gate, architecture budgets and a 45-min Docker job on 3-D changes; `plan.md` rewritten to 4,263 lines then 705 | Overhead is per-PR ceremony, not PR size |

### 2.3 Open PRs

| PR | Content | Recommendation |
|---|---|---|
| #58–#62 | stacked M0/M1/M5 work | superseded by #63; close with the merge link |
| #63 | integration batch (+698/−206, 24 files), green | fix the VJP NaN, re-qualify, merge first |
| #64 | my first plan (2.0 work order, README 317→209) | superseded by this PR; close |
| #65 | evidence-led channel plan (README 317→183) | reconcile with this PR (Section 4.13) |

---

## 3. What the research says we should do differently

Five focused surveys were run for this plan (numerics at high Ha; JAX/GPU solver
architecture; Q2D turbulence; open validation data; presentation and process).
Items marked [V] were verified by reading the source; [M] are from memory.

### 3.1 Numerics for high-Ha inductionless MHD

1. **Use the consistent-and-conservative scheme of Ni et al. (JCP 227:174 and 205,
   2007) [V].** Current is computed as a *face flux*
   `J_n,f = −σ_f (φ_N − φ_P)/d_PN + σ_f (u_f × B_f)·n_f`, the potential Poisson
   equation is assembled as the divergence of exactly that flux, and the Lorentz
   force is reconstructed at the cell centre by the face formula
   `(J×B)_c = (1/Ω) Σ_f J_n,f s_f (r_f − r_c) × B_f`. This is what HIMAG,
   MHD-UCLA, epotFoam (Tassone 2016), Blishchik et al. (2021), Siriano et al.
   (2024), Eardley-Brunt et al. (2024) and FreeMHD (Wynne 2025) implement [V].
   The reason it matters: in the core `−∇p + J×B = 0` to O(Ha^-2); any O(Δ)
   inconsistency between the two parts of J is amplified by Ha² and appears as
   spurious core currents. Ni shows only this scheme reproduces Shercliff/Hunt at
   Ha = 1000 (his Fig. 9). Our generic recurrence does not do this; the B2 lane
   partly does.
2. **Staggered (MAC) variables** (Ni Part III, JCP 231:281) [V abstract]: face
   velocities are exactly what `J_n,f` needs, and there is no need for Rhie–Chow
   or velocity clipping.
3. **Thin conducting-wall condition** (Walker 1981; KIT form
   `j·n = ∇_τ·(c ∇_τ φ_w)`) [V via Mistrangelo & Bühler 2017, Priede et al.]:
   the wall becomes a surface Laplacian on wall faces with conductance ratio
   `c = σ_w t_w/(σ a)`. Valid for `t_w ≪ a/√Ha` and c ≪ 1, which holds for B1
   (t_w/a = 0.056, c = 0.027) and B2 (0.14, 0.07). Removes the wall cells that our
   explicit shell needs, and MHD-UCLA reaches Ha = 15,000 on a 99² stretched mesh
   with it (Smolentsev 2015 Table I) [V].
4. **Layer resolution** [V, several sources]: ≥ 6–8 cells across the Hartmann
   layer `a/Ha`, ≥ 10 across the side layer `a/√Ha`, geometric stretching ratio
   ≤ 1.15, axial refinement across the fringe. For B2 that is ~120 cells across
   the half-width; for B1 at Ha 6600 (`δ_Ha/a = 1.5e-4`) ~150. Our uniform-ish
   generic meshes cannot do this; the fully developed mesh generator already
   stretches.
5. **Stiffness** [V Eardley-Brunt 2024; FreeMHD]: the velocity part of the Lorentz
   force is a damping `−σ(B²I − BBᵀ)u` with rate `σB²/ρ`, so explicit treatment
   needs `Δt ∝ Ha^-2` (UKAEA measured Δt falling from 5e-3 s at Ha 20 to 1e-6 s at
   Ha 1000). Treat it linearly implicitly as a *correction* to the conservative
   face force, `F^{n+1} ≈ F^n_cons − σ(B²I − BBᵀ)(u^{n+1} − u^n)`, which vanishes
   at steady state and keeps Ni's consistency. Diffusion Crank–Nicolson.
6. **Steady solves are linear at high N.** For the inertialess limit the coupled
   (u, p, φ) system is linear; the natural solver is matrix-free GMRES on the
   monolithic system with a block preconditioner (Stokes block + potential
   Laplacian; Planas/Badia FMS [V thesis]). With inertia, Newton–Krylov around the
   same operator. Pseudo-transient continuation only as a globaliser. This
   replaces the August approach (Anderson/Aitken/Schur retuning of a projection
   loop) that never converged.
7. **Fringing-field physics and observables** [V ANL/FPP/TM-228; HyPerComp 2008
   report]: the 3-D pressure drop comes from axial current loops closing through
   the side layers; the observable is `ΔP/(σ U B0² L)` between taps. The ANL
   parametric case (square duct, `x0 = 3`, c = 0.02) gives total 0.0932 vs
   fully developed 0.0754 and is an *analytic-model* reference we can code.
   Dimensional B2 set-up: 2a = 0.0878 m, t_w = 6.25 mm, σ_w = 1.51e6 S/m,
   B_max = 1.1 T, U = 0.337 m/s, Re = 15,574. B1: a = 0.0541 m, t_w = 3.01 mm,
   σ_w = 1.39e6 S/m, B_max = 2.08 T, U = 0.07 m/s, Re = 3,986.
8. **Field inputs**: the ALEX magnet profile was *measured*; no open fit exists.
   Use div- and curl-free analytic fringes (Votyakov, Kassinos & Albets-Chico 2009
   [V]) so that `B_x` is consistent with `B_y(x)`, and run B2 both with `B_y` only
   and with the solenoidal pair, as Smolentsev 2015 allows.

### 3.2 JAX and GPU solver architecture

1. **jax-cfd's design** [V source]: pytree containers with static grid/offset/BC
   metadata; boundary conditions applied by pad-then-slice (never `roll` at walls);
   pressure projection by *fast diagonalisation* (host-side `eigh` of the 1-D
   operators, tensor-product apply) with RHS mean subtraction and a nullspace
   cutoff; `lax.scan` (`funcutils.repeated`/`trajectory`) as the only loop.
   This is the model for the new core.
2. **JAX-Fluids 2.0** [V]: one-axis decomposition for two devices, halo exchange
   with `ppermute`, `jax.checkpoint` on every integration step inside `scan`,
   float64 on GPUs, float32 on TPUs; 2.5–3× fp32 speedup over fp64 on GA102.
3. **jaxDecomp** [V JOSS]: pure-JAX slab/pencil FFTs and halo exchange since
   v0.2.0; on two GPUs a slab split plus one `all_to_all` pair per Poisson solve.
   Native sharded `jnp.fft` also exists. Evaluate before writing our own.
4. **Fast diagonalisation on non-uniform tensor grids** [M, Lynch–Rice–Thomas]:
   two matmul passes per non-periodic axis; ~2·10⁹ flops at 128³, trivial in fp32
   on an A4000 but compute-bound in fp64 (0.3 TFLOPS). Hence the precision policy
   below. On uniform Neumann axes use `jax.scipy.fft.dctn` [V].
5. **Batched tridiagonal solves** [V]: `lax.linalg.tridiagonal_solve` is
   differentiable since JAX PR #25787 but calls cuSPARSE gtsv2 per system, not the
   batched kernel; for many short line solves a vectorised Thomas sweep or SOLVAX's
   own routines may win. Measure.
6. **Implicit differentiation** [V]: `jax.lax.custom_root` / `custom_linear_solve`
   (transpose solve required unless symmetric), optimistix `ImplicitAdjoint`,
   diffrax `RecursiveCheckpointAdjoint`; SOLVAX already exposes `root_solve`,
   `newton_krylov` and `checkpointed_fori_loop` (O(√n) memory). #65 notes the
   SOLVAX `root_solve` default tangent builds a dense Jacobian; supply the
   matrix-free tangent/transpose solves explicitly.
7. **Exponax / APEBench** [V]: ETDRK schemes integrate the linear operator
   exactly; a linear drag term is precisely the Hartmann friction of the Q2D model,
   so the high-Ha stiffness disappears. Kassam–Trefethen contour coefficients.
8. **GPU practice** [V JAX docs]: no per-step host sync, whole time loop under
   `jit` + `scan`, `donate_argnums`, `block_until_ready` timing, persistent cache,
   `jax.profiler` for peak memory. **RTX A4000: 19.2 TFLOPS fp32, fp64 = 1/64**
   [V NVIDIA whitepaper], 448 GB/s, 16 GB, PCIe 4.0 (no NVLink).
9. **Expected performance** [M, from bandwidth and JAX-Fluids numbers]: fp64
   single A4000 at 128³ ≈ 20–80 ms/step, 256³ ≈ 0.2–0.7 s/step; fp32 2–3× faster;
   10–40× over a laptop CPU at ≥ 128³ but only 2–5× at 64³ (launch floor);
   two-GPU efficiency 60–85 % at 256³, ~50–70 % at 128³, none at 64³. Memory:
   256³ fp64 with ~30 live fields ≈ 4 GB; √-checkpointing 1000 steps at 256³ fp64
   exceeds 16 GB, so checkpoint in fp32 or two-level.

### 3.3 Q2D turbulence

1. **Model** [V Pothérat–Sommeria–Moreau 2000]: `∂_t u + u·∇u = −∇p/ρ + ν∇²u −
   (n/t_H) u + f`, with `t_H = h²/(ν Ha_h)` and `n` the number of no-slip Hartmann
   walls (n = 2 for a duct). Validity: Ha ≫ 1, N ≫ 1, laminar Hartmann layer
   (Re/Ha ≲ 380), scales `l_⊥ ≳ h N^-1/3`. The current README states none of this.
2. **Blanket regime** [V Smolentsev & Moreau 2006; M properties]: PbLi at
   h = 0.1 m, B = 5 T, U = 0.1 m/s gives Ha ≈ 1.2e4, Re ≈ 7e4, N ≈ 2e3,
   Re/Ha ≈ 6 (laminar layer), t_H ≈ 5.6 s vs turnover 1 s. Friction-dominated
   Q2D is exactly the blanket regime, which justifies keeping this model.
3. **Cases with literature numbers** (Section 7): frozen decay (exact rescaling
   identity when ν → 0), electrically driven vortex lattice (Sommeria 1986:
   Kolmogorov constant 3–7, condensation), Kolmogorov flow with drag (Kochkov
   2021 reference fields at Re 1000/4000), Q2D Kolmogorov-like flow with Rayleigh
   friction (Tithof 2017: Re_c = 7.60, q_c = 0.465 for β = 1), MATUR-like shear
   layer (`δ/h ~ (Re/Ha)^1/2`), Q2D cylinder wake by volume penalisation
   (Hussam 2011 tables).
4. **Numerics** [V Kassam–Trefethen 2005]: ETDRK4 with contour-integral
   coefficients, friction inside the exponential, 2/3 dealiasing; cross-check
   against GeophysicalFlows.jl (same equation, ETDRK4) and Dedalus.

### 3.4 Validation data and comparators

1. **ALEX B1/B2**: digitised curves are "available upon request from the first
   author" (Smolentsev 2015) [V]; the underlying ANL report is open (OSTI 6789158)
   with Figs 11–13 for the square duct [V]. Plan: digitise both with
   WebPlotDigitizer, record axis calibration and repeat-digitisation error, and
   request the original files from the author in parallel.
2. **FreeMHD** [V]: MIT; validated on Shercliff/Hunt (Ha ≤ 1000, errors 0.3–3.9 %)
   and on KIT's fringing pipe (Ha 2000, N 200), *not* on ALEX; inputs on Zenodo
   10.5281/zenodo.13964055 (8.9 GB). Our pinned image builds and runs; its demo
   cases need physically long `endTime` to become validators.
3. **Other open comparators** [V]: GridapMHD (MIT, Julia, monolithic FEM, Hunt
   example; Urgorri 2024 charge-conservation analysis), epotFoam/mhdFoam in the
   same OpenFOAM image (Hartmann tutorial ships with OpenFOAM), Eardley-Brunt
   UKAEA input files (DOI 10.14468/43ys-0z67), Dedalus and GeophysicalFlows.jl
   for Q2D. Vertex-CFD and Nek5000 MHD are too heavy to build for this purpose.
4. **Recent physics targets** [V]: DNS at Ha = 2900, Re = 6299 in a decaying
   field (Acta Mech. Sin. 2025) shows transition inside B2 conditions; Jiang &
   Smolentsev (FED 2024) give a 3-D pressure-drop correlation
   `k(Ha, Re, ∂B/∂x)` over 80 COMSOL cases; Brynjell-Rahkola et al. (PRF 2025)
   and Camobreco et al. (PRF 2025) give Q2D/3-D transition targets.

### 3.5 Presentation and process

1. **README** [V, 13 READMEs analysed]: adopted codes put tagline → badges →
   `pip install` → a ≤ 15-line example or gallery → docs link in the first screen;
   300–1000 words; one hero ≤ 500 KB; caveats live in docs. The three longest
   READMEs in the sample were uwplasma's own (vmex 4590 words, LMX 1613).
   JAX-Fluids has 637 stars with no CI and no badges: adoption came from the
   paper, the images and five notebooks.
2. **Process** [V Google small-CLs; DESC workflows]: our PRs are already small
   (median 169 lines); the cost is the per-PR evidence template, the coverage gate
   and the plan log. DESC's pattern: `unit`/`regression` markers, regression tests
   only on PRs to `master`, benchmarks only with a `run_benchmarks` label, weekly
   cron. GitHub's guidance: never attach a self-hosted GPU runner to a public
   repo's PRs; use a cron on the box that pulls `main` and pushes results.
3. **Reference results as data** [V]: pytest-regressions (`num_regression`,
   `image_regression`) with committed reference files and provenance sidecars;
   pytest-benchmark for PR-vs-main deltas; asv for history.
4. **Publication** [V JOSS/pyOpenSci criteria]: JOSS needs ≥ 6 months of visible
   history (public history starts July 2026 → early 2027), a 750–1750-word paper
   with an AI-usage disclosure; the methods paper (CPC/SoftwareX) needs
   convergence plots at several Ha, one 3-D benchmark against an independent
   reference, gradient verification, a cost figure, and a Zenodo bundle.

---

## 4. Decisions (ADR style)

Each decision records context, the decision, consequences, and the position of
#64 and #65 so the reconciling reviewer can see the deltas.

### D1. Build one new 3-D core on the Ni/MAC/fast-diagonalisation recipe

*Context.* The generic recurrence is Stokes-like, clipped and unconverged; the
B2 lane is closer to Ni's scheme but is collocated, name-dispatched, tuned per
benchmark and never converged; neither runs on a GPU faster than a CPU.
*Decision.* Write `lmx/core3d` (Section 5) from the recipe in 3.1–3.2, reusing
the B2 lane's face-current and limited-convection pieces where they match, and
the existing mesh/materials/IO. Keep the fully developed solver (it works).
*Consequences.* One formulation for rect duct and pipe; steady and transient
share one residual; adjoints by IFT; convection included from the start with a
B = 0 Poiseuille test; ≈ 2,000 lines replaces ≈ 8,000.
*#64:* proposed tracing the existing loops and adding convection to them (weaker).
*#65:* proposes "one conservative inertial rectangular-channel residual" built by
repairing the retained collocated placement or MAC after a tiny-oracle comparison,
and defers pipes. Agreement on the goal; this plan picks MAC now on the strength
of Ni Part III and every ALEX-validated code, and includes the pipe because the
B1 geometry needs it and the O-grid pieces already exist.

### D2. Retire the specialised B1/B2 lane after the new core passes its gates

*Context.* 35 % of the code exists for a lane with no accepted result; #65 argues
"do not delete before their useful conservation, wall, pipe and validation
contracts have a verified replacement".
*Decision.* Freeze the lane now (no new work, tests kept), tag `lmx-b2-lane-v1`,
and delete it in Phase 4 once the new core reproduces Shercliff/Hunt at Ha 1000
(Table I flow rates), the ANL x0 = 3 analytic case, and the reduced FreeMHD B2
smoke. Keep specs, reference CSVs, provenance, and a ≤ 600-line FreeMHD
comparator.
*Consequences.* −≈ 11,000 lines; CI −1 Docker job on most PRs; no loss of
evidence (tag + PR links).
*#64:* delete first. *#65:* keep until replaced. This is the middle path with a
concrete trigger.

### D3. The time loop is `lax.scan` with per-step checkpointing; host code sees chunks

*Context.* #65: "absence of `lax.scan` does not establish a bottleneck; profile
first". Fair, and the 2026-09-04 traces already show the symptom (much time
outside numerical phases; 2 GPUs slower than 1; A4000 slower than laptop). The
JAX documentation is explicit that any per-step host read serialises dispatch.
*Decision.* Day-1 of Phase 2 is a profile that attributes time to host sync vs
kernels on the old code, to have a baseline. The new core is written with
`scan` from the start regardless; acceptance, restart and logging happen once
per chunk of K steps.
*Consequences.* Host-side diagnostics move into the scan carry; `float()` and
`bool()` are banned in `lmx/core3d` by a test that greps for them.

### D4. Explicit precision; no import-time x64

*Decision.* `jax_enable_x64` is enabled by `lmx.enable_x64()` or a case
`dtype` field; fully developed and validation runs default to float64; the 3-D
core and Q2D accept float32 with documented tolerances; fast-diagonalisation
transforms run in fp32 with one fp64 iterative-refinement step when the state is
fp64. `jax_default_matmul_precision='highest'` to avoid TF32.
*Consequence.* The A4000's fp64 penalty is contained to stencils (memory-bound)
rather than transforms.

### D5. Direct Poisson solves (fast diagonalisation) for potential and pressure

*Context.* The current code uses Jacobi/PCG/FGMRES with line preconditioners and
the trace showed "hundreds of tridiagonal kernels and thousands of copies".
*Decision.* Both Laplacians are separable on the tensor grid with thin walls:
solve them directly (host-side `eigh` once; two tensor-contractions per axis).
Keep PCG only for the non-separable cases (explicit conducting-wall stacks with
jumps) wrapped in `custom_linear_solve(symmetric=True)`.
*Consequence.* ~4 kernels per Poisson solve, exact linear map (AD is free),
deterministic cost.

### D6. Thin-wall boundary condition as the default wall model

*Decision.* Implement Walker's condition as a surface Laplacian on wall faces
with corner continuity; keep the explicit shell as an option and as a check.
Report the difference on Hunt at c = 0.05 and on B2 (`t_w/a = 0.14`).
*Consequence.* Meshes shrink by the wall cells; matches how MHD-UCLA and KIT
reach Ha 10⁴.

### D7. Q2D: ETDRK with exact friction, forced turbulence, movies

*Decision.* Replace IFRK4 with ETDRK2/4 (contour coefficients), keep the
existing spectral Poisson from SOLVAX, add forcing cases A–D of Section 7, add
spectra/budget diagnostics, and generate the movies at 1024² on the GPU.
*#65:* "qualify a forced example before adding another turbulence solver or
closure" — agreed; ETDRK is an integrator swap, not a closure.

### D8. Validation as data, comparators as nightly jobs

*Decision.* Reference profiles (analytic, Table I, FreeMHD long runs, digitised
ALEX curves) live as small CSV/NPZ with provenance sidecars and are compared by
fast tests (pytest-regressions). Docker comparators, GPU campaigns and refinement
studies run nightly on the office box via cron, never in PR CI.
*Consequence.* PR CI drops to ≤ 8 min with no coverage threshold; the coverage
report becomes informational at release.

### D9. Thermal transport enters as a passive scalar on the new core (Phase 5)

*Context.* #65 argues thermal physics is essential and should not wait for B2.
*Decision.* Agree, with sequencing: the energy equation is added to the new core
(same MAC faces, same scan) with conjugate walls and Joule heating, verified on
conduction slabs, advection–diffusion MMS and Nu = 48/11 / 3.66 limits, then the
Mistrangelo 2025 heated-duct benchmark as a stretch target. No buoyancy.

### D10. Design and device examples use tabulated fields, not live coil AD

*Decision.* A "stellarator channel" example samples an ESSOS Biot–Savart field
offline into an NPZ ≤ 200 KB (coil set from `ESSOS/examples`), with
gradients w.r.t. flow, wall conductance, field scale and geometry scale via the
IFT adjoint. Live coil-parameter derivatives stay out until the release after next.
*#65:* similar (offline bounded sampling first, then a provider contract).

### D11. Process: small PRs stay, ceremony goes

*Decision.* Adopt the process in Section 9: DESC-style markers, no per-PR coverage
gate, regression tests on PRs to `main` only, nightly GPU cron, ADRs in
`docs/adr/`, a one-page `ROADMAP.md`, and this `plan.md` frozen after
reconciliation.

### D12. Release cadence

*Decision.* Tag `v2.0.0` when Phases 0–4 are complete (Q2D and thermal may ship
as research-stage). PyPI upload and Zenodo DOI at that tag. JOSS submission not
before 2027-01 (history requirement); CPC/SoftwareX methods paper after Phase 6.
*#65:* "no promised version number"; this plan keeps a version because releases
are how users cite and how Zenodo mints DOIs, but ties it to gates, not dates.

### D13. Reconciliation with #64 and #65

| Topic | #64 | #65 | This plan |
|---|---|---|---|
| B1/B2 lane | delete now | keep until replaced | freeze now, delete on trigger (D2) |
| 3-D formulation | trace existing loops, add convection | one conservative residual, MAC vs collocated decided on a tiny oracle | new MAC core on Ni's recipe (D1) |
| Host loops | rewrite | profile first | profile day-1, new core uses scan (D3) |
| Precision | explicit dtype | explicit, no fp32 promise for high Ha | explicit; fp32 transforms + fp64 refinement (D4) |
| Poisson solver | unspecified | unspecified | fast diagonalisation (D5) |
| Walls | unspecified | thin-wall needs validity check | thin-wall default with shell check (D6) |
| Thermal | parked | essential, bounded laminar | passive scalar on new core, Phase 5 (D9) |
| Fully developed design example | not included | first deliverable | Phase 6, cheap, after core (it needs no new numerics) |
| Q2D | forced turbulence + movies | qualify forced example | ETDRK + cases A–D + movies (D7) |
| GPU | campaign after tracing | measure, report crossover | targets in G4/G5 with expected ranges (3.2.9) |
| Release | 2.0 at gates | hold | 2.0 at gates (D12) |
| Plan length | 190 lines | 600 lines | this document, then frozen |

---

## 5. Target architecture

```
src/lmx/
  grid.py        Grid (static shape/spacing/offsets), Field pytree, stretched 1-D coordinates
  bc.py          pad-with-ghosts per BC kind (periodic, no-slip, insulating, thin-wall, symmetry)
  ops.py         slice stencils on padded arrays: grad, div, laplacian, face interp, curl-free B sampling
  poisson.py     FastDiag (eigh on host, tensor apply), DCT path, PCG fallback via SOLVAX
  em.py          face currents J_n,f, Ohm's law, face Lorentz force, thin-wall surface Laplacian
  core3d.py      one step(state, params) -> state: predictor (CN diffusion, conservative advection,
                 implicit Lorentz damping correction), potential solve, pressure projection
  timeloop.py    scan over checkpoint(step); chunked host acceptance/restart/logging
  steady.py      residual R(u,p,phi;theta); Newton–Krylov via solvax.root_solve; IFT adjoint
  fully_developed.py   existing 2-D solver (renamed from cases/solvers), thin-wall option
  q2d.py         ETDRK2/4 spectral SM82 with forcing, budgets, spectra
  thermal.py     energy equation on the MAC grid, conjugate walls, Joule source (Phase 5)
  fields.py      analytic fringes (Votyakov), tabulated fields (NPZ), ESSOS offline sampler
  shard.py       mesh(('x',)), slab sharding, ppermute halos, all_to_all transposes
  io.py, specs.py, cli.py, physics.py   as today, trimmed
```

Data model: one frozen `Case` dataclass (geometry, materials, field, drive, walls,
solver controls, dtype), one `State` pytree (face velocities, p, φ), one `Result`
(state, diagnostics, status). Names are metadata; models are selected by fields.

Solver strategy: transient = projection step under `scan`; steady = Newton–Krylov
on the same residual with the projection step as preconditioner; both share
`em.py` and `poisson.py`. Adjoints: steady → IFT (one transposed solve);
transient → per-step `jax.checkpoint` in `scan` or SOLVAX `checkpointed_fori_loop`.

Sharding: slab along the axial direction for the 3-D core (halo = 1 for
second-order stencils, 2 with limited advection); Q2D uses sharded `jnp.fft` or
jaxDecomp; both selected by `num_devices`.

---

## 6. Phased work plan

Effort is in person-weeks for one developer using agents; phases marked ∥ run in
parallel tracks. Every phase ends with a PR that adds its figure to the gallery.

### Phase 0 — Integration and hygiene (week 1)

| Step | Action | Exit |
|---|---|---|
| 0.1 | Fix the VJP NaN in #63 (`jnp.where` double-branch in `_limited_linear_vector_face_weights_duct`), add resting and non-resting VJP tests | JVP/VJP finite; forward Jacobian unchanged |
| 0.2 | Merge #63; close #58–#62 with the link | `main` green |
| 0.3 | Reconcile #64, #65, this PR into one `plan.md` (independent reviewer); close the losers | one plan on `main` |
| 0.4 | Fix `outer_steps` so `max_steps` is honoured; document the change | a 600-step request runs 600 steps |
| 0.5 | Explicit precision (D4): `lmx.enable_x64()`, case `dtype`, remove the import-time switch behind a deprecation | suite green in both modes |
| 0.6 | Process (D11): markers, `ROADMAP.md`, `docs/adr/0001-core3d.md` … `0004-process.md`, CI split (Section 9) | PR CI ≤ 8 min |
| 0.7 | README v2 with the new Q2D animation and Hunt sweep figure (this PR's assets) | README ≤ 150 lines |

### Phase 1 — The new 3-D core (weeks 2–5, ∥ with 2 and 3)

| Step | Action | Exit |
|---|---|---|
| 1.1 | `grid.py`, `bc.py`, `ops.py`: MAC layout, stretched coordinates (tanh/geometric), pad-then-slice BCs; unit tests on constants/linear fields and adjoint identities (`⟨div u, p⟩ = −⟨u, grad p⟩`) | identities to 1e-14 |
| 1.2 | `poisson.py`: fast diagonalisation for Neumann/Dirichlet/periodic axes on non-uniform grids, DCT on uniform Neumann axes, PCG fallback; compare with dense on 8³ | residual 1e-12 fp64; ~4 kernels per solve |
| 1.3 | `em.py`: face currents, potential Poisson assembled from the same flux, face Lorentz force, thin-wall surface Laplacian, explicit shell option; test Σ_f J_n s_f = 0 per cell and the Hartmann/Shercliff/Hunt currents | charge residual 1e-13; Hunt wall current within 1 % of series |
| 1.4 | `core3d.py`: predictor with CN diffusion, conservative limited advection, implicit Lorentz damping correction; projection; B = 0 Poiseuille and manufactured solutions with convection | observed 2nd order in space; Poiseuille to 1e-8 |
| 1.5 | `timeloop.py`: `scan` + checkpoint; chunked acceptance; restart; a grep test forbidding `float(`/`bool(` in `core3d`/`timeloop` | CPU step count and time linear in steps; no host sync |
| 1.6 | `steady.py`: residual, Newton–Krylov (SOLVAX `root_solve` with explicit matrix-free tangent/transpose), projection-step preconditioner, IFT adjoint | Shercliff/Hunt Ha 20–1000 reproduce Table I flow rates within 0.5 %; adjoint vs FD 1e-6 |
| 1.7 | Pipe O-grid in the same code path (mapped metrics from the existing `_fringing_pipe` pieces, proper axis treatment: `r·u_r` flux form) | Gold/Chang–Lundgren pipe profile; translation/rotation invariance tests |
| 1.8 | Verification ladder rows 1–5 of Section 7; ANL x0 = 3 analytic case | ΔP within 3 % of 0.0932 on the medium mesh; 3 meshes reported |
| 1.9 | Switch `solve()` to the new core; mark the old generic and B1/B2 paths frozen | suite green; old paths unreachable from the public API |

Effort ≈ 3.5 weeks. Reuse: mesh generators, materials, IO/VTK, validation
helpers, SOLVAX solvers, the B2 lane's face-current and limited-convection code.

### Phase 2 — GPU and scaling (weeks 2–6, ∥)

| Step | Action | Exit |
|---|---|---|
| 2.1 | Day-1 profile of the *old* code on one A4000 (`jax.profiler`): host-sync vs kernel time, peak memory | baseline JSON committed under `benchmarks/` |
| 2.2 | Benchmark harness: pytest-benchmark cases (fully developed 128², 3-D 64³/128³/256³, Q2D 512²/1024²/2048²) × {fp32, fp64} × {CPU, 1 GPU, 2 GPU}; `lmx benchmark` writes JSON with hardware/versions/dtype | reproducible JSON + one figure script |
| 2.3 | Nightly cron on the office box (Section 9) pushing results to a `benchmarks` branch | first nightly run |
| 2.4 | Sharding: slab decomposition with `shard_map` + `ppermute` halos for the 3-D core; evaluate jaxDecomp for Q2D and the Poisson transposes | 1-GPU vs 2-GPU fields agree to 1e-12 (fp64) |
| 2.5 | Optimise to targets: fusion audit, donate buffers, fp32 transforms | G4/G5 numbers or a documented reason |

Effort ≈ 2 weeks of work spread across the phase; needs the office host.

### Phase 3 — Q2D turbulence (weeks 2–4, ∥)

| Step | Action | Exit |
|---|---|---|
| 3.1 | ETDRK2/4 with Kassam–Trefethen coefficients; friction and viscosity in the exponential; 2/3 dealiasing; energy/enstrophy budgets and spectra as scan outputs | Taylor–Green decay exact to 1e-10; budgets close to 1e-8 |
| 3.2 | Cases A (frozen decay), B (vortex lattice), C (Kolmogorov + drag), D (Tithof Kolmogorov-like) from Section 7 | numbers in Section 7 reproduced within stated tolerance |
| 3.3 | Cross-check against GeophysicalFlows.jl or Dedalus at 256² with identical initial data | relative L2 ≤ 1e-3 |
| 3.4 | Movies: 1024² decay with merging, forced lattice sweep, Kolmogorov condensate; README animation ≤ 500 KB; full MP4s as release assets | three movies produced on GPU |
| 3.5 | Regime statement in docs (t_H formula, n, Re/Ha < 380, N ≫ 1, PbLi example numbers) | tutorial updated |

Effort ≈ 1.5 weeks.

### Phase 4 — B2 validation and lane retirement (weeks 6–9)

| Step | Action | Exit |
|---|---|---|
| 4.1 | Digitise Smolentsev 2015 Figs 3–4 and ANL/FPP/TM-228 Figs 11–13 (WebPlotDigitizer; two independent passes); write CSV + provenance sidecar; email the author for the original files | uncertainty column filled from repeat digitisation |
| 4.2 | Matched FreeMHD B2 case in `freemhd_install` (blockMesh with stretching, analytic solenoidal fringe via `codedFixedValue`, thin conducting wall as a solid region, `endTime` ≥ 20 transit times); Shercliff/Hunt demos with proper run length and L2 assertions | FreeMHD B2 curve stored as reference data |
| 4.3 | LMX B2 on three meshes (≈ 60/90/120 cells across the half-width, geometric 1.12), thin wall vs explicit shell, `B_y`-only vs solenoidal field; GCI | numerical gates: refinement, conservation, independence |
| 4.4 | Comparison: axial pressure gradient at the side wall, transverse pressure difference, side-wall potential; acceptance if within experimental + digitisation uncertainty and within 5 % of FreeMHD | status `accepted` or `external_validation_open` with numbers |
| 4.5 | B1 cost estimate (mesh, steps, memory at Ha 6600); attempt only if < 1 GPU-day | documented either way |
| 4.6 | Delete the specialised lane (D2); collapse `validation/freemhd.py` + parity script into `lmx/freemhd.py` ≤ 600 lines | −≈ 11,000 lines; tag `lmx-b2-lane-v1` |

Effort ≈ 2.5 weeks plus Docker/GPU time.

### Phase 5 — Thermal passive scalar (weeks 8–10)

| Step | Action | Exit |
|---|---|---|
| 5.1 | `thermal.py`: `ρc_p(∂_t T + u·∇T) = ∇·(k∇T) + q_vol + J²/σ`, constant properties, conjugate walls (T and flux continuity), inlet/outlet conditions; scan-integrated with the core | conduction slab and advection–diffusion MMS at 2nd order |
| 5.2 | Fully developed laminar limits: Nu = 48/11 (uniform flux) and 3.66 (uniform T) for the pipe; rectangular duct values from Shah & London | within 0.5 % |
| 5.3 | Heated Hunt duct with Joule heating; energy balance (inlet/outlet enthalpy, heat input, Joule, viscous) | balance to 1e-6 |
| 5.4 | Stretch: Mistrangelo et al. 2025 heated conducting duct (Ha 235) | comparison figure |

Effort ≈ 1.5 weeks.

### Phase 6 — Design and device examples (weeks 9–11)

| Step | Action | Exit |
|---|---|---|
| 6.1 | Fully developed inverse design (from #65 B): drive for requested throughput by the linear response `f = Q_target/G`, profile fit, minimum work at fixed flow over wall conductance and aspect ratio with IFT gradients | Taylor/FD checks; held-out finer mesh; before/after figure |
| 6.2 | Stellarator channel: sample an ESSOS coil field (from `ESSOS/examples`) along a straight channel outside the plasma; NPZ ≤ 200 KB with provenance; PbLi properties; Ha/N/Re reported | 3-D pressure drop and current-loop movie |
| 6.3 | Gradient of pressure drop and peak wall current w.r.t. field scale, wall conductance, geometry scale, flow; one bounded optimisation | adjoint vs FD; feasible optimum re-solved on a finer mesh |
| 6.4 | Tokamak-like (1/R field) and mirror (two-coil) variants of the same example | same script, three configs |

Effort ≈ 1.5 weeks.

### Phase 7 — Release (week 12)

| Step | Action | Exit |
|---|---|---|
| 7.1 | Version 2.0.0, CITATION.cff, Zenodo integration, PyPI upload, release assets (movies, benchmark JSON, figures) | clean-install smoke on 3.10 and 3.13 |
| 7.2 | Docs pass: Diátaxis structure, equations → implementation → example → validation links, gallery | Sphinx `-W` |
| 7.3 | Methods-paper outline with figure list mapped to release assets | draft outline in `docs/paper/` |

---

## 7. Validation ladder

| Row | Case | Parameters | Reference | Tolerance | Phase |
|---|---|---|---|---|---|
| 1 | Poiseuille, B = 0 | rect duct, pipe | series / exact | 1e-8 | 1.4 |
| 2 | Hartmann | Ha 10–10⁴ | analytic | 1e-6 on 3 meshes | 1.6 |
| 3 | Shercliff | Ha 20–10⁴ | Table I flow rates (Smolentsev 2015) | 0.5 % | 1.6 |
| 4 | Hunt | Ha 20–10⁴, c 0.01–0.1 | series; Table I | 0.5 %; wall current 1 % | 1.6 |
| 5 | Conducting pipe | Ha 100–6600, c 0.027 | Chang–Lundgren / Gold | 0.5 % | 1.7 |
| 6 | Ni fringing case | Ha 100, Re 200, eqs. 87–88 field | Ni 2007 Figs 16–17 | qualitative + ΔP 3 % | 1.8 |
| 7 | ANL analytic fringe | square duct, x0 = 3, c = 0.02 | ΔP_total 0.0932, fd 0.0754 | 3 % | 1.8 |
| 8 | Manufactured 3-D | non-uniform grid, convection on | MMS | 2nd order | 1.4 |
| 9 | FreeMHD Shercliff/Hunt | Ha 20–1000 | long-run FreeMHD | 3 % | 4.2 |
| 10 | ALEX B2 | Ha 2900, N 540, c 0.07 | digitised experiment + FreeMHD | exp. + digitisation uncertainty; 5 % vs FreeMHD | 4.4 |
| 11 | ALEX B1 | Ha 6600, N 10700, c 0.027 | digitised experiment | feasibility first | 4.5 |
| 12 | Q2D frozen decay | ν → 0, γ t_eddy = 0.1–1 | exact rescaling identity | 1e-8 | 3.2 |
| 13 | Q2D vortex lattice | k_f = 6, Re_α 1–30 | Sommeria 1986: C = 3–7, condensation | qualitative + C range | 3.2 |
| 14 | Q2D Kolmogorov + drag | sin(4y), γ 0.1, Re 1000/4000 | Kochkov 2021 fields | spectra | 3.2 |
| 15 | Q2D Tithof flow | α 0.064, β 1 | Re_c 7.60, q_c 0.465 | 2 % | 3.2 |
| 16 | Q2D cross-code | 256², identical IC | GeophysicalFlows.jl / Dedalus | 1e-3 | 3.3 |
| 17 | Thermal limits | pipe/duct laminar | Nu 48/11, 3.66; Shah & London | 0.5 % | 5.2 |
| 18 | Heated conducting duct | Ha 235 | Mistrangelo 2025 | figure | 5.4 |
| 19 | Derivatives | every continuous input | FD sweeps, Taylor, duality | 1e-6 fp64 | 1.6, 3.1, 6.3 |
| 20 | Performance | 3-D 64³–256³; Q2D 512²–2048² | JSON + figure | G4/G5 | 2.5 |

---

## 8. Showcase figures and movies

All produced by `scripts/make_showcase_figures.py` (this PR adds the first two)
at demo settings on CPU and at production settings on the office GPU.

| Asset | Content | Budget | Status |
|---|---|---|---|
| `docs/_static/q2d_turbulence_256.webp` | 256² decaying Q2D turbulence, 30 frames, 320 px | 493 KB | **in this PR** |
| `docs/_static/q2d_turbulence_poster.webp` | three snapshots + energy spectrum with k^-3 guide | 84 KB | **in this PR** |
| `docs/_static/hunt_side_layers.webp` | Hunt profiles Ha 20–1000, 2-D map at Ha 1000, jet position vs Ha^-1/2 | 44 KB | **in this PR** |
| `analytic_velocity_profiles.webp` | existing Hartmann/Shercliff/Hunt validation | 72 KB | regenerate with consistent style |
| 3-D fringing duct | current loops and axial pressure at Ha 500, 128×64×64 | ≤ 150 KB poster + MP4 asset | Phase 1.8 |
| Stellarator channel | coil field along the channel, ΔP, wall current | ≤ 150 KB + MP4 | Phase 6.2 |
| Q2D forced lattice and condensate | 1024², movie | MP4 asset + 300 KB animation | Phase 3.4 |
| GPU scaling | time/step vs size, CPU / 1 GPU / 2 GPU, fp32/fp64 | 60 KB | Phase 2.5 |
| B2 comparison | ΔP curves: LMX 3 meshes, FreeMHD, experiment with error bars | 80 KB | Phase 4.4 |

Git media budget: ≤ 1 MB total; movies and raw data as release assets with
checksums.

---

## 9. How we work

- **Branches and PRs.** `main` always releasable; short-lived branches; squash
  merge; ≤ 400 changed lines per PR, one concern, tests in the same PR;
  description = what, why, how verified (three lines). No per-PR evidence template.
- **Tests.** Markers `unit`, `regression`, `slow`, `gpu`, `external`. PR CI runs
  `unit` on one Python (≤ 5 min) and `regression` only for PRs to `main`
  (≤ 8 min); no coverage threshold on PRs; coverage reported at release. Reference
  data via pytest-regressions with provenance sidecars.
- **Nightly (office box, cron, never a PR runner).** `git pull main`; `slow`,
  `gpu`, `external` (FreeMHD Docker); benchmark JSON pushed to a `benchmarks`
  branch; failures open one issue.
- **Benchmarks.** pytest-benchmark on PRs carrying a `run_benchmarks` label
  (DESC pattern); asv history from the nightly.
- **Planning.** `ROADMAP.md` one page (next three milestones); ADRs in
  `docs/adr/` for decisions that change equations, tolerances, formats or public
  API; work logs live in PRs and issues. This `plan.md` is frozen after
  reconciliation and updated only at phase boundaries.
- **AI-assisted work.** Same rules; the reviewer reads the diff; CONTRIBUTING
  carries the disclosure line JOSS requires.
- **Authorship.** All commits authored by the project owner; no tool attribution.

---

## 10. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Office GPU unavailable for weeks | Phases 1, 3, 5, 6 are CPU-developable; Phase 2 targets slip, not the release |
| Fast diagonalisation inapplicable with explicit conducting shells | PCG fallback (D5); thin wall is the default anyway |
| Newton–Krylov stalls at B2 | pseudo-transient continuation from the projection loop; report as `external_validation_open` with the numbers rather than retuning |
| B1 unaffordable | documented estimate; not a release blocker (G7) |
| Digitised data disputed | request originals from the author; publish digitisation error; keep FreeMHD comparison independent |
| Session/rate limits on agent work | small PRs; each phase's steps are independently mergeable |
| Two competing plans persist | D13 table gives the reconciler the deltas; the reconciled plan supersedes all three |

---

## 11. References (read at the depth stated in Section 3)

Numerics: Ni et al., JCP 227:174 (2007) [PDF](https://bpb-us-w2.wpmucdn.com/research.seas.ucla.edu/dist/d/39/files/2019/08/JCP-v227-NiCurrentPart1.pdf); JCP 227:205 (2007); JCP 231:281 (2012) · Hua, Walker, Picologlou, Reed, ANL/FPP/TM-228 (1988) [OSTI](https://www.osti.gov/servlets/purl/6789158) · Smolentsev et al., FED 100:65 (2015) [PDF](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf) · Munipalli et al., HyPerComp/UCLA report (2008) [OSTI](https://www.osti.gov/servlets/purl/929194) · Eardley-Brunt, Dubas, Davis, PPCF 66:015015 (2024) [PDF](https://scientific-publications.ukaea.uk/wp-content/uploads/Eardley-Brunt_2024_Plasma_Phys._Control._Fusion_66_015015.pdf) · Blishchik, van der Lans, Kenjereš, IJHFF 90:108800 (2021) · Tassone, epotFoam report (2016) · Siriano et al. (2024) · Urgorri et al., PPCF 66:105007 (2024) · Planas, Badia, Codina, JCP 230:2977 (2011) · Votyakov, Kassinos, Albets-Chico, [arXiv:0901.0624](https://arxiv.org/abs/0901.0624) · Pothérat, Sommeria, Moreau, [arXiv:2006.06973](https://arxiv.org/abs/2006.06973) · Jiang & Smolentsev, FED 201:114262 (2024) · Wynne et al., Phys. Plasmas 32:013907 (2025), [arXiv:2409.08950](https://arxiv.org/abs/2409.08950) · Mistrangelo et al., Nucl. Fusion 65:116006 (2025).

JAX/GPU: jax-cfd source ([fast_diagonalization.py](https://raw.githubusercontent.com/google/jax-cfd/main/jax_cfd/base/fast_diagonalization.py), [pressure.py](https://raw.githubusercontent.com/google/jax-cfd/main/jax_cfd/base/pressure.py), [funcutils.py](https://raw.githubusercontent.com/google/jax-cfd/main/jax_cfd/base/funcutils.py)) · JAX-Fluids 2.0 [arXiv:2402.05193](https://arxiv.org/abs/2402.05193) and 1.0 [arXiv:2203.13760](https://arxiv.org/abs/2203.13760) · jaxDecomp [JOSS 10.21105/joss.08852](https://joss.theoj.org/papers/10.21105/joss.08852.pdf) · JAX docs: [async dispatch](https://docs.jax.dev/en/latest/async_dispatch.html), [autodiff_remat](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html), [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html), [custom_linear_solve](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html) · optimistix/lineax/diffrax adjoint docs · Exponax [`_navier_stokes.py`](https://raw.githubusercontent.com/Ceyron/exponax/main/exponax/stepper/_navier_stokes.py), APEBench [arXiv:2411.00180](https://arxiv.org/abs/2411.00180) · Kassam & Trefethen, SIAM J. Sci. Comput. 26:1214 (2005) · NVIDIA RTX A4000 datasheet and GA102 whitepaper · SOLVAX README.

Q2D: Sommeria & Moreau, JFM 118:507 (1982) · Pothérat, Sommeria, Moreau, JFM 424:75 (2000) [arXiv:2006.15468](https://arxiv.org/abs/2006.15468) · Pothérat & Klein, JFM 761:168 (2014) [arXiv:1305.7105](https://arxiv.org/abs/1305.7105) · Baker et al., PRL 120:224502 (2018) · Sommeria, JFM 170:139 (1986) · Boffetta & Ecke, Annu. Rev. Fluid Mech. 44:427 (2012) · Cassells et al., JFM 861:382 (2019) · Camobreco, Pothérat, Sheard, PRF 6:013901 (2021); JFM Rapids (2023); PRF 10:023905 (2025) · Tithof et al., JFM 828:837 (2017) · Kochkov et al., PNAS 118 (2021) · Smolentsev & Moreau, CTR Proceedings (2006) · Zikanov et al., Appl. Mech. Rev. 66:030802 (2014) · Brynjell-Rahkola, Duguet, Boeck, PRF 10:023903 (2025).

Data and comparators: FreeMHD [repo](https://github.com/PlasmaControl/FreeMHD), [Zenodo 13964055](https://zenodo.org/records/13964055) · `rogeriojorge/freemhd_install` · GridapMHD · Eardley-Brunt input files DOI 10.14468/43ys-0z67 · GeophysicalFlows.jl · Dedalus · Acta Mech. Sin. (2025) DOI 10.1007/s10409-025-25324-x.

Presentation and process: pyOpenSci README guide · JOSS review criteria · Google small CLs · DESC workflows (`regression_tests.yml`, `benchmark.yml`) · GitHub security hardening (self-hosted runners) · pytest-regressions, pytest-benchmark, asv · Nygard ADRs · FAIR4RS.

---

## 12. Log

| Date | Work | Next |
|---|---|---|
| 2026-09-06 | Full review (code, tests, PRs #1–#65, CI, FreeMHD Docker); five literature/software surveys; fresh runs: 508/508 tests, seven examples, Hunt Ha sweep to 1000, Q2D 256² turbulence, 3-D step-control bug, FreeMHD B2 smoke; this plan and README with two new showcase assets | Reconcile #64/#65/this PR; then Phase 0 |

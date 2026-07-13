# Performance and Scaling

LMX keeps the fully developed duct solver on a JAX-native operator path and
benchmarks the production ALEX B2 solve directly. The goal is practical
research throughput:

- keep the routine validation lane under five minutes
- preserve a differentiable fixed-iteration lane
- provide explicit CPU and multi-GPU scaling evidence for larger studies

## Performance closeout gates

LMX should separate user-facing runtime feedback from benchmark claims.

Required user-facing behavior:

- progress output with current step, simulated time, residuals, flow rate,
  current diagnostics, elapsed time, and estimated remaining time
- explicit grid sizes, write stride, and output paths in summaries
- memory-relevant problem dimensions in benchmark JSON files
- compile time and warm runtime reported separately for JAX paths

Required benchmark behavior:

- CPU scaling tied to the executable `extruded_inductionless` projection
  arithmetic or a documented higher-intensity surrogate
- GPU scaling run on problem sizes large enough that sharding overhead does not
  dominate
- profiler traces before promoting a new strong-scaling figure
- memory-allocation and host-to-device placement checks for large arrays
- persistent JAX compilation cache on long artifact workflows where supported

Archived surrogate figures remain reproducibility evidence, not a current
performance claim. A paper-facing panel must use the production solve and a
profile that identifies memory bandwidth, communication, compilation, and
kernel-launch costs.

## What is benchmarked

The scaling workflow has two explicit benchmark kinds:

- `extruded_solve`
  - runs the actual rectangular `solve_extruded_inductionless(...)` path
  - applies named axial sharding to every production ALEX B2 3-D field
  - fails if a requested multi-device solve does not return that shard count
  - records grid size, estimated array memory, warm cell-updates per second,
    and optional JAX trace directory
  - is the only solver-faithful timing gate for release candidates
- `extruded3d`
  - runs a sharded fixed-iteration MHD operator surrogate
  - keeps explicit multi-device sharding over the fixed global grid
  - remains useful for isolating CPU/GPU stencil and communication behavior

This distinction matters for publication claims. `extruded_solve` is the
research-code runtime evidence because it uses the same projection loop as the
examples. `extruded3d` is the scaling-algorithm evidence because it exposes the
current multi-device sharding behavior on a denser fixed operator.

The older public `stencil2d` microbenchmark has been removed. It duplicated
sharding/reporting machinery without exercising the 3-D inductionless solver;
the retained surrogate isolates operator costs, while `extruded_solve` is the
only release-facing performance gate.

### Production-sharding checkpoint (12 July 2026)

The first end-to-end ALEX B2 checkpoint uses a deliberately small
`24 x 24 x 24` total grid and two outer steps. One RTX A4000 took `29.10 s`; two
A4000s took `40.72 s`, so this overhead-dominated probe is a measured scaling
miss, not a speedup claim. It is nevertheless a correctness gate: relative
one/two-GPU differences in the velocity, potential, and current L2 signatures
were `2.4e-9`, `7.3e-7`, and `1.4e-6`, respectively, and every returned 3-D
field retained two axial shards. The production wrappers now reuse a
process-stable device mesh and cached JIT kernels, so repeated two-GPU solves
remain physics-identical. Warm time is `8.79 s` on one GPU and `21.43 s` on two
for the small case; a `48 x 36 x 36` case measures `10.48 s` versus `38.28 s`.
An early pressure-only cross-section line-block experiment did not improve that
result and was rejected. The coarse-scale `102 x 77 x 77` footprint runs two outer steps in
`34.08 s` warm. The earlier `15.7 GiB` observation mostly measured JAX's
default memory preallocation; production campaigns now disable preallocation
unless the user overrides it. Global PCG reductions and halo traffic remain
the multi-device bottlenecks. Independent variants therefore occupy the two
GPUs concurrently, one process per GPU, in restart-aware waves. Those workers
share a source-fingerprinted persistent compilation cache under the system
temporary directory, while an explicit `JAX_COMPILATION_CACHE_DIR` override is
preserved.
Long solves also retain only the configured Anderson history window instead of
keeping every prior three-dimensional state alive.
The campaign stops before tolerance/iteration variants when baseline or wall
prerequisites miss a frozen physics gate, so GPU time is not spent refining an
invalid state.
Worker entry points prepend and verify their own repository root before
importing LMX, preventing an installed package or external `PYTHONPATH` from
being mislabeled with the copied source fingerprint.

SOLVAX 0.7.0 adds an opt-in algebraically equivalent single-reduction PCG
recurrence, and sharded B2 duct solves now use it for momentum, projection, and
electric solves. On two A4000s its compiled per-step scalar products form one
tuple all-reduce. A stricter conservative-residual audit supersedes the earlier
apparent L2 parity: on the matched `48 x 36 x 36`, two-step probe, one GPU has a
maximum charge residual of `6.66e-5`, while two GPUs produce
`1.91e-3`--`2.53e-3` and shift velocity/current L2 by about 1.8%/0.24%.
Standard PCG reproduces the failure, excluding the single-reduction recurrence
as the cause. These measurements diagnose the sharded domain path; they are not
strong-scaling evidence.

A symmetric point constraint was tested as a way to remove the rank-one gauge
all-reduce. It reconstructs the manufactured solution on one device, but the
real two-GPU B2 probe becomes unstable and reaches a charge residual near
`1e9`. That branch is rejected. A replacement nullspace treatment must be
designed together with the explicit distributed operator and pass the same
conservative production gate.

The subsequent isolation audit found that the axial neighbor exchange and
variable-coefficient finite-volume operator are bitwise identical on one and
two GPUs. Manufactured cold and warm pressure solves agree to `7.4e-15`, and
the exact B2 conductivity-jump electric solve agrees to `1.3e-11` cold and
`4.1e-12` warm relative. The earlier boundary-current failure was diagnostic,
not physical: it mixed global inlet/outlet fluxes into every axial station and
sliced the remote endpoint shard. LMX now evaluates each slab through the exact
finite-volume divergence theorem. On a near-converged `102 x 77 x 77` restart,
one/two-GPU potential and current signatures agree to about `4.2e-9` and
`6.2e-7` relative; charge residuals are `2.56e-5` and `2.79e-5`, and boundary
residuals are below `2e-14`. Velocity L2 differs by `3.2e-6` relative on the
two-step probe. A six-step two-GPU continuation passes the complete steady
conjunction with update residual `3.28e-5`, charge `2.50e-5`, boundary flux
`1.76e-14`, and gauge-invariant potential updates near `1e-6`. Matched
records store an explicit signature limit of `2.5e-5`, half the frozen
nonlinear tolerance, and are grouped by the actual executed update count. This
establishes the production scaling workload; a common converged initializer
and uncontended paired timing are still required before any speedup claim.

The source-bound retry on commit `3d5de4e` (`c6485085...`) closes correctness
from the same restart (`75097639...`). Aitken acceleration is now suspended
only after the primary fields meet the nonlinear tolerance, preventing a
global reduction from amplifying decomposition-order roundoff in the converged
tail. One and two GPUs both stop after four updates; their largest recorded
velocity, potential, or current L2 difference is `2.2e-15` relative. Charge is
below `2.7e-5`, boundary flux below `2.0e-14`, and every electric solve passes.
Reusing the already computed conservative current diagnostics also makes two
repeated sharded solves safe in one process and removes a duplicate final
evaluation.

Correctness does not imply strong scaling. The initial paired RTX A4000 row was
`37.78 s` warm on one GPU and `109.23 s` on two. Commit `9e0d1dc` first batches
14 outer-step scalars into one host transfer, reducing two-GPU time to
`50.18 s`. Commit `036d26b` then keeps the line preconditioner transverse to
the sharded axis: the axial block crossed devices and diluted the dominant
wall-normal y/z blocks. The final source-matched three-repeat row is `36.96 s`
on one GPU and `22.23 s` on two, for speedup `1.662` and two-device efficiency
`0.831`. Both warm two-GPU repeats agree within 0.12%. Cold times are `66.71 s`
and `65.75 s`; main L2 signatures agree within `1.8e-16`, and every steady,
mass, current, boundary, and electric gate passes. This meets the two-device
efficiency target but not the M5 exit requirement, because a four-device host
has not yet been measured. The compact source-bound record is
`benchmarks/results/gpu-strong-scaling-20260713.json`.

Solver-faithful workers accept `--restart` and record its SHA-256. Restart and
cold-start rows are separate scaling groups. Production scaling should use a
verified, evenly shardable steady restart; cold-start transients remain useful
for debugging but are not release performance evidence.

On the Mac CPU backend, two forced virtual devices are also slower than JAX's
normal single CPU device (`5.03 s` versus `3.19 s` warm on the small probe).
Normal CPU execution already uses threaded kernels: a production-path check on
the M-series development Mac consumed `30.7` CPU-seconds in `12.5` wall-seconds
even with `OMP_NUM_THREADS=1`, or about 2.5 cores on average. Setting
`OMP_NUM_THREADS` to 1, 4, and 8 left the warm time statistically unchanged at
`3.44`, `3.45`, and `3.58 s`, respectively. Virtual-device sharding is therefore
not the recommended local acceleration path. Use normal one-device JAX for one
solve and process-level concurrency for independent cases.

An HLO audit of the named-sharded axial stencil shows that XLA already emits
two one-plane `collective-permute` exchanges and no all-gather. A manual
`jax.shard_map` rewrite would reproduce that communication pattern, so it is
deferred unless a GPU trace later exposes a different partition. A geometric
multigrid preconditioner is the primary time-to-solution experiment:
the exact coarse B2 checkpoint currently needs about 722--723 electric PCG
iterations per outer update, so reducing Krylov iterations also removes the
same number of global synchronization points. The next scaling checkpoint must
profile the multigrid-preconditioned production solve and its remaining scalar
collectives before changing halo code.

Skipping the local-residual refinement after the first electric PCG is also
rejected. It reduced the recorded solve from about 720 to 600 iterations but
did not improve baseline wall time and worsened maximum charge residual from
`2.63e-5` to `7.76e-4`. Although still below the frozen `1e-3` limit, that loss
of verification margin is not acceptable.

A later exact B2 ablation applied the same cross-section line-block choice
consistently to momentum, projection, and electric PCG. This reduced baseline
electric work from 720 to 571--572 iterations and wall time from `63.9` to
`52.1 s` (18.5%) while keeping charge residual near `2.7e-5`. Major-field L2
differences were around `1e-7`; transverse components changed by less than
`5.8e-7` absolute, and the primary observable changed by `8.38e-9` absolute.
The single-device B2 path therefore omits the axial line block; B1 and generic
duct solves retain their existing preconditioners. A two-device HLO audit of
the same cross-only pressure solve
reduced `collective-permute` operations from 17 to 7, all-reduces from 25 to
15, and all-gathers from 12 to zero, but the resulting two-GPU velocity/current
signatures missed parity by about 1.8%/0.24%. Restoring axial line blocks for
sharded projection and electric solves leaves the same conservative-residual
and signature failure. The reduced collective graph and the restored-line run
are both rejected diagnostic evidence, not active scaling paths. The next
implementation step is an explicitly verified halo/domain-decomposition path,
with multigrid and stable compiled closures reducing its synchronization cost.
On the matched small Mac production solve, warm time improves from `3.44` to
`3.15 s` (8.5%); velocity and current L2 signatures change by only about
`1.1e-8` and `3.5e-7` relative, respectively.
An idle-device matched `48 x 36 x 36` A4000 row measures `9.71 s` warm, down
from the earlier `10.23 s` one-GPU row. Rejected two-GPU runs take
`42.99`--`45.15 s`; neither their timings nor their signatures may appear in an
accepted scaling table.

Clean follow-up runs close the line-smoother search. The symmetric
multiplicative y--z--y SOLVAX smoother reduced electric PCG to 380--381
iterations with a `2.48e-5` charge residual but took `64.5 s`, 24% slower than
the accepted additive y/z block. A one-direction block exceeded 95 seconds
without completing. The
multigrid design target follows the structured finite-volume evidence of
[Singh et al.](https://doi.org/10.1002/fld.4277), where multigrid was the best
of the tested parallel pressure-Poisson preconditioners on nonuniform grids:
cell-volume restriction, geometry-aware prolongation, and semicoarsening where
the wall-normal anisotropy requires it.

An external warm-run JAX trace (123 MiB, intentionally not tracked) shows that
rebuilding an identical solver call still triggers 102 compile events totaling
`7.10 s` and 932 cache misses totaling `1.30 s`; six PCG while calls total
`2.29 s`, while host/device copies are negligible. The next package must
therefore make compiled solver closures stable across calls and fuse larger
regions independently of the multigrid work.

A newer accepted-path two-GPU trace is also kept external (218 MiB). Its wall
time is not scaling evidence because an unrelated SPECTRAX process held both
GPUs at 100% utilization. It nevertheless provides useful structural evidence:
the projection occupies `103.6 s` of the `166.5 s` public solve span, 580
`pjit` cache misses total `6.43 s`, 62 backend compile/load events total
`3.80 s`, and the two pressure-response calls occupy `4.11 s`. Collective
launches are much smaller (`0.078 s` of all-reduce starts and `0.103 s` of
collective-permute starts), so the next accepted work targets host orchestration
and compiled-region reuse before lower-level collective tuning.

The first such cleanup batches the 102-station diagnostic table into one JAX
stack and one host transfer. On the same restart, the old path takes `1.283 s`
cold and `0.636`--`0.692 s` warm; the vectorized path takes `0.476 s` cold and
`0.003`--`0.006 s` warm with identical station count and charge residual.
This is commit `8a069fe`. Whole-worker timings collected during the SPECTRAX
contention are quarantined and do not replace the accepted scaling row.

The pressure-response span is not a useful optimization target after isolated
measurement. Solving one axial copy instead of all 102 reduces that kernel from
`0.0236 s` to `0.0011 s` warm, but introduces a separate roughly one-second
compilation. The complete one-GPU restart regresses from the accepted `37.78 s`
to `39.72 s` warm and increases the maximum charge residual by 6.2%, although
the velocity, potential, and current L2 signatures remain exact. The prototype
is rejected and removed; the trace's `4.11 s` nested span primarily reflects
tracing/compilation rather than repeatable response work.

Vectorizing the three independent momentum PCG solves is also rejected. It
removes 36 source lines and could batch scalar reductions, but JAX's batched
while-loop must remain active until the slowest component converges. On an
otherwise idle A4000, the two-repeat one-GPU worker was still running after
135 seconds versus about 100 seconds for the accepted cold-plus-warm row, so it
was terminated before the longer two-GPU lane. Independent component solves
remain the accepted path; future fusion must not couple their stopping times.

The accepted alternative stacks only outer-step diagnostics. It reduces about
14 scalar transfers to one per update while leaving the three momentum solves
independent. The complete portable gate passes 899 tests with 8 expected
optional-data skips and 95.06% branch coverage in `504.16 s`, below the
ten-minute hard limit but above the `450 s` warning target on this Mac run.

A fresh accepted-source two-GPU trace after this change is kept external
(207 MiB). Profiling inflates the worker to `91.41 s`, so that number is not a
scaling row. The trace places `71.24 s` of the `73.49 s` public solve span in
the projection path and still records 580 `pjit` cache misses (`6.20 s`) and
62 compile/load events (`4.11 s`). It contains 554 all-reduce-labelled spans
totalling `0.091 s`, 92 collective-permute spans totalling `0.110 s`, and 324
all-gather-labelled spans totalling `0.002 s`; these are launch/trace spans,
not complete asynchronous device-loop costs. SOLVAX's sharded PCG already uses
the accepted single-reduction recurrence, so the next change requires a
better preconditioner or a device-resolved timeline, not another ungrounded
inner-product rearrangement.

A stable top-level JIT for all B2 outer-step reductions was tested next. It
preserves every Mac CPU and one/two-GPU physics gate, but adds 115 net source
and test lines without improving runtime: one GPU is `36.82 s` warm versus the
accepted `36.68 s`, and two GPUs are `50.26 s` versus `50.18 s`. The full-size
Mac cold restart passes in `48.03 s`. This prototype is rejected and removed;
the eager `pjit` events visible under tracing are not a profitable standalone
fusion target after the accepted one-transfer stack.

The accepted solver-side change removes axial line relaxation from sharded
projection/electric preconditioners while retaining transverse y/z line solves
and SOLVAX single-reduction PCG. This revisits an earlier ablation only after
the decomposition-safe current diagnostics closed its false failure. It adds
three net package lines, preserves exact one/two-GPU physics, and changes the
two-GPU warm row from `50.18 s` to `22.23 s`. The complete portable gate passes
899 tests with 8 expected skips and 95.06% branch coverage in `512.04 s`.

B1 needs a different decomposition. Its frozen coarse mesh has 101 axial
stations, and its cylindrical preconditioner solves complete axial and radial
lines. An even reduced CPU probe verifies that axial sharding can preserve the
fields, but the corresponding two-A4000 worker exceeded two minutes with only
GPU 0 active; the one-GPU row was `26.94 s` cold and `7.37 s` warm. Sharding
the periodic azimuthal axis likewise passed a forced two-CPU-device probe
(`6.48 s` to `6.04 s`, velocity agreement `9.2e-15` relative), but exceeded
70 seconds on two A4000s while GPU 1 remained idle, versus `26.62 s` cold and
`6.85 s` warm on one. Both prototypes are rejected and removed. Adding theta
to the existing non-cyclic line preconditioner is also rejected: the
manufactured local residual regresses to `4.58e-8` against its `1e-8` gate.
Until a partition-preserving cyclic SPD line solve has a reusable, efficient
accelerator application, run independent B1 variants one per GPU with the
campaign runner rather than claiming spatial scaling for the mapped-pipe path.

SOLVAX branch checkpoint `47831dd` supplies the missing exact cyclic
tridiagonal algebra and passes 241 tests with 98.00% branch coverage. LMX's
manufactured pipe gates pass when it is used as a theta line, but the reduced
one-A4000 worker exceeds 100 seconds with the GPU saturated, versus `26.62 s`
cold and `6.85 s` warm without that line. The integration is rejected and
removed: refactoring the cyclic system inside every PCG application costs more
than the additional block saves. Revisit it only with persistent reusable
factors or a demonstrably cheaper periodic accelerator path.
The existing SOLVAX periodic-banded factors were also tested as a reusable
theta inverse. Manufactured gates pass, but the reduced worker is CPU-bound
with the A4000 idle after 66 seconds. That integration is likewise rejected;
the required reuse must retain a fused accelerator apply rather than falling
back to scanned general-banded LU.

The accepted B1 optimization keeps cylindrical coefficient and wall-system
preparation eager, then reuses one compiled SOLVAX PCG system for the three
momentum components. This boundary matters: compiling the entire diffusion
operator was faster but amplified the short-run divergence diagnostic and was
rejected. In a paired ten-update `32 x 16 x 32` full-physics run on the two
A4000s, the accepted split reduces runtime from `62.68 s` to `50.56 s`
(`1.240x`). Velocity, potential, pressure, and pressure-gradient L2 signatures
agree within `6.9e-11` relative; divergence improves from `3.37e-3` to
`3.16e-3`, and charge residual improves from `2.05e-4` to `2.02e-4`. The
compact record is
`benchmarks/results/b1-momentum-jit-20260713.json`. This accelerates each B1
worker; independent variants remain the supported way to occupy both GPUs. The
source-matched portable gate passes 899 tests with 8 expected skips and 95.07%
branch coverage in 141.6 seconds, comfortably inside the 600-second CI limit.

A first stride-four SOLVAX Galerkin prototype used linear prolongation and its
exact transpose. It reduced a manufactured solve from 39 to 27 PCG iterations,
but the approximate diagonal coarse solve is not production-safe: one sweep
stalls B2 at the 1200-iteration cap with charge residual near 52, and four
sweeps trigger PCG preconditioner breakdown. It is rejected. The replacement
must construct a stronger SPD coarse solve once for the fixed conductivity and
reuse it across electric right-hand sides. Capturing conductivity and mask in
the compiled closure is also rejected: physics remains exact and one-GPU warm
time is `37.37 s`, but two-GPU warm time regresses from `109.23 s` to
`118.66 s`.

Dense spectral Galerkin correction is likewise rejected. A 27-mode cosine
space raises production work from 573 to 586--588 iterations. Expanding to 64
modes lowers it to 514--515 with exact physics, but setup/application overhead
increases one-GPU warm time from `37.78 s` to `57.98 s`. The accepted design
therefore requires near-linear coarse transfers and persistent hierarchy reuse.

Unsmoothed piecewise-constant aggregation is not sufficient. Factor-four
aggregates worsen a manufactured solve from 39 to 41 iterations. Factor-two
aggregates improve that probe to 35 but force both production B2 electric
stages to consume the full 1200 combined iterations. This branch is removed;
future work requires smoothed aggregation or rediscretization with a true
coarse line solve.

A rediscretized factor-two coarse line solve was then formed by summing fine
face conductances across aggregate boundaries and using SOLVAX tridiagonal
blocks. The action is SPD, but adding it to the fine line inverse worsens the
manufactured solve from 39 to 54 iterations and again consumes all 1200 B2
iterations, with charge residual `6.7e-5`. Additive overlap is rejected; a
coarse line solve must be embedded in a balanced symmetric V-cycle.

The balanced form `L + (I-LA) C (I-AL)` is stable, but piecewise-constant
aggregates still worsen the manufactured solve from 39 to 41 iterations and B2
from 573 to 852. This closes unsmoothed aggregation. M5 now prioritizes
reduction-count and fused-kernel profiling; only smoothed interpolation remains
a viable future multigrid direction.

At the outer-coupling level, checkpoint-matched B2 probes selected SOLVAX
vector Aitken relaxation capped at two. It approximately doubles the local
residual decay rate while remaining monotone over the probe; caps three and
four oscillate and are rejected. This reduces continuation time, but it does
not address the dominant per-update PCG and collective costs targeted by
multigrid and larger fused solver regions.

## Run the benchmark

Local CPU only:

```bash
python examples/strong_scaling_demo.py --output artifacts/examples/strong_scaling_cpu
```

Solver-faithful rectangular `extruded_inductionless` timing with a JAX trace:

```bash
python examples/strong_scaling_demo.py \
  --benchmark-kind extruded_solve \
  --cpu-counts 1,2,4 \
  --repeats 2 \
  --profile \
  --output artifacts/examples/extruded_solve_scaling
```

For release scaling, initialize each worker from the same verified,
evenly-shardable restart and preserve its checksum in the JSON record:

```bash
python scripts/run_strong_scaling_worker.py \
  --benchmark-kind extruded_solve \
  --nx 102 --ny 65 --nz 65 --iterations 6 --repeats 2 \
  --num-devices 1 --platform GPU \
  --restart /path/to/b2_102_steady.npz \
  --output artifacts/scaling/gpu_1.json
```

Run the matching two-device worker with only `--num-devices` and the output
path changed. The worker rejects a restart whose case or mesh metadata do not
match the requested problem.

Local CPU plus a remote GPU host reachable over SSH:

```bash
python examples/strong_scaling_demo.py \
  --remote-host office \
  --output artifacts/examples/strong_scaling_full
```

The example writes:

- raw JSON timing records
- optional JAX trace directories under `profiles/`
- `strong_scaling_summary.json`
- `strong_scaling_diagnostics.json`
- `strong_scaling_table.csv`
- `strong_scaling.png`
- `strong_scaling.pdf`

`strong_scaling_table.csv` is the compact artifact intended for CI/release
review. It reports fixed-problem speedup, parallel efficiency, warm
Mcell-updates/s, estimated array memory, profiler-trace availability, and
whether each row came from the actual `solve_extruded_inductionless` path. The
JSON diagnostics carry the same rows plus summary counters, so release gates can
reject surrogate-only timing evidence when a solver-faithful scaling claim is
being made. Solver-faithful workers also enforce the frozen `1e-3` charge,
boundary-current, and electric local-residual limits and require every electric
solve to converge. A failed physics gate aborts the worker instead of emitting
a publishable timing row.

## Backend selection for CLI runs

Standard `lmx input.toml` runs inherit the active JAX backend from the shell
environment. Typical launch patterns are:

CPU:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 lmx examples/hartmann_case.toml
```

Single GPU:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx cases/ducts/hunt_case.toml
```

Those commands select the execution device for the normal CLI solver run. Use
`examples/strong_scaling_demo.py --benchmark-kind extruded_solve` for a
production-path device-count sweep. For remote GPU runs on the `office` host,
the same pattern works over SSH:

```bash
ssh office 'cd /home/rjorge/tmp/lmx_scaling_repo && PYTHONPATH=/home/rjorge/tmp/lmx_scaling_repo CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python3 -m lmx cases/ducts/hunt_case.toml'
```

## Current conclusion

The production `extruded_solve` checkpoints above supersede the archived
surrogate curve as the current scaling result. They prove device placement and
physics equivalence but miss the strong-scaling target on two A4000s. Forced
virtual-device sharding also slows the Mac CPU path, whose normal JAX kernels
already use host threads. Until a production solve meets the frozen efficiency
gate, the supported throughput strategy is one solve per GPU and normal
single-device JAX execution on the Mac. Treat shard placement alone as an
implementation check, never as a physics or performance result.

The `auto` path resolves to released SOLVAX 0.5.1 PCG. Reproduce its
native/SOLVAX forward, implicit-gradient, independent-transpose, compile,
warm-time, and compiler-memory comparison on CPU with:

```bash
JAX_ENABLE_X64=true uv run --locked --extra dev \
  python scripts/benchmark_solvax_pcg_backend.py --expected-backend cpu
```

The tracked CPU and RTX A4000 x64 records are
`benchmarks/results/solvax-pcg-equivalence-cpu.json` and
`benchmarks/results/solvax-pcg-equivalence-gpu.json`. Both pass. On the recorded
GPU, SOLVAX has field relative difference `1.54e-12`, implicit-gradient error
`1.13e-16`, transpose residual `2.54e-13`, warm-time ratio `0.230`, and
temporary-memory ratio `1.000` relative to native CG. The four-level Ha=20
FreeMHD ladder and all eight high-Ha Table I rows pass at the same promoted
solver-core fingerprint in `benchmarks/results/solvax-pcg-acceptance.json`.

## Recent compatibility and platform validation

The current tree has also been validated on:

- local Python `3.13` with JAX `0.9.2`
- remote Python `3.10.12` on the `office` host with JAX `0.6.2`

That remote run confirms both the broad JAX-version compatibility work and the
Python `3.10` TOML fallback path on a real two-GPU machine.

Recent small validation run on `office`:

- CPU `256 x 256`, `32` iterations:
  - `1` device warm runtime: `1.46e-3 s`
  - `2` devices warm runtime: `2.00e-3 s`
- GPU `512 x 512`, `32` iterations:
  - `1` GPU warm runtime: `6.11e-4 s`
  - `2` GPUs warm runtime: `3.91e-3 s`

Those numbers are not the main scaling artifact; they are a smoke validation
that the remote-GPU orchestration still works on a
live two-GPU host.

Recent matched one-device comparison used to confirm the expected CPU/GPU
direction of travel for the current tree:

- local CPU `512 x 512`, `32` iterations:
  - `warm_seconds ≈ 4.31e-3`
- office GPU `512 x 512`, `32` iterations:
  - `warm_seconds ≈ 6.65e-4`

At that problem size the remote single-GPU warm runtime is faster than the
local one-device CPU warm runtime. Small two-GPU runs are still dominated by
overhead, so multi-GPU strong scaling should be judged on the larger committed
main artifact, not on these smoke runs.

## Design notes

- `lmx/scaling.py`
  - sharded stencil benchmark kernel
- `scripts/run_strong_scaling_worker.py`
  - single-environment worker used by the example/orchestrator
- `examples/strong_scaling_demo.py`
  - explicit user-facing orchestration for CPU and remote GPU runs
- `lmx/plotting.py`
  - scaling plot generation

## References

- [JAX distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [JAX sharded computation guide](https://docs.jax.dev/en/latest/sharded-computation.html)

---
orphan: true
---

# Media provenance

Feature pages embed the relevant figures directly. This compact index records
where the compressed derivatives live. The checksummed source bundle is in the
versioned [research-assets release](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1);
other raw intermediates remain outside Git. The package wheel contains no media.

## Visual evidence coverage

Every curated journey and every accepted result with meaningful spatial,
transient, convergence, comparison, sensitivity, or scaling structure owns a
visual in its feature page. CLI/configuration mechanics and scalar-only
contracts may share the physical result they drive; they do not receive
decorative charts. Each published visual must name its maturity, source record
or command, observable, and acceptance limit or uncertainty.

| Workflow or claim | Documentation visual | Coverage |
|---|---|---|
| Hartmann CLI/Python and Shercliff/Hunt cases | analytical profiles and Hunt startup | complete |
| manufactured operators | observed-order convergence | complete |
| FreeMHD closed-channel parity | accepted observable comparison | complete; add validation-ladder composite |
| portable fringing workflow | bounded solver-family response and B2 panel | complete; research-stage |
| exact restart/replay | direct-versus-resumed state comparison | complete |
| custom fields and mapped geometry | field response and geometry panels | complete |
| differentiable design | checked sensitivity and bounded design trace | complete |
| CPU/GPU parallelism | forced-device calibration plus physical-core pilot | CPU pilot and 1/2-GPU topology current; steady scaling promotion open |
| mapped-pipe external profiles | visible FreeMHD-profile mismatch | complete; not ALEX-B1 acceptance |
| Samper high-Ha literature ladder | frozen eight-row acceptance composite | complete |
| Q2D, obstacle, and blanket studies | loops, detailed diagnostic composites, and strict Votyakov mismatch | complete; quantitative parity remains open |

The next derivative is evidence-driven: exact ALEX-B1 pressure only after its
acceptance record exists. Prefer existing writers and compact accepted JSON;
do not rerun a solver merely to change presentation. One composite should
serve one claim family.

## Tracked web set

Twenty files in `docs/_static/` total 1,207,298 bytes. The `showcase` section of
[`release-assets.json`](release-assets.json) records every byte count and
SHA-256; `python scripts/manage_release_assets.py --check` verifies it locally.

| Derivative | Bytes | Display | Provenance |
|---|---:|---|---|
| duct profiles | 71,620 | 1,338 × 816 WebP | released analytical-profile PNG `f55c380c...` |
| exact restart | 24,514 | 1,200 × 629 WebP | portable 3+3 versus direct six-step state; all field differences zero |
| accepted FreeMHD observables | 21,086 | 1,200 × 641 WebP | acceptance record `9f94ea15...`; generated at `72f2049` |
| fringing solver family | 46,678 | 1,200 × 843 WebP | released source `5fd83110...`; bounded internal diagnosis only |
| mapped-pipe profiles | 52,768 | 1,100 × 655 WebP | released source `d3343a09...`; failing FreeMHD-profile diagnostic, not ALEX |
| Votyakov obstacle target | 32,966 | 1,100 × 458 WebP | released source `7bb3ca51...`; strict reverse-flow mismatch |
| Q2D external diagnostics | 87,350 | 1,000 × 1,509 WebP | released sources `5bc14089...` and `11f452c1...`; external parity explicitly open |
| manufactured operators | 41,120 | 1,600 × 624 WebP | bounded example; orders 2.00/1.94/1.91; `test_example_runner.py` / `test_plotting.py` |
| Samper Benchmark A | 74,438 | 1,405 × 913 WebP | accepted aggregate `9f94ea15...`; eight rows; JSON-only writer/test |
| B2 schema-6 CPU evidence | 55,186 | 1,400 × 1,329 WebP | forced-device calibration plus accepted 32-update, 2/4/8-CPU sustained fixed-work scaling from `b2-schema6-cpu-scaling-20260716.json`; not steady-state evidence |
| Hunt startup | 114,866 | 440 × 287 animated WebP, 7 s | Python/Pillow derivative of source GIF `12f30a38...`; 42 frames |
| checked sensitivities | 32,612 | compressed WebP | released derivative `f2add9fe...` |
| geometry gallery | 61,600 | compressed WebP | released derivative `e844f069...` |
| variable-field response | 41,648 | compressed WebP | released derivative `5d4b593c...` |
| magnetic obstacle | 48,630 | compressed WebP | released derivative `6e351e80...` |
| blanket-flow loop | 61,776 | 480 × 222 animated WebP, 7 s | Python/Pillow derivative of released GIF `1a23d23...`; 42 frames |
| Q2D loop | 117,102 | 440 × 381 animated WebP, 7 s | Python/Pillow derivative of released GIF `feb9e145...`; 42 frames |
| B2 field and pressure | 43,174 | compressed WebP | released derivative `f186f109...` |
| curved-pipe validation | 94,236 | compressed WebP | released derivative `d32d666a...` |
| blanket current and pressure | 89,356 | 1,476 × 1,573 WebP | released sources `48f4ba58...` and `4be86176...`; research-stage |

Regenerate the Samper derivative from tracked evidence only with
`python scripts/manage_release_assets.py --write-benchmark-a-plot docs/_static/samper_benchmark_a.webp`.
The plot path is covered by `test_acceptance_plot_uses_only_frozen_json`.

The three directly embedded motion derivatives are regenerated with
`python scripts/manage_release_assets.py --write-animated-webp SOURCE OUTPUT`.
They sample existing physical frames to 42 frames over 7 seconds and use lossy
animated WebP compression; no solver or motion interpolation is involved. The
full-quality MP4s and source GIFs remain in the release or ignored artifacts.

## Release-hosted README set

The README embeds every plot and motion derivative directly. Release URLs are
retained only for full-quality source inspection; the standalone documentation
does not depend on release authentication.

| Derivative | SHA-256 |
|---|---|
| `readme-autodiff.webp` | `f2add9fe96a044a082a3b0ca28a3de50b71a0961399ac362caf2ebcaa049aa28` |
| `readme-geometries.webp` | `e844f069475fa625279e229a3e8499ae816308b88c8d1d33afb4fb14077bf700` |
| `readme-variable-field.webp` | `5d4b593c26a55cb031678988133ba2f6ddeae5e0cbee16a0e166282584bce569` |
| `readme-magnetic-obstacle.webp` | `6e351e80531742b1aed6f951d6155cc55d3f5fd4f3e9b556a15336f76a74bc84` |
| `readme-blanket-flow-poster.webp` | `f40dff36a41bdfc1fa9df7371cf9c4907a367d70c5603a4658a18d27f708260d` |
| `readme-blanket-flow-7s.mp4` | `ca7d72cebb564a28bc6f91b395824cd1a9417ad271dc8ba3c0cbf2c365f18c51` |
| `readme-alex-b2-field-pressure.webp` | `f186f109ed9f0dc25134baaee22069e521140134ff984be94a60fe1bc825bf2f` |
| `readme-q2d-turbulence-poster.webp` | `97a9e5c418012f1591344ea131a846933a2c60fc2ba47eaf9f93f86e79e1bfb5` |
| `readme-q2d-turbulence.mp4` | `fd6e60bfdeb99f1c0123a8dc68f2198cfde91904b09cfdcde7caa34667492a62` |
| `readme-li-aln-multilayer-convergence.webp` | `e3c8bc3c143cca9c0e1a30c2892005b2be6e0d3c46efda4c00704d81f57e66e6` |
| `readme-curved-pipes.webp` | `d32d666a3bb34b2845c549ff303a3865b06b543e00e3364a0c705fddc831ebb6` |

The 71,043-byte Q2D H.264 loop and 13,724-byte poster derive from the existing
72-frame source GIF (SHA-256
`feb9e145cee2a5a87f13381803855a34ad16d44f4d63d0c31c155ad07f74bd96`),
stretched from 5.04 to 7.00 seconds at 600 × 520 and 12 fps. No solver was
rerun; quantitative Q2D-MHDfoam parity remains open.

The 86,808-byte blanket H.264 loop is exactly 7.00 seconds at 1,000 × 462 and
12 fps. Motion interpolation retimes the existing 5.75-second derivative
(`fb426241...`) without changing its physical frames; no solver was rerun. The
underlying centerline pressure-velocity model remains research-stage.

The 95,818-byte Li/AlN WebP derives from the 2,532 × 1,732 convergence figure
(SHA-256 `0a5d4d0a9a97cff78eb15efdb79934515fcabb5918c23148c8c0653ce198ea94`).
It reports research-stage mesh-step evidence; no experimental-validation claim
is attached to it.

The 94,236-byte curved-pipe WebP stacks resized copies of the released
`bent_pipe_overview.png` (`20d2ea12...`) and
`dean_literature_validation.png` (`06a214ab...`). Only deterministic layout,
resizing, metadata removal, and WebP compression were applied; no solve was
rerun. It shows a low-De inductionless baseline and the still-open Dean-vortex
literature gate.

New examples write beneath ignored `artifacts/`. Publish new large media only
with the generating commit, command, environment, input hashes, and output
hashes; add a docs-local derivative only while the tracked checkout remains
inside its architecture budget.

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
| portable fringing workflow | B2 field/pressure panel | representative; portable six-panel result queued |
| exact restart/replay | direct-versus-resumed writer | queued for the case cookbook |
| custom fields and mapped geometry | field response and geometry panels | complete |
| differentiable design | checked sensitivity and bounded design trace | complete |
| CPU/GPU parallelism | current schema-6 CPU calibration | CPU current; two-GPU replay open |
| ALEX B1 pipe reference | mapped-pipe comparison writer | queued; research-stage |
| Samper high-Ha literature ladder | frozen eight-row result records | priority composite queued |
| Q2D, obstacle, and blanket studies | compressed loops/panels | workflow coverage complete; quantitative parity open |

The next tracked derivatives are ordered by evidentiary value: the frozen
Samper/Benchmark-A ladder, restart equivalence, and the mapped-pipe comparison.
Prefer existing writers and compact accepted JSON; do not rerun a solver merely
to change presentation. One composite should serve one claim family.

## Tracked web set

Thirteen files in `docs/_static/` total 622,321 bytes. The `showcase` section of
[`release-assets.json`](release-assets.json) records every byte count and
SHA-256; `python scripts/manage_release_assets.py --check` verifies it locally.

| Derivative | Bytes | Display | Provenance |
|---|---:|---|---|
| duct profiles | 71,620 | 1,338 × 816 WebP | released analytical-profile PNG `f55c380c...` |
| accepted FreeMHD observables | 21,086 | 1,200 × 641 WebP | acceptance record `9f94ea15...`; generated at `72f2049` |
| manufactured operators | 41,120 | 1,600 × 624 WebP | bounded example; orders 2.00/1.94/1.91; `test_example_runner.py` / `test_plotting.py` |
| B2 schema-6 CPU calibration | 31,616 | 1,400 × 653 WebP | accepted record `b2-schema6-cpu-scaling-20260716.json`; `test_plotting.py`; GPU replay omitted while open |
| Hunt startup | 35,171 | 640 × 416 H.264, 7 s | existing 24.08-second transient; no solver rerun |
| Hunt poster | 13,000 | 640 × 416 WebP | released poster PNG `cac2125e...` |
| checked sensitivities | 32,612 | compressed WebP | released derivative `f2add9fe...` |
| geometry gallery | 61,600 | compressed WebP | released derivative `e844f069...` |
| variable-field response | 41,648 | compressed WebP | released derivative `5d4b593c...` |
| magnetic obstacle | 48,630 | compressed WebP | released derivative `6e351e80...` |
| blanket-flow loop | 86,808 | 1,000 × 462 H.264, 7 s | released derivative `ca7d72ce...` |
| B2 field and pressure | 43,174 | compressed WebP | released derivative `f186f109...` |
| curved-pipe validation | 94,236 | compressed WebP | released derivative `d32d666a...` |

The 35,171-byte Hunt H.264 loop (SHA-256
`ad1095279f4af84ebdb5c2b1d4677aed322c8b54a11107cdc7ceda40268563b7`)
compresses the complete 24.08-second tracked transient into 7.00 seconds at
640 × 416 and 12 fps. Its 13,000-byte WebP poster has SHA-256
`a4ada2e53c72cd2a57da99db072fc47a5477784aaf4938fc8efa83318b3456a4`.
The 10,635,673-byte source GIF (`12f30a38...`) remains outside Git and the
release; no solver was rerun.

## Release-hosted README set

The README retains release URLs so GitHub readers can inspect the versioned
asset collection. The eight derivatives used by the standalone documentation
are mirrored locally above so a public docs build does not depend on private
release authentication.

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

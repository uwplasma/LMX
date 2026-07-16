---
orphan: true
---

# Media provenance

Feature pages embed the relevant figures directly. This compact index records
where the compressed derivatives live. The checksummed source bundle is in the
versioned [research-assets release](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1);
other raw intermediates remain outside Git. The package wheel contains no media.

## Tracked web set

Five files in `docs/_static/` total 168,237 bytes. The `showcase` section of
[`release-assets.json`](release-assets.json) records every byte count and
SHA-256; `python scripts/manage_release_assets.py --check` verifies it locally.

| Derivative | Bytes | Display | Provenance |
|---|---:|---|---|
| duct profiles | 71,620 | 1,338 × 816 WebP | released analytical-profile PNG `f55c380c...` |
| accepted FreeMHD observables | 21,086 | 1,200 × 641 WebP | acceptance record `9f94ea15...`; generated at `72f2049` |
| B2 scaling evidence | 27,360 | 1,400 × 661 WebP | CPU/GPU records `cf9fafb8...` / `5f032be4...`; generated at `72f2049` |
| Hunt startup | 35,171 | 640 × 416 H.264, 7 s | existing 24.08-second transient; no solver rerun |
| Hunt poster | 13,000 | 640 × 416 WebP | released poster PNG `cac2125e...` |

The 35,171-byte Hunt H.264 loop (SHA-256
`ad1095279f4af84ebdb5c2b1d4677aed322c8b54a11107cdc7ceda40268563b7`)
compresses the complete 24.08-second tracked transient into 7.00 seconds at
640 × 416 and 12 fps. Its 13,000-byte WebP poster has SHA-256
`a4ada2e53c72cd2a57da99db072fc47a5477784aaf4938fc8efa83318b3456a4`.
The 10,635,673-byte source GIF (`12f30a38...`) remains outside Git and the
release; no solver was rerun.

## Release-hosted showcase set

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

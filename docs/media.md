---
orphan: true
---

# Media provenance

Feature pages embed the relevant figures directly. This compact index records
where the compressed derivatives live; raw fields, meshes, and full-resolution
media remain in the versioned [research-assets release](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1).
The package wheel contains no media.

## Tracked web set

Six files in `docs/_static/` total 390,625 bytes: duct profiles, FreeMHD parity,
the B1 comparison, current CPU/GPU scaling, and the Hunt movie/poster pair. The
44,582-byte scaling WebP has SHA-256
`60c83033c11239384cbf4a6fea7bb5ca5a4a7b98f1e1224b56961730637f4740`.

The 35,171-byte Hunt H.264 loop (SHA-256
`ad1095279f4af84ebdb5c2b1d4677aed322c8b54a11107cdc7ceda40268563b7`)
compresses the complete 24.08-second tracked transient into 7.00 seconds at
640 × 416 and 12 fps. Its 13,000-byte WebP poster has SHA-256
`a4ada2e53c72cd2a57da99db072fc47a5477784aaf4938fc8efa83318b3456a4`.
No solver was rerun.

## Release-hosted showcase set

| Derivative | SHA-256 |
|---|---|
| `readme-autodiff.webp` | `f2add9fe96a044a082a3b0ca28a3de50b71a0961399ac362caf2ebcaa049aa28` |
| `readme-geometries.webp` | `e844f069475fa625279e229a3e8499ae816308b88c8d1d33afb4fb14077bf700` |
| `readme-variable-field.webp` | `5d4b593c26a55cb031678988133ba2f6ddeae5e0cbee16a0e166282584bce569` |
| `readme-magnetic-obstacle.webp` | `6e351e80531742b1aed6f951d6155cc55d3f5fd4f3e9b556a15336f76a74bc84` |
| `readme-blanket-flow-poster.webp` | `f40dff36a41bdfc1fa9df7371cf9c4907a367d70c5603a4658a18d27f708260d` |
| `readme-blanket-flow.mp4` | `fb426241143c3e6f4d726b0d40b9918de55271b5990888077d9181834035b2cc` |
| `readme-alex-b2-field-pressure.webp` | `f186f109ed9f0dc25134baaee22069e521140134ff984be94a60fe1bc825bf2f` |
| `readme-q2d-turbulence-poster.webp` | `97a9e5c418012f1591344ea131a846933a2c60fc2ba47eaf9f93f86e79e1bfb5` |
| `readme-q2d-turbulence.mp4` | `fd6e60bfdeb99f1c0123a8dc68f2198cfde91904b09cfdcde7caa34667492a62` |
| `readme-li-aln-multilayer-convergence.webp` | `e3c8bc3c143cca9c0e1a30c2892005b2be6e0d3c46efda4c00704d81f57e66e6` |

The 71,043-byte Q2D H.264 loop and 13,724-byte poster derive from the existing
72-frame source GIF (SHA-256
`feb9e145cee2a5a87f13381803855a34ad16d44f4d63d0c31c155ad07f74bd96`),
stretched from 5.04 to 7.00 seconds at 600 × 520 and 12 fps. No solver was
rerun; quantitative Q2D-MHDfoam parity remains open.

The 95,818-byte Li/AlN WebP derives from the 2,532 × 1,732 convergence figure
(SHA-256 `0a5d4d0a9a97cff78eb15efdb79934515fcabb5918c23148c8c0653ce198ea94`).
It reports research-stage mesh-step evidence; no experimental-validation claim
is attached to it.

New examples write beneath ignored `artifacts/`. Publish new large media only
with the generating commit, command, environment, input hashes, and output
hashes; add a docs-local derivative only while the tracked checkout remains
inside its architecture budget.

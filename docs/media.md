---
orphan: true
---

# Media provenance

Feature pages embed the relevant figures directly. This compact index records
where the compressed derivatives live; raw fields, meshes, and full-resolution
media remain in the versioned [research-assets release](https://github.com/uwplasma/LMX/releases/tag/lmx-research-assets-v1).
The package wheel contains no media.

## Tracked web set

Six files in `docs/_static/` total 413,409 bytes: duct profiles, FreeMHD parity,
the B1 comparison, current CPU/GPU scaling, and the Hunt movie/poster pair. The
67,366-byte scaling WebP has SHA-256
`4d4acb251fb9be7d17817f656facb227d078f556296077286dae778cc7598a35`.

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

New examples write beneath ignored `artifacts/`. Publish new large media only
with the generating commit, command, environment, input hashes, and output
hashes; add a docs-local derivative only while the tracked checkout remains
inside its architecture budget.

# Figures and movies

LMX keeps source, compact reference observables, tests, and a sub-megabyte web
media set in Git. Generated fields, meshes, full tables, and full-resolution
media are checksummed release assets. The package wheel contains no media.

## Selected results

![Analytical and LMX duct-flow profiles](_static/analytic_velocity_profiles.webp)

![LMX versus FreeMHD observable parity](_static/freemhd_closed_channel_observable_parity.webp)

![B1 pipe reference comparison](_static/pipe_reference_comparison.webp)

![GPU strong scaling](_static/strong_scaling.webp)

## Movies

<p align="center">
  <a href="_static/readme_hunt_startup_2d.mp4"><img src="_static/readme_hunt_startup_2d_poster.webp" alt="Hunt-flow startup movie" width="60%"></a>
</p>

The Hunt poster links to a 130 KB H.264 movie derived from the original
10.6 MB GIF with the duration preserved (SHA-256
`6dfcb77f5c849b9a3858ce0b82df05bae1bf0294a4111abd32798696f0a4c073`).

Examples write new media beneath `artifacts/`. Before publication, compress
PNG files, encode movies for the web, create a poster image, and upload the
bundle to a versioned GitHub or Zenodo release. The release record must include
the generating commit, command, environment, input hashes, and output hashes.
Only add a docs-local derivative when it is required for anonymous access and
the complete web-media set remains within the architecture byte budget.

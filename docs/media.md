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

The README also uses seven release-hosted derivatives so geometry, nonuniform
fields, differentiation, magnetic obstacles, B2 evidence, and blanket transients are visible
without enlarging the checkout. Together they are about 330 KB: WebP stills
are 28--64 KB and the H.264 movie is 72 KB. Their SHA-256 values are:

- `readme-autodiff.webp`: `f2add9fe96a044a082a3b0ca28a3de50b71a0961399ac362caf2ebcaa049aa28`
- `readme-geometries.webp`: `e844f069475fa625279e229a3e8499ae816308b88c8d1d33afb4fb14077bf700`
- `readme-variable-field.webp`: `5d4b593c26a55cb031678988133ba2f6ddeae5e0cbee16a0e166282584bce569`
- `readme-magnetic-obstacle.webp`: `6e351e80531742b1aed6f951d6155cc55d3f5fd4f3e9b556a15336f76a74bc84`
- `readme-blanket-flow-poster.webp`: `f40dff36a41bdfc1fa9df7371cf9c4907a367d70c5603a4658a18d27f708260d`
- `readme-blanket-flow.mp4`: `fb426241143c3e6f4d726b0d40b9918de55271b5990888077d9181834035b2cc`
- `readme-alex-b2-field-pressure.webp`: `f186f109ed9f0dc25134baaee22069e521140134ff984be94a60fe1bc825bf2f`

The 43,174-byte B2 panel was generated at commit `15db9d0` with the plot-only
Benchmark B command; no solver or restart was loaded. Input SHA-256 values are:

- fine transverse record: `cbdb645c95aeee46e223f0c03ae4e47941957c3530db0846c956fe990b5e54e3`
- coarse Maxwell-consistent record: `8162fc8603d320619b98207afa56b13239affdf3f17a68125d467ddc274df5d9`
- Maxwell-consistent field: `987e1e1fd4430e32eef2b3217914a9b6df3394054a9dee3c43476ae6b490b5e5`

Examples write new media beneath `artifacts/`. Before publication, compress
PNG files, encode movies for the web, create a poster image, and upload the
bundle to a versioned GitHub or Zenodo release. The release record must include
the generating commit, command, environment, input hashes, and output hashes.
Only add a docs-local derivative when it is required for anonymous access and
the complete web-media set remains within the architecture byte budget.

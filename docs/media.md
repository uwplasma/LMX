# Figures and movies

LMX keeps source, compact reference observables, and tests in Git. Generated
fields, meshes, full tables, and movies are checksummed release assets. This
keeps clones and wheels small while preserving reproducibility.

## Selected results

![Analytical and LMX duct-flow profiles](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/analytic_velocity_profiles.png)

![LMX versus FreeMHD observable parity](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/freemhd_closed_channel_observable_parity.png)

![B1 pipe reference comparison](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/pipe_reference_comparison.png)

![GPU strong scaling](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/strong_scaling.png)

## Movies

<p align="center">
  <img src="https://github.com/uwplasma/LMX/releases/download/v1.0.2/readme_hunt_startup_2d.gif" alt="Hunt-flow startup" width="46%">
  <img src="https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/wham_blanket_flow.gif" alt="Blanket-flow response" width="46%">
</p>

Examples write new media beneath `artifacts/`. Before publication, compress
PNG files, encode movies for the web, create a poster image, and upload the
bundle to a versioned GitHub or Zenodo release. The release record must include
the generating commit, command, environment, input hashes, and output hashes.
Do not copy generated media back into `docs/_static/`.

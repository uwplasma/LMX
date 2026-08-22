# References

The versioned benchmark manifest is `benchmarks/provenance.json`. It records the
exact repository evidence, checksums for local literature artifacts, and the
tests and workflows that consume each source.

Core references used by the implemented models and validation contracts include:

- U. Müller and L. Bühler, *Magnetofluiddynamics in Channels and Containers*,
  Springer, 2001 — inductionless duct equations, wall effects, and canonical
  Hartmann/Shercliff/Hunt regimes.
- S. Smolentsev et al., “An approach to verification and validation of MHD
  codes for fusion applications,” *Fusion Engineering and Design*,
  [doi:10.1016/j.fusengdes.2014.04.049](https://doi.org/10.1016/j.fusengdes.2014.04.049).
- S. Smolentsev, R. Moreau, L. Bühler, and C. Mistrangelo, “MHD thermofluid
  issues of liquid-metal blankets,” *Fusion Engineering and Design*, 2010,
  [doi:10.1016/j.fusengdes.2010.02.038](https://doi.org/10.1016/j.fusengdes.2010.02.038).
- ALEX results, “A comparison of measurements from a round and a rectangular
  duct with 3-D code predictions,” 1987 — Benchmark B1/B2 pressure and flow
  observables.
- [FreeMHD](https://github.com/ukaea/FreeMHD) — independent finite-volume MHD
  implementation used by the executable external comparison.

Each numerical result should cite the LMX version/commit and the specific
benchmark source used for its physical claim.

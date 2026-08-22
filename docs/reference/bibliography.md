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
- J. Sommeria and R. Moreau, “Why, how, and when, MHD turbulence becomes
  two-dimensional,” *Journal of Fluid Mechanics* 118, 1982,
  [doi:10.1017/S0022112082001177](https://doi.org/10.1017/S0022112082001177) —
  basic Q2D Hartmann-friction closure.
- A. Pothérat, J. Sommeria, and R. Moreau, “An effective two-dimensional model
  for MHD flows with transverse magnetic field,” *Journal of Fluid Mechanics*
  424, 2000,
  [doi:10.1017/S0022112000001944](https://doi.org/10.1017/S0022112000001944) —
  higher-order Q2D corrections and model limits.
- ALEX results, “A comparison of measurements from a round and a rectangular
  duct with 3-D code predictions,” 1987 — Benchmark B1/B2 pressure and flow
  observables.
- S. Benavides and C. Gissinger, “FreeMHD: Validation and application of a new
  open-source MHD solver for OpenFOAM,” *Physics of Plasmas* 32, 2025,
  [doi:10.1063/5.0230242](https://doi.org/10.1063/5.0230242) — the
  [FreeMHD source](https://github.com/PlasmaControl/FreeMHD) supplies the
  independent finite-volume implementation used by the executable comparison.

Each numerical result should cite the LMX version/commit and the specific
benchmark source used for its physical claim.

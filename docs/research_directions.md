# Research Directions

LMX is intended to support both benchmark-quality verification and new research
 studies in liquid-metal MHD.

## Immediate directions

- fully developed conducting-wall duct benchmark and sensitivity studies
- laminar fringing-field benchmarks in ducts and pipes
- differentiable inverse problems over magnetic field strength, geometry, and
  wall conductance ratio

## Near-term literature anchors

- [Samper et al., validation and verification ladder for fusion MHD codes](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf)
- [Differentiable simulation review in IEEE Access](https://mpan31415.github.io/assets/pdf/papers/2024/IEEEAccess24_DiffSim.pdf)
- [Φ-Flow: differentiable PDE tooling](https://proceedings.mlr.press/v235/holl24a.html)
- [Lineax documentation](https://docs.kidger.site/lineax/api/solvers/)
- [Diffrax adjoint documentation](https://docs.kidger.site/diffrax/api/adjoints/)

## Why this matters for LMX

The codebase is being shaped so that:

- the solver core is suitable for reproducible V&V studies
- the same core can be differentiated for inverse and optimization tasks
- external benchmark comparisons remain possible without coupling the governing
  implementation to an external codebase

## Recommended next paper-scale milestones

1. finalize the fully developed duct solver acceptance set
2. land the first `extruded_inductionless` laminar fringing benchmark
3. ship a differentiable inverse example that optimizes one benchmark parameter
   against a target profile or integral observable

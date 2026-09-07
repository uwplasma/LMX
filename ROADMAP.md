# LMX roadmap

[plan.md](plan.md) is authoritative. Issues and PRs record daily evidence; dates
below are planning targets, not validation or release promises.

1. **Integration and hygiene — September 2026.** Complete Phase 0 precision and
   CI qualification (#74, #75). Preserve the tagged B1/B2 reference and its tests.
2. **Trustworthy 3-D core and useful inverse design — October–November 2026.**
   Run the 8³ MAC/collocated oracle before choosing the momentum discretization.
   Verify conservation, convergence and implicit derivatives. Ship the fully
   developed fixed-flow design examples without waiting for new 3-D numerics.
3. **Independent validation and research applications — gate-driven, then release.**
   Qualify B2 against analytical and FreeMHD evidence before retiring reference
   formulations. Add measured CPU/GPU scaling, Q2D and passive heat in their
   documented envelopes. Use exterior ESSOS fields and independent blanket
   geometry for device channels; do not extrapolate VMEX interior fields.

No release until the plan's acceptance gates pass. No speed, turbulence or heat
transfer claim without its corresponding numerical and physical evidence.

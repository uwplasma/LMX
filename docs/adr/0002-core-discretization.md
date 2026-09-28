# 0002 — Conservative 3-D discretization

Status: accepted. MAC is the default under plan D1, and the oracle result is
recorded below (#181, 2026-09-28). Until then this line read "oracle result pending".

Use compatible face currents and a shared steady/transient residual. Before
implementing momentum, compare MAC and repaired collocated layouts on an 8³
problem: nullspaces, volume-weighted adjoint identity, energy, derivative cost
and source size. Record measured results here; do not interpret this ADR as a
completed comparison. Retain the B1/B2 reference until all D2 replacement gates.

Basis: Ni et al., JCP 227:174 (2007), the
[charge-conservative formulation](https://bpb-us-w2.wpmucdn.com/research.seas.ucla.edu/dist/d/39/files/2019/08/JCP-v227-NiCurrentPart1.pdf).

## Oracle result (plan step 1.4, 2026-09-28)

The tiny oracle is `tests/test_momentum_placement_oracle.py`. It builds the
pressure operator of each layout on a uniform 8³ box with walls on every side
(512 cells). The staggered operator comes from the production stencils of
`lmhdx.ops`. The collocated one is the wide centred difference, mirrored at the
walls, which is the classic unrepaired layout; it is assembled in the test and
not shipped. Every number below is asserted there.

| Property | MAC (staggered) | Collocated | Why it matters |
|---|---|---|---|
| Nullspace of the pressure operator | 1 (the constant) | 8 (one constant per even/odd sublattice) | seven checkerboard modes the projection cannot see |
| Gradient–divergence adjoint defect, `\|<p, D u> + <u, G p>\| / \|<p, D u>\|` | ≤ 1e-14 | 0.82 | a projection is idempotent and orthogonal only when it holds |
| Divergence left by the projection, relative | 2.6e-15 (production fast-diagonal solve) | 6.1 % (least-squares pressure; the right-hand side is outside the operator's range) | whether the projected field is solenoidal at all |
| Energy identity, `\|u*\|² = \|u\|² + \|G p\|²`, relative defect | 9.8e-16 | 9.4e4 | projection must only remove energy |
| Volume-weighted asymmetry of the pressure operator | 0 | 0.67 | the direct factorization applies, and the adjoint solve of a derivative reuses it |
| Stencil entries per row (mean / max) | 6.25 / 7 | 7 / 7 (reaching two cells) | cost of one apply and of one derivative apply |

**Derivative cost.**
- Reverse mode through `L p = b` solves `Lᵀ λ = g`.
- MAC is symmetric under the cell volumes, so that adjoint is the same fast-diagonal solve. One host factorization serves the value and the gradient; `lmhdx.steady` relies on this for its symmetric `custom_linear_solve`.
- The collocated operator would need a second factorization or a nonsymmetric Krylov solve for every adjoint. It would also need a Rhie–Chow-type interpolation to remove the seven spurious modes first, and that interpolation is not adjoint either.
- The cost difference is therefore at least one extra factorization and a non-symmetric solver, not a stencil-width factor. A wall-clock comparison would compare a working solver against one that does not project, so none is quoted.

**Decision.** MAC stays the default (D1). No repaired collocated prototype is built; the oracle is the evidence the plan asked for.

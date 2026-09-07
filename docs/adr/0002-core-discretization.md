# 0002 — Conservative 3-D discretization

Status: MAC is the accepted default under plan D1; oracle result pending.

Use compatible face currents and a shared steady/transient residual. Before
implementing momentum, compare MAC and repaired collocated layouts on an 8³
problem: nullspaces, volume-weighted adjoint identity, energy, derivative cost
and source size. Record measured results here; do not interpret this ADR as a
completed comparison. Retain the B1/B2 reference until all D2 replacement gates.

Basis: Ni et al., JCP 227:174 (2007), the
[charge-conservative formulation](https://bpb-us-w2.wpmucdn.com/research.seas.ucla.edu/dist/d/39/files/2019/08/JCP-v227-NiCurrentPart1.pdf).

# Fringing-field workflows

LMX includes a research-stage 3D inductionless solver for liquid metal moving
through an imposed spatially varying magnetic field. It is suitable for bounded
laminar studies and method development; it is not yet a generally validated
production fringing-field solver.

## Start with a curated case

```bash
lmx examples/cases/fringing/fringing_rect_case.toml
lmx examples/cases/fringing/fringing_layered_case.toml
lmx examples/cases/fringing/fringing_pipe_case.toml
```

Or run the benchmark-oriented Python workflow:

```bash
python examples/fringing_benchmark_demo.py --help
```

The paired restart input is
`examples/cases/fringing/fringing_layered_restart_case.toml`; a minimal Python
restart workflow is `examples/extruded_restart_demo.py`.

## Model

For prescribed magnetic field `B(x)` and low magnetic Reynolds number,

```text
div(u) = 0
rho (du/dt + u.grad(u)) = -grad(p) + rho nu laplacian(u) + J x B
J = sigma (-grad(phi) + u x B)
div(J) = 0
```

The electric-potential solve enforces current continuity at each update. The
field is imposed and is not evolved. See [Theory](theory.md) for assumptions and
[Numerics](numerics.md) for discretization.

## Supported geometry and fields

- extruded rectangular and layered rectangular ducts;
- mapped pipe O-grid construction and preview;
- analytical axial fringe profiles;
- divergence-free cross-section fields;
- checked tabulated fields from NPZ data;
- restartable steady/transient outer iterations.

The tabulated example is:

```bash
lmx examples/cases/fringing/fringing_tabulated_case.toml
```

LMX validates coordinates, units, shape, finite values, interpolation bounds,
and field-quality metrics before a tabulated field reaches the solver.

## Diagnostics and acceptance

Every reported run should include:

- normalized `div(J)` and boundary-normal current;
- normalized `div(u)` and mass-flux mismatch;
- electric, pressure, and momentum solver convergence;
- pressure drop and flow-rate station histories;
- viscous, Lorentz, pressure, and Joule power terms;
- mesh, time-step, source, and restart fingerprints.

Small conservation residuals establish internal consistency, not agreement with
an experiment. Experimental promotion additionally requires mesh/time
convergence and a frozen pressure observable from independent data.

## ALEX benchmarks

The retained specifications and references are:

| Case | Geometry | Reference | Status |
|---|---|---|---|
| B1 | conducting circular pipe | `alex-b1-pipe.csv` | pressure solver accepted; experimental observable open |
| B2 | conducting square duct | `alex-b2-square.csv` | physics research-stage; 2-GPU scaling passes |

Files live in `benchmarks/specs/` and `benchmarks/references/`. Construction and
observable extraction are implemented in `lmx/benchmarks.py`.

B1 uses an exact periodic azimuthal line and retained modal setup factors. A
one-cycle physical-convergence pilot now reduces the large solve-plus-restart
pressure ceiling from 768 to 669 Krylov iterations while preserving divergence,
fixed-flow, charge, and restart gates. Experimental pressure-observable and mesh
acceptance remain open.

The compatible retained-modal solver is the sole frozen B1 pressure path after
small factor parity, medium and large field/pressure-observable parity, and a
large solve/restart gate passed. No B1 environment switch is required.

B2 supports named axial sharding. On two RTX A4000 GPUs the fixed-size scaling
gate improves from 36.96 s to 22.23 s. The fine-checkpoint transverse Galerkin
gate separately reduces electric iterations 5.18x and matched two-update time
1.87x with equivalent fields and residuals. The resulting fine baseline plus
doubled-iteration and wall confirmations pass; its tighter-tolerance variant
remains open. These are solver and numerical results, not experimental
validation.

## Limitations

- laminar inductionless formulation only;
- no evolved magnetic field or full MHD induction;
- no validated turbulence, heat transfer, or free surface;
- mapped-pipe and fringe-pressure validation remains incomplete;
- only solver paths with explicit equivalence gates may use multiple devices.

Large field dumps and movies are release assets. Compact specifications,
references, and accepted summaries remain in Git.

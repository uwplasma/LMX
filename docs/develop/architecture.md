# Architecture

LMX is organized by physical ownership:

```text
CaseSpec
  ├─ fully developed duct ──> mesh + MHD coefficients ──> SOLVAX ──> Solution
  └─ extruded/fringing ─────> mapped metrics + MHD coupling ─> SOLVAX ─> ExtrudedSolution
                                                                  │
                                       diagnostics + validation <─┘
Q2DProblem ──> vorticity dynamics ──> SOLVAX periodic Poisson ──> Q2DResult
```

The stable package root contains the common 2-D and Q2D workflows. Three-dimensional,
field, differentiation, output, and validation APIs live in their named
modules. Imports remain one-way: specifications and data containers do not
depend on solvers; plotting dependencies load only when an output function
requests them.

`lmx.fringing` is the public 3-D interface. Its private implementation is
partitioned by mathematical ownership: the public module owns solve
orchestration, while private modules hold shared mapped, rectangular, and
cylindrical operators. The private modules are not separate user APIs, and
architecture gates prevent the numerical kernels from becoming an
undifferentiated mega-module.

Package sources live under `src/lmx`, so an editable installation and a wheel
resolve the same module tree. The wheel includes `lmx/py.typed`; every root API
callable has an explicit signature, and distribution audits require the typing
marker and reject files outside the package and metadata roots.

Reusable algebra belongs in SOLVAX when it is independent of LMX geometry,
units, boundaries, and terminology and has its own correctness, gradient,
convergence, documentation, and performance tests. LMX retains coefficient
assembly, gauges that express physical constraints, coupling, and physical
acceptance residuals.

FreeMHD execution is repository tooling and is excluded from the runtime wheel.
Only benchmark specifications and compact reference arrays ship as package data.

Structural budgets are enforced by `scripts/audit_architecture.py`: tracked
size, source/module count, test/script/example count, root API size, import
latency, and distribution contents.

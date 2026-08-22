# Architecture

LMX is organized by physical ownership:

```text
CaseSpec
  ├─ fully developed duct ──> mesh + MHD coefficients ──> SOLVAX ──> Solution
  └─ extruded/fringing ─────> mapped metrics + MHD coupling ─> SOLVAX ─> ExtrudedSolution
                                                                  │
                                       diagnostics + validation <─┘
```

The stable package root contains the common 2-D workflow. Three-dimensional,
field, differentiation, output, and validation APIs live in their named
modules. Imports remain one-way: specifications and data containers do not
depend on solvers; plotting dependencies load only when an output function
requests them.

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

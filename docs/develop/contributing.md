# Contributing

Install the development dependencies, then use the change-aware gate while
editing:

```console
python -m pip install -e ".[dev]"
python scripts/run_full_test_suite.py --changed-from HEAD
```

The command includes working-tree changes, selects a conservative affected test
set, and keeps JAX's persistent compilation cache outside the repository. Run
exact test node IDs when the edit has a narrower contract. Before merging one
source candidate, run the complete covered suite and architecture audit once:

```console
python scripts/run_full_test_suite.py
python scripts/audit_architecture.py --check --measure-import
```

Build Sphinx HTML only when user documentation or its imported API changes.
External links, distributions, clean-wheel smoke, Docker, production meshes,
and accelerator measurements belong to scheduled, release, or directly
affected numerical boundaries rather than every edit.

Changes should reduce or preserve conceptual surface area. Prefer one clear
path, immutable typed inputs, explicit units and shapes, bounded allocation,
JAX-compatible array operations, and physical residuals that fail closed.
Generated outputs, environment directories, downloaded literature, Docker case
trees, and benchmark dumps do not belong in Git.

For numerical changes, add the smallest test that establishes the relevant
invariant or convergence result. A change to the B2 discretization, benchmark,
or validation boundary requires the reduced FreeMHD B2 Docker smoke; unrelated
pipe, Q2D, documentation, and test-only changes do not. Combined line/branch
coverage must remain above 95% at the source-candidate boundary.

Public functions need type annotations and docstrings that state units, shapes,
defaults, return semantics, and failure modes. Keep user documentation focused
on the supported current interface.

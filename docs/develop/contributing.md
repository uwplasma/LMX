# Contributing

Install the development and documentation dependencies, then run the same gates
used by CI:

```console
python -m pip install -e ".[dev,docs]"
python scripts/audit_architecture.py --check --measure-import
python scripts/run_full_test_suite.py
python -m sphinx -W -b html docs docs/_build/html
```

Changes should reduce or preserve conceptual surface area. Prefer one clear
path, immutable typed inputs, explicit units and shapes, bounded allocation,
JAX-compatible array operations, and physical residuals that fail closed.
Generated outputs, environment directories, downloaded literature, Docker case
trees, and benchmark dumps do not belong in Git.

For numerical changes, add the smallest test that establishes the relevant
invariant or convergence result. A structural change to 3-D/fringing code also
requires the reduced FreeMHD B2 Docker smoke. Combined line/branch coverage must
remain above 95%.

Public functions need type annotations and docstrings that state units, shapes,
defaults, return semantics, and failure modes. Keep user documentation focused
on the supported current interface.

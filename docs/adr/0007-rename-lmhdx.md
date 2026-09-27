# 0007 — The code is renamed LMhdX

Status: accepted 2026-09-23; supersedes D28 of ADR 0006.

**D30. LMX is renamed LMhdX: repository, distribution, import package and command.**
The name is written LMhdX, with a lower-case "hd", so the code can still be
called LMX informally.
PyPI refuses the name `lmx` as too close to existing projects, and the
first PyPI release needs a name that is the same everywhere. From version 1.5.0:

- repository `uwplasma/LMhdX`; GitHub redirects `uwplasma/LMX` and its clone URL;
- distribution `pip install lmhdx`, published from `release.yml` by trusted publishing;
- import package `lmhdx` (`import lmx` becomes `import lmhdx`), command `lmhdx`,
  environment variables `LMHDX_COMPILATION_CACHE` and `LMHDX_XLA_DEFAULTS`, and
  the default cache directory `~/.cache/lmhdx`.

No compatibility module named `lmx` is shipped, because it would occupy the
import name the rename frees. The documentation keeps its current address until
the Read the Docs project is renamed. Historical text (this directory, the log
rows and past decisions in `plan.md`) keeps the old name, as it describes what
was true then. D28's rule carries over: public text keeps the code distinct
from PPPL's LMX-U experiment.

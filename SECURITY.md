# Security policy

## Supported versions

Security fixes target the latest released minor version and the current default
branch. Older research snapshots and external campaign artifacts are retained for
reproducibility but are not supported runtime environments.

## Reporting a vulnerability

Use GitHub's private security-advisory interface for `uwplasma/LMX`. Do not open
a public issue containing exploit details, credentials, private paths, or other
sensitive information.

Include the affected version or commit, platform, minimal reproduction, impact,
and any proposed mitigation. Maintainers will acknowledge a report, assess
severity and affected versions, coordinate a fix and disclosure, and credit the
reporter unless anonymity is requested.

LMX processes user-provided TOML, NPZ, CSV, JSON, VTK, and external-solver data.
Treat untrusted files as untrusted input and run third-party Docker images or
campaign scripts with the isolation appropriate to the data and host.

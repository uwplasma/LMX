# 0004 — Bounded evidence and trusted scheduled work

Status: accepted under plan D12 and Section 9; deployment qualification pending.

Use disjoint PR unit/regression selections without coverage instrumentation.
Retain main/release coverage and all numerical tolerances. Measure queue and
execution time separately. Collection tests must prove complete tier coverage.
Before costly campaigns record hypothesis, budget and exit; a failed bounded
attempt requires a decision rather than an automatic longer run.

Office scheduling must execute trusted main only, never PR code. The repository
currently has no registered self-hosted runner; do not register a persistent
runner on this public repository to make an untested nightly workflow appear
operational. Keep external validation active until the trusted replacement is
deployed and verified. Benchmark publication and one-issue failure reporting
remain deployment acceptance gates in #75.

Basis: [GitHub runner security](https://docs.github.com/en/actions/concepts/security/compromised-runners).

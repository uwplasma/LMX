# FreeMHD comparison

FreeMHD is an external validator, not an LMX runtime dependency. LMX retains a
small boundary for four tasks:

1. describe matched equations, geometry, properties, drive, and controls;
2. observe declared fields and scalar quantities from native outputs;
3. record immutable source, image, and artifact identities;
4. compare the same normalized observables with explicit tolerances.

The B2 square-duct smoke runs FreeMHD in Docker and then evaluates LMX from the
matched input. It checks that the solver actually ran, that output timestamps
advanced, and that the observed files belong to the recorded execution.

```console
python scripts/run_freemhd_parity_suite.py \
  --matched-b2-smoke \
  --freemhd-image freemhd-install:latest \
  --output artifacts/freemhd-b2
```

Resolve and record the immutable image ID before interpreting a comparison:

```console
docker image inspect freemhd-install:latest --format '{{.Id}}'
```

The accepted record contains source commit, image ID, case and evaluator
checksums, mesh and precision, solver exit metadata, artifact hashes,
normalizations, observables, tolerances, and individual gate results.

FreeMHD fixture parsing is useful for unit tests but cannot satisfy an executed
parity gate. B1 pipe acceptance also requires the mapped-pipe formulation,
conducting annulus, three mesh levels, tolerance/iteration independence, and a
matched executable FreeMHD case.

![LMX and FreeMHD observable parity](../_static/freemhd_closed_channel_observable_parity.webp)

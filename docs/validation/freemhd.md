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

The reproducible build uses
[`freemhd_install`](https://github.com/rogeriojorge/freemhd_install) commit
`36f409d294ba3170d64d4073378d5ef68401072f`, FreeMHD commit
`14b54a3e8e1a05b6ee4c98331995abaaae96e7a5`, and OpenFOAM v2206. The weekly
external-validation workflow rebuilds that image, checks the two source pins,
runs the comparison, and uploads the complete evidence directory. Relevant
3-D/validation pull requests and release validation call the same workflow.

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

On the frozen harness grid, the normalized transverse-pressure comparison has
RMS error 0.004518 and maximum error 0.01092. Its fixed harness limits are 0.16
and 0.32. Because the harness role executes two updates on a small mesh, a
passing record verifies integration and matched-observable plumbing but cannot
promote B2 to production-mesh acceptance.

FreeMHD fixture parsing is useful for unit tests but cannot satisfy an executed
parity gate. B1 pipe acceptance also requires the mapped-pipe formulation,
conducting annulus, three mesh levels, tolerance/iteration independence, and a
matched executable FreeMHD case.

# FreeMHD Container Bundle

This bundle is the local handoff for building and running FreeMHD parity cases beside LMX.

## Files

- `Dockerfile`: OpenFOAM v2206 based image scaffold for FreeMHD.
- `run_freemhd_case.sh`: container entrypoint for `Allrun` or direct solver execution.

## Typical usage

```bash
docker build -t lmx-freemhd ./docker
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd --case-dir /absolute/path/to/case
```

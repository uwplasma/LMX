# Examples

These scripts run one native LMX duct case and write:

- ParaView output
- CSV midplane profiles
- `example_report.json`
- publication-style `overview.png/.pdf`
- publication-style `diagnostics.png/.pdf`

Run them directly from the repo root:

```bash
/Users/rogerio/base_env/bin/python3 examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
/Users/rogerio/base_env/bin/python3 examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
/Users/rogerio/base_env/bin/python3 examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
```

Shercliff and Hunt automatically use the default closed-channel analytical
reference root when it exists under `./external/FreeMHDPaperAllFigures/...`.
Override that with `--reference-root` if needed.

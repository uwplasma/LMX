# Examples

These scripts run one native LMX duct case and write:

- ParaView output
- CSV midplane profiles
- `example_report.json`
- publication-style `overview.png/.pdf`
- publication-style `diagnostics.png/.pdf`
- for the meeting demo, 2D/3D Hunt startup movies and poster frames

Run them directly from the repo root:

```bash
/Users/rogerio/base_env/bin/python3 examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
/Users/rogerio/base_env/bin/python3 examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
/Users/rogerio/base_env/bin/python3 examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
/Users/rogerio/base_env/bin/python3 examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo --resolution 32 --hunt-resolution 24 --hunt-dt 5e-6 --hunt-t-final 8e-5 --hunt-frames 6
```

Shercliff and Hunt automatically use the default closed-channel analytical
reference root when it exists under `./external/FreeMHDPaperAllFigures/...`.
Override that with `--reference-root` if needed.

`theory_meeting_demo.py` is the highest-signal presentation example. It writes
steady Hartmann, Shercliff, and Hunt comparison plots plus Hunt startup movies
that visualize `u - <u>_fluid`, which makes the evolving boundary layers
visible even when the bulk startup field is nearly uniform. The Hunt movie
assets are written as:

- `hunt/hunt_boundary_layers_2d.gif`
- `hunt/hunt_boundary_layers_3d.gif`
- `hunt/hunt_boundary_layers_2d_poster.png`
- `hunt/hunt_boundary_layers_3d_poster.png`

The Hunt movie assets land under `hunt/` as:

- `hunt_boundary_layers_2d.gif`
- `hunt_boundary_layers_3d.gif`
- `hunt_boundary_layers_2d_poster.png/.pdf`
- `hunt_boundary_layers_3d_poster.png/.pdf`

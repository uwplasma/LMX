# Examples

These scripts run native LMX duct cases and write:

- ParaView output
- CSV midplane profiles
- compressed `.npz` solution dumps with mesh, fields, material arrays, and diagnostics
- `example_report.json`
- publication-style Matplotlib plots generated from those `.npz` files
- for the meeting demo, 2D/3D startup movies and poster frames for one selected case

Run them directly from the repo root:

```bash
/Users/rogerio/base_env/bin/python3 examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
/Users/rogerio/base_env/bin/python3 examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
/Users/rogerio/base_env/bin/python3 examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
/Users/rogerio/base_env/bin/python3 examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
```

Shercliff and Hunt automatically use the default closed-channel analytical
reference root when it exists under `./external/FreeMHDPaperAllFigures/...`.
Override that with `--reference-root` if needed.

`theory_meeting_demo.py` is the highest-signal presentation example. It writes
verbose setup and solver-progress logs, steady Hartmann, Shercliff, and Hunt
NPZ dumps, comparison plots, and startup movies for a selected case. The
retained default is Shercliff because it gives the clearest startup structure
at modest runtime. The default movie assets are written as:

- `shercliff/shercliff_startup_snapshots.npz`
- `shercliff/movie/shercliff_startup_2d.gif`
- `shercliff/movie/shercliff_startup_3d.gif`
- `shercliff/movie/shercliff_startup_2d_poster.png`
- `shercliff/movie/shercliff_startup_3d_poster.png`

Use `--movie-case hunt` or `--movie-case hartmann` to switch the movie path.
The Hunt mode renders `u - <u>_fluid` so the boundary layers remain visible
even when the bulk startup field is nearly uniform.

The script is intentionally written as a teaching example, not a one-line
wrapper. It defines local functions for:

- case construction
- solver-control overrides
- verbose logging
- NPZ serialization
- replotting and movie generation

`plot_npz_results.py` is a standalone reader/plotter for those saved files:

```bash
/Users/rogerio/base_env/bin/python3 examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_ha20_results.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot
/Users/rogerio/base_env/bin/python3 examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_startup_snapshots.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot_movie --movies --stem shercliff_replot
```

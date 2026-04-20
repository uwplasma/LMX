from __future__ import annotations

from pathlib import Path

import jax


def enable_compilation_cache(
    cache_dir: str | Path | None = None,
    *,
    min_compile_time_secs: float = 0.0,
    min_entry_size_bytes: int = -1,
) -> Path:
    """Enable JAX's persistent compilation cache before the first heavy compile.

    The default location is local to the user cache tree so repeated example,
    benchmark, and validation reruns on the same host reuse compiled artifacts.
    """

    target = Path(cache_dir or (Path.home() / ".cache" / "lmx" / "jax_compilation"))
    target.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(target))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_size_bytes)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_time_secs)
    return target

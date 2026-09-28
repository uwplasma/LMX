"""Programs compiled once per shape, with the arrays they close over passed as arguments (2b.1)."""

from __future__ import annotations

import collections
import contextlib
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["shape_program"]


def _constants_as_constvars():
    """Keep array constants as jaxpr constants while tracing, which :func:`shape_program` packs.

    JAX's simplified constants (on with the default compilation cache, where the installed JAX
    has them) put them inline as literals instead, and lowering then embeds numpy arrays.
    """
    from jax._src import config

    state = getattr(config, "use_simplified_jaxpr_constants", None)
    return contextlib.nullcontext() if state is None else state(False)


_EXECUTABLES: collections.OrderedDict = collections.OrderedDict()
_MAX_EXECUTABLES = 64


def shape_program(function, *arguments):
    """Compile ``function`` of ``arguments`` (shapes and dtypes) once per program shape (2b.1).

    Tracing leaves every array the solve closes over -- the grid's metric, the
    field, the factorizations -- as a constant of the program. Embedded, a new
    Hartmann number, field or conductance on the same mesh is a new program and
    compiles again, 4-5 s on a 48-cell duct. Here the constants are packed into
    one device buffer per dtype, which the program takes as an argument, and the
    executable is looked up by a hash of the lowered program, so two problems
    whose programs differ only in those arrays share it, in the process and, as
    the lowered program is the same, in the persistent compilation cache. A
    scalar that the trace keeps as a literal still selects its own program, so
    the sharing is exact by construction. Returns a function of ``arguments``.
    """
    with _constants_as_constvars():
        closed, shapes = jax.make_jaxpr(function, return_shape=True)(*arguments)
    tree = jax.tree.structure(shapes)
    groups: dict[str, list[np.ndarray]] = {}
    sizes: dict[str, int] = {}
    layout = []
    for constant in closed.consts:
        value = np.asarray(constant)
        name = value.dtype.str
        layout.append((name, sizes.get(name, 0), value.shape))
        groups.setdefault(name, []).append(value.ravel())
        sizes[name] = sizes.get(name, 0) + value.size
    names = sorted(groups)
    with jax.ensure_compile_time_eval():
        packed = tuple(jnp.asarray(np.concatenate(groups[name])) for name in names)
    jaxpr = closed.jaxpr

    def run(buffers, *values):
        by_name = dict(zip(names, buffers, strict=True))
        constants = [
            by_name[name][start : start + int(np.prod(shape, dtype=int))].reshape(shape)
            for name, start, shape in layout
        ]
        return jax.core.eval_jaxpr(jaxpr, constants, *values)

    with _constants_as_constvars():
        lowered = jax.jit(run).lower(packed, *arguments)
    # The module text does not name the backend, so the platform the buffers live on is part of the key.
    platforms = sorted({device.platform for buffer in packed for device in buffer.devices()})
    key = hashlib.sha256((repr(platforms) + lowered.as_text()).encode()).hexdigest()
    executable = _EXECUTABLES.pop(key, None) or lowered.compile()
    _EXECUTABLES[key] = executable
    while len(_EXECUTABLES) > _MAX_EXECUTABLES:
        _EXECUTABLES.popitem(last=False)
    return lambda *values: jax.tree.unflatten(tree, executable(packed, *values))

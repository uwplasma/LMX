"""Programs compiled once per shape, with the arrays they close over passed as arguments (2b.1)."""

from __future__ import annotations

import collections
import contextlib
import contextvars
import hashlib

import jax
import jax.numpy as jnp
import numpy as np

from .grid import Field, Grid

__all__ = [
    "ShapeProgram",
    "Unbound",
    "attribute",
    "bound",
    "discovering",
    "host_array",
    "host_scalar",
    "problem_arrays",
    "shape_program",
]


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


# Stage 5 of 2b.1: every array a solve reads is built on the host by a function of the problem,
# so a program traced once per shape runs any problem of that shape without tracing it again.

ROOT = ("problem",)
_TRACE: contextvars.ContextVar = contextvars.ContextVar("lmhdx_problem_trace", default=None)


class Unbound(Exception):
    """A traced solve read an array that the problem's arrays do not provide."""


def _grid(problem):
    return problem.grid


def _item(values, index):
    return values[index]


def _attribute(owner, name):
    return getattr(owner, name)


class _Trace:
    """Where the host arrays of one problem come from while a solve over it is traced.

    Without ``table`` it only notes the key of every array (discovery, on the
    first problem of a shape, whose program embeds them); with one, each array
    is the matching entry of ``table``, a traced argument.
    """

    def __init__(self, problem, table=None, objects=None):
        self.problem, self.table, self.objects = problem, table, objects or {}
        self.origins: dict[int, tuple] = {}
        self.alive: list = []
        self.keys: dict[tuple, None] = {}
        self.complete = True
        self._register(problem, ROOT)
        self._register(problem.grid, (ROOT, _grid, ()))

    def _register(self, value, origin) -> None:
        self.origins[id(value)] = origin
        self.alive.append(value)
        if type(value) is tuple:
            for index, item in enumerate(value):
                self._register(item, (origin, _item, (index,)))

    def origin(self, owner):
        origin = self.origins.get(id(owner))
        if origin is None and isinstance(owner, Grid) and owner == self.problem.grid:
            origin = self.origins[id(self.problem.grid)]
        return origin

    def key(self, owner, build, static, kind):
        origin = self.origin(owner)
        if origin is None:
            if self.table is not None:
                raise Unbound(f"{build.__qualname__} read an array of an object the problem does not name")
            self.complete = False
            return None
        key = (origin, build, static, kind)
        if self.table is not None and key not in self.table:
            raise Unbound(f"{build.__qualname__}{static} is not among the problem's arrays")
        self.keys[key] = None
        return key


@contextlib.contextmanager
def _tracing(trace):
    token = _TRACE.set(trace)
    try:
        yield trace
    finally:
        _TRACE.reset(token)


def _kind(dtype) -> str | None:
    return None if dtype is None else jnp.dtype(dtype).name


def host_array(owner, build, *static, dtype=None) -> jnp.ndarray:
    """``jnp.asarray(build(owner, *static), dtype)``: an array of the problem, built on the host.

    ``owner`` is the problem, its grid or an object :func:`bound` returned, and
    ``static`` holds only what the program's shape fixes (axes, conditions,
    offsets). Traced for a shared program, this is an argument of the program.
    """
    trace = _TRACE.get()
    if trace is not None and trace.table is not None:
        return trace.table[trace.key(owner, build, static, _kind(dtype))]
    value = jnp.asarray(build(owner, *static), dtype=dtype)
    if trace is not None:
        trace.key(owner, build, static, _kind(dtype))
    return value


def host_scalar(owner, build, *static):
    """A number of the problem, ``build(owner, *static)``: a literal embedded, an argument shared."""
    trace = _TRACE.get()
    if trace is not None and trace.table is not None:
        return trace.table[trace.key(owner, build, static, "scalar")]
    if trace is not None:
        trace.key(owner, build, static, "scalar")
    return build(owner, *static)


def attribute(owner, name: str):
    """``getattr(owner, name)`` for a number the traced solve uses, through :func:`host_scalar`."""
    return host_scalar(owner, _attribute, name)


def bound(owner, build, *static):
    """An object built on the host from ``owner`` (a factorization), whose arrays the solve reads.

    Built at trace time as before; while a solve is traced for a shared program
    the object records where it came from, so its arrays can be rebuilt for
    another problem of the shape.
    """
    trace = _TRACE.get()
    origin = None if trace is None else trace.origin(owner)
    key = None if origin is None else (origin, build, static)
    if trace is not None and key in trace.objects:
        value = trace.objects[key]
    else:
        with _tracing(None), jax.ensure_compile_time_eval():
            value = build(owner, *static)
    if trace is not None:
        if key is None:
            if trace.table is not None:
                raise Unbound(f"{build.__qualname__} was built from an object the problem does not name")
            trace.complete = False
        else:
            trace._register(value, key)
    return value


def problem_arrays(problem, keys, objects=None) -> list:
    """Build the arrays ``keys`` name for ``problem`` on the host: one factorization, no trace."""
    objects = {ROOT: problem} if objects is None else objects
    objects.setdefault(ROOT, problem)

    def resolve(origin):
        if origin not in objects:
            parent, build, static = origin
            with jax.ensure_compile_time_eval():
                objects[origin] = build(resolve(parent), *static)
        return objects[origin]

    values = []
    for origin, build, static, kind in keys:
        value = build(resolve(origin), *static)
        values.append(float(value) if kind == "scalar" else jnp.asarray(value, dtype=kind))
    return values


def _constants(closed) -> list:
    """Every constant of a traced program, its inner programs' included."""
    found = list(closed.consts)

    def walk(jaxpr):
        for equation in jaxpr.eqns:
            for parameter in equation.params.values():
                for inner in parameter if isinstance(parameter, (tuple, list)) else (parameter,):
                    if hasattr(inner, "consts") and hasattr(inner, "jaxpr"):
                        found.extend(inner.consts)
                        walk(inner.jaxpr)
                    elif hasattr(inner, "eqns"):
                        walk(inner)

    walk(closed.jaxpr)
    return found


def _regrid(tree, grid):
    """Put the fields a program returns on ``grid``: the program was traced on another problem's."""
    return jax.tree.map(
        lambda leaf: Field(leaf.data, leaf.offset, grid) if isinstance(leaf, Field) else leaf,
        tree,
        is_leaf=lambda leaf: isinstance(leaf, Field),
    )


class ShapeProgram:
    """A solve traced once per shape, whose problem arrays are its arguments (2b.1 stage 5).

    ``build(problem)`` returns the function a program runs. It is traced on
    one problem with every :func:`host_array`, :func:`host_scalar` and
    :func:`bound` object an argument, keyed by how the host builds it from the
    problem: ``keys``, the ones the first problem of the shape named. Another
    problem of the shape then costs its host arrays (:func:`problem_arrays`) and
    no trace. A traced array constant would be one problem's data in every
    problem's program, so a trace that leaves a floating-point array constant
    raises :class:`Unbound`, as does an array the keys do not name.
    """

    def __init__(self, build, problem, arguments, keys):
        objects = {ROOT: problem}
        values = problem_arrays(problem, keys, objects)

        def traced(values, *arguments):
            with _tracing(_Trace(problem, dict(zip(keys, values, strict=True)), objects)):
                return build(problem)(*arguments)

        with _constants_as_constvars():
            staged = jax.jit(traced).trace(values, *arguments)
        leaked = [
            value
            for value in _constants(staged.jaxpr)
            if np.issubdtype(np.asarray(value).dtype, np.inexact) and np.size(value) > 1
        ]
        if leaked:
            raise Unbound(f"the trace kept {len(leaked)} floating-point array constants")
        self.keys = keys
        self.signature = [(np.shape(value), _signature_dtype(value)) for value in values]
        lowered = staged.lower()
        # Equal for every problem of the shape; a problem's value in the program would change it.
        self.fingerprint = hashlib.sha256(lowered.as_text().encode()).hexdigest()
        self.executable = lowered.compile()
        self._first = (problem, values)

    def bind(self, problem):
        """Return the program of ``problem`` as a function of the arguments; raise if its arrays differ in shape."""
        if self._first is not None and self._first[0] is problem:
            values = self._first[1]
        else:
            values = problem_arrays(problem, self.keys)
        self._first = None
        if [(np.shape(value), _signature_dtype(value)) for value in values] != self.signature:
            raise Unbound("the problem's arrays differ in shape from the program's")
        executable, grid = self.executable, problem.grid
        return lambda *arguments: _regrid(executable(values, *arguments), grid)


def _signature_dtype(value) -> str:
    return "scalar" if isinstance(value, float) else str(value.dtype)


def discovering(problem):
    """Note the keys of every problem array a trace reads (the first problem of a shape)."""
    return _tracing(_Trace(problem))

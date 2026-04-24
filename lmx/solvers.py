"""Compatibility facade for fully developed steady/transient solvers."""

from __future__ import annotations

import sys as _sys

from . import _solvers as _implementation

_sys.modules[__name__] = _implementation

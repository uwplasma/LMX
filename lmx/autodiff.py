"""Compatibility facade for differentiable objectives and inverse-design tools."""

from __future__ import annotations

import sys as _sys

from . import _autodiff as _implementation

_sys.modules[__name__] = _implementation

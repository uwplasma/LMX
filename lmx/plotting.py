"""Compatibility facade for plotting, media, and publication-figure helpers."""

from __future__ import annotations

import sys as _sys

from . import _plotting as _implementation

_sys.modules[__name__] = _implementation

"""Compatibility facade for validation profiles, reports, and mesh gates."""

from __future__ import annotations

import sys as _sys

from . import _validation as _implementation

_sys.modules[__name__] = _implementation

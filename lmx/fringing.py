"""Compatibility facade for the 3D inductionless fringing solver family.

Implementation currently lives in :mod:`lmx._fringing` while the public
``lmx.fringing`` import path remains stable during the module split.
"""

from __future__ import annotations

import sys as _sys

from . import _fringing as _implementation

_sys.modules[__name__] = _implementation

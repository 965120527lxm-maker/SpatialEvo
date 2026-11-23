"""Compatibility shim — implementation lives in lxm/tools/ppt/svg.py."""

from __future__ import annotations

import os
import sys

_LXM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _LXM_ROOT not in sys.path:
    sys.path.insert(0, _LXM_ROOT)

from tools.ppt.svg import (  # noqa: E402
    Svg,
    SvgImagePart,
    apply_pptx_svg_support,
    _svg_viewport_px_size,
    _is_svg,
)

__all__ = [
    "Svg",
    "SvgImagePart",
    "apply_pptx_svg_support",
    "_svg_viewport_px_size",
    "_is_svg",
]

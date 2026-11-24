from __future__ import annotations

import os
from typing import Tuple


DEFAULT_DPI: int = 300

EXPORT_FORMATS: Tuple[str, ...] = ("pdf", "png")

EXPORT_SVG_BY_DEFAULT: bool = False

DEFAULT_BBOX_INCHES: str = "tight"

DEFAULT_BBOX_PAD_INCHES: float = 0.02

DEFAULT_FACECOLOR: str = "white"


DEFAULT_FONT_FAMILY: Tuple[str, ...] = (
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
)

DEFAULT_MATHTEXT_FONTSET: str = "stix"

EMBED_FONTS_IN_PDF: bool = True


_PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_OUTPUT_ROOT: str = os.environ.get(
    "PAPER_PLT_OUTPUT_ROOT",
    os.path.join(_PROJECT_ROOT, "paper_img"),
)


def output_root() -> str:
    """Return the configured paper-figure output root."""
    return DEFAULT_OUTPUT_ROOT

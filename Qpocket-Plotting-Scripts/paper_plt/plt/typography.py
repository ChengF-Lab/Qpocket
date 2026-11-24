from __future__ import annotations

from typing import Dict

from .sizes import FIGURE_SIZE_ALIASES, FIGURE_SIZES


BASE_FONT_SIZES: Dict[str, float] = {
    "title": 9.0,
    "axis_label": 8.0,
    "tick_label": 7.0,
    "legend": 7.0,
    "annotation": 7.0,
    "panel_label": 10.0,
    "colorbar_label": 7.5,
    "colorbar_tick": 7.0,
}


FONT_SCALE: Dict[str, float] = {
    "main_single": 0.95,
    "main_wide": 1.00,
    "main_double": 1.00,
    "main_multi": 1.00,
    "main_tall": 0.95,
    "supp_single": 0.95,
    "supp_double": 1.00,
    "supp_multi": 0.95,
    "schematic": 1.10,
    "square": 1.00,
    "tall_strip": 0.90,
}


def _resolve_figure_type(figure_type: str) -> str:
    return FIGURE_SIZE_ALIASES.get(figure_type, figure_type)


def scaled_fonts(figure_type: str = "main_double") -> Dict[str, float]:
    """Return the font hierarchy scaled for ``figure_type``."""
    key = _resolve_figure_type(figure_type)
    if key not in FIGURE_SIZES:
        raise KeyError(
            f"Unknown figure_type {figure_type!r}; " f"valid: {sorted(FIGURE_SIZES)}"
        )
    scale = FONT_SCALE.get(key, 1.0)
    return {k: round(v * scale, 2) for k, v in BASE_FONT_SIZES.items()}


def scale_for(figure_type: str) -> float:
    """Just the scalar font-scale for a given figure type."""
    key = _resolve_figure_type(figure_type)
    return FONT_SCALE.get(key, 1.0)

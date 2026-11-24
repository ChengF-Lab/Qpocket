from __future__ import annotations

from typing import Dict, Tuple

FIGURE_SIZES: Dict[str, Tuple[float, float]] = {
    "main_single": (3.42, 2.80),
    "main_wide": (4.50, 3.20),
    "main_double": (7.00, 4.30),
    "main_multi": (7.00, 7.50),
    "main_tall": (3.42, 5.20),
    "supp_single": (3.50, 3.00),
    "supp_double": (7.00, 4.50),
    "supp_multi": (7.50, 8.50),
    "schematic": (7.00, 3.50),
    "square": (4.00, 4.00),
    "tall_strip": (3.00, 6.00),
}

FIGURE_SIZE_ALIASES: Dict[str, str] = {
    "single": "main_single",
    "wide": "main_wide",
    "double": "main_double",
    "multi": "main_multi",
    "supp": "supp_single",
    "supp_wide": "supp_double",
}


def get_figsize(figure_type: str) -> Tuple[float, float]:
    """Resolve ``figure_type`` (or alias) to a ``(w, h)`` inches tuple."""
    key = FIGURE_SIZE_ALIASES.get(figure_type, figure_type)
    if key not in FIGURE_SIZES:
        raise KeyError(
            f"Unknown figure_type {figure_type!r}. "
            f"Valid options: {sorted(FIGURE_SIZES)} "
            f"(aliases: {sorted(FIGURE_SIZE_ALIASES)})."
        )
    return FIGURE_SIZES[key]


def list_figure_types() -> list:
    """Return the canonical figure-type names (no aliases)."""
    return sorted(FIGURE_SIZES)

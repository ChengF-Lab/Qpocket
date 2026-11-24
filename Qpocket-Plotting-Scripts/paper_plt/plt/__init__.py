from __future__ import annotations

import matplotlib as _mpl

_mpl.rcParams["pdf.fonttype"] = 42
_mpl.rcParams["ps.fonttype"] = 42


from .config import (
    DEFAULT_DPI,
    EXPORT_FORMATS,
    DEFAULT_FONT_FAMILY,
    DEFAULT_OUTPUT_ROOT,
    output_root,
)
from .sizes import (
    FIGURE_SIZES,
    FIGURE_SIZE_ALIASES,
    get_figsize,
    list_figure_types,
)
from .typography import (
    BASE_FONT_SIZES,
    FONT_SCALE,
    scaled_fonts,
    scale_for,
)
from .colors import (
    FRAGMENT_COLORS,
    FRAGMENT_ORDER,
    MODULE_COLORS,
    MODULE_ORDER,
    FRAGMENT_TO_MODULE,
    MUTATION_COLORS,
    MUTATION_ORDER,
    LIGAND_COLORS,
    LIGAND_ORDER,
    NEUTRAL_REFERENCE,
    GRID_GREY,
    ANNOT_GREY,
    HIGHLIGHT,
    GOOD,
    BAD,
    color_for_fragment,
    color_for_module,
    color_for_mutation,
    color_for_ligand,
    palette_for,
)

from .theme import apply_theme, get_theme
from .helpers import (
    new_figure,
    add_panel_label,
    style_axes,
    clean_spines,
    apply_grid,
    maybe_rotate_xticks,
    add_zero_line,
    add_reference_line,
    compact_legend,
    add_section_label,
    add_subpanel_label,
    add_metadata_box,
    add_dashed_border,
)
from .save import (
    save_figure,
    ensure_parent_dir,
    maybe_save_svg,
    resolve_output_path,
)
from .compose import (
    compose_figure1_workflow,
    compose_figure2_performance,
    compose_figure3_landscape,
    compose_figure4_module_response,
    compose_mosaic,
)

__all__ = [
    "DEFAULT_DPI",
    "EXPORT_FORMATS",
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_OUTPUT_ROOT",
    "output_root",
    "FIGURE_SIZES",
    "FIGURE_SIZE_ALIASES",
    "get_figsize",
    "list_figure_types",
    "BASE_FONT_SIZES",
    "FONT_SCALE",
    "scaled_fonts",
    "scale_for",
    "FRAGMENT_COLORS",
    "FRAGMENT_ORDER",
    "MODULE_COLORS",
    "MODULE_ORDER",
    "FRAGMENT_TO_MODULE",
    "MUTATION_COLORS",
    "MUTATION_ORDER",
    "LIGAND_COLORS",
    "LIGAND_ORDER",
    "NEUTRAL_REFERENCE",
    "GRID_GREY",
    "ANNOT_GREY",
    "HIGHLIGHT",
    "GOOD",
    "BAD",
    "color_for_fragment",
    "color_for_module",
    "color_for_mutation",
    "color_for_ligand",
    "palette_for",
    "apply_theme",
    "get_theme",
    "new_figure",
    "add_panel_label",
    "style_axes",
    "clean_spines",
    "apply_grid",
    "maybe_rotate_xticks",
    "add_zero_line",
    "add_reference_line",
    "compact_legend",
    "add_section_label",
    "add_subpanel_label",
    "add_metadata_box",
    "add_dashed_border",
    "save_figure",
    "ensure_parent_dir",
    "maybe_save_svg",
    "resolve_output_path",
    "compose_figure1_workflow",
    "compose_figure2_performance",
    "compose_figure3_landscape",
    "compose_figure4_module_response",
    "compose_mosaic",
]

__version__ = "0.1.0"

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from . import colors as _colors
from .sizes import get_figsize
from .typography import scaled_fonts


def new_figure(
    figure_type: str = "main_double",
    *,
    nrows: int = 1,
    ncols: int = 1,
    constrained_layout: bool = True,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Optional[Sequence[float]] = None,
    width_ratios: Optional[Sequence[float]] = None,
    **subplots_kwargs,
) -> Tuple[Figure, "object"]:
    """Create a figure with the canonical size for ``figure_type``."""
    figsize = get_figsize(figure_type)
    gridspec_kw = {}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = list(height_ratios)
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = list(width_ratios)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        constrained_layout=constrained_layout,
        gridspec_kw=gridspec_kw or None,
        **subplots_kwargs,
    )
    return fig, axes


def add_panel_label(
    ax: Axes,
    label: str,
    *,
    figure_type: str = "main_double",
    x: float = -0.16,
    y: float = 1.04,
    weight: str = "normal",
    ha: str = "left",
    va: str = "bottom",
) -> None:
    """Add an ``A``/``B``/... panel label to the top-left of ``ax``."""
    fonts = scaled_fonts(figure_type)
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fonts["panel_label"],
        fontweight=weight,
        ha=ha,
        va=va,
    )


def style_axes(
    ax: Axes,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    despine: bool = True,
    grid: bool = False,
) -> None:
    """One-call axis styling: labels, limits, optional grid, despine."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if despine:
        clean_spines(ax)
    if grid:
        apply_grid(ax)


def clean_spines(ax: Axes, *, keep: Iterable[str] = ("left", "bottom")) -> None:
    """Hide the spines other than those listed in ``keep``."""
    keep_set = set(keep)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep_set)


def apply_grid(
    ax: Axes,
    *,
    axis: str = "y",
    color: str = _colors.GRID_GREY,
    linewidth: float = 0.4,
    linestyle: str = "-",
    alpha: float = 0.7,
    zorder: int = 0,
) -> None:
    """Add a subtle grid (default: horizontal lines only, behind data)."""
    ax.grid(
        True,
        axis=axis,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.set_axisbelow(True)


def maybe_rotate_xticks(
    ax: Axes,
    *,
    angle: float = 30.0,
    ha: str = "right",
    threshold: int = 6,
) -> None:
    """Rotate x-tick labels if there are more than ``threshold`` of them."""
    labels = ax.get_xticklabels()
    if len(labels) > threshold:
        for lbl in labels:
            lbl.set_rotation(angle)
            lbl.set_ha(ha)


def add_zero_line(
    ax: Axes,
    *,
    axis: str = "y",
    color: str = _colors.NEUTRAL_REFERENCE,
    linewidth: float = 0.6,
    linestyle: str = "--",
    alpha: float = 0.8,
    zorder: int = 1,
) -> None:

    if axis == "y":
        ax.axhline(
            0.0,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
    elif axis == "x":
        ax.axvline(
            0.0,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def add_reference_line(
    ax: Axes,
    value: float,
    *,
    axis: str = "y",
    color: str = _colors.ANNOT_GREY,
    linewidth: float = 0.6,
    linestyle: str = ":",
    alpha: float = 0.8,
    label: Optional[str] = None,
    zorder: int = 1,
) -> None:

    fn = ax.axhline if axis == "y" else ax.axvline
    fn(
        value,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )


def compact_legend(
    ax: Axes,
    *,
    loc: str = "best",
    ncol: int = 1,
    title: Optional[str] = None,
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
) -> None:
    """Create a legend from labeled entries."""
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
    if not keep:
        return
    h, l = zip(*keep)
    ax.legend(
        h,
        l,
        loc=loc,
        ncol=ncol,
        title=title,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
    )


_CORNER_PRESETS = {
    "bottom-left": (0.03, 0.04, "left", "bottom"),
    "bottom-right": (0.97, 0.04, "right", "bottom"),
    "top-left": (0.03, 0.96, "left", "top"),
    "top-right": (0.97, 0.96, "right", "top"),
}


def _resolve_bbox(
    target: Union[SubplotSpec, Axes, Tuple[float, float, float, float]],
    fig: Optional[Figure] = None,
) -> Bbox:
    """Resolve a SubplotSpec / Axes / 4-tuple to a figure-coord ``Bbox``."""
    if isinstance(target, SubplotSpec):
        if fig is None:
            raise ValueError("fig is required when target is a SubplotSpec")
        return target.get_position(fig)
    if isinstance(target, Axes):
        return target.get_position()
    if isinstance(target, tuple) and len(target) == 4:
        return Bbox.from_extents(*target)
    raise TypeError(
        f"target must be SubplotSpec / Axes / (x0,y0,x1,y1); got {type(target)!r}"
    )


def add_section_label(
    fig: Figure,
    target: Union[SubplotSpec, Axes, Tuple[float, float, float, float]],
    label: str,
    *,
    fontsize: float = 14.0,
    fontweight: str = "bold",
    x_offset: float = -0.005,
    y_offset: float = 0.012,
    color: str = "#111111",
) -> None:

    bbox = _resolve_bbox(target, fig)
    fig.text(
        bbox.x0 + x_offset,
        bbox.y1 + y_offset,
        label,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha="left",
        va="bottom",
    )


def add_subpanel_label(
    ax: Axes,
    label: str,
    *,
    position: str = "bottom-left",
    fontsize: float = 8.0,
    fontweight: str = "bold",
    style: str = "parens",
    color: str = "#222222",
    pad: Optional[float] = None,
) -> None:
    """Place a SUB-PANEL letter (``a``, ``b``, ...) inside a single axes."""
    if position not in _CORNER_PRESETS:
        raise ValueError(
            f"position must be one of {list(_CORNER_PRESETS)}, got {position!r}"
        )
    x, y, ha, va = _CORNER_PRESETS[position]
    text = f"({label})" if style == "parens" else str(label)
    bbox_kw = None
    if pad is not None:
        bbox_kw = dict(
            boxstyle=f"round,pad={pad}",
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        )
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha=ha,
        va=va,
        bbox=bbox_kw,
    )


_MetadataRow = Union[str, Tuple[str, ...]]


def _format_metadata_line(row: _MetadataRow, italicize_emphasis: bool) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, tuple):
        return str(row)
    if len(row) == 2:
        key, val = row
        return f"{key}: {val}"
    if len(row) == 3:
        key, val, emph = row
        if not emph:
            return f"{key}: {val}"
        if italicize_emphasis:
            return f"{key}: {val}  $\\mathit{{{emph}}}$"
        return f"{key}: {val}  {emph}"
    raise ValueError(f"row must be str / 2-tuple / 3-tuple, got {row!r}")


def add_metadata_box(
    ax: Axes,
    rows: Sequence[_MetadataRow],
    *,
    position: str = "bottom-left",
    fontsize: float = 8.5,
    italicize_emphasis: bool = True,
    facecolor: str = "white",
    edgecolor: str = "#222222",
    linewidth: float = 0.6,
    pad: float = 0.40,
    line_spacing: float = 1.20,
    color: str = "#111111",
) -> None:

    if position not in _CORNER_PRESETS:
        raise ValueError(
            f"position must be one of {list(_CORNER_PRESETS)}, got {position!r}"
        )
    text = "\n".join(_format_metadata_line(r, italicize_emphasis) for r in rows)
    x, y, ha, va = _CORNER_PRESETS[position]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        linespacing=line_spacing,
        bbox=dict(
            boxstyle=f"round,pad={pad}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        ),
    )


def add_dashed_border(
    fig: Figure,
    target: Union[SubplotSpec, Axes, Tuple[float, float, float, float]],
    *,
    color: str = "#3a3a3a",
    linewidth: float = 1.0,
    linestyle: Union[str, Tuple] = (0, (4, 3)),
    pad: float = 0.005,
    zorder: float = -1.0,
) -> None:

    bbox = _resolve_bbox(target, fig)
    rect = Rectangle(
        (bbox.x0 - pad, bbox.y0 - pad),
        bbox.width + 2.0 * pad,
        bbox.height + 2.0 * pad,
        transform=fig.transFigure,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    fig.add_artist(rect)

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .helpers import add_panel_label
from .sizes import get_figsize


PanelDict = Dict[str, Axes]


def _label_panels(
    panels: PanelDict,
    figure_type: str,
    *,
    x: float = -0.18,
    y: float = 1.05,
) -> None:
    for label, ax in panels.items():
        add_panel_label(ax, label, figure_type=figure_type, x=x, y=y)


def compose_figure1_workflow(
    *,
    figure_type: str = "main_double",
    panel_labels: bool = True,
    constrained_layout: bool = True,
) -> Tuple[Figure, PanelDict]:
    """Frame for Figure 1: 2 rows × 3 cols."""
    figsize = get_figsize(figure_type)
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2])

    panels: PanelDict = {
        "A": fig.add_subplot(gs[0, :]),
        "B": fig.add_subplot(gs[1, 0]),
        "C": fig.add_subplot(gs[1, 1]),
        "D": fig.add_subplot(gs[1, 2]),
    }
    if panel_labels:
        _label_panels(panels, figure_type)
    return fig, panels


def compose_figure2_performance(
    *,
    figure_type: str = "main_double",
    panel_labels: bool = True,
    constrained_layout: bool = True,
) -> Tuple[Figure, PanelDict]:
    """Frame for Figure 2: 2 × 2 grid for performance metrics."""
    figsize = get_figsize(figure_type)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        constrained_layout=constrained_layout,
    )
    panels: PanelDict = {
        "A": axes[0, 0],
        "B": axes[0, 1],
        "C": axes[1, 0],
        "D": axes[1, 1],
    }
    if panel_labels:
        _label_panels(panels, figure_type)
    return fig, panels


def compose_figure3_landscape(
    *,
    figure_type: str = "main_double",
    panel_labels: bool = True,
    constrained_layout: bool = True,
) -> Tuple[Figure, PanelDict]:
    """Frame for Figure 3: large landscape panel + 2-3 quantitative panels."""
    figsize = get_figsize(figure_type)
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.0, 1.0])

    panels: PanelDict = {
        "A": fig.add_subplot(gs[:, 0]),
        "B": fig.add_subplot(gs[0, 1:]),
        "C": fig.add_subplot(gs[1, 1:]),
    }
    if panel_labels:
        _label_panels(panels, figure_type)
    return fig, panels


def compose_figure4_module_response(
    *,
    figure_type: str = "main_multi",
    panel_labels: bool = True,
    constrained_layout: bool = True,
) -> Tuple[Figure, PanelDict]:
    """Frame for Figure 4: 3-row module response."""
    figsize = get_figsize(figure_type)
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs = fig.add_gridspec(3, 2)

    panels: PanelDict = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
        "E": fig.add_subplot(gs[2, :]),
    }
    if panel_labels:
        _label_panels(panels, figure_type)
    return fig, panels


def compose_mosaic(
    mosaic: str,
    *,
    figure_type: str = "main_double",
    panel_labels: bool = True,
    constrained_layout: bool = True,
    label_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Figure, PanelDict]:
    """Compose an arbitrary panel mosaic from a string spec."""
    figsize = get_figsize(figure_type)
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    panels = fig.subplot_mosaic(mosaic)

    if panel_labels:
        labelled: PanelDict = {}
        for key, ax in panels.items():
            label = (label_overrides or {}).get(key, key)
            labelled[label] = ax
        _label_panels(labelled, figure_type)

    return fig, panels

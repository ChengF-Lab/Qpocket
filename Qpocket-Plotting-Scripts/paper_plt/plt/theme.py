from __future__ import annotations

from typing import Dict

import matplotlib as mpl
from matplotlib import font_manager

from . import config as _cfg
from .typography import scaled_fonts


def _available_fonts(candidates):
    """Filter ``candidates`` down to font families actually installed."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    out = [c for c in candidates if c in installed]
    if "DejaVu Sans" not in out:
        out.append("DejaVu Sans")
    return out


def _line_widths(scale: float) -> Dict[str, float]:

    return {
        "axes": round(0.8 * scale, 3),
        "tick": round(0.6 * scale, 3),
        "line": round(1.2 * scale, 3),
        "marker_edge": round(0.5 * scale, 3),
    }


def _build_rc(figure_type: str) -> Dict:
    fonts = scaled_fonts(figure_type)
    from .typography import scale_for

    scale = scale_for(figure_type)
    lw = _line_widths(scale)

    rc: Dict = {
        "font.family": _available_fonts(_cfg.DEFAULT_FONT_FAMILY),
        "font.size": fonts["tick_label"],
        "mathtext.fontset": _cfg.DEFAULT_MATHTEXT_FONTSET,
        "pdf.fonttype": 42 if _cfg.EMBED_FONTS_IN_PDF else 3,
        "ps.fonttype": 42 if _cfg.EMBED_FONTS_IN_PDF else 3,
        "svg.fonttype": "none",
        "axes.titlesize": fonts["title"],
        "axes.labelsize": fonts["axis_label"],
        "axes.linewidth": lw["axes"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.titleweight": "normal",
        "axes.labelpad": 2.5,
        "xtick.labelsize": fonts["tick_label"],
        "ytick.labelsize": fonts["tick_label"],
        "xtick.major.width": lw["tick"],
        "ytick.major.width": lw["tick"],
        "xtick.minor.width": lw["tick"] * 0.7,
        "ytick.minor.width": lw["tick"] * 0.7,
        "xtick.major.size": 3.0 * scale,
        "ytick.major.size": 3.0 * scale,
        "xtick.minor.size": 1.5 * scale,
        "ytick.minor.size": 1.5 * scale,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": fonts["legend"],
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "lines.linewidth": lw["line"],
        "lines.markersize": 3.5 * scale,
        "lines.markeredgewidth": lw["marker_edge"],
        "patch.linewidth": lw["axes"],
        "figure.dpi": 100,
        "figure.facecolor": _cfg.DEFAULT_FACECOLOR,
        "savefig.dpi": _cfg.DEFAULT_DPI,
        "savefig.facecolor": _cfg.DEFAULT_FACECOLOR,
        "savefig.bbox": _cfg.DEFAULT_BBOX_INCHES,
        "savefig.pad_inches": _cfg.DEFAULT_BBOX_PAD_INCHES,
        "savefig.transparent": False,
    }
    return rc


def get_theme(figure_type: str = "main_double") -> Dict:
    """Return the rcParams dict that ``apply_theme`` would install."""
    return _build_rc(figure_type)


def apply_theme(figure_type: str = "main_double") -> Dict[str, float]:
    """Install the project rcParams and return the scaled font hierarchy."""
    rc = _build_rc(figure_type)
    mpl.rcParams.update(rc)
    return scaled_fonts(figure_type)

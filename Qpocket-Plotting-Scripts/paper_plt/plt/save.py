from __future__ import annotations

import os
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import config as _cfg


def resolve_output_path(
    out_path: str,
    root: Union[str, bool, None] = None,
) -> str:
    """Resolve a relative output path against the figure directory."""
    if os.path.isabs(out_path):
        return out_path
    if root is False:
        return out_path

    base_root = root if isinstance(root, str) else _cfg.DEFAULT_OUTPUT_ROOT

    rel = out_path
    if rel.startswith("./"):
        rel = rel[2:]

    root_basename = os.path.basename(base_root.rstrip(os.sep))
    parts = rel.split(os.sep, 1)
    if root_basename and parts and parts[0] == root_basename:
        rel = parts[1] if len(parts) > 1 else ""

    return os.path.join(base_root, rel) if rel else base_root


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if missing."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _strip_ext(path: str) -> str:
    base, _ = os.path.splitext(path)
    return base


def save_figure(
    fig: Figure,
    out_path: str,
    *,
    formats: Optional[Iterable[str]] = None,
    dpi: Optional[int] = None,
    bbox_inches: Optional[str] = None,
    pad_inches: Optional[float] = None,
    facecolor: Optional[str] = None,
    transparent: bool = False,
    close: bool = True,
    root: Union[str, bool, None] = None,
    preview_png: bool = True,
    preview_dpi: int = 300,
) -> list:
    """Save ``fig`` to one or more formats."""
    fmts = tuple(formats) if formats is not None else _cfg.EXPORT_FORMATS
    dpi = dpi if dpi is not None else _cfg.DEFAULT_DPI
    bbox = bbox_inches if bbox_inches is not None else _cfg.DEFAULT_BBOX_INCHES
    pad = pad_inches if pad_inches is not None else _cfg.DEFAULT_BBOX_PAD_INCHES
    fc = facecolor if facecolor is not None else _cfg.DEFAULT_FACECOLOR

    rooted = resolve_output_path(out_path, root=root)
    base = _strip_ext(rooted)
    ensure_parent_dir(base + ".tmp")

    written: list = []
    fmt_set_lower = {f.lstrip(".").lower() for f in fmts}

    for fmt in fmts:
        ext = fmt.lstrip(".").lower()
        target = f"{base}.{ext}"
        fig.savefig(
            target,
            dpi=dpi,
            bbox_inches=bbox,
            pad_inches=pad,
            facecolor=fc,
            transparent=transparent,
        )
        written.append(target)

    if _cfg.EXPORT_SVG_BY_DEFAULT and "svg" not in fmt_set_lower:
        target = f"{base}.svg"
        fig.savefig(
            target,
            dpi=dpi,
            bbox_inches=bbox,
            pad_inches=pad,
            facecolor=fc,
            transparent=transparent,
        )
        written.append(target)
        fmt_set_lower.add("svg")

    if preview_png and "png" not in fmt_set_lower:
        preview_target = f"{base}.png"
        fig.savefig(
            preview_target,
            dpi=preview_dpi,
            bbox_inches=bbox,
            pad_inches=pad,
            facecolor=fc,
            transparent=transparent,
        )
        written.append(preview_target)

    if close:
        plt.close(fig)

    return written


def maybe_save_svg(
    fig: Figure,
    out_path: str,
    root: Union[str, bool, None] = None,
    **save_kwargs,
) -> Optional[str]:
    """Save an SVG copy regardless of ``EXPORT_SVG_BY_DEFAULT``."""
    rooted = resolve_output_path(out_path, root=root)
    base = _strip_ext(rooted)
    target = base + ".svg"
    ensure_parent_dir(target)
    fig.savefig(
        target,
        dpi=save_kwargs.get("dpi", _cfg.DEFAULT_DPI),
        bbox_inches=save_kwargs.get("bbox_inches", _cfg.DEFAULT_BBOX_INCHES),
        pad_inches=save_kwargs.get("pad_inches", _cfg.DEFAULT_BBOX_PAD_INCHES),
        facecolor=save_kwargs.get("facecolor", _cfg.DEFAULT_FACECOLOR),
        transparent=save_kwargs.get("transparent", False),
    )
    return target

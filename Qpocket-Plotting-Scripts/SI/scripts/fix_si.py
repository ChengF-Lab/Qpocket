"""Render SI galleries using the rectangular contour renderer."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def log(message):
    print(message)


def render():
    import make_si_landscapes as M
    from landscape.config import LandscapeConfig
    from landscape.builder import LandscapeBuilder, _estimate_basins
    from landscape.visualizer import LandscapePlotter

    cfg = LandscapeConfig()
    builder = LandscapeBuilder(cfg)
    plotter = LandscapePlotter(cfg)
    cmap = mpl.colormaps[cfg.colormap]
    SYS_ORDER, FRAG_ORDER = M.SYS_ORDER, M.FRAG_ORDER
    RUN, OUT, OUTPC = M.RUN_V2_DIR, M.OUT_FIG, M.OUT_PERCASE
    OUT.mkdir(parents=True, exist_ok=True)
    OUTPC.mkdir(parents=True, exist_ok=True)

    cases = sorted(
        [
            p
            for p in RUN.iterdir()
            if p.is_dir() and "_Frag-" in p.name and not p.name.endswith("_Frag-B")
        ]
    )
    by_frag = defaultdict(dict)
    for p in cases:
        s, v, f = M.parse(p.name)
        by_frag[f][s] = (p, v)

    cache = {}

    def get(cdir):
        if cdir.name in cache:
            return cache[cdir.name]
        rng = np.random.default_rng(M.RNG_SEED)
        try:
            data, extras, _ = M.build_landscape_data(
                cdir, cfg, builder, _estimate_basins, rng
            )
        except Exception as e:
            log("Could not build the landscape for %s: %r" % (cdir.name, e))
            data = None
        cache[cdir.name] = data
        return data

    def panel(ax, d, title=None, fs=0.85, title_fs=11):

        energies = d.energies
        e_min = float(energies.min())
        e_max_display = float(np.percentile(energies, cfg.energy_percentile_clip))
        norm = mcolors.Normalize(vmin=e_min, vmax=e_max_display)
        plotter._panel_2d_contour(ax, d, norm, cmap, e_max_display, fs)
        ax.set_title(title or "", fontsize=title_fs, pad=6)

    for p in cases:
        get(p)

    for frag in FRAG_ORDER:
        items = [(s, by_frag[frag][s]) for s in SYS_ORDER if s in by_frag.get(frag, {})]
        if not items:
            continue
        n = len(items)
        ncol = min(4, n)
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(4.5 * ncol, 4.1 * nrow),
            squeeze=False,
            constrained_layout=True,
        )
        for ax in axes.flat:
            ax.set_axis_off()
        ok = 0
        for k, (s, (cdir, v)) in enumerate(items):
            ax = axes[k // ncol][k % ncol]
            ax.set_axis_on()
            d = get(cdir)
            if d is None:
                ax.text(
                    0.5,
                    0.5,
                    "%s %s\n(no data)" % (s, v),
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_axis_off()
                continue
            panel(ax, d, title="%s %s" % (s, v), fs=0.85)
            ok += 1
        stem = OUT / ("figS_landscape_Frag-%s" % frag)
        fig.savefig("%s.png" % stem, dpi=200, bbox_inches="tight")
        fig.savefig("%s.pdf" % stem, bbox_inches="tight")
        plt.close(fig)
        log("Wrote %s.{png,pdf} (%d/%d)" % (stem.name, ok, n))

    rows = [s for s in SYS_ORDER if any(s in by_frag.get(f, {}) for f in FRAG_ORDER)]
    ncol = len(FRAG_ORDER)
    nrow = len(rows)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(4.0 * ncol, 3.7 * nrow),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_axis_off()
    filled = 0
    for r, s in enumerate(rows):
        for c, frag in enumerate(FRAG_ORDER):
            ent = by_frag.get(frag, {}).get(s)
            if ent is None:
                continue
            cdir, v = ent
            d = get(cdir)
            ax = axes[r][c]
            ax.set_axis_on()
            if d is None:
                ax.set_axis_off()
                continue
            panel(ax, d, title="%s %s · Frag-%s" % (s, v, frag), fs=0.72, title_fs=10)
            filled += 1
    stem = OUT / "figS_landscape_all_remaining_master"
    fig.savefig("%s.png" % stem, dpi=160, bbox_inches="tight")
    fig.savefig("%s.pdf" % stem, bbox_inches="tight")
    plt.close(fig)
    log("Wrote %s.{png,pdf} (%d panels)" % (stem.name, filled))

    nc = 0
    for p in cases:
        d = cache.get(p.name)
        if d is None:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
        panel(ax, d, title=p.name.replace("_", " "), fs=0.95)
        fig.savefig(
            OUTPC / ("landscape2d_%s.png" % p.name), dpi=200, bbox_inches="tight"
        )
        fig.savefig(OUTPC / ("landscape2d_%s.pdf" % p.name), bbox_inches="tight")
        plt.close(fig)
        nc += 1
    log("Wrote %d per-case PNG/PDF" % nc)


if __name__ == "__main__":
    render()

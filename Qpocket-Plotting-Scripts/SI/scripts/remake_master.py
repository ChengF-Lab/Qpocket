"""Render the SI overview matrix from per-case coordinates."""

import sys
import math
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_si_landscapes as M
from landscape.config import LandscapeConfig
from landscape.builder import LandscapeBuilder, _estimate_basins
from landscape.visualizer import LandscapePlotter


def main():
    cfg = LandscapeConfig()
    builder = LandscapeBuilder(cfg)
    plotter = LandscapePlotter(cfg)

    all_cases = sorted(
        [
            p
            for p in M.RUN_V2_DIR.iterdir()
            if p.is_dir() and "_Frag-" in p.name and not p.name.endswith("_Frag-B")
        ]
    )
    by_frag = defaultdict(dict)
    for p in all_cases:
        sysid, var, frag = M.parse(p.name)
        by_frag[frag][sysid] = (p, var)

    rows = [
        s for s in M.SYS_ORDER if any(s in by_frag.get(f, {}) for f in M.FRAG_ORDER)
    ]
    ncol = len(M.FRAG_ORDER)
    nrow = len(rows)
    print(f"Overview layout: {nrow} system rows by {ncol} fragment columns")

    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.4 * ncol, 3.0 * nrow),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_axis_off()

    filled = 0
    for r, sysid in enumerate(rows):
        for c, frag in enumerate(M.FRAG_ORDER):
            entry = by_frag.get(frag, {}).get(sysid)
            if entry is None:
                continue
            cdir, var = entry
            rng = np.random.default_rng(M.RNG_SEED)
            try:
                data, extras, _ = M.build_landscape_data(
                    cdir, cfg, builder, _estimate_basins, rng
                )
            except Exception as e:
                print(f"Could not build the landscape for {cdir.name}: {e!r}")
                data = None
            ax = axes[r][c]
            ax.set_axis_on()
            if data is None:
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_axis_off()
                continue
            try:
                M.draw_2d(ax, data, cfg, plotter, fs=0.6)
                ax.set_title(f"{sysid} {var}  ·  Frag-{frag}", fontsize=8.5, pad=3)
                filled += 1
            except Exception as e:
                print(f"Could not render the overview panel for {cdir.name}: {e!r}")

    stem = M.OUT_FIG / "figS_landscape_all_remaining_master"
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.name}.{{png,pdf}}  ({filled} panels)")


if __name__ == "__main__":
    main()

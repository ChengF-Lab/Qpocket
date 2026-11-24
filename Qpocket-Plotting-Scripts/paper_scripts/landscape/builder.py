from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LandscapeConfig
from .data_types import LandscapeData, SampledConformation


_LOG = logging.getLogger(__name__)


def _estimate_basins(
    pca_coords: np.ndarray,
    energies: np.ndarray,
    n_grid: int = 20,
    min_cells: int = 4,
    min_samples: int = 5,
) -> int:

    from scipy.ndimage import label as sp_label

    x, y = pca_coords[:, 0], pca_coords[:, 1]
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())

    if x_range < 1e-6 or y_range < 1e-6:
        return 1

    grid = np.full((n_grid, n_grid), np.inf)
    xi = np.clip(((x - x.min()) / x_range * (n_grid - 1)).astype(int), 0, n_grid - 1)
    yi = np.clip(((y - y.min()) / y_range * (n_grid - 1)).astype(int), 0, n_grid - 1)
    for k in range(len(energies)):
        if energies[k] < grid[xi[k], yi[k]]:
            grid[xi[k], yi[k]] = energies[k]

    filled_mask = grid < np.inf
    if not filled_mask.any():
        return 1

    threshold = float(np.percentile(grid[filled_mask], 35))
    low_mask = (grid <= threshold) & filled_mask
    labeled, n_features = sp_label(low_mask)
    if n_features == 0:
        return 1

    sample_label = labeled[xi, yi]

    n_valid = 0
    for lab in range(1, int(n_features) + 1):
        cell_count = int(np.sum(labeled == lab))
        if cell_count < min_cells:
            continue
        sample_count = int(np.sum(sample_label == lab))
        if sample_count < min_samples:
            continue
        n_valid += 1

    return max(1, n_valid)


class LandscapeBuilder:

    def __init__(self, cfg: Optional[LandscapeConfig] = None) -> None:
        self.cfg = cfg or LandscapeConfig()

    def _interpolate_surface(
        self,
        pca_coords: np.ndarray,
        energies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a smooth energy surface over the 2-D PCA space."""
        e_clip_val = float(np.percentile(energies, self.cfg.energy_percentile_clip))
        e_plot = np.clip(energies, None, e_clip_val)

        x = pca_coords[:, 0]
        y = pca_coords[:, 1]

        x_range = float(x.max() - x.min())
        y_range = float(y.max() - y.min())
        margin_x = max(0.08 * x_range, 1e-3)
        margin_y = max(0.08 * y_range, 1e-3)

        G = self.cfg.grid_resolution
        xi = np.linspace(x.min() - margin_x, x.max() + margin_x, G)
        yi = np.linspace(y.min() - margin_y, y.max() + margin_y, G)
        GX, GY = np.meshgrid(xi, yi)
        grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

        try:
            from scipy.interpolate import RBFInterpolator

            smoothing = self.cfg.rbf_smoothing_factor * len(pca_coords)
            rbf = RBFInterpolator(
                pca_coords,
                e_plot,
                kernel=self.cfg.rbf_kernel,
                smoothing=smoothing,
            )
            GZ = rbf(grid_pts).reshape(GX.shape)
        except Exception as exc:
            _LOG.debug(f"RBFInterpolator failed ({exc}), falling back to griddata.")
            from scipy.interpolate import griddata

            GZ = griddata(
                pca_coords,
                e_plot,
                (GX, GY),
                method="linear",
                fill_value=float(e_clip_val),
            )

        return GX, GY, GZ

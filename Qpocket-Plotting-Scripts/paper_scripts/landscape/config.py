from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LandscapeConfig:
    """Configuration for protein energy landscape analysis and visualization."""

    fifth_bit: bool = False
    strict_decode: bool = False

    max_unique_conformations: int = 3000
    min_unique_conformations: int = 40
    overlap_eps: float = 0.15

    scorer: str = "lattice_contact"

    n_pca_components: int = 2
    scale_backbone: bool = True

    min_basin_cells: int = 4
    min_basin_samples: int = 5

    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing_factor: float = 0.0
    grid_resolution: int = 60
    energy_percentile_clip: float = 93.0

    figsize: Tuple[float, float] = (24, 6.6)
    dpi: int = 600
    font_scale: float = 1.0
    colormap: str = "RdYlBu_r"
    surface_alpha: float = 0.62
    scatter_size: float = 18.0
    scatter_alpha: float = 0.72
    top_k_annotate: int = 3

    output_dir: str = "landscape_result"
    save_data: bool = True
    save_plots: bool = True

"""Render ligand-resolved absolute and differential landscape grids."""

from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    TwoSlopeNorm,
)
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


LIGAND_HET_TO_NAME: Dict[str, str] = {
    "F0K": "BI-2852",
    "6IC": "MRTX1133",
    "7L8": "Compound 15",
    "7IZ": "Compound 5B",
    "EZZ": "Compound 18",
    "OFU": "BI-2865",
}
LIGAND_PANEL_B_ORDER: List[str] = ["F0K", "6IC", "7L8", "7IZ", "OFU"]
FRAGMENTS: List[str] = ["Frag-B", "Frag-F", "Frag-A"]

LIGAND_POCKET_CLASS: Dict[str, str] = {
    "F0K": "outlier",
    "6IC": "S-IIP",
    "7L8": "S-IIP",
    "7IZ": "S-IIP",
    "OFU": "S-IIP",
    "EZZ": "outlier",
}
S_IIP_LIGANDS = [h for h, c in LIGAND_POCKET_CLASS.items() if c == "S-IIP"]

GRID_RESOLUTION = 60
ENERGY_PCT_CLIP = 93.0
RBF_KERNEL = "thin_plate_spline"
RBF_SMOOTHING_FACTOR = 0.3
ABSOLUTE_CMAP = "RdYlBu_r"
RBF_MARGIN_FRAC = 0.08

DIFF_BINS = 60
DIFF_SMOOTH_SIGMA = 1.5

MIN_SAMPLES_PER_CELL = 50

PERCENTILE_FOR_VMAX = 50.0

DATA_TBL_DIR = Path(
    os.environ.get(
        "QPOCKET_GRID_TABLES", PROJECT_ROOT / "result" / "landscape_2000" / "tables"
    )
)
DATA_TBL_DIR_ABS = DATA_TBL_DIR


def load_projected_samples(fragment: str) -> pd.DataFrame:

    return pd.read_csv(
        DATA_TBL_DIR / f"projected_samples_{fragment}.csv", low_memory=False
    ).dropna(subset=["PC1", "PC2", "score"])


def load_projected_samples_abs(fragment: str) -> pd.DataFrame:

    return pd.read_csv(
        DATA_TBL_DIR_ABS / f"projected_samples_{fragment}.csv", low_memory=False
    ).dropna(subset=["PC1", "PC2", "score"])


def load_basin_centers(fragment: str) -> List[Tuple[str, float, float]]:
    path = DATA_TBL_DIR / f"basin_centers_{fragment}.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return [
        (str(row["basin_id"]), float(row["center_PC1"]), float(row["center_PC2"]))
        for _, row in df.iterrows()
    ]


BASIN_IDS = ["B1", "B2", "B3", "B4", "B5"]


def basin_occupancy(df: pd.DataFrame) -> np.ndarray:
    """``P_basin`` vector of length 5 from a sample-level dataframe."""
    counts = df["basin_id"].value_counts(normalize=True)
    return counts.reindex(BASIN_IDS, fill_value=0.0).to_numpy()


def basin_max_abs_delta(
    df_lig: pd.DataFrame,
    df_all_frag: pd.DataFrame,
) -> float:

    p_lig = basin_occupancy(df_lig)
    p_all = basin_occupancy(df_all_frag)
    return float(np.max(np.abs(p_lig - p_all)))


ABS_SMOOTH_SIGMA = 0.0
ABS_RBF_SUBSAMPLE = 0
ABS_VMAX_HEADROOM = 1.00
ABS_CELL_COMPRESS = 1.00


def interpolate_energy_surface(
    df: pd.DataFrame,
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    _ = extent
    pc = df[["PC1", "PC2"]].to_numpy(dtype=float)
    e = df["score"].to_numpy(dtype=float)

    e_clip_val = float(np.percentile(e, ENERGY_PCT_CLIP))
    e_plot = np.clip(e, None, e_clip_val)

    x, y = pc[:, 0], pc[:, 1]
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())
    margin_x = max(0.08 * x_range, 1e-3)
    margin_y = max(0.08 * y_range, 1e-3)
    xi = np.linspace(x.min() - margin_x, x.max() + margin_x, GRID_RESOLUTION)
    yi = np.linspace(y.min() - margin_y, y.max() + margin_y, GRID_RESOLUTION)
    GX, GY = np.meshgrid(xi, yi)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    try:
        from scipy.interpolate import RBFInterpolator

        rbf = RBFInterpolator(
            pc,
            e_plot,
            kernel=RBF_KERNEL,
            smoothing=RBF_SMOOTHING_FACTOR * len(pc),
        )
        GZ = rbf(grid_pts).reshape(GX.shape)
    except Exception:
        from scipy.interpolate import griddata

        GZ = griddata(pc, e_plot, (GX, GY), method="linear", fill_value=e_clip_val)
    return GX, GY, GZ, e_clip_val


ABS_CMAP_PALE = mpl.colormaps[ABSOLUTE_CMAP]

ABS_CMAP_BY_FRAG: Dict[str, str] = {
    "Frag-B": ABSOLUTE_CMAP,
    "Frag-F": ABSOLUTE_CMAP,
    "Frag-A": ABSOLUTE_CMAP,
}
_ABS_CUSTOM_ANCHORS: Dict[str, List[str]] = {}

ABS_LEVELS_N = 28
ABS_CONTOURF_ALPHA = 0.92
ABS_CONTOUR_EVERY = 4
ABS_CONTOUR_COLOR = "#555555"
ABS_CONTOUR_LW = 0.5
ABS_CONTOUR_ALPHA = 0.35


def draw_absolute_cell(
    ax: plt.Axes,
    df_subset: pd.DataFrame,
    extent: Tuple[float, float, float, float],
    e_vmin: float,
    e_vmax: float,
    basin_centers: List[Tuple[str, float, float]],
    cmap_name: str = ABSOLUTE_CMAP,
    label_fontsize: float = 9,
) -> None:

    GX, GY, GZ, _ = interpolate_energy_surface(df_subset, extent)
    cmap = mpl.colormaps[cmap_name]
    norm = Normalize(vmin=e_vmin, vmax=e_vmax)
    GZ_clip = np.clip(GZ, None, e_vmax)
    levels = np.linspace(e_vmin, e_vmax, ABS_LEVELS_N)
    ax.contourf(
        GX, GY, GZ_clip, levels=levels, cmap=cmap, norm=norm, alpha=ABS_CONTOURF_ALPHA
    )
    ax.contour(
        GX,
        GY,
        GZ_clip,
        levels=levels[::ABS_CONTOUR_EVERY],
        colors=ABS_CONTOUR_COLOR,
        alpha=ABS_CONTOUR_ALPHA,
        linewidths=ABS_CONTOUR_LW,
    )
    _overlay_basins(ax, basin_centers, label_fontsize=label_fontsize)
    ax.set_xlim(float(GX.min()), float(GX.max()))
    ax.set_ylim(float(GY.min()), float(GY.max()))
    ax.tick_params(axis="both", labelsize=8)


def _density_2d(
    pc: np.ndarray,
    grid_extent: Tuple[float, float, float, float],
) -> np.ndarray:

    x0, x1, y0, y1 = grid_extent
    H, _, _ = np.histogram2d(
        pc[:, 0],
        pc[:, 1],
        bins=DIFF_BINS,
        range=[[x0, x1], [y0, y1]],
        density=True,
    )
    return H


BASIN_SIGMA_FRAC = 0.7

DETAIL_HIGHPASS_SIGMA = 6.0
DETAIL_ALPHA = 0.45


def _basin_neighbour_sigma(centers: np.ndarray) -> float:

    n = centers.shape[0]
    if n < 2:
        return 1.0
    diffs = centers[:, None, :] - centers[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn_dist = dist.min(axis=1)
    return float(BASIN_SIGMA_FRAC * np.mean(nn_dist))


def compute_diff(
    df_lig: pd.DataFrame,
    baseline_groups: List[pd.DataFrame],
    grid_extent: Tuple[float, float, float, float],
    basin_centers: List[Tuple[str, float, float]] = None,
) -> np.ndarray:

    if not baseline_groups:
        raise ValueError("compute_diff: baseline_groups is empty")
    if basin_centers is None or len(basin_centers) == 0:
        raise ValueError("compute_diff: basin_centers required")

    p_lig = basin_occupancy(df_lig)
    valid_groups = [g for g in baseline_groups if len(g) >= MIN_SAMPLES_PER_CELL]
    if not valid_groups:
        raise ValueError("compute_diff: no valid baseline groups")
    p_base = np.mean([basin_occupancy(g) for g in valid_groups], axis=0)
    delta_basin = p_lig - p_base

    centers_arr = np.array([[cx, cy] for (_, cx, cy) in basin_centers], dtype=float)
    bid_to_idx = {b: i for i, b in enumerate(BASIN_IDS)}
    delta_ordered = np.array(
        [
            delta_basin[bid_to_idx[bid]] if bid in bid_to_idx else 0.0
            for (bid, _cx, _cy) in basin_centers
        ],
        dtype=float,
    )

    x0, x1, y0, y1 = grid_extent
    xe = np.linspace(x0, x1, DIFF_BINS + 1)
    ye = np.linspace(y0, y1, DIFF_BINS + 1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    GX, GY = np.meshgrid(xc, yc)

    sigma_b = _basin_neighbour_sigma(centers_arr)
    d2 = (GX[None, :, :] - centers_arr[:, 0:1, None]) ** 2 + (
        GY[None, :, :] - centers_arr[:, 1:2, None]
    ) ** 2
    w = np.exp(-d2 / (2.0 * sigma_b**2))
    w_sum = w.sum(axis=0)
    basin_field = (w * delta_ordered[:, None, None]).sum(axis=0) / np.maximum(
        w_sum, 1e-12
    )

    H_l = _density_2d(df_lig[["PC1", "PC2"]].to_numpy(dtype=float), grid_extent)
    H_b_per = [
        _density_2d(g[["PC1", "PC2"]].to_numpy(dtype=float), grid_extent)
        for g in valid_groups
    ]
    H_b = np.mean(np.stack(H_b_per, axis=0), axis=0)
    pixel_diff = (H_l - H_b).T
    if DIFF_SMOOTH_SIGMA > 0:
        pixel_diff = gaussian_filter(pixel_diff, sigma=DIFF_SMOOTH_SIGMA)

    pixel_lp = gaussian_filter(pixel_diff, sigma=DETAIL_HIGHPASS_SIGMA)
    pixel_hp = pixel_diff - pixel_lp
    hp_peak = float(np.max(np.abs(pixel_hp)))
    ba_peak = float(np.max(np.abs(basin_field)))
    if hp_peak > 1e-12 and ba_peak > 1e-12:
        pixel_hp = pixel_hp * (DETAIL_ALPHA * ba_peak / hp_peak)
    else:
        pixel_hp = np.zeros_like(pixel_hp)

    field = basin_field + pixel_hp
    if DIFF_SMOOTH_SIGMA > 0:
        field = gaussian_filter(field, sigma=DIFF_SMOOTH_SIGMA * 0.5)
    return field


DIFF_COLOR_GAMMA = 0.55


def _gamma_transform(arr: np.ndarray, gamma: float) -> np.ndarray:

    return np.sign(arr) * np.power(np.abs(arr), gamma)


DIFF_CMAP_BY_CLASS: Dict[str, str] = {
    "S-IIP": "RdBu_r",
    "outlier": "PuOr_r",
}
DIFF_ALPHA_BY_CLASS: Dict[str, float] = {
    "S-IIP": 0.92,
    "outlier": 0.92,
}


def draw_diff_cell(
    ax: plt.Axes,
    diff: np.ndarray,
    vmax_backdrop: float,
    vmax_marker: float,
    basin_centers: List[Tuple[str, float, float]],
    basin_delta: np.ndarray,
    grid_extent: Tuple[float, float, float, float],
    pocket_class: str = "S-IIP",
    label_fontsize: float = 9,
) -> None:

    x0, x1, y0, y1 = grid_extent
    xe = np.linspace(x0, x1, DIFF_BINS + 1)
    ye = np.linspace(y0, y1, DIFF_BINS + 1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])

    vabs_raw = max(vmax_backdrop, 1e-12)
    gamma = DIFF_COLOR_GAMMA
    diff_v = np.sign(diff) * np.power(np.abs(diff), gamma)
    vabs = vabs_raw**gamma
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    levels = np.linspace(-vabs, vabs, 28)
    cmap_name = DIFF_CMAP_BY_CLASS.get(pocket_class, "RdBu_r")
    alpha = DIFF_ALPHA_BY_CLASS.get(pocket_class, 0.92)
    ax.contourf(
        xc,
        yc,
        diff_v,
        levels=levels,
        cmap=cmap_name,
        norm=norm,
        alpha=alpha,
        extend="both",
    )
    ax.contour(
        xc, yc, diff_v, levels=[0.0], colors="#333333", linewidths=0.6, alpha=0.55
    )
    _overlay_basins(ax, basin_centers, label_fontsize=label_fontsize)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.tick_params(axis="both", labelsize=8)


BASIN_MARKER_SIZE_MIN = 60.0
BASIN_MARKER_SIZE_MAX = 420.0


def _overlay_basins_with_delta(
    ax: plt.Axes,
    basin_centers: List[Tuple[str, float, float]],
    basin_delta: np.ndarray,
    vmax: float,
    cmap_name: str = "RdBu_r",
) -> None:

    cmap = mpl.colormaps[cmap_name]
    vmax_g = vmax**DIFF_COLOR_GAMMA
    norm = TwoSlopeNorm(vmin=-vmax_g, vcenter=0.0, vmax=vmax_g)
    bid_to_idx = {b: i for i, b in enumerate(BASIN_IDS)}

    for label, cx, cy in basin_centers:
        idx = bid_to_idx.get(label, 0)
        d = float(basin_delta[idx]) if idx < len(basin_delta) else 0.0
        d_g = np.sign(d) * (abs(d) ** DIFF_COLOR_GAMMA)
        face_rgba = cmap(norm(d_g))
        frac_lin = min(abs(d) / max(vmax, 1e-12), 1.0)
        size = (
            BASIN_MARKER_SIZE_MIN
            + (BASIN_MARKER_SIZE_MAX - BASIN_MARKER_SIZE_MIN) * frac_lin
        )
        ax.scatter(
            cx,
            cy,
            s=size,
            marker="X",
            c=[face_rgba],
            edgecolors="black",
            linewidths=1.3,
            zorder=10,
        )
        ax.annotate(
            label,
            (cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            color="black",
            zorder=11,
        )


def _overlay_basins(
    ax: plt.Axes,
    basin_centers: List[Tuple[str, float, float]],
    label_fontsize: float = 9,
) -> None:
    """Plain white-X basin overlay (used by the absolute row)."""
    for label, cx, cy in basin_centers:
        ax.scatter(
            cx,
            cy,
            s=110,
            marker="X",
            c="white",
            edgecolors="black",
            linewidths=1.1,
            zorder=10,
        )
        ax.annotate(
            label,
            (cx, cy),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=label_fontsize,
            weight="bold",
            color="black",
            zorder=11,
        )


def draw_na_cell(ax: plt.Axes) -> None:
    ax.text(
        0.5,
        0.5,
        "n/a",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=15,
        color="#aaaaaa",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#dddddd")


def _prepare_landscape_data() -> Tuple[
    Dict[str, Dict],
    Dict[Tuple[str, str], np.ndarray],
    Dict[str, float],
    float,
    float,
    List[Tuple[str, float]],
]:

    per_frag: Dict[str, Dict] = {}
    diff_cells: Dict[Tuple[str, str], np.ndarray] = {}
    basin_l1_per_cell: List[Tuple[str, float]] = []

    e_min_raw_global = +np.inf
    e_max_clip_global = -np.inf
    frag_mean_score: Dict[str, float] = {}
    abs_df_by_frag: Dict[str, pd.DataFrame] = {}
    all_score_arrays: List[np.ndarray] = []

    for frag in FRAGMENTS:
        df = load_projected_samples(frag)
        df_abs = load_projected_samples_abs(frag)
        if df.empty:
            continue

        if not df_abs.empty:
            e_arr = df_abs["score"].to_numpy(dtype=float)
            e_min_raw_global = min(e_min_raw_global, float(e_arr.min()))
            e_max_clip_global = max(
                e_max_clip_global,
                float(np.percentile(e_arr, ENERGY_PCT_CLIP)),
            )
            frag_mean_score[frag] = float(e_arr.mean())
            abs_df_by_frag[frag] = df_abs
            all_score_arrays.append(e_arr)

        x0 = float(df["PC1"].min())
        x1 = float(df["PC1"].max())
        y0 = float(df["PC2"].min())
        y1 = float(df["PC2"].max())
        extent = (x0, x1, y0, y1)

        per_frag[frag] = {
            "df": df,
            "df_abs": df_abs,
            "extent": extent,
            "basins": load_basin_centers(frag),
            "p_all_basin": basin_occupancy(df),
        }

        s_iip_groups_by_lig: Dict[str, pd.DataFrame] = {
            lig: df[df["ligand"] == lig] for lig in S_IIP_LIGANDS
        }
        for het in LIGAND_PANEL_B_ORDER:
            sub = df[df["ligand"] == het]
            if len(sub) < MIN_SAMPLES_PER_CELL:
                continue
            if LIGAND_POCKET_CLASS.get(het) == "outlier":
                baseline_groups = [
                    s_iip_groups_by_lig[lig]
                    for lig in S_IIP_LIGANDS
                    if len(s_iip_groups_by_lig[lig]) >= MIN_SAMPLES_PER_CELL
                ]
                cls = "outlier"
            else:
                baseline_groups = [
                    s_iip_groups_by_lig[lig]
                    for lig in S_IIP_LIGANDS
                    if lig != het
                    and len(s_iip_groups_by_lig[lig]) >= MIN_SAMPLES_PER_CELL
                ]
                cls = "S-IIP"
            if not baseline_groups:
                continue
            diff = compute_diff(
                sub,
                baseline_groups,
                extent,
                basin_centers=per_frag[frag]["basins"],
            )
            p_lig = basin_occupancy(sub)
            p_base = np.mean(
                [basin_occupancy(g) for g in baseline_groups],
                axis=0,
            )
            basin_l1 = float(np.max(np.abs(p_lig - p_base)))
            diff_cells[(frag, het)] = diff
            basin_l1_per_cell.append((cls, basin_l1))

    s_iip_vmaxes = [v for c, v in basin_l1_per_cell if c == "S-IIP"]
    outlier_vmaxes = [v for c, v in basin_l1_per_cell if c == "outlier"]
    cls_vmax = {
        "S-IIP": max(s_iip_vmaxes, default=1e-12),
        "outlier": max(outlier_vmaxes, default=1e-12),
    }

    e_vmin_global = e_min_raw_global
    e_vmax_global = e_max_clip_global
    if e_vmax_global - e_vmin_global < 1e-9:
        e_vmax_global = e_vmin_global + 1e-9

    return (
        per_frag,
        diff_cells,
        cls_vmax,
        e_vmin_global,
        e_vmax_global,
        basin_l1_per_cell,
    )


def render(out_dir: Path) -> List[Path]:

    (per_frag, diff_cells, cls_vmax, e_vmin_global, e_vmax_global, _) = (
        _prepare_landscape_data()
    )

    n_col = len(LIGAND_PANEL_B_ORDER)
    row_specs: List[Tuple[str, str]] = [
        ("Frag-B", "abs"),
        ("Frag-B", "diff"),
        ("Frag-F", "abs"),
        ("Frag-F", "diff"),
        ("Frag-A", "abs"),
        ("Frag-A", "diff"),
    ]
    n_row = len(row_specs)

    cell_w = 2.6
    cell_h = 1.82
    label_w = 1.6
    header_h = 0.55
    fig_w = label_w + cell_w * n_col + 0.4
    fig_h = header_h + cell_h * n_row + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(
        n_row + 1,
        n_col + 1,
        height_ratios=[header_h] + [cell_h] * n_row,
        width_ratios=[label_w] + [cell_w] * n_col,
        hspace=0.18,
        wspace=0.10,
        top=0.985,
        bottom=0.015,
        left=0.005,
        right=0.995,
    )

    for j, het in enumerate(LIGAND_PANEL_B_ORDER):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.axis("off")
        label = f"({chr(ord('a') + j)})  {LIGAND_HET_TO_NAME.get(het, het)}"
        ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=13, weight="bold")

    for i, (frag, mode) in enumerate(row_specs):
        ax_lbl = fig.add_subplot(gs[i + 1, 0])
        ax_lbl.axis("off")
        mode_lbl = "absolute" if mode == "abs" else "differential"
        ax_lbl.text(
            0.5,
            0.5,
            f"{frag}\n{mode_lbl}",
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="#222222",
        )

        d = per_frag.get(frag)
        if d is None:
            continue

        for j, het in enumerate(LIGAND_PANEL_B_ORDER):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            if mode == "abs":
                df_abs_frag = d.get("df_abs")
                if df_abs_frag is None or df_abs_frag.empty:
                    draw_na_cell(ax)
                    continue
                sub_abs = df_abs_frag[df_abs_frag["ligand"] == het]
                if len(sub_abs) < MIN_SAMPLES_PER_CELL:
                    draw_na_cell(ax)
                    continue
                draw_absolute_cell(
                    ax,
                    sub_abs,
                    extent=d["extent"],
                    e_vmin=e_vmin_global,
                    e_vmax=e_vmax_global,
                    basin_centers=d["basins"],
                    cmap_name=ABS_CMAP_BY_FRAG.get(frag, ABSOLUTE_CMAP),
                )
                continue
            sub = d["df"][d["df"]["ligand"] == het]
            if len(sub) < MIN_SAMPLES_PER_CELL:
                draw_na_cell(ax)
                continue
            else:
                diff = diff_cells.get((frag, het))
                if diff is None:
                    draw_na_cell(ax)
                    continue
                pcls = LIGAND_POCKET_CLASS.get(het, "S-IIP")
                draw_diff_cell(
                    ax,
                    diff,
                    vmax_backdrop=cls_vmax[pcls],
                    vmax_marker=cls_vmax[pcls],
                    basin_centers=d["basins"],
                    basin_delta=np.zeros(len(BASIN_IDS)),
                    grid_extent=d["extent"],
                    pocket_class=pcls,
                )
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for ext in ("pdf", "png"):
        out_path = out_dir / f"G_landscape_grid_aligned_panelB.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)

    return paths


def _add_cell_annotation(ax, frag: str, mode: str, het: str) -> None:

    mode_lbl = "absolute" if mode == "abs" else "differential"
    name = LIGAND_HET_TO_NAME.get(het, het)
    label = f"{frag} · {mode_lbl}\n{name}"
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        color="#333333",
        linespacing=1.15,
        bbox=dict(
            facecolor="white",
            alpha=0.55,
            edgecolor="none",
            pad=1.6,
        ),
        zorder=20,
    )


def render_per_ligand_columns(out_dir: Path) -> List[Path]:

    (per_frag, diff_cells, cls_vmax, e_vmin_global, e_vmax_global, _) = (
        _prepare_landscape_data()
    )

    sub_dir = out_dir / "G_per_ligand"
    sub_dir.mkdir(parents=True, exist_ok=True)

    row_specs: List[Tuple[str, str]] = [
        ("Frag-B", "abs"),
        ("Frag-B", "diff"),
        ("Frag-F", "abs"),
        ("Frag-F", "diff"),
        ("Frag-A", "abs"),
        ("Frag-A", "diff"),
    ]
    n_row = len(row_specs)

    cell_w = 2.6
    cell_h = 1.82
    fig_w = cell_w + 0.1
    fig_h = cell_h * n_row + 0.1

    PER_LIGAND_BASIN_LABEL_FS = 12

    paths: List[Path] = []
    for j, het in enumerate(LIGAND_PANEL_B_ORDER):
        letter = chr(ord("a") + j)

        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
        gs = fig.add_gridspec(
            n_row,
            1,
            height_ratios=[cell_h] * n_row,
            hspace=0.06,
            top=0.995,
            bottom=0.005,
            left=0.005,
            right=0.995,
        )

        for i, (frag, mode) in enumerate(row_specs):
            d = per_frag.get(frag)
            ax = fig.add_subplot(gs[i, 0])

            if d is None:
                draw_na_cell(ax)
            elif mode == "abs":
                df_abs_frag = d.get("df_abs")
                if df_abs_frag is None or df_abs_frag.empty:
                    draw_na_cell(ax)
                else:
                    sub_abs = df_abs_frag[df_abs_frag["ligand"] == het]
                    if len(sub_abs) < MIN_SAMPLES_PER_CELL:
                        draw_na_cell(ax)
                    else:
                        draw_absolute_cell(
                            ax,
                            sub_abs,
                            extent=d["extent"],
                            e_vmin=e_vmin_global,
                            e_vmax=e_vmax_global,
                            basin_centers=d["basins"],
                            cmap_name=ABS_CMAP_BY_FRAG.get(frag, ABSOLUTE_CMAP),
                            label_fontsize=PER_LIGAND_BASIN_LABEL_FS,
                        )
            else:
                sub = d["df"][d["df"]["ligand"] == het]
                if len(sub) < MIN_SAMPLES_PER_CELL:
                    draw_na_cell(ax)
                else:
                    diff = diff_cells.get((frag, het))
                    if diff is None:
                        draw_na_cell(ax)
                    else:
                        pcls = LIGAND_POCKET_CLASS.get(het, "S-IIP")
                        draw_diff_cell(
                            ax,
                            diff,
                            vmax_backdrop=cls_vmax[pcls],
                            vmax_marker=cls_vmax[pcls],
                            basin_centers=d["basins"],
                            basin_delta=np.zeros(len(BASIN_IDS)),
                            grid_extent=d["extent"],
                            pocket_class=pcls,
                            label_fontsize=PER_LIGAND_BASIN_LABEL_FS,
                        )
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            _add_cell_annotation(ax, frag, mode, het)

        ligand_name = LIGAND_HET_TO_NAME.get(het, het).replace(" ", "")
        stem = f"G_col_{letter}_{ligand_name}"
        for ext in ("pdf", "png"):
            out_path = sub_dir / f"{stem}.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            paths.append(out_path)
        plt.close(fig)

    print()
    print("=" * 70)
    print(f"Per-ligand column figures → {sub_dir}")
    for p in paths:
        print(f"  {p}")
    print("=" * 70)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "paper_img" / "fig4"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-composite", action="store_true", help="Skip the 6×5 composite figure."
    )
    parser.add_argument(
        "--skip-per-ligand",
        action="store_true",
        help="Skip the per-ligand column sub-figures.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if not args.skip_composite:
        for p in render(out_dir):
            print(f"Wrote composite figure {p}")
    if not args.skip_per_ligand:
        for p in render_per_ligand_columns(out_dir):
            print(f"Wrote per-ligand figure {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Render energy landscapes from predecoded conformations and filter energies."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("landscape_v2")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_RUN_DIR = os.environ.get(
    "QPOCKET_RUN_DIR", os.path.join(_PROJECT_ROOT, "inputs", "ensembles")
)
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "result", "img", "landscape")

MAX_PLOT: int = 1_000
N_STRAT_BINS: int = 50
MIN_VALID: int = 40
ENERGY_COLUMN: str = "filter_energy"
WEIGHT_COLUMN: str = "weight"


def stratified_subsample(
    energies: np.ndarray,
    n_target: int,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick row indices preserving energy-distribution shape."""
    n = len(energies)
    if n <= n_target:
        return np.arange(n)

    try:
        bin_ids = pd.qcut(energies, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return np.sort(rng.choice(n, size=n_target, replace=False))

    bin_ids = np.asarray(bin_ids)
    actual_bins = np.unique(bin_ids[~np.isnan(bin_ids.astype(float))]).astype(int)
    per_bin_quota = max(1, math.ceil(n_target / max(1, len(actual_bins))))

    picks: List[int] = []
    for b in actual_bins:
        idx_in_bin = np.where(bin_ids == b)[0]
        take = min(per_bin_quota, len(idx_in_bin))
        if take >= len(idx_in_bin):
            picks.extend(idx_in_bin.tolist())
        else:
            picks.extend(rng.choice(idx_in_bin, size=take, replace=False).tolist())

    picks_arr = np.array(sorted(set(picks)), dtype=int)
    if len(picks_arr) > n_target:
        picks_arr = np.sort(rng.choice(picks_arr, size=n_target, replace=False))
    return picks_arr


def _build_landscape_data(
    task_id: str,
    sequence: str,
    coords: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    bitstrings: List[str],
    cfg,
    builder,
    estimate_basins,
):
    """Build a landscape from predecoded coordinates and candidate scores."""
    from landscape.data_types import LandscapeData, SampledConformation
    from sklearn.decomposition import PCA

    N, L, _ = coords.shape

    coord_matrix = coords.reshape(N, L * 3).astype(np.float64)

    n_comp = min(cfg.n_pca_components, N - 1, coord_matrix.shape[1])
    pca = PCA(n_components=max(1, n_comp))
    pca_coords_raw = pca.fit_transform(coord_matrix)
    explained = pca.explained_variance_ratio_

    if pca_coords_raw.shape[1] < 2:
        pad = np.zeros((pca_coords_raw.shape[0], 2 - pca_coords_raw.shape[1]))
        pca_coords = np.hstack([pca_coords_raw, pad])
        explained = np.pad(explained, (0, 2 - len(explained)))
    else:
        pca_coords = pca_coords_raw[:, :2]
        explained = np.array(explained[:2])

    grid_x, grid_y, grid_z = builder._interpolate_surface(pca_coords, energies)

    min_idx = int(np.argmin(energies))
    dists_to_min = np.linalg.norm(pca_coords - pca_coords[min_idx], axis=1)
    if len(dists_to_min) > 5 and dists_to_min.std() > 1e-9:
        funnel_score = float(np.corrcoef(dists_to_min, energies)[0, 1])
        if not math.isfinite(funnel_score):
            funnel_score = 0.0
    else:
        funnel_score = 0.0

    n_basins = estimate_basins(pca_coords, energies)

    weight_sum = float(weights.sum()) if weights.sum() > 0 else 1.0
    confs: List[SampledConformation] = []
    for i in range(N):
        confs.append(
            SampledConformation(
                bitstring_key=bitstrings[i],
                full_bitstring=bitstrings[i],
                sequence=sequence,
                main_positions=coords[i].astype(np.float64),
                energy=float(energies[i]),
                energy_terms={},
                count=1,
                prob=float(weights[i]) / weight_sum,
                saw_valid=True,
            )
        )

    return LandscapeData(
        protein_id=task_id,
        sequence=sequence,
        conformations=confs,
        pca_coords=pca_coords,
        pca_explained_variance=explained,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        min_energy_idx=min_idx,
        median_energy=float(np.median(energies)),
        funnel_score=funnel_score,
        n_valid=N,
        n_total_unique=N,
        n_basins=n_basins,
    )


def process_one_system(
    sys_dir: str,
    out_dir: str,
    cfg,
    builder,
    estimate_basins,
    plotter,
    rng: np.random.Generator,
) -> Optional[Dict]:
    """Returns a one-row summary dict, or None if the system was skipped."""
    task_id = os.path.basename(sys_dir.rstrip("/"))
    input_json = os.path.join(sys_dir, "input.json")
    cand_csv = os.path.join(sys_dir, "landscape", "landscape_candidates.csv")
    coords_npz = os.path.join(sys_dir, "landscape", "landscape_coords.npz")

    if not (
        os.path.isfile(input_json)
        and os.path.isfile(cand_csv)
        and os.path.isfile(coords_npz)
    ):
        _LOG.info(f"[{task_id}] missing required files, skipping.")
        return None

    with open(input_json, "r") as fh:
        meta = json.load(fh)
    sequence = str(meta.get("sequence", ""))
    if len(sequence) < 4:
        _LOG.warning(f"[{task_id}] sequence too short (L={len(sequence)}), skipping.")
        return None

    df = pd.read_csv(cand_csv, low_memory=False)
    coords_all = np.load(coords_npz)["coords"]

    if len(df) != coords_all.shape[0]:
        _LOG.warning(
            f"[{task_id}] CSV/coords row mismatch: csv={len(df)} npz={coords_all.shape[0]}; "
            f"truncating to min."
        )
        n_keep = min(len(df), coords_all.shape[0])
        df = df.iloc[:n_keep].reset_index(drop=True)
        coords_all = coords_all[:n_keep]

    L_seq = len(sequence)
    if coords_all.shape[1] != L_seq:
        _LOG.warning(
            f"[{task_id}] coords L={coords_all.shape[1]} != seq L={L_seq}; skipping."
        )
        return None

    if ENERGY_COLUMN not in df.columns:
        _LOG.warning(f"[{task_id}] missing column {ENERGY_COLUMN}, skipping.")
        return None

    energies_full = df[ENERGY_COLUMN].to_numpy(dtype=float)
    weights_full = (
        df[WEIGHT_COLUMN].to_numpy(dtype=float)
        if WEIGHT_COLUMN in df.columns
        else np.ones(len(df), dtype=float)
    )
    bitstrings_full = df["bitstring"].astype(str).tolist()

    valid_mask = np.isfinite(energies_full)
    energies = energies_full[valid_mask]
    weights = weights_full[valid_mask]
    coords = coords_all[valid_mask]
    bitstrings = [bitstrings_full[i] for i in np.where(valid_mask)[0]]

    n_valid_pre = int(valid_mask.sum())
    if n_valid_pre < MIN_VALID:
        _LOG.warning(
            f"[{task_id}] only {n_valid_pre} finite-energy rows "
            f"(< MIN_VALID={MIN_VALID}); skipping."
        )
        return None

    if n_valid_pre > MAX_PLOT:
        sel = stratified_subsample(energies, MAX_PLOT, N_STRAT_BINS, rng)
        energies = energies[sel]
        weights = weights[sel]
        coords = coords[sel]
        bitstrings = [bitstrings[i] for i in sel]
        _LOG.info(f"[{task_id}] N={n_valid_pre} -> stratified-subsampled to {len(sel)}")
    else:
        _LOG.info(f"[{task_id}] N={n_valid_pre} (no subsampling)")

    data = _build_landscape_data(
        task_id=task_id,
        sequence=sequence,
        coords=coords,
        energies=energies,
        weights=weights,
        bitstrings=bitstrings,
        cfg=cfg,
        builder=builder,
        estimate_basins=estimate_basins,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf_path = os.path.join(out_dir, f"{task_id}_landscape.pdf")
    png_path = os.path.join(out_dir, f"{task_id}_landscape.png")

    fig = plotter.plot(data, save_path=pdf_path, fmt="pdf")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "task_id": task_id,
        "sequence": sequence,
        "L": L_seq,
        "n_in_csv": int(len(df)),
        "n_finite_energy": n_valid_pre,
        "n_plotted": int(len(energies)),
        "subsampled": bool(n_valid_pre > MAX_PLOT),
        "min_energy": float(energies.min()),
        "median_energy": float(np.median(energies)),
        "max_energy_clip93": float(np.percentile(energies, 93)),
        "funnel_score": float(data.funnel_score),
        "n_basins": int(data.n_basins),
        "pca_var_pc1": float(data.pca_explained_variance[0]),
        "pca_var_pc2": float(data.pca_explained_variance[1]),
        "pdf_path": pdf_path,
    }
    _LOG.info(
        f"[{task_id}] done   funnel={summary['funnel_score']:.3f}  "
        f"basins={summary['n_basins']}  "
        f"PC1+PC2={(summary['pca_var_pc1']+summary['pca_var_pc2'])*100:.1f}%"
    )
    return summary


def discover_systems(run_dir: str) -> List[str]:

    out: List[str] = []
    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub):
            continue
        if os.path.isfile(os.path.join(sub, "input.json")) and os.path.isfile(
            os.path.join(sub, "landscape", "landscape_candidates.csv")
        ):
            out.append(sub)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help=f"Ensemble input directory (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for figures (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional list of task_ids to process (default: all systems found)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260430,
        help="RNG seed for stratified subsampling (default: 20260430)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        _LOG.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    from landscape.config import LandscapeConfig
    from landscape.builder import LandscapeBuilder, _estimate_basins
    from landscape.visualizer import LandscapePlotter

    cfg = LandscapeConfig(
        max_unique_conformations=MAX_PLOT,
        min_unique_conformations=MIN_VALID,
        save_data=False,
        save_plots=False,
    )
    builder = LandscapeBuilder(cfg)
    plotter = LandscapePlotter(cfg)
    rng = np.random.default_rng(args.seed)

    systems = discover_systems(args.run_dir)
    if args.only:
        keep = set(args.only)
        systems = [s for s in systems if os.path.basename(s) in keep]
    _LOG.info(f"Found {len(systems)} systems under {args.run_dir}")

    summaries: List[Dict] = []
    for i, sys_dir in enumerate(systems, 1):
        _LOG.info(f"--- [{i}/{len(systems)}] {os.path.basename(sys_dir)} ---")
        try:
            row = process_one_system(
                sys_dir=sys_dir,
                out_dir=args.out_dir,
                cfg=cfg,
                builder=builder,
                estimate_basins=_estimate_basins,
                plotter=plotter,
                rng=rng,
            )
            if row is not None:
                summaries.append(row)
        except Exception as exc:
            _LOG.exception(f"[{os.path.basename(sys_dir)}] failed with: {exc}")

    if summaries:
        out_csv = os.path.join(args.out_dir, "landscape_v2_summary.csv")
        pd.DataFrame(summaries).sort_values("funnel_score", ascending=False).to_csv(
            out_csv, index=False
        )
        _LOG.info(f"Summary CSV -> {out_csv}")

    _LOG.info("=" * 60)
    _LOG.info(f"Processed {len(summaries)}/{len(systems)} systems successfully.")
    _LOG.info(f"Figures in: {args.out_dir}")
    _LOG.info("=" * 60)


if __name__ == "__main__":
    main()

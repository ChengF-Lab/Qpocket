"""Render per-case and per-fragment landscape galleries."""

from __future__ import annotations

import os
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "paper_scripts", ROOT / "paper_plt"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUN_V2_DIR = Path(os.environ.get("QPOCKET_RUN_DIR", ROOT / "inputs" / "ensembles"))
OUT_FIG = ROOT / "SI" / "figs_landscape"
OUT_PERCASE = OUT_FIG / "per_case"
OUT_DATA = ROOT / "SI" / "fig3"
for d in (OUT_FIG, OUT_PERCASE, OUT_DATA):
    d.mkdir(parents=True, exist_ok=True)

LANDSCAPE_N_REPRESENTATIVES = 200
LANDSCAPE_STRAT_BINS = 25
RNG_SEED = 20260504
ORACLE_TAU_A = 3.0

SYS_ORDER = ["7RT4", "7RT1", "7RPZ", "6GJ6", "6GJ8", "8AZV", "8AZX", "8AZY", "8AZZ"]
FRAG_ORDER = ["A", "C", "D", "E", "F"]


def stratified_subsample(energies, n_target, n_bins, rng):
    import pandas as pd

    n = len(energies)
    if n <= n_target:
        return np.arange(n)
    try:
        bin_ids = pd.qcut(energies, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return np.sort(rng.choice(n, size=n_target, replace=False))
    bin_ids = np.asarray(bin_ids)
    actual = np.unique(bin_ids[~np.isnan(bin_ids.astype(float))]).astype(int)
    per_bin = max(1, math.ceil(n_target / max(1, len(actual))))
    picks = []
    for b in actual:
        idx = np.where(bin_ids == b)[0]
        take = min(per_bin, len(idx))
        if take >= len(idx):
            picks.extend(idx.tolist())
        else:
            picks.extend(rng.choice(idx, size=take, replace=False).tolist())
    arr = np.array(sorted(set(picks)), dtype=int)
    if len(arr) > n_target:
        arr = np.sort(rng.choice(arr, size=n_target, replace=False))
    return arr


def build_landscape_data(
    sys_dir, cfg, builder, estimate_basins, rng, n_subsample=LANDSCAPE_N_REPRESENTATIVES
):
    import json
    import pandas as pd
    from landscape.data_types import LandscapeData, SampledConformation
    from sklearn.decomposition import PCA

    task_id = sys_dir.name
    input_json = sys_dir / "input.json"
    cand_csv = sys_dir / "landscape" / "landscape_candidates.csv"
    coords_npz = sys_dir / "landscape" / "landscape_coords.npz"
    if not (input_json.is_file() and cand_csv.is_file() and coords_npz.is_file()):
        return None, None, task_id

    sequence = str(json.loads(input_json.read_text()).get("sequence", ""))
    if len(sequence) < 4:
        return None, None, task_id

    df = pd.read_csv(cand_csv, low_memory=False)
    coords_all = np.load(coords_npz)["coords"]
    n_keep = min(len(df), coords_all.shape[0])
    df = df.iloc[:n_keep].reset_index(drop=True)
    coords_all = coords_all[:n_keep]
    if coords_all.shape[1] != len(sequence):
        return None, None, task_id
    if "filter_energy" not in df.columns:
        return None, None, task_id

    energies_full = df["filter_energy"].to_numpy(dtype=float)
    weights_full = (
        df["weight"].to_numpy(dtype=float)
        if "weight" in df.columns
        else np.ones(len(df))
    )
    bitstrings_full = df["bitstring"].astype(str).tolist()
    rmsds_full = (
        df["kabsch_rmsd"].to_numpy(dtype=float)
        if "kabsch_rmsd" in df.columns
        else np.full(len(df), np.nan)
    )
    anchor_full = (
        df["is_oracle_anchor"].astype(bool).to_numpy()
        if "is_oracle_anchor" in df.columns
        else np.zeros(len(df), bool)
    )

    valid = np.isfinite(energies_full)
    energies = energies_full[valid]
    weights = weights_full[valid]
    coords = coords_all[valid]
    bitstrings = [bitstrings_full[i] for i in np.where(valid)[0]]
    rmsds = rmsds_full[valid]
    is_anchor = anchor_full[valid]
    if len(energies) < 40:
        return None, None, task_id

    if len(energies) > n_subsample:
        sel = stratified_subsample(energies, n_subsample, LANDSCAPE_STRAT_BINS, rng)
        energies, weights, coords = energies[sel], weights[sel], coords[sel]
        bitstrings = [bitstrings[i] for i in sel]
        rmsds, is_anchor = rmsds[sel], is_anchor[sel]

    N, L, _ = coords.shape
    coord_matrix = coords.reshape(N, L * 3).astype(np.float64)
    n_comp = min(getattr(cfg, "n_pca_components", 2), N - 1, coord_matrix.shape[1])
    pca = PCA(n_components=max(1, n_comp))
    pca_raw = pca.fit_transform(coord_matrix)
    explained = pca.explained_variance_ratio_
    if pca_raw.shape[1] < 2:
        pad = np.zeros((pca_raw.shape[0], 2 - pca_raw.shape[1]))
        pca_coords = np.hstack([pca_raw, pad])
        explained = np.pad(explained, (0, 2 - len(explained)))
    else:
        pca_coords = pca_raw[:, :2]
        explained = np.array(explained[:2])

    _uniq_idx = np.sort(
        np.unique(np.round(pca_coords, 6), axis=0, return_index=True)[1]
    )
    grid_x, grid_y, grid_z = builder._interpolate_surface(
        pca_coords[_uniq_idx], energies[_uniq_idx]
    )
    min_idx = int(np.argmin(energies))
    dists = np.linalg.norm(pca_coords - pca_coords[min_idx], axis=1)
    funnel = 0.0
    if len(dists) > 5 and dists.std() > 1e-9:
        c = float(np.corrcoef(dists, energies)[0, 1])
        funnel = c if math.isfinite(c) else 0.0
    n_basins = estimate_basins(pca_coords, energies)

    weight_sum = float(weights.sum()) if weights.sum() > 0 else 1.0
    confs = [
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
        for i in range(N)
    ]

    data = LandscapeData(
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
        funnel_score=funnel,
        n_valid=N,
        n_total_unique=N,
        n_basins=n_basins,
    )
    extras = {
        "kabsch_rmsds": np.asarray(rmsds, float),
        "is_anchor": np.asarray(is_anchor, bool),
        "energies": np.asarray(energies, float),
    }
    return data, extras, task_id


def draw_2d(ax, data, cfg, plotter, fs=0.7):
    """Draw the energy contours using the color limits for this panel."""
    energies = data.energies
    e_min = float(energies.min())
    e_max_display = float(np.percentile(energies, cfg.energy_percentile_clip))
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max_display)
    cmap = mpl.colormaps[cfg.colormap]
    plotter._panel_2d_contour(ax, data, norm, cmap, e_max_display, fs)


def parse(name):
    sysid, var, frag = name.split("_")
    return sysid, var, frag.replace("Frag-", "")


def main() -> int:
    from landscape.config import LandscapeConfig
    from landscape.builder import LandscapeBuilder, _estimate_basins
    from landscape.visualizer import LandscapePlotter

    cfg = LandscapeConfig()
    builder = LandscapeBuilder(cfg)
    plotter = LandscapePlotter(cfg)

    all_cases = sorted(
        [p for p in RUN_V2_DIR.iterdir() if p.is_dir() and "_Frag-" in p.name]
    )
    by_frag = defaultdict(dict)
    for p in all_cases:
        sysid, var, frag = parse(p.name)
        by_frag[frag][sysid] = (p, var)

    remaining = sorted([p for p in all_cases if not p.name.endswith("_Frag-B")])
    print(
        f"Found {len(all_cases)} cases, including {len(remaining)} outside Frag-B."
    )

    cache: dict[str, tuple] = {}
    sample_rows: list[dict] = []

    def get(case_dir):
        if case_dir.name in cache:
            return cache[case_dir.name]
        rng = np.random.default_rng(RNG_SEED)
        try:
            data, extras, _ = build_landscape_data(
                case_dir, cfg, builder, _estimate_basins, rng
            )
        except Exception as e:
            print(f"Could not build the landscape for {case_dir.name}: {e!r}")
            data, extras = None, None
        cache[case_dir.name] = (data, extras)
        if data is not None:
            sysid, var, frag = parse(case_dir.name)
            pca = data.pca_coords
            en = np.asarray(data.energies, float)
            rm = extras["kabsch_rmsds"]
            an = extras["is_anchor"]
            prox = np.where(np.isfinite(rm), np.exp(-rm / ORACLE_TAU_A), np.nan)
            for i in range(len(en)):
                sample_rows.append(
                    dict(
                        system=sysid,
                        mutation=var,
                        fragment=frag,
                        case=case_dir.name,
                        sample_idx=i,
                        PC1=float(pca[i, 0]),
                        PC2=float(pca[i, 1]),
                        energy=float(en[i]),
                        kabsch_rmsd_A=float(rm[i]),
                        oracle_proximity=float(prox[i]),
                        is_oracle_anchor=bool(an[i]),
                        pc1_var_ratio=float(data.pca_explained_variance[0]),
                        pc2_var_ratio=float(data.pca_explained_variance[1]),
                    )
                )
        return cache[case_dir.name]

    def title_for(case_dir):
        sysid, var, frag = parse(case_dir.name)
        return f"{sysid} {var}"

    for p in all_cases:
        d, _ = get(p)
        print(f"{'Loaded' if d is not None else 'Skipped'} {p.name}")

    for frag in FRAG_ORDER:
        sysmap = by_frag.get(frag, {})
        items = [(s, sysmap[s]) for s in SYS_ORDER if s in sysmap]
        if not items:
            continue
        n = len(items)
        ncol = min(4, n)
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(3.7 * ncol, 3.2 * nrow),
            squeeze=False,
            constrained_layout=True,
        )
        for ax in axes.flat:
            ax.set_axis_off()
        ok = 0
        for k, (sysid, (cdir, var)) in enumerate(items):
            ax = axes[k // ncol][k % ncol]
            ax.set_axis_on()
            data, _ = get(cdir)
            if data is None:
                ax.text(
                    0.5,
                    0.5,
                    f"{sysid} {var}\n(no data)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_axis_off()
                continue
            try:
                draw_2d(ax, data, cfg, plotter)
                ax.set_title(f"{sysid} {var}", fontsize=10, pad=4)
                ok += 1
            except Exception as e:
                print(f"Could not render the panel for {cdir.name}: {e!r}")
                ax.text(
                    0.5,
                    0.5,
                    f"{sysid} {var}\n(draw error)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_axis_off()
        fig.suptitle(
            f"Energy landscapes — Frag-{frag} (remaining systems, "
            f"2-D PCA energy contours)",
            fontsize=13,
            y=1.01,
        )
        stem = OUT_FIG / f"figS_landscape_Frag-{frag}"
        fig.savefig(f"{stem}.png", dpi=170, bbox_inches="tight")
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {stem.name}.{{png,pdf}}  ({ok}/{n} panels)")

    rows = [s for s in SYS_ORDER if any(s in by_frag.get(f, {}) for f in FRAG_ORDER)]
    ncol = len(FRAG_ORDER)
    nrow = len(rows)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(3.3 * ncol, 2.9 * nrow),
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.flat:
        ax.set_axis_off()
    filled = 0
    for r, sysid in enumerate(rows):
        for c, frag in enumerate(FRAG_ORDER):
            ax = axes[r][c]
            entry = by_frag.get(frag, {}).get(sysid)
            if entry is None:
                continue
            cdir, var = entry
            data, _ = get(cdir)
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
                draw_2d(ax, data, cfg, plotter, fs=0.6)
                filled += 1
            except Exception as e:
                print(f"Could not render the overview panel for {cdir.name}: {e!r}")
            if r == 0:
                ax.annotate(
                    f"Frag-{frag}",
                    xy=(0.5, 1.18),
                    xycoords="axes fraction",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                )
            if (
                c == 0
                or by_frag.get(frag, {}).get(sysid) is not None
                and all(sysid not in by_frag.get(FRAG_ORDER[cc], {}) for cc in range(c))
            ):
                ax.set_ylabel(f"{sysid} {var}", fontsize=10, fontweight="bold")
    fig.suptitle(
        "Energy landscapes for non-main-text fragments "
        "(rows = KRAS system, cols = fragment; 2-D PCA energy contours)",
        fontsize=15,
        y=1.005,
    )
    stem = OUT_FIG / "figS_landscape_all_remaining_master"
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem.name}.{{png,pdf}}  ({filled} panels)")

    for p in remaining:
        data, _ = cache.get(p.name, (None, None))
        if data is None:
            continue
        fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
        try:
            draw_2d(ax, data, cfg, plotter, fs=0.85)
            ax.set_title(p.name.replace("_", " "), fontsize=11)
            fig.savefig(
                OUT_PERCASE / f"landscape2d_{p.name}.png", dpi=160, bbox_inches="tight"
            )
        except Exception as e:
            print(f"Could not render the individual plot for {p.name}: {e!r}")
        plt.close(fig)

    import pandas as pd

    df = pd.DataFrame(sample_rows)
    if not df.empty:
        rem = df[df.fragment != "B"].sort_values(["fragment", "system", "sample_idx"])
        fb = df[df.fragment == "B"].sort_values(["system", "sample_idx"])
        rem.to_csv(OUT_DATA / "fig3_SI_landscape_samples_remaining.csv", index=False)
        fb.to_csv(OUT_DATA / "fig3_landscape_samples_FragB.csv", index=False)
        print(
            f"Wrote landscape sample CSVs: "
            f"other fragments, {len(rem)} rows; Frag-B, {len(fb)} rows"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

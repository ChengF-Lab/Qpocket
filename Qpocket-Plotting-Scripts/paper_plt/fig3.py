"""Render Frag-B structures and coordinate-PCA landscapes."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PAPER_SCRIPTS_DIR = PROJECT_ROOT / "paper_scripts"
if str(PAPER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS_DIR))

from paper_plt.plt import save_figure
from paper_plt.pymol_vis.runner import find_pymol, run_pml


KRAS_PDB_DIR = PROJECT_ROOT / "kras_select_systems"
SEQUENCE_CSV = PROJECT_ROOT / "inputs" / "protein_sequence_summary.csv"
RUN_V2_DIR = Path(
    os.environ.get("QPOCKET_RUN_DIR", PROJECT_ROOT / "inputs" / "ensembles")
)

PDB_ORDER: List[str] = [
    "7RT4",
    "7RT1",
    "7RPZ",
    "6GJ6",
    "6GJ8",
    "8AZV",
    "8AZX",
    "8AZY",
    "8AZZ",
]

PDB_LIGAND: Dict[str, str] = {
    "7RT4": "Compound 5B",
    "7RT1": "Compound 15",
    "7RPZ": "MRTX1133",
    "6GJ6": "Compound 18",
    "6GJ8": "BI-2852",
    "8AZV": "BI-2865",
    "8AZX": "BI-2865",
    "8AZY": "BI-2865",
    "8AZZ": "BI-2865",
}

PDB_MUTATION: Dict[str, str] = {
    "7RT4": "G12D",
    "7RT1": "G12D",
    "7RPZ": "G12D",
    "6GJ6": "G12D",
    "6GJ8": "G12D",
    "8AZV": "WT",
    "8AZX": "G12C",
    "8AZY": "G12D",
    "8AZZ": "G12V",
}

MUTATION_RGB: Dict[str, Tuple[int, int, int]] = {
    "WT": (130, 170, 130),
    "G12D": (220, 60, 50),
    "G12C": (240, 170, 30),
    "G12V": (170, 60, 200),
}

MUTATION_RESI_RANGE: Tuple[int, int] = (10, 17)

LIGAND_RGB: Dict[str, Tuple[int, int, int]] = {
    "Compound 5B": (74, 95, 170),
    "Compound 15": (224, 122, 43),
    "MRTX1133": (61, 140, 90),
    "Compound 18": (199, 62, 62),
    "BI-2852": (142, 91, 168),
    "BI-2865": (210, 111, 160),
}


RGB = Tuple[int, int, int]
PALETTE_NON_PRED_GRAY: RGB = (220, 220, 220)
PALETTE_FRAG_DEEP_BLUE: RGB = (16, 55, 160)
PALETTE_FRAG_LIGHT_BLUE: RGB = (132, 190, 255)
PALETTE_LIGHT_RED: RGB = (220, 90, 110)

CONTACT_DIST_A = 3.0
PYMOL_RAY_PX = 1200

POCKET_BG_THRESHOLD: float = 0.97
POCKET_CROP_PAD_FRAC: float = 0.025

LANDSCAPE_N_REPRESENTATIVES = 200
LANDSCAPE_STRAT_BINS = 25
RNG_SEED = 20260504


_AA_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def chain_residue_aa(pdb_path: Path, chain: str = "A") -> Dict[int, str]:

    seen: Dict[int, str] = {}
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if (line[21].strip() or "_") != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            resn = line[17:20].strip()
            if resn in _AA_3TO1:
                seen[resi] = _AA_3TO1[resn]
    return seen


def find_fragment_residues(
    resi_aa: Dict[int, str],
    fragment_seq: str,
    *,
    max_mm: int = 2,
) -> Optional[Tuple[int, int]]:
    """Locate ``fragment_seq`` in the chain's amino-acid string."""
    sorted_resis = sorted(resi_aa.keys())
    aa_string = "".join(resi_aa[r] for r in sorted_resis)
    L = len(fragment_seq)
    if L == 0 or L > len(aa_string):
        return None

    pos = aa_string.find(fragment_seq)
    if pos >= 0:
        return sorted_resis[pos], sorted_resis[pos + L - 1]

    best_pos, best_mm = -1, max_mm + 1
    for i in range(len(aa_string) - L + 1):
        win = aa_string[i : i + L]
        mm = sum(1 for a, b in zip(win, fragment_seq) if a != b)
        if mm < best_mm:
            best_pos, best_mm = i, mm
    if best_pos < 0:
        return None
    return sorted_resis[best_pos], sorted_resis[best_pos + L - 1]


def _rgb_str(rgb: RGB) -> str:
    return f"[{int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}]"


def _darken(rgb: RGB, factor: float = 0.45) -> RGB:
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))


def _frag_resi_clause(fragments: List[Tuple[int, int]]) -> str:
    """Compose a PyMOL ``resi a-b or resi c-d ...`` clause."""
    return " or ".join(f"resi {a}-{b}" for a, b in fragments)


def build_pocket_pml(
    pdb_path: Path,
    chain: str,
    fragments: List[Tuple[int, int]],
    ligand_rgb: RGB,
    output_png: Path,
    *,
    mutation_rgb: RGB = (170, 170, 170),
    mutation_resi_range: Tuple[int, int] = MUTATION_RESI_RANGE,
    contact_dist_a: float = CONTACT_DIST_A,
    image_size: Tuple[int, int] = (PYMOL_RAY_PX, PYMOL_RAY_PX),
) -> str:
    """Generate the PyMOL commands for the pocket view."""
    img_w, img_h = image_size
    frag_clause = _frag_resi_clause(fragments)
    ligand_main_str = _rgb_str(ligand_rgb)
    ligand_edge_str = _rgb_str(_darken(ligand_rgb, 0.40))
    mutation_str = _rgb_str(mutation_rgb)
    mut_lo, mut_hi = mutation_resi_range

    return f"""
load {pdb_path}, complex
remove solvent
hide everything

set specular, 0
set direct, 1
set ambient, 0.5
set ray_trace_mode, 1
set ray_trace_depth_factor, 1
set ray_trace_disco_factor, 1
set ray_trace_gain, 0.005
set depth_cue, 1
set fog, on
set fog_start, 0.3
set ray_trace_fog, 1
set ray_shadow, off
set antialias, 2
set line_smooth, 1
bg_color white
set orthoscopic, on
set auto_zoom, off

set_color non_pred_gray, {_rgb_str(PALETTE_NON_PRED_GRAY)}
set_color frag_deep_blue, {_rgb_str(PALETTE_FRAG_DEEP_BLUE)}
set_color contact_pink, {_rgb_str(PALETTE_LIGHT_RED)}
set_color ligand_main, {ligand_main_str}
set_color ligand_edge, {ligand_edge_str}

show cartoon, polymer.protein
color non_pred_gray, polymer.protein
set cartoon_transparency, 0.10, polymer.protein

select frag_all, polymer.protein and chain {chain} and ({frag_clause})
color frag_deep_blue, %frag_all
set cartoon_transparency, 0.00, %frag_all

select contacts, (polymer.protein and chain {chain}) and not %frag_all \\
                 and byres (all within {contact_dist_a} of %frag_all)
color contact_pink, %contacts
set cartoon_transparency, 0.05, %contacts

set_color mutation_color, {mutation_str}
select mut_segment, polymer.protein and chain {chain} and resi {mut_lo}-{mut_hi}
color mutation_color, %mut_segment
set cartoon_transparency, 0.00, %mut_segment

select ligand, hetatm and not resn HOH+WAT+SO4+PO4+GOL+EDO+CL+NA+MG+ZN+CA \\
                and not polymer.protein
create ligand_outline, %ligand
show sticks, %ligand_outline
color ligand_edge, %ligand_outline
set stick_radius, 0.18, %ligand_outline
set stick_transparency, 0.20, %ligand_outline

show sticks, %ligand
color ligand_main, %ligand
set stick_radius, 0.10, %ligand
set stick_transparency, 0.00, %ligand

h_add %ligand
distance hbonds, %ligand, %frag_all, 3.3, mode=2
hide labels, hbonds
set dash_as_cylinders, 0
set dash_width, 2.5, hbonds
set dash_gap, 0.28, hbonds
set dash_length, 0.20, hbonds
set dash_color, yellow, hbonds
hide sticks, (%ligand or %ligand_outline) and elem H

select frag_polar, %frag_all and (elem N or elem O or elem S)
select hb_resi, byres (%frag_polar and (all within 3.3 of %ligand))
select hb_sidechain, %hb_resi and not (name N+CA+C+O+H+HA)
show sticks, %hb_sidechain
color contact_pink, %hb_sidechain and elem C
set stick_radius, 0.10, %hb_sidechain
set stick_transparency, 0.00, %hb_sidechain

select focus_sel, %frag_all or %ligand
center focus_sel
orient focus_sel
zoom focus_sel, 5.0
set field_of_view, 22
turn y, 15
turn x, -8
turn z, 2

viewport {img_w}, {img_h}
ray {min(img_w, img_h)}
png {output_png}, {img_w}, {img_h}, 300, ray=1
quit
"""


def load_fragment_ranges(
    csv_path: Path = SEQUENCE_CSV,
) -> Dict[str, List[Tuple[str, Tuple[int, int]]]]:

    pdb_frags: Dict[str, List[Tuple[str, str]]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            protein_id = (row.get("protein_id") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if not protein_id or not seq:
                continue
            parts = protein_id.split("_")
            if len(parts) < 3:
                continue
            pdb_frags.setdefault(parts[0], []).append((parts[-1], seq))

    out: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {}
    for pdb_id, frag_list in pdb_frags.items():
        pdb_path = KRAS_PDB_DIR / f"{pdb_id}.pdb"
        if not pdb_path.is_file():
            print(f"[fig3] {pdb_id}: structure file missing — excluded")
            continue
        resi_aa = chain_residue_aa(pdb_path, "A")
        if not resi_aa:
            print(f"[fig3] {pdb_id}: chain A has no resolved residues — excluded")
            continue

        resolved: List[Tuple[str, Tuple[int, int]]] = []
        skip = False
        for frag_name, seq in frag_list:
            rng = find_fragment_residues(resi_aa, seq)
            if rng is None:
                print(
                    f"[fig3] {pdb_id}: fragment {frag_name!r} ({seq}) not "
                    f"locatable in structure — excluding PDB entirely"
                )
                skip = True
                break
            resolved.append((frag_name, rng))
        if not skip:
            out[pdb_id] = resolved
    return out


def crop_to_content(
    img: np.ndarray,
    *,
    bg_threshold: float = POCKET_BG_THRESHOLD,
    pad_frac: float = POCKET_CROP_PAD_FRAC,
    make_square: bool = True,
) -> np.ndarray:
    """Crop a PyMOL render to its non-background bounding box."""
    if img.size == 0 or img.ndim < 2:
        return img

    if img.ndim == 3 and img.shape[2] >= 4:
        alpha = img[..., 3]
        rgb = img[..., :3]
        mask = (alpha > 0.0) & (rgb.mean(axis=2) < bg_threshold)
    elif img.ndim == 3:
        rgb = img[..., :3]
        mask = rgb.mean(axis=2) < bg_threshold
    else:
        mask = img < bg_threshold

    if not mask.any():
        return img

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    pad_r = max(1, int(img.shape[0] * pad_frac))
    pad_c = max(1, int(img.shape[1] * pad_frac))
    r0 = max(0, r0 - pad_r)
    r1 = min(img.shape[0], r1 + pad_r)
    c0 = max(0, c0 - pad_c)
    c1 = min(img.shape[1], c1 + pad_c)
    cropped = img[r0:r1, c0:c1]

    if make_square:
        h, w = cropped.shape[:2]
        if h != w:
            if h > w:
                pad_l = (h - w) // 2
                pad_r2 = (h - w) - pad_l
                pad_spec = ((0, 0), (pad_l, pad_r2))
            else:
                pad_t = (w - h) // 2
                pad_b = (w - h) - pad_t
                pad_spec = ((pad_t, pad_b), (0, 0))
            if cropped.ndim == 3:
                pad_spec = pad_spec + ((0, 0),)
            cropped = np.pad(
                cropped,
                pad_spec,
                mode="constant",
                constant_values=1.0,
            )

    return cropped


HIGHLIGHT_SAT_THRESHOLD: float = 0.20
HIGHLIGHT_CROP_PAD_FRAC: float = 0.05


def crop_to_highlight(
    img: np.ndarray,
    *,
    saturation_threshold: float = HIGHLIGHT_SAT_THRESHOLD,
    pad_frac: float = HIGHLIGHT_CROP_PAD_FRAC,
    make_square: bool = True,
) -> np.ndarray:
    """Crop a PyMOL render to its **highlighted** region only."""
    if img.size == 0 or img.ndim != 3 or img.shape[2] < 3:
        return img

    rgb = img[..., :3]
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb = rgb.astype(np.float32) / 255.0

    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    if img.ndim == 3 and img.shape[2] >= 4:
        alpha = img[..., 3]
        if alpha.dtype != np.float32 and alpha.dtype != np.float64:
            alpha = alpha.astype(np.float32) / 255.0
        mask = (saturation > saturation_threshold) & (alpha > 0.0)
    else:
        mask = saturation > saturation_threshold

    if not mask.any():
        return crop_to_content(img, make_square=make_square)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    pad_r = max(1, int(img.shape[0] * pad_frac))
    pad_c = max(1, int(img.shape[1] * pad_frac))
    r0 = max(0, r0 - pad_r)
    r1 = min(img.shape[0], r1 + pad_r)
    c0 = max(0, c0 - pad_c)
    c1 = min(img.shape[1], c1 + pad_c)
    cropped = img[r0:r1, c0:c1]

    if make_square:
        h, w = cropped.shape[:2]
        if h != w:
            if h > w:
                pad_l = (h - w) // 2
                pad_r2 = (h - w) - pad_l
                pad_spec = ((0, 0), (pad_l, pad_r2))
            else:
                pad_t = (w - h) // 2
                pad_b = (w - h) - pad_t
                pad_spec = ((pad_t, pad_b), (0, 0))
            if cropped.ndim == 3:
                pad_spec = pad_spec + ((0, 0),)
            cropped = np.pad(
                cropped,
                pad_spec,
                mode="constant",
                constant_values=1.0,
            )

    return cropped


def render_pocket(
    pdb_id: str,
    fragments: List[Tuple[int, int]],
    output_png: Path,
    *,
    chain: str = "A",
    reuse_existing: bool = True,
) -> bool:

    if reuse_existing and output_png.is_file():
        print(f"Using existing pocket image {output_png.name}")
        return True

    pdb_path = KRAS_PDB_DIR / f"{pdb_id}.pdb"
    if not pdb_path.is_file():
        print(f"[fig3] missing PDB: {pdb_path}")
        return False

    ligand_chem = PDB_LIGAND.get(pdb_id, "BI-2865")
    ligand_rgb = LIGAND_RGB.get(ligand_chem, (180, 180, 180))

    mutation = PDB_MUTATION.get(pdb_id, "WT")
    mutation_rgb = MUTATION_RGB.get(mutation, MUTATION_RGB["WT"])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    pml = build_pocket_pml(
        pdb_path=pdb_path.resolve(),
        chain=chain,
        fragments=fragments,
        ligand_rgb=ligand_rgb,
        mutation_rgb=mutation_rgb,
        output_png=output_png.resolve(),
    )
    try:
        run_pml(pml, timeout_s=300.0)
    except FileNotFoundError as e:
        print(f"[fig3] PyMOL not found: {e}")
        return False
    except RuntimeError as e:
        print(f"[fig3] PyMOL run failed for {pdb_id}: {e!s:.300s}")
        return False
    return output_png.is_file()


def stratified_subsample(
    energies: np.ndarray,
    n_target: int,
    n_bins: int,
    rng: np.random.Generator,
) -> np.ndarray:

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

    picks: List[int] = []
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


def _build_landscape_data_from_system(
    sys_dir: Path,
    cfg,
    builder,
    estimate_basins,
    rng: np.random.Generator,
    n_subsample: int = LANDSCAPE_N_REPRESENTATIVES,
):

    import json
    import pandas as pd

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
        else np.ones(len(df), dtype=float)
    )
    bitstrings_full = df["bitstring"].astype(str).tolist()
    rmsds_full = (
        df["kabsch_rmsd"].to_numpy(dtype=float)
        if "kabsch_rmsd" in df.columns
        else np.full(len(df), np.nan, dtype=float)
    )
    anchor_full = (
        df["is_oracle_anchor"].astype(bool).to_numpy()
        if "is_oracle_anchor" in df.columns
        else np.zeros(len(df), dtype=bool)
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
        sel = stratified_subsample(
            energies,
            n_subsample,
            LANDSCAPE_STRAT_BINS,
            rng,
        )
        energies = energies[sel]
        weights = weights[sel]
        coords = coords[sel]
        bitstrings = [bitstrings[i] for i in sel]
        rmsds = rmsds[sel]
        is_anchor = is_anchor[sel]

    from landscape.data_types import LandscapeData, SampledConformation
    from sklearn.decomposition import PCA

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

    grid_x, grid_y, grid_z = builder._interpolate_surface(pca_coords, energies)

    min_idx = int(np.argmin(energies))
    dists = np.linalg.norm(pca_coords - pca_coords[min_idx], axis=1)
    if len(dists) > 5 and dists.std() > 1e-9:
        funnel = float(np.corrcoef(dists, energies)[0, 1])
        if not math.isfinite(funnel):
            funnel = 0.0
    else:
        funnel = 0.0
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
        "kabsch_rmsds": np.asarray(rmsds, dtype=float),
        "is_anchor": np.asarray(is_anchor, dtype=bool),
        "energies": np.asarray(energies, dtype=float),
    }
    return data, extras, task_id


def _draw_landscape_pair(
    fig: plt.Figure,
    ax3d: plt.Axes,
    ax2d: plt.Axes,
    data,
    plotter,
    cfg,
    *,
    fs: float = 0.65,
):

    import matplotlib.colors as mcolors
    from matplotlib import cm

    energies = data.energies
    e_min = float(energies.min())
    e_max_display = float(np.percentile(energies, cfg.energy_percentile_clip))
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max_display)
    cmap = mpl.colormaps[cfg.colormap]

    plotter._panel_3d_surface(ax3d, data, norm, cmap, e_max_display, fs)
    plotter._panel_2d_contour(ax2d, data, norm, cmap, e_max_display, fs)


def _systems_for_fragb() -> List[Path]:

    if not RUN_V2_DIR.is_dir():
        return []
    out = []
    for p in sorted(RUN_V2_DIR.iterdir()):
        if p.is_dir() and p.name.endswith("_Frag-B"):
            out.append(p)
    return out


def render_pockets(
    fragment_ranges: Dict[str, List[Tuple[str, Tuple[int, int]]]],
    out_dir: Path,
) -> Dict[str, Path]:
    """Render each PDB's pocket panel; returns ``{pdb: png_path}``."""
    out: Dict[str, Path] = {}
    for pdb_id in PDB_ORDER:
        rngs = fragment_ranges.get(pdb_id)
        if not rngs:
            print(f"[fig3] {pdb_id}: no fragments resolved — skipped")
            continue
        png = out_dir / f"pocket_{pdb_id}.png"
        ok = render_pocket(pdb_id, [r for _, r in rngs], png)
        if ok:
            out[pdb_id] = png
            print(f"[fig3] pocket -> {png.name}")
    return out


def render_landscapes_standalone(
    out_dir: Path,
    *,
    cfg,
    builder,
    estimate_basins,
    plotter,
    rng: np.random.Generator,
    allowed_pdbs: Optional[set] = None,
) -> Dict[str, object]:

    landscapes: Dict[str, object] = {}
    for sys_dir in _systems_for_fragb():
        task_id = sys_dir.name
        if allowed_pdbs is not None:
            pdb_prefix = task_id.split("_", 1)[0]
            if pdb_prefix not in allowed_pdbs:
                print(f"[fig3] {task_id}: skipped because the reference structure was excluded")
                continue

        data, extras, _ = _build_landscape_data_from_system(
            sys_dir,
            cfg,
            builder,
            estimate_basins,
            rng,
        )
        if data is None:
            print(f"[fig3] {task_id}: skipped (insufficient data)")
            continue
        data._extras = extras

        fig = plt.figure(figsize=(9.0, 4.4), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d = fig.add_subplot(gs[0, 1])
        _draw_landscape_pair(fig, ax3d, ax2d, data, plotter, cfg, fs=0.85)
        ax3d.set_title("3-D energy surface", fontsize=10, pad=6)
        ax2d.set_title("2-D energy landscape", fontsize=10, pad=6)
        fig.suptitle(task_id.replace("_", " "), fontsize=11, y=1.005)
        save_figure(fig, f"fig3/landscape_{task_id}", formats=("pdf",))
        landscapes[task_id] = data
        print(f"[fig3] landscape -> landscape_{task_id}.pdf")
    return landscapes


REPRESENTATIVE_FRAGB_SYSTEMS: List[str] = [
    "7RT4_G12D_Frag-B",
    "8AZV_WT_Frag-B",
]

ORACLE_PROXIMITY_TAU_A: float = 3.0


def oracle_proximity(rmsd: np.ndarray) -> np.ndarray:
    """Map a kabsch RMSD array (Å) to oracle proximity in (0, 1]."""
    arr = np.asarray(rmsd, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    finite = np.isfinite(arr)
    out[finite] = np.exp(-arr[finite] / ORACLE_PROXIMITY_TAU_A)
    return out


PROXIMITY_CMAP = "RdBu"
RMSD_CMAP = "RdYlBu_r"
ENERGY_CMAP = "RdYlBu_r"


def _kmeans_basins(
    pca_coords: np.ndarray,
    n_basins: int,
    *,
    seed: int = RNG_SEED,
) -> np.ndarray:

    from sklearn.cluster import KMeans

    n = len(pca_coords)
    k = max(1, min(int(n_basins), max(1, n - 1)))
    if k <= 1 or n <= 2:
        return np.zeros(n, dtype=int)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    raw_labels = km.fit_predict(pca_coords)
    return raw_labels.astype(int)


def _basin_stats(
    labels: np.ndarray,
    energies: np.ndarray,
    rmsds: np.ndarray,
    valid_mask: np.ndarray,
) -> "pd.DataFrame":

    import pandas as pd

    rows: List[Dict] = []
    uniq = sorted(set(int(l) for l in labels[valid_mask]))
    for b in uniq:
        sel = (labels == b) & valid_mask
        if not sel.any():
            continue
        rows.append(
            {
                "basin_id": int(b),
                "n": int(sel.sum()),
                "mean_energy": float(np.mean(energies[sel])),
                "mean_rmsd": float(np.mean(rmsds[sel])),
                "median_rmsd": float(np.median(rmsds[sel])),
                "min_rmsd": float(np.min(rmsds[sel])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_energy").reset_index(drop=True)


def _basin_proximity_stats(
    labels: np.ndarray,
    energies: np.ndarray,
    proximities: np.ndarray,
    valid_mask: np.ndarray,
) -> "pd.DataFrame":
    """Aggregate per-basin (n, mean energy, mean proximity, max proximity)."""
    import pandas as pd

    rows: List[Dict] = []
    uniq = sorted(set(int(l) for l in labels[valid_mask]))
    for b in uniq:
        sel = (labels == b) & valid_mask
        if not sel.any():
            continue
        rows.append(
            {
                "basin_id": int(b),
                "n": int(sel.sum()),
                "mean_energy": float(np.mean(energies[sel])),
                "mean_proximity": float(np.mean(proximities[sel])),
                "max_proximity": float(np.max(proximities[sel])),
                "median_proximity": float(np.median(proximities[sel])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_energy").reset_index(drop=True)


def _panel_proximity_projection(
    ax: plt.Axes,
    data,
    *,
    fs: float = 0.85,
    title: Optional[str] = None,
    add_colorbar: bool = True,
) -> None:
    """B-panel: scatter of PC1 / PC2 coloured by **oracle proximity**."""
    extras = getattr(data, "_extras", None)
    if extras is None:
        ax.text(
            0.5,
            0.5,
            "no kabsch_rmsd / is_anchor data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rmsds = extras["kabsch_rmsds"]
    is_anchor = extras["is_anchor"]
    pca = data.pca_coords

    keep = (~is_anchor) & np.isfinite(rmsds)
    if not keep.any():
        ax.text(
            0.5,
            0.5,
            "no usable RMSD samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    x = pca[keep, 0]
    y = pca[keep, 1]
    p = oracle_proximity(rmsds[keep])
    r = rmsds[keep]

    sc = ax.scatter(
        x,
        y,
        c=p,
        cmap=PROXIMITY_CMAP,
        vmin=0.0,
        vmax=1.0,
        s=18 * fs,
        edgecolors="white",
        linewidths=0.3,
        alpha=0.95,
        zorder=3,
    )

    bi = int(np.argmax(p))
    ax.scatter(
        [x[bi]],
        [y[bi]],
        marker="*",
        s=180 * fs,
        facecolor="#ffffff",
        edgecolor="#222222",
        linewidths=1.0,
        zorder=5,
        label=f"best  prox={p[bi]:.2f}  (RMSD {r[bi]:.2f} \u00c5)",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8.0 * fs)

    ax.set_xlabel(
        f"PC1 ({100 * data.pca_explained_variance[0]:.1f}%)", fontsize=9.0 * fs
    )
    ax.set_ylabel(
        f"PC2 ({100 * data.pca_explained_variance[1]:.1f}%)", fontsize=9.0 * fs
    )
    ax.tick_params(axis="both", labelsize=8.0 * fs)
    if title is not None:
        ax.set_title(title, fontsize=10.0 * fs, pad=4)

    if add_colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=8.0 * fs)
        cbar.set_label(
            f"oracle proximity  exp(\u2212RMSD/{ORACLE_PROXIMITY_TAU_A:g} \u00c5)",
            fontsize=9.0 * fs,
        )


def _panel_energy_filter_sweep(
    ax: plt.Axes,
    data,
    *,
    fs: float = 0.85,
    title: Optional[str] = None,
    rmsd_hit_threshold_a: float = 3.0,
    q_grid: Optional[np.ndarray] = None,
) -> None:
    """C-panel (indicator 1): energy-percentile native-likeness sweep."""
    extras = getattr(data, "_extras", None)
    if extras is None:
        ax.text(
            0.5,
            0.5,
            "no kabsch_rmsd / is_anchor data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rmsds = extras["kabsch_rmsds"]
    is_anchor = extras["is_anchor"]
    energies = extras["energies"]

    valid = (~is_anchor) & np.isfinite(rmsds) & np.isfinite(energies)
    if valid.sum() < 10:
        ax.text(
            0.5,
            0.5,
            "too few non-anchor samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    e = energies[valid]
    r = rmsds[valid]
    p = oracle_proximity(r)
    n = len(e)

    if q_grid is None:
        q_grid = np.array(
            [1, 2, 5, 10, 15, 20, 25, 33, 50, 67, 75, 90, 100], dtype=float
        )

    order = np.argsort(e)
    p_sorted = p[order]
    r_sorted = r[order]

    means: List[float] = []
    bests: List[float] = []
    fracs: List[float] = []
    for q in q_grid:
        k = max(1, int(round(n * q / 100.0)))
        sub_p = p_sorted[:k]
        sub_r = r_sorted[:k]
        means.append(float(np.mean(sub_p)))
        bests.append(float(np.max(sub_p)))
        fracs.append(float(np.mean(sub_r <= rmsd_hit_threshold_a)))
    means = np.asarray(means)
    bests = np.asarray(bests)
    fracs = np.asarray(fracs)

    mean_all = float(np.mean(p))
    frac_all = float(np.mean(r <= rmsd_hit_threshold_a))

    color_mean = "#0072b2"
    color_best = "#d55e00"
    color_frac = "#009e73"

    ax.plot(
        q_grid,
        means,
        "-o",
        color=color_mean,
        lw=1.4,
        ms=4.5 * fs,
        label="mean proximity",
        zorder=4,
    )
    ax.plot(
        q_grid,
        bests,
        "--^",
        color=color_best,
        lw=1.1,
        ms=4.5 * fs,
        label="best proximity",
        zorder=4,
    )
    ax.plot(
        q_grid,
        fracs,
        ":s",
        color=color_frac,
        lw=1.1,
        ms=4.5 * fs,
        label=f"frac RMSD \u2264 {rmsd_hit_threshold_a:g} \u00c5",
        zorder=4,
    )

    ax.axhline(mean_all, color=color_mean, lw=0.6, alpha=0.45, linestyle="-", zorder=2)
    ax.axhline(frac_all, color=color_frac, lw=0.6, alpha=0.45, linestyle=":", zorder=2)

    if 10 in list(q_grid):
        i10 = int(np.where(q_grid == 10)[0][0])
        en_mean = (means[i10] / mean_all) if mean_all > 1e-9 else float("nan")
        en_frac = (fracs[i10] / frac_all) if frac_all > 1e-9 else float("nan")
        ax.text(
            0.97,
            0.04,
            f"@ top-10%\nmean enrich = {en_mean:.2f}\u00d7\n"
            f"hit-rate enrich = {en_frac:.2f}\u00d7",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0 * fs,
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5, alpha=0.9
            ),
        )

    ax.set_xscale("log")
    ax.set_xlim(0.9, 110.0)
    ax.set_xticks([1, 5, 10, 25, 50, 100])
    ax.set_xticklabels(["1%", "5%", "10%", "25%", "50%", "100%"])
    y_top = max(1.02, float(bests.max()) * 1.10)
    ax.set_ylim(0.0, y_top)

    ax.set_xlabel("energy filter — top q% lowest energy retained", fontsize=9.0 * fs)
    ax.set_ylabel(
        f"oracle proximity / hit fraction  (\u03c4={ORACLE_PROXIMITY_TAU_A:g} \u00c5)",
        fontsize=9.0 * fs,
    )
    ax.tick_params(axis="both", labelsize=8.0 * fs)
    ax.legend(loc="upper left", frameon=False, fontsize=8.0 * fs)

    if title is not None:
        ax.set_title(title, fontsize=10.0 * fs, pad=4)


def _panel_rmsd_projection(
    ax: plt.Axes,
    data,
    *,
    fs: float = 0.85,
    title: Optional[str] = None,
    add_colorbar: bool = True,
) -> None:

    extras = getattr(data, "_extras", None)
    if extras is None:
        ax.text(
            0.5,
            0.5,
            "no kabsch_rmsd / is_anchor data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rmsds = extras["kabsch_rmsds"]
    is_anchor = extras["is_anchor"]
    pca = data.pca_coords

    keep = (~is_anchor) & np.isfinite(rmsds)
    if not keep.any():
        ax.text(
            0.5,
            0.5,
            "no usable RMSD samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    x = pca[keep, 0]
    y = pca[keep, 1]
    r = rmsds[keep]

    sc = ax.scatter(
        x,
        y,
        c=r,
        cmap=RMSD_CMAP,
        vmin=float(np.min(r)),
        vmax=float(np.max(r)),
        s=18 * fs,
        edgecolors="white",
        linewidths=0.3,
        alpha=0.95,
        zorder=3,
    )

    bi = int(np.argmin(r))
    ax.scatter(
        [x[bi]],
        [y[bi]],
        marker="*",
        s=180 * fs,
        facecolor="#ffffff",
        edgecolor="#222222",
        linewidths=1.0,
        zorder=5,
        label=f"best  RMSD = {r[bi]:.2f} \u00c5",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8.0 * fs)

    ax.set_xlabel(
        f"PC1 ({100 * data.pca_explained_variance[0]:.1f}%)", fontsize=9.0 * fs
    )
    ax.set_ylabel(
        f"PC2 ({100 * data.pca_explained_variance[1]:.1f}%)", fontsize=9.0 * fs
    )
    ax.tick_params(axis="both", labelsize=8.0 * fs)
    if title is not None:
        ax.set_title(title, fontsize=10.0 * fs, pad=4)

    if add_colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=8.0 * fs)
        cbar.set_label("kabsch RMSD (\u00c5)", fontsize=9.0 * fs)


def _panel_basin_2axes_scatter(
    ax: plt.Axes,
    data,
    *,
    fs: float = 0.85,
    title: Optional[str] = None,
    add_colorbar: bool = True,
) -> None:

    extras = getattr(data, "_extras", None)
    if extras is None:
        ax.text(
            0.5,
            0.5,
            "no kabsch_rmsd data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rmsds = extras["kabsch_rmsds"]
    is_anchor = extras["is_anchor"]
    energies = extras["energies"]
    valid = (~is_anchor) & np.isfinite(rmsds) & np.isfinite(energies)
    if valid.sum() < 4:
        ax.text(
            0.5,
            0.5,
            "too few non-anchor samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    n_basins = int(getattr(data, "n_basins", 1))
    labels = _kmeans_basins(data.pca_coords, n_basins)
    stats = _basin_stats(labels, energies, rmsds, valid)
    if stats.empty:
        ax.text(
            0.5,
            0.5,
            "no valid basins",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    e_vals = stats["mean_energy"].to_numpy()
    r_vals = stats["mean_rmsd"].to_numpy()
    n_vals = stats["n"].to_numpy()

    n_max = max(int(n_vals.max()), 1)
    sizes = 40.0 + 240.0 * (n_vals / n_max)

    sc = ax.scatter(
        e_vals,
        r_vals,
        c=e_vals,
        cmap=ENERGY_CMAP,
        vmin=float(np.min(e_vals)),
        vmax=float(np.max(e_vals)),
        s=sizes * fs,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
        zorder=3,
    )

    if len(e_vals) >= 3:
        slope, intercept = np.polyfit(e_vals, r_vals, 1)
        xs = np.linspace(e_vals.min(), e_vals.max(), 50)
        ax.plot(
            xs,
            slope * xs + intercept,
            linestyle="--",
            color="#666666",
            lw=0.8,
            alpha=0.7,
            zorder=2,
        )
        if len(e_vals) > 1:
            r_pearson = float(np.corrcoef(e_vals, r_vals)[0, 1])
        else:
            r_pearson = float("nan")
        ax.text(
            0.97,
            0.96,
            f"basins n = {len(e_vals)}\nr = {r_pearson:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0 * fs,
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5, alpha=0.9
            ),
        )

    ax.set_xlabel("basin mean energy (a.u.)", fontsize=9.0 * fs)
    ax.set_ylabel("basin mean RMSD (\u00c5)", fontsize=9.0 * fs)
    ax.tick_params(axis="both", labelsize=8.0 * fs)
    if title is not None:
        ax.set_title(title, fontsize=10.0 * fs, pad=4)
    if add_colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=8.0 * fs)
        cbar.set_label("basin mean energy", fontsize=9.0 * fs)


def _panel_basin_proximity_scatter(
    ax: plt.Axes,
    data,
    *,
    fs: float = 0.85,
    title: Optional[str] = None,
    add_colorbar: bool = True,
) -> None:
    """Per-basin scatter of (mean energy, mean oracle proximity)."""
    extras = getattr(data, "_extras", None)
    if extras is None:
        ax.text(
            0.5,
            0.5,
            "no kabsch_rmsd data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    rmsds = extras["kabsch_rmsds"]
    is_anchor = extras["is_anchor"]
    energies = extras["energies"]
    valid = (~is_anchor) & np.isfinite(rmsds) & np.isfinite(energies)
    if valid.sum() < 4:
        ax.text(
            0.5,
            0.5,
            "too few non-anchor samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    proximities = oracle_proximity(rmsds)

    n_basins = int(getattr(data, "n_basins", 1))
    labels = _kmeans_basins(data.pca_coords, n_basins)
    stats = _basin_proximity_stats(labels, energies, proximities, valid)
    if stats.empty:
        ax.text(
            0.5,
            0.5,
            "no valid basins",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    e_vals = stats["mean_energy"].to_numpy()
    p_vals = stats["mean_proximity"].to_numpy()
    n_vals = stats["n"].to_numpy()

    n_max = max(int(n_vals.max()), 1)
    sizes = 40.0 + 240.0 * (n_vals / n_max)

    sc = ax.scatter(
        e_vals,
        p_vals,
        c=e_vals,
        cmap=ENERGY_CMAP,
        vmin=float(np.min(e_vals)),
        vmax=float(np.max(e_vals)),
        s=sizes * fs,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
        zorder=3,
    )

    if len(e_vals) >= 3:
        slope, intercept = np.polyfit(e_vals, p_vals, 1)
        xs = np.linspace(e_vals.min(), e_vals.max(), 50)
        ax.plot(
            xs,
            slope * xs + intercept,
            linestyle="--",
            color="#666666",
            lw=0.8,
            alpha=0.7,
            zorder=2,
        )
        if len(e_vals) > 1:
            r_pearson = float(np.corrcoef(e_vals, p_vals)[0, 1])
        else:
            r_pearson = float("nan")
        ax.text(
            0.97,
            0.96,
            f"basins n = {len(e_vals)}\nr = {r_pearson:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.0 * fs,
            bbox=dict(
                boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.5, alpha=0.9
            ),
        )

    ax.set_xlabel("basin mean energy (a.u.)", fontsize=9.0 * fs)
    ax.set_ylabel(
        f"basin mean oracle proximity  (\u03c4={ORACLE_PROXIMITY_TAU_A:g} \u00c5)",
        fontsize=9.0 * fs,
    )
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=8.0 * fs)
    if title is not None:
        ax.set_title(title, fontsize=10.0 * fs, pad=4)
    if add_colorbar:
        cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=8.0 * fs)
        cbar.set_label("basin mean energy", fontsize=9.0 * fs)


_panel_basin_2axes = _panel_basin_2axes_scatter
_panel_basin_proximity = _panel_basin_proximity_scatter


def render_landscape_2d_standalones(
    landscapes: Dict[str, object],
    *,
    cfg,
    plotter,
) -> Dict[str, List[str]]:
    """Save each 2-D landscape contour as its own standalone PDF/PNG."""
    out: Dict[str, List[str]] = {}
    for sys_id, data in landscapes.items():
        fig = plt.figure(figsize=(4.6, 4.2), constrained_layout=True)
        ax = fig.add_subplot(111)
        _draw_landscape_2d_only(ax, data, plotter, cfg, fs=1.25)
        ax.set_title(sys_id.replace("_", " "), fontsize=16, pad=8)
        paths = save_figure(
            fig,
            f"fig3/landscape_2d_{sys_id}",
            formats=("pdf", "png"),
        )
        out[sys_id] = paths
        for p in paths:
            print(f"[fig3] 2D-only -> {p}")
    return out


def render_per_system_extras(
    landscapes: Dict[str, object],
) -> None:

    panels = [
        ("proximity_projection", _panel_proximity_projection),
        ("rmsd_projection", _panel_rmsd_projection),
        ("basin_2axes", _panel_basin_2axes_scatter),
        ("basin_proximity", _panel_basin_proximity_scatter),
        ("energy_filter", _panel_energy_filter_sweep),
    ]
    for sys_id, data in landscapes.items():
        for fname, fn in panels:
            fig = plt.figure(figsize=(5.6, 4.4), constrained_layout=True)
            ax = fig.add_subplot(111)
            fn(
                ax,
                data,
                fs=0.95,
                title=f"{sys_id.replace('_', ' ')} \u2014 "
                f"{fname.replace('_', ' ')}",
            )
            paths = save_figure(
                fig,
                f"fig3/{fname}_{sys_id}",
                formats=("pdf", "png"),
            )
            for p in paths:
                print(f"[fig3] {fname} -> {p}")
            plt.close(fig)


def _draw_landscape_2d_only(
    ax2d: plt.Axes,
    data,
    plotter,
    cfg,
    *,
    fs: float = 0.85,
):
    """Render only the 2-D contour panel into ``ax2d``."""
    import matplotlib.colors as mcolors

    energies = data.energies
    e_min = float(energies.min())
    e_max_display = float(np.percentile(energies, cfg.energy_percentile_clip))
    norm = mcolors.Normalize(vmin=e_min, vmax=e_max_display)
    cmap = mpl.colormaps[cfg.colormap]
    plotter._panel_2d_contour(ax2d, data, norm, cmap, e_max_display, fs)


def render_composite(
    pocket_pngs: Dict[str, Path],
    landscapes: Dict[str, "tuple"],
    *,
    cfg,
    plotter,
) -> plt.Figure:
    """Stitch the pocket row + landscape grid into one composite figure."""
    pdbs = [p for p in PDB_ORDER if p in pocket_pngs]
    n_pockets = len(pdbs)

    landscape_items: List[Tuple[str, object]] = []
    for pdb in pdbs:
        for sys_id, data in landscapes.items():
            if sys_id.startswith(pdb + "_") and sys_id.endswith("_Frag-B"):
                landscape_items.append((sys_id, data))
                break
    n_land = len(landscape_items)

    pockets_row0 = [
        p for p in ["7RT4", "7RT1", "7RPZ", "6GJ6", "6GJ8"] if p in pocket_pngs
    ][:4]
    pockets_row1 = [p for p in ["8AZV", "8AZX", "8AZY", "8AZZ"] if p in pocket_pngs][:4]
    placed = set(pockets_row0 + pockets_row1)
    leftover = [p for p in pdbs if p not in placed]
    pockets_row0 = (pockets_row0 + leftover)[:4]

    n_outer_cols = 4
    fig_w = 16.0
    fig_h = 16.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer = fig.add_gridspec(
        nrows=4,
        ncols=n_outer_cols,
        height_ratios=[1.0, 1.0, 1.05, 1.05],
        hspace=0.18,
        wspace=0.15,
        left=0.05,
        right=0.97,
        top=0.975,
        bottom=0.04,
    )

    pocket_axes: List[plt.Axes] = []
    for r, pdbs_in_row in enumerate([pockets_row0, pockets_row1]):
        for i, pdb in enumerate(pdbs_in_row):
            ax = fig.add_subplot(outer[r, i])
            ax.set_axis_off()
            png = pocket_pngs[pdb]
            try:
                img = crop_to_highlight(plt.imread(png))
                ax.imshow(img, aspect="equal")
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"{pdb}\n(no image)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                print(f"[fig3] composite: failed to embed {png}: {e}")
            ligand = PDB_LIGAND.get(pdb, "")
            mutation = PDB_MUTATION.get(pdb, "")
            parts = (
                [pdb] + ([ligand] if ligand else []) + ([mutation] if mutation else [])
            )
            title = "\n".join(parts)
            ax.set_title(title, fontsize=9.5, pad=4, linespacing=1.05)
            pocket_axes.append(ax)

    for k, (sys_id, data) in enumerate(landscape_items):
        row = 2 + (k // n_outer_cols)
        col = k % n_outer_cols
        ax2d = fig.add_subplot(outer[row, col])
        _draw_landscape_2d_only(ax2d, data, plotter, cfg, fs=0.85)
        ax2d.set_title(sys_id.replace("_", " "), fontsize=10, pad=5)

    return fig


def main() -> int:
    out_root = PROJECT_ROOT / "paper_img" / "fig3"
    out_root.mkdir(parents=True, exist_ok=True)

    print("[fig3] Resolving fragment ranges from CSV ...")
    frag_ranges = load_fragment_ranges()
    if not frag_ranges:
        print("[fig3] No fragments resolved; aborting.")
        return 1
    print(f"[fig3] resolved {len(frag_ranges)} PDBs: " f"{sorted(frag_ranges.keys())}")

    print("[fig3] Rendering pockets via PyMOL ...")
    try:
        find_pymol()
    except FileNotFoundError as e:
        print(f"[fig3] PyMOL unavailable: {e}")
        pocket_pngs = {}
    else:
        pocket_pngs = render_pockets(frag_ranges, out_root)

    print("[fig3] Building FragB landscapes ...")
    from landscape.config import LandscapeConfig
    from landscape.builder import LandscapeBuilder, _estimate_basins
    from landscape.visualizer import LandscapePlotter

    cfg = LandscapeConfig()
    builder = LandscapeBuilder(cfg)
    plotter = LandscapePlotter(cfg)
    rng = np.random.default_rng(RNG_SEED)

    landscapes = render_landscapes_standalone(
        out_root,
        cfg=cfg,
        builder=builder,
        estimate_basins=_estimate_basins,
        plotter=plotter,
        rng=rng,
        allowed_pdbs=set(frag_ranges.keys()),
    )

    render_landscape_2d_standalones(landscapes, cfg=cfg, plotter=plotter)

    render_per_system_extras(landscapes)

    print("[fig3] Assembling composite ...")
    if not pocket_pngs and not landscapes:
        print("[fig3] no panels available; skipping composite.")
        return 1
    fig = render_composite(pocket_pngs, landscapes, cfg=cfg, plotter=plotter)
    paths = save_figure(fig, "fig3", formats=("pdf",))
    print(f"[fig3] composite -> {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

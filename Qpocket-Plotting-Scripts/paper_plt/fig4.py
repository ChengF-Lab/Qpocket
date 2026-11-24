"""Render pocket and ensemble-response panels."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_plt.plt import (
    apply_theme,
    save_figure,
    output_root,
    FRAGMENT_COLORS,
    MODULE_COLORS,
    LIGAND_COLORS,
    LIGAND_ORDER,
    GRID_GREY,
    ANNOT_GREY,
    add_panel_label,
    style_axes,
    clean_spines,
    apply_grid,
)
from paper_plt.pymol_vis.runner import find_pymol, run_pml


TBL_FRAG_ANNOT = (
    PROJECT_ROOT
    / "result"
    / "kras_correlation_result_package"
    / "tables"
    / "fragment_functional_annotation_paper.csv"
)
TBL_FRAG_CORE = (
    PROJECT_ROOT
    / "result"
    / "kras_correlation_result_package"
    / "tables"
    / "fragment_core_response_summary.csv"
)
TBL_FRAGB_HEATMAP = (
    PROJECT_ROOT
    / "result"
    / "landscape_rmsd_enriched"
    / "rmsd_top1000"
    / "tables"
    / "ligand_basin_response_Frag-B.csv"
)
LIGAND_BASIN_TBL_DIR = (
    PROJECT_ROOT / "result" / "landscape_rmsd_enriched" / "rmsd_top1000" / "tables"
)
TBL_FRAGB_PROJECTED = (
    PROJECT_ROOT
    / "result"
    / "landscape_rmsd_enriched"
    / "rmsd_top1000"
    / "tables"
    / "projected_samples_Frag-B.csv"
)
TBL_POOL_SUMMARY = (
    PROJECT_ROOT
    / "result"
    / "kras_correlation_result_package"
    / "tables"
    / "pool_summary_for_fragB_fragF_fragA.csv"
)

POOL_ORDER: List[str] = [
    "sample500",
    "rmsd_top200",
    "rmsd_top500",
    "rmsd_top800",
    "rmsd_top1000",
    "rmsd_top1500",
    "rmsd_top2000",
    "rmsd_top3000",
    "rmsd_top5000",
    "rmsd_top10pct",
    "rmsd_top20pct",
    "rmsd_top30pct",
    "rmsd_top40pct",
]
POOL_DISPLAY: Dict[str, str] = {
    "sample500": "s-500",
    "rmsd_top200": "top200",
    "rmsd_top500": "top500",
    "rmsd_top800": "top800",
    "rmsd_top1000": "top1k",
    "rmsd_top1500": "top1.5k",
    "rmsd_top2000": "top2k",
    "rmsd_top3000": "top3k",
    "rmsd_top5000": "top5k",
    "rmsd_top10pct": "top10%",
    "rmsd_top20pct": "top20%",
    "rmsd_top30pct": "top30%",
    "rmsd_top40pct": "top40%",
}

DIFF_DENSITY_DIR = PROJECT_ROOT / "result" / "landscape_2000" / "figures"
DIFF_DENSITY_TRIPTYCH: List[Tuple[str, str, str]] = [
    (
        "ΔP — G12D − G12C",
        "differential_density_G12D_vs_G12C_Frag-B.png",
        "mutation contrast",
    ),
    (
        "ΔP — OFU − EZZ",
        "differential_density_OFU_vs_EZZ_Frag-B.png",
        "ligand contrast (Switch-II vs Switch-II groove)",
    ),
    (
        "ΔP — OFU − F0K",
        "differential_density_OFU_vs_F0K_Frag-B.png",
        "ligand contrast (Switch-II vs BI-2852)",
    ),
]

KRAS_PDB_DIR = PROJECT_ROOT / "kras_select_systems"
RUN_V2_DIR = Path(
    os.environ.get("QPOCKET_RUN_DIR", PROJECT_ROOT / "inputs" / "ensembles")
)
POCKET_FULL_DIR = Path(os.environ.get("QPOCKET_POCKET_DIR", PROJECT_ROOT / "inputs" / "pockets"))

OUT_BASE = Path(output_root())
OUT_FIG4_DIR = OUT_BASE / "fig4"
OUT_PYMOL_CACHE = OUT_FIG4_DIR / "_pymol_cache"

REPRESENTATIVE_POOL_FRAGB = "rmsd_top1000"

HEATMAP_CMAP = "RdBu_r"

PANEL_A_PDB = "8AZV"

PANEL_B_SYSTEMS: List[Tuple[str, str, str, str]] = [
    ("7RT4", "6IC", "OFU", "Compound 5B"),
    ("7RT1", "7L8", "6IC", "Compound 15"),
    ("7RPZ", "OFU", "7L8", "MRTX1133"),
    ("6GJ8", "F0K", "F0K", "BI-2852"),
    ("8AZV", "7IZ", "7IZ", "BI-2865 (WT)"),
]

FRAGB_START_RESI = 54
FRAGB_END_RESI = 68
FRAGB_KEY_RESIDUES = [60, 61, 64]
FRAGB_KEY_RESIDUE_LABELS = {60: "G60", 61: "Q61", 64: "Y64"}

PYMOL_RAY_PX = 1200
PYMOL_RAY_PX_SMALL = 900

PANEL_D_FRAGMENTS = ["Frag-B", "Frag-F", "Frag-A"]

_MODULE_GROUP_TO_KEY: Dict[str, str] = {
    "mutation_site_module": "P-loop_mutation_site",
    "ligand_contact_module": "Switch-II_ligand_contact",
    "allosteric_module": "Alpha3_allosteric",
    "peripheral_low_power_module": "Peripheral",
}
_MODULE_DISPLAY: Dict[str, str] = {
    "mutation_site_module": "P-loop mutation site",
    "ligand_contact_module": "Switch-II ligand-contact",
    "allosteric_module": "α3 allosteric",
    "peripheral_low_power_module": "peripheral / low-power",
}


def _parse_residues(rng: str) -> Tuple[int, int]:
    rng = str(rng).replace("\u2013", "-").replace("\u2014", "-")
    a, b = rng.split("-")
    return int(a), int(b)


def load_fragment_annotation() -> pd.DataFrame:
    df = pd.read_csv(TBL_FRAG_ANNOT)
    df["start_resi"], df["end_resi"] = zip(*df["residues"].map(_parse_residues))
    return df


def load_fragment_core_summary() -> pd.DataFrame:
    return pd.read_csv(TBL_FRAG_CORE)


def load_basin_response(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    delta_cols = [c for c in df.columns if c.startswith("delta_B")]
    out = df.set_index("ligand")[delta_cols].copy()
    out.columns = [c.replace("delta_", "") for c in out.columns]
    return out


def load_fragb_projected_samples() -> pd.DataFrame:
    return pd.read_csv(TBL_FRAGB_PROJECTED, low_memory=False)


def load_pool_summary() -> pd.DataFrame:

    return pd.read_csv(TBL_POOL_SUMMARY)


def load_ligand_basin_response(fragment: str) -> pd.DataFrame:
    """Per-ligand basin-occupancy fingerprint for one fragment."""
    path = LIGAND_BASIN_TBL_DIR / f"ligand_basin_response_{fragment}.csv"
    return pd.read_csv(path)


def load_projected_samples(fragment: str) -> pd.DataFrame:
    """Per-sample (PC1, PC2, ligand, ...) projection for one fragment."""
    path = (
        PROJECT_ROOT
        / "result"
        / "landscape_2000"
        / "tables"
        / f"projected_samples_{fragment}.csv"
    )
    return pd.read_csv(path, low_memory=False)


RGB = Tuple[int, int, int]


def _hex_to_rgb255(hex_str: str) -> RGB:
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_str(rgb: RGB) -> str:
    return f"[{int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}]"


def _darken(rgb: RGB, factor: float = 0.45) -> RGB:
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))


def _aa_residues_in_chain(pdb_path: Path, chain: str = "A") -> Dict[int, str]:
    """``{resi: 1-letter-aa}`` for chain CA lines. Used to locate fragments."""
    aa3to1 = {
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
            if resn in aa3to1:
                seen[resi] = aa3to1[resn]
    return seen


def _find_fragment_residues(
    resi_aa: Dict[int, str],
    fragment_seq: str,
    *,
    max_mm: int = 2,
) -> Optional[Tuple[int, int]]:
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


def _resolve_fragment_ranges_for_pdb(
    pdb_id: str,
    fragments: pd.DataFrame,
) -> Dict[str, Tuple[int, int]]:
    """Returns ``{Frag-X: (start_resi, end_resi)}`` for the deposited PDB."""
    seq_csv = PROJECT_ROOT / "inputs" / "protein_sequence_summary.csv"
    if not seq_csv.is_file():
        return {
            f["fragment"]: (int(f["start_resi"]), int(f["end_resi"]))
            for _, f in fragments.iterrows()
        }
    df = pd.read_csv(seq_csv)
    df = df[df["protein_id"].str.startswith(pdb_id + "_")]
    pdb_path = KRAS_PDB_DIR / f"{pdb_id}.pdb"
    if not pdb_path.is_file():
        return {}
    resi_aa = _aa_residues_in_chain(pdb_path, "A")
    out: Dict[str, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        frag = row["protein_id"].split("_")[-1]
        seq = str(row.get("sequence", "")).strip()
        if not seq:
            continue
        rng = _find_fragment_residues(resi_aa, seq)
        if rng is not None:
            out[frag] = rng
    return out


def _build_panel_a_pml(
    pdb_path: Path,
    chain: str,
    fragments: Dict[str, Tuple[int, int]],
    output_png: Path,
    *,
    image_size: Tuple[int, int] = (PYMOL_RAY_PX, PYMOL_RAY_PX),
) -> str:
    img_w, img_h = image_size
    PALETTE_GRAY: RGB = (220, 220, 220)
    color_blocks: List[str] = []
    show_blocks: List[str] = []
    for i, (frag, (s, e)) in enumerate(fragments.items()):
        rgb = _hex_to_rgb255(FRAGMENT_COLORS.get(frag, "#888888"))
        color_blocks.append(f"set_color frag_color_{i}, {_rgb_str(rgb)}")
        show_blocks.append(
            f"select frag_{i}, polymer.protein and chain {chain} and resi {s}-{e}\n"
            f"color frag_color_{i}, %frag_{i}\n"
            f"set cartoon_transparency, 0.0, %frag_{i}"
        )
    color_block_str = "\n".join(color_blocks)
    show_block_str = "\n".join(show_blocks)
    return f"""
load {pdb_path}, complex
remove solvent
remove hetatm
hide everything

set specular, 0
set direct, 1
set ambient, 0.5
set ray_trace_mode, 1
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

set_color non_pred_gray, {_rgb_str(PALETTE_GRAY)}
{color_block_str}

show cartoon, polymer.protein
color non_pred_gray, polymer.protein
set cartoon_transparency, 0.20, polymer.protein

{show_block_str}

orient polymer.protein
zoom polymer.protein, 4.0
turn y, 12
turn x, -8

viewport {img_w}, {img_h}
ray {min(img_w, img_h)}
png {output_png}, {img_w}, {img_h}, 300, ray=1
quit
"""


def _build_panel_a_pml_focused(
    pdb_path: Path,
    chain: str,
    fragments: Dict[str, Tuple[int, int]],
    focus_resi: Tuple[int, int],
    output_png: Path,
    *,
    image_size: Tuple[int, int] = (PYMOL_RAY_PX, PYMOL_RAY_PX),
) -> str:

    img_w, img_h = image_size
    fs, fe = focus_resi
    PALETTE_GRAY: RGB = (220, 220, 220)
    color_blocks: List[str] = []
    show_blocks: List[str] = []
    for i, (frag, (s, e)) in enumerate(fragments.items()):
        rgb = _hex_to_rgb255(FRAGMENT_COLORS.get(frag, "#888888"))
        color_blocks.append(f"set_color frag_color_{i}, {_rgb_str(rgb)}")
        show_blocks.append(
            f"select frag_{i}, polymer.protein and chain {chain} and resi {s}-{e}\n"
            f"color frag_color_{i}, %frag_{i}\n"
            f"set cartoon_transparency, 0.0, %frag_{i}"
        )
    color_block_str = "\n".join(color_blocks)
    show_block_str = "\n".join(show_blocks)
    return f"""
load {pdb_path}, complex
remove solvent
remove hetatm
hide everything

set specular, 0
set direct, 1
set ambient, 0.5
set ray_trace_mode, 1
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

set_color non_pred_gray, {_rgb_str(PALETTE_GRAY)}
{color_block_str}

show cartoon, polymer.protein
color non_pred_gray, polymer.protein
set cartoon_transparency, 0.30, polymer.protein

{show_block_str}

select focus_sel, polymer.protein and chain {chain} and resi {fs}-{fe}
center focus_sel
orient focus_sel
zoom focus_sel, 6.0
turn y, 8
turn x, -4

viewport {img_w}, {img_h}
ray {min(img_w, img_h)}
png {output_png}, {img_w}, {img_h}, 300, ray=1
quit
"""


def render_panel_a_pymol_per_fragment(
    fragments_df: pd.DataFrame,
    *,
    force: bool = False,
) -> Dict[str, Path]:

    out: Dict[str, Path] = {}
    OUT_FIG4_DIR.mkdir(parents=True, exist_ok=True)
    pdb_path = KRAS_PDB_DIR / f"{PANEL_A_PDB}.pdb"
    if not pdb_path.is_file():
        print(f"[fig4] Panel A focus: missing PDB {pdb_path}")
        return out

    canonical: Dict[str, Tuple[int, int]] = {
        str(row["fragment"]): (int(row["start_resi"]), int(row["end_resi"]))
        for _, row in fragments_df.iterrows()
    }

    chain_resi = _aa_residues_in_chain(pdb_path, "A")
    if not chain_resi:
        print(f"[fig4] Panel A focus: chain A empty in {pdb_path.name}")
        return out

    try:
        find_pymol()
    except FileNotFoundError as e:
        print(f"[fig4] Panel A focus: PyMOL not found: {e}")
        return out

    for frag, (fs, fe) in canonical.items():
        if not any(r in chain_resi for r in range(fs, fe + 1)):
            print(
                f"[fig4] Panel A focus: {frag} residues {fs}-{fe} "
                f"not in {PANEL_A_PDB} chain A — skipped"
            )
            continue
        out_png = OUT_FIG4_DIR / f"panelA_focus_{frag}.png"
        if out_png.is_file() and not force:
            print(f"Using existing panel A image {out_png.name}")
            out[frag] = out_png
            continue
        pml = _build_panel_a_pml_focused(
            pdb_path=pdb_path.resolve(),
            chain="A",
            fragments=canonical,
            focus_resi=(fs, fe),
            output_png=out_png.resolve(),
        )
        try:
            run_pml(pml, timeout_s=300.0)
        except RuntimeError as e:
            print(f"[fig4] Panel A focus PyMOL failed for {frag}: {e!s:.250s}")
            continue
        if out_png.is_file():
            out[frag] = out_png
            print(f"[fig4] Panel A focus -> {out_png.name}")
    return out


def render_panel_a_pymol(
    fragments_df: pd.DataFrame,
    *,
    force: bool = False,
) -> Optional[Path]:
    OUT_PYMOL_CACHE.mkdir(parents=True, exist_ok=True)
    out_png = OUT_PYMOL_CACHE / f"panelA_{PANEL_A_PDB}_modules.png"
    if out_png.is_file() and not force:
        return out_png
    pdb_path = KRAS_PDB_DIR / f"{PANEL_A_PDB}.pdb"
    if not pdb_path.is_file():
        return None
    fragments_in_pdb = _resolve_fragment_ranges_for_pdb(
        PANEL_A_PDB,
        fragments_df,
    )
    if not fragments_in_pdb:
        return None
    pml = _build_panel_a_pml(
        pdb_path=pdb_path.resolve(),
        chain="A",
        fragments=fragments_in_pdb,
        output_png=out_png.resolve(),
    )
    try:
        find_pymol()
        run_pml(pml, timeout_s=300.0)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[fig4] Panel A PyMOL failed: {e}")
        return None
    return out_png if out_png.is_file() else None


def _crop_to_highlight(
    img: np.ndarray, *, saturation_threshold: float = 0.15, pad_frac: float = 0.04
) -> np.ndarray:
    if img.size == 0 or img.ndim != 3 or img.shape[2] < 3:
        return img
    rgb = img[..., :3].astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    if img.shape[2] >= 4:
        alpha = img[..., 3]
        if alpha.dtype not in (np.float32, np.float64):
            alpha = alpha.astype(np.float32) / 255.0
        mask = (sat > saturation_threshold) & (alpha > 0.0)
    else:
        mask = sat > saturation_threshold
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
    return img[r0:r1, c0:c1]


def _draw_panel_a_image(ax: plt.Axes, fragments_df: pd.DataFrame) -> bool:

    png = render_panel_a_pymol(fragments_df)
    ax.set_axis_off()
    if png is None or not png.is_file():
        ax.text(
            0.5,
            0.5,
            f"PyMOL render unavailable\n(expected at {png})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.0,
        )
        return False
    img = _crop_to_highlight(
        plt.imread(str(png)),
        saturation_threshold=0.10,
        pad_frac=0.02,
    )
    ax.imshow(img, aspect="equal")
    return True


def _draw_panel_a_legend(ax: plt.Axes, fragments_df: pd.DataFrame) -> None:

    ax.set_axis_off()
    rows: List[Tuple[str, str, str]] = []
    for _, row in fragments_df.iterrows():
        rows.append(
            (
                row["fragment"],
                FRAGMENT_COLORS.get(row["fragment"], "#888888"),
                _MODULE_DISPLAY.get(row["module_group"], row["module_group"]),
            )
        )

    n_cols = 3
    col_xs = [0.025, 0.355, 0.685]
    row_ys = [0.74, 0.30]
    swatch_w, swatch_h = 0.055, 0.13

    for i, (frag, hex_c, mod_label) in enumerate(rows):
        col = i % n_cols
        row = i // n_cols
        if row >= len(row_ys):
            break
        x = col_xs[col]
        y = row_ys[row]
        ax.add_patch(
            Rectangle(
                (x, y - swatch_h / 2),
                swatch_w,
                swatch_h,
                facecolor=hex_c,
                edgecolor="black",
                linewidth=0.4,
                transform=ax.transAxes,
            )
        )
        ax.text(
            x + swatch_w + 0.012,
            y + 0.030,
            frag,
            transform=ax.transAxes,
            fontsize=8.0,
            va="center",
            ha="left",
            color=hex_c,
            fontweight="normal",
        )
        ax.text(
            x + swatch_w + 0.012,
            y - 0.045,
            mod_label,
            transform=ax.transAxes,
            fontsize=6.5,
            va="center",
            ha="left",
            color="#444444",
            style="italic",
        )


def draw_panel_a(
    fig: plt.Figure,
    gs_cell,
    fragments_df: pd.DataFrame,
    *,
    title_size: float = 9.5,
    title: str = "KRAS functional modules and fragment annotation",
) -> None:

    ax = fig.add_subplot(gs_cell)
    ax.set_axis_off()
    _draw_panel_a_image(ax, fragments_df)

    rows: List[Tuple[str, str]] = [
        (row["fragment"], FRAGMENT_COLORS.get(row["fragment"], "#888888"))
        for _, row in fragments_df.iterrows()
    ]
    n_per_col = 3
    swatch_w = 0.045
    swatch_h = 0.040
    col_dx = 0.22
    row_dy = 0.060
    base_x = 0.020
    base_y = 0.980
    n_cols = (len(rows) + n_per_col - 1) // n_per_col
    bg_w = col_dx * (n_cols - 1) + 0.18
    bg_h = row_dy * n_per_col + 0.010
    bg_x = base_x - 0.012
    bg_y = base_y - bg_h + 0.010
    ax.add_patch(
        Rectangle(
            (bg_x, bg_y),
            bg_w,
            bg_h,
            facecolor="white",
            edgecolor="#bbbbbb",
            linewidth=0.4,
            alpha=0.85,
            transform=ax.transAxes,
            zorder=9,
        )
    )
    for i, (frag, hex_c) in enumerate(rows):
        col = i // n_per_col
        row = i % n_per_col
        x = base_x + col * col_dx
        y = base_y - row * row_dy
        ax.add_patch(
            Rectangle(
                (x, y - swatch_h),
                swatch_w,
                swatch_h,
                facecolor=hex_c,
                edgecolor="black",
                linewidth=0.4,
                transform=ax.transAxes,
                zorder=10,
            )
        )
        ax.text(
            x + swatch_w + 0.008,
            y - swatch_h / 2,
            frag,
            transform=ax.transAxes,
            fontsize=6.8,
            va="center",
            ha="left",
            color=hex_c,
            fontweight="normal",
            zorder=10,
        )

    bbox = gs_cell.get_position(fig)
    fig.text(
        (bbox.x0 + bbox.x1) / 2.0,
        bbox.y1 + 0.012,
        title,
        ha="center",
        va="bottom",
        fontsize=title_size,
    )


def _build_panel_b_pml(
    pdb_native: Path,
    pdb_predicted: Optional[Path],
    chain: str,
    fragb_resi: Tuple[int, int],
    key_residues: List[int],
    fragb_color: RGB,
    output_png: Path,
    *,
    image_size: Tuple[int, int] = (PYMOL_RAY_PX_SMALL, PYMOL_RAY_PX_SMALL),
) -> str:
    img_w, img_h = image_size
    s, e = fragb_resi
    fragb_color_str = _rgb_str(fragb_color)
    fragb_edge_str = _rgb_str(_darken(fragb_color, 0.55))
    PALETTE_GRAY: RGB = (220, 220, 220)
    PALETTE_LIGAND: RGB = (110, 200, 110)
    PALETTE_LIG_EDGE: RGB = (50, 110, 50)
    key_resi_clause = "+".join(str(r) for r in key_residues)

    if pdb_predicted is not None:
        pdb_to_show_fragb = str(pdb_predicted)
    else:
        pdb_to_show_fragb = str(pdb_native)
    return f"""
load {pdb_to_show_fragb}, fragb_obj
load {pdb_native}, native_obj
remove solvent
hide everything

set specular, 0
set direct, 1
set ambient, 0.5
set ray_trace_mode, 1
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

set_color non_pred_gray, {_rgb_str(PALETTE_GRAY)}
set_color fragb_color, {fragb_color_str}
set_color fragb_edge, {fragb_edge_str}
set_color ligand_main, {_rgb_str(PALETTE_LIGAND)}
set_color ligand_edge, {_rgb_str(PALETTE_LIG_EDGE)}

show cartoon, fragb_obj and polymer.protein and chain {chain}
color non_pred_gray, fragb_obj and polymer.protein and chain {chain}
set cartoon_transparency, 0.45, fragb_obj and polymer.protein and chain {chain}

select fragb_sel, fragb_obj and polymer.protein and chain {chain} and resi {s}-{e}
color fragb_color, %fragb_sel
set cartoon_transparency, 0.0, %fragb_sel

select key_resi, fragb_obj and polymer.protein and chain {chain} and resi {key_resi_clause}
show sticks, %key_resi
color fragb_color, %key_resi and elem C
set stick_radius, 0.18, %key_resi
set stick_transparency, 0.0, %key_resi

select native_lig, native_obj and hetatm and not resn HOH+WAT+SO4+PO4+GOL+EDO+CL+NA+MG+ZN+CA \\
                   and not polymer.protein
align fragb_obj and polymer.protein and chain {chain}, \\
      native_obj and polymer.protein and chain {chain}
create lig_outline, %native_lig
show sticks, %lig_outline
color ligand_edge, %lig_outline
set stick_radius, 0.30, %lig_outline
set stick_transparency, 0.20, %lig_outline
show sticks, %native_lig
color ligand_main, %native_lig and elem C
set stick_radius, 0.18, %native_lig
set stick_transparency, 0.00, %native_lig

select focus_sel, %fragb_sel or %native_lig
center focus_sel
orient focus_sel
zoom focus_sel, 4.5

viewport {img_w}, {img_h}
ray {min(img_w, img_h)}
png {output_png}, {img_w}, {img_h}, 300, ray=1
quit
"""


def render_panel_b_pymol_one(
    pdb_id: str,
    ligand_resname: str,
    ligand_color_key: str,
    label: str,
    fragments_df: pd.DataFrame,
    *,
    force: bool = False,
) -> Optional[Path]:
    OUT_PYMOL_CACHE.mkdir(parents=True, exist_ok=True)
    out_png = OUT_PYMOL_CACHE / f"panelB_{pdb_id}_{ligand_resname}.png"
    if out_png.is_file() and not force:
        return out_png

    pdb_native = KRAS_PDB_DIR / f"{pdb_id}.pdb"
    pdb_predicted = POCKET_FULL_DIR / pdb_id / "embedded_full_pocket.pdb"
    if not pdb_native.is_file():
        return None

    use_pdb = pdb_predicted if pdb_predicted.is_file() else pdb_native
    resi_aa = _aa_residues_in_chain(use_pdb, "A")
    fragb_seq = ""
    seq_csv = PROJECT_ROOT / "inputs" / "protein_sequence_summary.csv"
    if seq_csv.is_file():
        df = pd.read_csv(seq_csv)
        sub = df[df["protein_id"] == f"{pdb_id}_Frag-B"]
        if not sub.empty:
            fragb_seq = str(sub["sequence"].iloc[0])
    rng = _find_fragment_residues(resi_aa, fragb_seq) if fragb_seq else None
    if rng is None:
        rng = (FRAGB_START_RESI, FRAGB_END_RESI)

    fragb_color = _hex_to_rgb255(LIGAND_COLORS.get(ligand_color_key, "#888888"))
    pml = _build_panel_b_pml(
        pdb_native=pdb_native.resolve(),
        pdb_predicted=pdb_predicted.resolve() if pdb_predicted.is_file() else None,
        chain="A",
        fragb_resi=rng,
        key_residues=FRAGB_KEY_RESIDUES,
        fragb_color=fragb_color,
        output_png=out_png.resolve(),
    )
    try:
        find_pymol()
        run_pml(pml, timeout_s=300.0)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[fig4] Panel B PyMOL failed for {pdb_id}: {e}")
        return None
    return out_png if out_png.is_file() else None


def draw_panel_b(
    fig: plt.Figure,
    gs_cell,
    fragments_df: pd.DataFrame,
    *,
    title: str = (
        "Frag-B predicted structure across ligand contexts "
        "(ligand-coloured cartoon + native ligand sticks)"
    ),
    title_size: float = 9.5,
    layout: str = "3+2",
) -> None:
    """Render the 5 per-ligand PyMOL panels into ``gs_cell``."""
    if layout == "1x5":
        inner = gs_cell.subgridspec(1, len(PANEL_B_SYSTEMS), wspace=0.10)
        positions = [(0, slice(i, i + 1)) for i in range(len(PANEL_B_SYSTEMS))]
    elif layout == "3+2":
        inner = gs_cell.subgridspec(2, 6, wspace=0.12, hspace=0.30)
        positions = [
            (0, slice(0, 2)),
            (0, slice(2, 4)),
            (0, slice(4, 6)),
            (1, slice(1, 3)),
            (1, slice(3, 5)),
        ]
    else:
        raise ValueError(f"Unknown Panel B layout: {layout!r}")

    for i, (pdb_id, lig_resname, lig_key, label) in enumerate(PANEL_B_SYSTEMS):
        r, c = positions[i]
        ax = fig.add_subplot(inner[r, c])
        ax.set_axis_off()
        png = render_panel_b_pymol_one(
            pdb_id,
            lig_resname,
            lig_key,
            label,
            fragments_df,
        )
        if png is None or not png.is_file():
            ax.text(
                0.5,
                0.5,
                f"{pdb_id}\n(no render)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8.0,
            )
        else:
            img = _crop_to_highlight(
                plt.imread(str(png)), saturation_threshold=0.12, pad_frac=0.03
            )
            ax.imshow(img, aspect="equal")
        color = LIGAND_COLORS.get(lig_key, "#444444")
        ax.set_title(label, fontsize=8.0, color=color, pad=3)

    bbox = inner.get_topmost_subplotspec().get_position(fig)
    fig.text(
        (bbox.x0 + bbox.x1) / 2.0,
        bbox.y1 + 0.015,
        title,
        ha="center",
        va="bottom",
        fontsize=title_size,
    )


def draw_panel_c_diff_density(
    fig: plt.Figure,
    gs_cell,
    *,
    title: str = "Frag-B differential density · ligand and mutation contrasts",
    title_size: float = 9.0,
) -> None:
    """Panel C — three pre-computed Frag-B ΔP(z) difference plots."""
    inner = gs_cell.subgridspec(1, 3, wspace=0.06)

    for i, (label, fname, _) in enumerate(DIFF_DENSITY_TRIPTYCH):
        ax = fig.add_subplot(inner[0, i])
        ax.set_axis_off()
        png_path = DIFF_DENSITY_DIR / fname
        if png_path.is_file():
            try:
                img = plt.imread(str(png_path))
                ax.imshow(img, aspect="equal")
            except Exception as e:
                ax.text(
                    0.5,
                    0.5,
                    f"({label})\nload error",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7.5,
                )
                print(f"[fig4] Panel C: failed to read {png_path}: {e}")
        else:
            ax.text(
                0.5,
                0.5,
                f"({label})\nmissing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7.5,
            )
        ax.set_title(label, fontsize=7.5, pad=3)

    bbox = inner.get_topmost_subplotspec().get_position(fig)
    fig.text(
        (bbox.x0 + bbox.x1) / 2.0,
        bbox.y1 + 0.012,
        title,
        ha="center",
        va="bottom",
        fontsize=title_size,
    )


def draw_panel_c(
    ax: plt.Axes,
    *,
    title: str = "Frag-B provides the strongest projected ligand-response readout",
    title_size: float = 9.5,
) -> None:
    df = load_basin_response(TBL_FRAGB_HEATMAP)
    rows = [l for l in LIGAND_ORDER if l in df.index]
    rest = [l for l in df.index if l not in rows]
    df = df.loc[rows + rest]

    M = df.to_numpy()
    vmax = max(0.05, float(np.nanmax(np.abs(M))))
    im = ax.imshow(M, cmap=HEATMAP_CMAP, vmin=-vmax, vmax=+vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(v) > 0.55 * vmax else "#222222"
            ax.text(
                j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color=color
            )

    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=7.0)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(df.index, fontsize=7.0)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("projected basin", fontsize=8.0)
    ax.set_ylabel("ligand context", fontsize=8.0)
    ax.set_title(title, fontsize=title_size, pad=4)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("occupancy deviation\n(vs baseline)", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    try:
        df_proj = load_fragb_projected_samples()
    except Exception:
        return
    inset = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    for lig in LIGAND_ORDER:
        sub = df_proj[df_proj["ligand"] == lig]
        if sub.empty:
            continue
        inset.scatter(
            sub["PC1"],
            sub["PC2"],
            s=2.0,
            alpha=0.55,
            c=LIGAND_COLORS.get(lig, "#888888"),
            edgecolors="none",
        )
    inset.set_xlabel("PC1", fontsize=6.0, labelpad=1)
    inset.set_ylabel("PC2", fontsize=6.0, labelpad=1)
    inset.tick_params(axis="both", labelsize=5.5, length=2, pad=1)
    inset.set_title("Frag-B PCA · ligand", fontsize=6.5, pad=2)
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    inset.set_facecolor("#fdfdfd")


PANEL_D_POOL_SUBSET: List[str] = [
    "rmsd_top200",
    "rmsd_top500",
    "rmsd_top800",
    "rmsd_top1000",
    "rmsd_top1500",
    "rmsd_top2000",
    "rmsd_top3000",
    "rmsd_top5000",
    "rmsd_top10pct",
    "rmsd_top20pct",
    "rmsd_top30pct",
    "rmsd_top40pct",
]


def _draw_panel_d_bars(
    ax: plt.Axes,
    df_core: pd.DataFrame,
    df_pool: pd.DataFrame,
) -> None:

    from matplotlib.patches import Patch

    rows: List[Tuple[str, float, float, str]] = []
    for frag in PANEL_D_FRAGMENTS:
        core_sub = df_core[df_core["fragment"] == frag]
        pool_sub = df_pool[df_pool["fragment"] == frag]
        if core_sub.empty or pool_sub.empty:
            continue
        lig_v = float(core_sub["ligand_max_L1_best"].iloc[0])
        mut_v = float(pool_sub["max_mutation_L1"].max())
        rows.append((frag, lig_v, mut_v, FRAGMENT_COLORS.get(frag, "#888888")))

    labels = [r[0] for r in rows]
    lig_values = [r[1] for r in rows]
    mut_values = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    xs = np.arange(len(labels), dtype=float)
    width = 0.36
    offset = width / 2 + 0.02

    for x, v, c in zip(xs, lig_values, colors):
        ax.bar(x - offset, v, width=width, color=c, edgecolor="black", linewidth=0.9)
    for x, v, c in zip(xs, mut_values, colors):
        ax.bar(
            x + offset,
            v,
            width=width,
            color=c,
            edgecolor="black",
            linewidth=0.9,
            hatch="////",
            alpha=0.85,
        )

    for x, v in zip(xs, lig_values):
        ax.text(
            x - offset,
            v + 0.03,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
        )
    for x, v in zip(xs, mut_values):
        ax.text(
            x + offset,
            v + 0.03,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
        )

    motif_text: Dict[str, str] = {
        "Frag-A": "P-loop",
        "Frag-B": "Switch-II",
        "Frag-F": "Switch-II",
    }
    for x, frag, c in zip(xs, labels, colors):
        ax.text(
            x,
            -0.16,
            motif_text.get(frag, ""),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.0,
            style="italic",
            color=c,
            weight="bold",
        )

    ref_color = FRAGMENT_COLORS.get("Frag-B", "#1f4e79")
    legend_handles = [
        Patch(facecolor=ref_color, edgecolor="black", linewidth=0.9, label="ligand"),
        Patch(
            facecolor=ref_color,
            edgecolor="black",
            linewidth=0.9,
            hatch="////",
            alpha=0.85,
            label="mutation",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=11.0,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.30,
    )

    apply_grid(ax, axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.7)
    clean_spines(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11.0)
    ax.tick_params(axis="y", labelsize=10.0)
    ax.set_ylabel("max L1 (best across pools)", fontsize=11.0)
    ymax = max(lig_values) if lig_values else 1.0
    ax.set_ylim(0, ymax * 1.25)
    ax.set_title("response amplitude", fontsize=12.0, pad=4)


def _bh_fdr_q(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjustment."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    valid = p[mask]
    m = valid.size
    order = np.argsort(valid)
    ranked = valid[order]
    factor = m / np.arange(1, m + 1, dtype=float)
    q_sorted = np.minimum.accumulate((ranked * factor)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_valid = np.empty_like(q_sorted)
    q_valid[order] = q_sorted
    out[mask] = q_valid
    return out


def _draw_panel_d_amplitude(ax: plt.Axes, df_pool: pd.DataFrame) -> None:

    rmsd_pools = PANEL_D_POOL_SUBSET

    for frag in PANEL_D_FRAGMENTS:
        sub = df_pool[df_pool["fragment"] == frag]
        if sub.empty:
            continue
        sub_idx = sub.set_index("pool_label")
        sub_rmsd = sub_idx.reindex(rmsd_pools)
        ys = sub_rmsd["mean_ligand_L1"].to_numpy(dtype=float)
        es = sub_rmsd["bootstrap_mean_std"].to_numpy(dtype=float)
        ps = _bh_fdr_q(sub_rmsd["ligand_perm_p"].to_numpy(dtype=float))
        color = FRAGMENT_COLORS.get(frag, "#888888")

        xs = np.arange(len(rmsd_pools))
        valid = np.isfinite(ys) & np.isfinite(es)
        if valid.any():
            xs_v = xs[valid]
            ys_v = ys[valid]
            es_v = es[valid]
            ax.fill_between(
                xs_v,
                ys_v - es_v,
                ys_v + es_v,
                color=color,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                xs_v,
                ys_v,
                color=color,
                marker="o",
                markersize=8.0,
                linewidth=2.2,
                label=frag,
                zorder=4,
            )

        for x_pos, y, p in zip(xs, ys, ps):
            if not np.isfinite(p) or not np.isfinite(y):
                continue
            if p <= 0.01:
                ax.scatter(
                    x_pos,
                    y,
                    s=180,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.8,
                    zorder=5,
                )
            elif p <= 0.05:
                ax.scatter(
                    x_pos,
                    y,
                    s=150,
                    facecolor="none",
                    edgecolor="#555555",
                    linewidth=1.1,
                    zorder=5,
                )

    apply_grid(ax, axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.6)
    clean_spines(ax)
    ax.set_xticks(range(len(rmsd_pools)))
    ax.set_xticklabels(
        [POOL_DISPLAY.get(p, p) for p in rmsd_pools],
        fontsize=9.5,
        rotation=45,
        ha="right",
    )
    ax.set_ylabel(
        "mean ligand L1\n(band = bootstrap SE)", fontsize=11.0, linespacing=1.0
    )
    ax.set_xlabel("ligand-conformer pool", fontsize=11.0, labelpad=4)
    ax.set_title("response amplitude vs enrichment", fontsize=12.0, pad=4)
    ax.tick_params(axis="both", labelsize=10.0)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=11.0,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.30,
    )
    ax.text(
        0.02,
        0.04,
        "● q≤0.01    ○ q≤0.05",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#333333",
        va="bottom",
        ha="left",
    )


def _draw_panel_d_significance(ax: plt.Axes, df_pool: pd.DataFrame) -> None:

    rmsd_pools = PANEL_D_POOL_SUBSET
    eps = 1e-6

    def _rolling_median_5(arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        out = np.empty(n, dtype=float)
        half = 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            out[i] = float(np.nanmedian(arr[lo:hi]))
        return out

    for frag in PANEL_D_FRAGMENTS:
        sub = df_pool[df_pool["fragment"] == frag]
        if sub.empty:
            continue
        sub = sub.set_index("pool_label").reindex(rmsd_pools)
        p_raw = sub["ligand_perm_p"].to_numpy(dtype=float)
        q = _bh_fdr_q(p_raw)
        q = np.where(np.isfinite(q) & (q > 0), q, eps)
        ys_raw = -np.log10(q)
        ys = _rolling_median_5(ys_raw)
        color = FRAGMENT_COLORS.get(frag, "#888888")
        xs = np.arange(len(rmsd_pools))
        ax.plot(
            xs,
            ys,
            color=color,
            marker="o",
            markersize=8.0,
            linewidth=2.2,
            label=frag,
            zorder=4,
        )

    for thresh, label in [(0.05, "q = 0.05"), (0.01, "q = 0.01")]:
        y = -np.log10(thresh)
        ax.axhline(y, color=ANNOT_GREY, linewidth=1.0, linestyle="--", alpha=0.75)
        ax.text(
            0.55,
            y,
            label,
            fontsize=9.5,
            color=ANNOT_GREY,
            ha="center",
            va="center",
            transform=ax.get_yaxis_transform(),
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.92),
        )

    apply_grid(ax, axis="y", color=GRID_GREY, linewidth=0.5, alpha=0.6)
    clean_spines(ax)
    ax.set_xticks(range(len(rmsd_pools)))
    ax.set_xticklabels(
        [POOL_DISPLAY.get(p, p) for p in rmsd_pools],
        fontsize=9.5,
        rotation=45,
        ha="right",
    )
    ax.set_xlabel("ligand-conformer pool", fontsize=11.0, labelpad=4)
    ax.set_ylabel("−log\u2081\u2080(BH q)", fontsize=11.0)
    ax.set_title("significance vs enrichment", fontsize=12.0, pad=4)
    ax.tick_params(axis="both", labelsize=10.0)
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.36),
        frameon=False,
        fontsize=10.5,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def draw_panel_d(
    fig: plt.Figure,
    gs_cell,
    df_core: pd.DataFrame,
    df_pool: pd.DataFrame,
    *,
    title: str = "Cross-ligand response concentrates in Switch-II",
    title_size: float = 9.0,
) -> None:

    inner = gs_cell.subgridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.50,
    )
    ax_b = fig.add_subplot(inner[0, 0])
    ax_a = fig.add_subplot(inner[0, 1])
    ax_p = fig.add_subplot(inner[0, 2])

    _draw_panel_d_bars(ax_b, df_core, df_pool)
    _draw_panel_d_amplitude(ax_a, df_pool)
    _draw_panel_d_significance(ax_p, df_pool)

    for ax in (ax_b, ax_a, ax_p):
        ax.set_box_aspect(1.0)

    bbox = inner.get_topmost_subplotspec().get_position(fig)
    fig.text(
        (bbox.x0 + bbox.x1) / 2.0,
        bbox.y1 + 0.012,
        title,
        ha="center",
        va="bottom",
        fontsize=title_size,
    )


def _save_one_panel(
    out_name: str,
    figsize: Tuple[float, float],
    drawer,
    panel_letter: str,
) -> List[str]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    drawer(ax)
    return save_figure(fig, f"fig4/{out_name}", formats=("pdf", "png"))


def _save_panel_b_standalone(
    fragments_df: pd.DataFrame,
) -> List[str]:

    fig = plt.figure(figsize=(8.0, 2.4), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig, left=0.02, right=0.98, top=0.83, bottom=0.05)
    draw_panel_b(
        fig,
        gs[0, 0],
        fragments_df,
        title=(
            "Frag-B predicted structure across ligand " "contexts (ligand-coloured)"
        ),
    )
    return save_figure(
        fig, "fig4/B_fragb_per_ligand_structures", formats=("pdf", "png")
    )


def _save_panel_c_subpanels_standalone() -> List[List[str]]:
    """One standalone PDF/PNG per diff-density image (C1, C2, C3)."""
    out: List[List[str]] = []
    name_map = [
        "C1_fragb_dP_G12D_vs_G12C",
        "C2_fragb_dP_OFU_vs_EZZ",
        "C3_fragb_dP_OFU_vs_F0K",
    ]
    for (label, fname, _), out_name in zip(DIFF_DENSITY_TRIPTYCH, name_map):
        fig = plt.figure(figsize=(4.0, 4.0), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        png_path = DIFF_DENSITY_DIR / fname
        if png_path.is_file():
            img = plt.imread(str(png_path))
            ax.imshow(img, aspect="equal")
        else:
            ax.text(
                0.5,
                0.5,
                f"({label})\nmissing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
        ax.set_title(label, fontsize=10, pad=4)
        paths = save_figure(fig, f"fig4/{out_name}", formats=("pdf", "png"))
        out.append(paths)
    return out


def _save_panel_c_standalone() -> List[str]:

    fig = plt.figure(figsize=(7.6, 2.6), constrained_layout=False)
    gs = GridSpec(
        1,
        1,
        figure=fig,
        left=0.02,
        right=0.98,
        top=0.85,
        bottom=0.04,
    )
    draw_panel_c_diff_density(
        fig,
        gs[0, 0],
        title="Frag-B differential density · ligand and mutation contrasts",
        title_size=9.0,
    )
    return save_figure(
        fig, "fig4/C_fragb_diff_density_triptych", formats=("pdf", "png")
    )


def _save_panel_a_legend_standalone(fragments_df: pd.DataFrame) -> List[str]:
    """Standalone legend-only PDF/PNG with **transparent background**."""
    fig = plt.figure(figsize=(5.4, 1.4), constrained_layout=False)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    _draw_panel_a_legend(ax, fragments_df)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return save_figure(
        fig,
        "fig4/A_legend",
        formats=("pdf", "png"),
        transparent=True,
    )


def _save_panel_a_standalone(fragments_df: pd.DataFrame) -> List[str]:
    fig = plt.figure(figsize=(4.6, 4.6), constrained_layout=False)
    gs = GridSpec(
        1,
        1,
        figure=fig,
        left=0.04,
        right=0.98,
        top=0.92,
        bottom=0.04,
    )
    draw_panel_a(
        fig,
        gs[0, 0],
        fragments_df,
        title="KRAS functional modules · WT 8AZV",
        title_size=10.0,
    )
    return save_figure(fig, "fig4/A_kras_module_schematic", formats=("pdf", "png"))


LANDSCAPE_ABS_SUBDIR = "E_landscape_per_ligand"
LANDSCAPE_DIFF_SUBDIR = "F_diff_landscape_per_ligand"

LIGAND_HET_TO_NAME: Dict[str, str] = {
    "F0K": "BI-2852",
    "6IC": "MRTX1133",
    "7L8": "Compound 15",
    "7IZ": "Compound 5B",
    "EZZ": "Compound 18",
    "OFU": "BI-2865",
}

LIGAND_PANEL_B_ORDER: List[str] = ["F0K", "6IC", "7L8", "7IZ", "OFU"]
_LANDSCAPE_BUILDER = None
_LANDSCAPE_PLOTTER = None
_LANDSCAPE_CFG = None


def _get_landscape_machinery():

    global _LANDSCAPE_BUILDER, _LANDSCAPE_PLOTTER, _LANDSCAPE_CFG
    if _LANDSCAPE_BUILDER is None:
        import sys as _sys

        ps = str(PROJECT_ROOT / "paper_scripts")
        if ps not in _sys.path:
            _sys.path.insert(0, ps)
        from landscape.config import LandscapeConfig
        from landscape.builder import LandscapeBuilder
        from landscape.visualizer import LandscapePlotter

        _LANDSCAPE_CFG = LandscapeConfig()
        _LANDSCAPE_BUILDER = LandscapeBuilder(_LANDSCAPE_CFG)
        _LANDSCAPE_PLOTTER = LandscapePlotter(_LANDSCAPE_CFG)
    return _LANDSCAPE_BUILDER, _LANDSCAPE_PLOTTER, _LANDSCAPE_CFG


def _build_per_ligand_landscape_data(
    df_subset: pd.DataFrame,
    fragment: str,
    *,
    energy_col: str = "score",
) -> "object":

    import sys as _sys

    ps = str(PROJECT_ROOT / "paper_scripts")
    if ps not in _sys.path:
        _sys.path.insert(0, ps)
    from landscape.data_types import LandscapeData, SampledConformation

    builder, _, _cfg = _get_landscape_machinery()

    pca_coords = df_subset[["PC1", "PC2"]].to_numpy(dtype=float)
    energies = df_subset[energy_col].to_numpy(dtype=float)
    N = len(energies)
    if N < 5:
        return None

    grid_x, grid_y, grid_z = builder._interpolate_surface(pca_coords, energies)
    min_idx = int(np.argmin(energies))

    weights = np.ones(N, dtype=float)
    w_sum = float(weights.sum())
    confs = [
        SampledConformation(
            bitstring_key=f"r{i}",
            full_bitstring=f"r{i}",
            sequence="",
            main_positions=np.zeros((1, 3)),
            energy=float(energies[i]),
            energy_terms={},
            count=1,
            prob=float(weights[i]) / w_sum,
            saw_valid=True,
        )
        for i in range(N)
    ]
    pca_var = np.array([np.nan, np.nan])

    return LandscapeData(
        protein_id=fragment,
        sequence="",
        conformations=confs,
        pca_coords=pca_coords,
        pca_explained_variance=pca_var,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        min_energy_idx=min_idx,
        median_energy=float(np.median(energies)),
        funnel_score=0.0,
        n_valid=N,
        n_total_unique=N,
        n_basins=5,
    )


def _load_basin_centers(fragment: str) -> List[Tuple[str, float, float]]:
    path = (
        PROJECT_ROOT
        / "result"
        / "landscape_2000"
        / "tables"
        / f"basin_centers_{fragment}.csv"
    )
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return [
        (str(row["basin_id"]), float(row["center_PC1"]), float(row["center_PC2"]))
        for _, row in df.iterrows()
    ]


def _draw_absolute_landscape(
    ax: plt.Axes,
    df_subset: pd.DataFrame,
    fragment: str,
    ligand: str,
    e_max_display_global: float,
    e_min_display_global: float,
    basin_centers: List[Tuple[str, float, float]],
) -> None:

    import matplotlib.colors as mcolors

    builder, _, cfg = _get_landscape_machinery()
    data = _build_per_ligand_landscape_data(df_subset, fragment)
    if data is None:
        ax.text(
            0.5,
            0.5,
            f"{ligand}: insufficient samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#888888",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    cmap = mpl.colormaps[cfg.colormap]
    norm = mcolors.Normalize(vmin=e_min_display_global, vmax=e_max_display_global)

    GX, GY, GZ = data.grid_x, data.grid_y, data.grid_z
    GZ_clip = np.clip(GZ, None, e_max_display_global)
    levels = np.linspace(float(e_min_display_global), float(e_max_display_global), 28)
    ax.contourf(GX, GY, GZ_clip, levels=levels, cmap=cmap, norm=norm, alpha=0.92)
    ax.contour(
        GX,
        GY,
        GZ_clip,
        levels=levels[::4],
        colors="#555555",
        alpha=0.35,
        linewidths=0.5,
    )

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
            fontsize=9,
            weight="bold",
            color="black",
            zorder=11,
        )

    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"{ligand}  (n={data.n_valid})", fontsize=10, pad=4)


def _draw_diff_landscape(
    ax: plt.Axes,
    df_subset: pd.DataFrame,
    df_all_frag: pd.DataFrame,
    fragment: str,
    ligand: str,
    vabs_global: float,
    basin_centers: List[Tuple[str, float, float]],
    grid_extent: Tuple[float, float, float, float],
) -> None:

    from matplotlib.colors import TwoSlopeNorm

    nb = 60
    sigma = 1.5

    pc_lig = df_subset[["PC1", "PC2"]].to_numpy(dtype=float)
    pc_all = df_all_frag[["PC1", "PC2"]].to_numpy(dtype=float)
    if len(pc_lig) < 50 or len(pc_all) < 50:
        ax.text(
            0.5,
            0.5,
            f"{ligand}: too few samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#888888",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    x0, x1, y0, y1 = grid_extent
    H_l, xe, ye = np.histogram2d(
        pc_lig[:, 0],
        pc_lig[:, 1],
        bins=nb,
        range=[[x0, x1], [y0, y1]],
        density=True,
    )
    H_b, _, _ = np.histogram2d(
        pc_all[:, 0],
        pc_all[:, 1],
        bins=nb,
        range=[[x0, x1], [y0, y1]],
        density=True,
    )
    try:
        from scipy.ndimage import gaussian_filter

        H_l = gaussian_filter(H_l, sigma=sigma)
        H_b = gaussian_filter(H_b, sigma=sigma)
    except Exception:
        pass

    diff = H_l.T - H_b.T
    vabs = max(vabs_global, 1e-12)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = np.linspace(-vabs, vabs, 28)
    ax.contourf(
        xc, yc, diff, levels=levels, cmap="RdBu_r", norm=norm, alpha=0.92, extend="both"
    )
    ax.contour(xc, yc, diff, levels=[0.0], colors="#333333", linewidths=0.6, alpha=0.55)

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
            fontsize=9,
            weight="bold",
            color="black",
            zorder=11,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(f"{ligand}  (n={len(pc_lig)})", fontsize=10, pad=4)


def _save_per_ligand_landscapes(
    *,
    diff: bool,
) -> List[str]:

    fragments = ["Frag-B", "Frag-F", "Frag-A"]
    subdir = LANDSCAPE_DIFF_SUBDIR if diff else LANDSCAPE_ABS_SUBDIR
    out_dir = PROJECT_ROOT / "paper_img" / "fig4" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    builder, _plotter, cfg = _get_landscape_machinery()

    per_frag_data: Dict[str, Dict] = {}
    global_e_min = +np.inf
    global_e_max_clip = -np.inf
    global_vabs = 1e-9

    for frag in fragments:
        df_frag = load_projected_samples(frag)
        if df_frag.empty:
            print(f"[fig4] {frag}: no projected samples — skipped")
            continue
        df_frag = df_frag.dropna(subset=["PC1", "PC2", "score"])

        e = df_frag["score"].to_numpy(dtype=float)
        if len(e) == 0:
            continue
        e_min = float(np.min(e))
        e_max_clip = float(np.percentile(e, cfg.energy_percentile_clip))
        global_e_min = min(global_e_min, e_min)
        global_e_max_clip = max(global_e_max_clip, e_max_clip)

        ligands = sorted(df_frag["ligand"].dropna().unique().tolist())
        if diff:
            x0, x1 = float(df_frag["PC1"].min()), float(df_frag["PC1"].max())
            y0, y1 = float(df_frag["PC2"].min()), float(df_frag["PC2"].max())
            H_b, _, _ = np.histogram2d(
                df_frag["PC1"],
                df_frag["PC2"],
                bins=60,
                range=[[x0, x1], [y0, y1]],
                density=True,
            )
            try:
                from scipy.ndimage import gaussian_filter

                H_b = gaussian_filter(H_b, sigma=1.5)
            except Exception:
                pass
            for lig in ligands:
                sub = df_frag[df_frag["ligand"] == lig]
                if len(sub) < 50:
                    continue
                H_l, _, _ = np.histogram2d(
                    sub["PC1"],
                    sub["PC2"],
                    bins=60,
                    range=[[x0, x1], [y0, y1]],
                    density=True,
                )
                try:
                    from scipy.ndimage import gaussian_filter

                    H_l = gaussian_filter(H_l, sigma=1.5)
                except Exception:
                    pass
                global_vabs = max(global_vabs, float(np.nanmax(np.abs(H_l - H_b))))

        per_frag_data[frag] = {
            "df_frag": df_frag,
            "ligands": ligands,
            "basins": _load_basin_centers(frag),
            "extent": (
                float(df_frag["PC1"].min()),
                float(df_frag["PC1"].max()),
                float(df_frag["PC2"].min()),
                float(df_frag["PC2"].max()),
            ),
        }

    written: List[str] = []
    for frag in fragments:
        d = per_frag_data.get(frag)
        if d is None:
            continue
        for lig in d["ligands"]:
            sub = d["df_frag"][d["df_frag"]["ligand"] == lig]
            if len(sub) < 5:
                continue
            fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
            if diff:
                _draw_diff_landscape(
                    ax,
                    sub,
                    d["df_frag"],
                    frag,
                    lig,
                    vabs_global=global_vabs,
                    basin_centers=d["basins"],
                    grid_extent=d["extent"],
                )
            else:
                _draw_absolute_landscape(
                    ax,
                    sub,
                    frag,
                    lig,
                    e_max_display_global=global_e_max_clip,
                    e_min_display_global=global_e_min,
                    basin_centers=d["basins"],
                )

            friendly = LIGAND_HET_TO_NAME.get(lig, lig)
            safe = friendly.replace(" ", "_")
            base_name = f"{frag}_{safe}__{lig}"
            for ext in ("pdf", "png"):
                out_path = out_dir / f"{base_name}.{ext}"
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                written.append(str(out_path))
            plt.close(fig)
            print(
                f"[fig4] {'diff' if diff else 'abs'} landscape -> "
                f"{frag}/{friendly} ({lig})"
            )

    return written


def _save_landscape_grid_aligned_panelB() -> List[str]:

    builder, _plotter, cfg = _get_landscape_machinery()

    fragments = ["Frag-B", "Frag-F", "Frag-A"]
    per_frag: Dict[str, Dict] = {}
    global_e_min = +np.inf
    global_e_max_clip = -np.inf
    global_vabs = 1e-9
    for frag in fragments:
        df = load_projected_samples(frag).dropna(subset=["PC1", "PC2", "score"])
        if df.empty:
            continue
        e = df["score"].to_numpy(dtype=float)
        global_e_min = min(global_e_min, float(e.min()))
        global_e_max_clip = max(
            global_e_max_clip,
            float(np.percentile(e, cfg.energy_percentile_clip)),
        )

        x0, x1 = float(df["PC1"].min()), float(df["PC1"].max())
        y0, y1 = float(df["PC2"].min()), float(df["PC2"].max())
        H_b, _, _ = np.histogram2d(
            df["PC1"],
            df["PC2"],
            bins=60,
            range=[[x0, x1], [y0, y1]],
            density=True,
        )
        try:
            from scipy.ndimage import gaussian_filter

            H_b = gaussian_filter(H_b, sigma=1.5)
        except Exception:
            pass
        for het in LIGAND_PANEL_B_ORDER:
            sub = df[df["ligand"] == het]
            if len(sub) < 50:
                continue
            H_l, _, _ = np.histogram2d(
                sub["PC1"],
                sub["PC2"],
                bins=60,
                range=[[x0, x1], [y0, y1]],
                density=True,
            )
            try:
                from scipy.ndimage import gaussian_filter

                H_l = gaussian_filter(H_l, sigma=1.5)
            except Exception:
                pass
            global_vabs = max(global_vabs, float(np.nanmax(np.abs(H_l - H_b))))

        per_frag[frag] = {
            "df": df,
            "extent": (x0, x1, y0, y1),
            "basins": _load_basin_centers(frag),
            "baseline": H_b,
        }

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
        ax_lbl.text(
            0.5,
            0.5,
            f"{frag}\n{('absolute' if mode == 'abs' else 'differential')}",
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
            sub = d["df"][d["df"]["ligand"] == het]
            if len(sub) < 50:
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
                for spine in ax.spines.values():
                    spine.set_color("#dddddd")
                continue

            if mode == "abs":
                _draw_absolute_landscape(
                    ax,
                    sub,
                    frag,
                    het,
                    e_max_display_global=global_e_max_clip,
                    e_min_display_global=global_e_min,
                    basin_centers=d["basins"],
                )
            else:
                _draw_diff_landscape(
                    ax,
                    sub,
                    d["df"],
                    frag,
                    het,
                    vabs_global=global_vabs,
                    basin_centers=d["basins"],
                    grid_extent=d["extent"],
                )
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

    paths = save_figure(
        fig, "fig4/G_landscape_grid_aligned_panelB", formats=("pdf", "png")
    )
    plt.close(fig)
    return paths


def _draw_basin_fingerprint(
    ax: plt.Axes,
    fragment: str,
    df_basin: pd.DataFrame,
    df_pool: pd.DataFrame,
    *,
    pool_label: str = "rmsd_top1000",
    role_label: str = "",
) -> None:
    """Per-fragment **ligand × basin** Δ-occupancy heatmap."""
    basin_cols = [f"delta_B{i}" for i in range(1, 6)]
    ligands = df_basin["ligand"].tolist()
    mat = df_basin[basin_cols].to_numpy(dtype=float)

    vabs = 0.50
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-vabs, vmax=vabs)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            if abs(v) < 0.005:
                ax.text(
                    j, i, "·", ha="center", va="center", fontsize=8, color="#888888"
                )
                continue
            txt_color = "white" if abs(v) > vabs * 0.55 else "#222222"
            ax.text(
                j,
                i,
                f"{v:+.02f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color=txt_color,
            )

    ax.set_xticks(range(5))
    ax.set_xticklabels([f"B{i}" for i in range(1, 6)], fontsize=8.5)
    ax.set_yticks(range(len(ligands)))
    ax.set_yticklabels(ligands, fontsize=8.5)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("basin", fontsize=8.0, labelpad=3)
    ax.set_ylabel("ligand", fontsize=8.0, labelpad=3)

    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color="#dddddd", linewidth=0.4)
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color="#dddddd", linewidth=0.4)

    row = df_pool[
        (df_pool["fragment"] == fragment) & (df_pool["pool_label"] == pool_label)
    ]
    title_parts = [f"{fragment}"]
    if role_label:
        title_parts.append(role_label)
    if not row.empty:
        l1max = float(row["max_ligand_L1"].iloc[0])
        p_val = float(row["ligand_perm_p"].iloc[0])
        p_str = f"p < 1e-3" if p_val < 1e-3 else f"p = {p_val:.3f}"
        title_parts.append(f"L1max = {l1max:.2f}")
        title_parts.append(p_str)
    ax.set_title("   ·   ".join(title_parts), fontsize=9.0, pad=6)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7.0)
    cbar.set_label("Δ basin occupancy", fontsize=7.5)
    cbar.outline.set_visible(False)


PANEL_D_FINGERPRINT_POOL = "rmsd_top1000"

PANEL_D_FRAGMENT_ROLES: Dict[str, str] = {
    "Frag-B": "Switch-II  (core)",
    "Frag-F": "α3 helix  (support)",
    "Frag-A": "loop  (control)",
}


def _save_panel_d_subpanels_standalone(
    df_core: pd.DataFrame,
    df_pool: pd.DataFrame,
) -> List[List[str]]:
    """One standalone PDF/PNG per Panel-D sub-axes."""
    fingerprint_specs: List[Tuple[str, str]] = [
        ("D4_basin_fingerprint_FragB", "Frag-B"),
        ("D4_basin_fingerprint_FragF", "Frag-F"),
        ("D4_basin_fingerprint_FragA", "Frag-A"),
    ]
    specs = [
        (
            "D1_response_amplitude_bars",
            lambda ax: _draw_panel_d_bars(ax, df_core, df_pool),
            "D1",
        ),
        ("D2_amplitude_vs_pool", lambda ax: _draw_panel_d_amplitude(ax, df_pool), "D2"),
        (
            "D3_significance_vs_pool",
            lambda ax: _draw_panel_d_significance(ax, df_pool),
            "D3",
        ),
    ]
    for out_name, frag in fingerprint_specs:
        df_basin = load_ligand_basin_response(frag)
        specs.append(
            (
                out_name,
                lambda ax, _frag=frag, _df=df_basin: _draw_basin_fingerprint(
                    ax,
                    _frag,
                    _df,
                    df_pool,
                    pool_label=PANEL_D_FINGERPRINT_POOL,
                    role_label=PANEL_D_FRAGMENT_ROLES.get(_frag, ""),
                ),
                "D4",
            )
        )
    out: List[List[str]] = []
    for out_name, drawer, letter in specs:
        is_d1 = out_name == "D1_response_amplitude_bars"
        if is_d1:
            fig = plt.figure(figsize=(4.1, 3.0), constrained_layout=True)
        else:
            fig = plt.figure(figsize=(3.0, 3.0), constrained_layout=True)
        ax = fig.add_subplot(111)
        drawer(ax)
        if not is_d1:
            ax.set_box_aspect(1.0)
        if is_d1:
            ax.set_title(
                "Frag-F supports the Switch-II pattern; " "Frag-A remains weak",
                fontsize=10.5,
                pad=8,
            )
        paths = save_figure(fig, f"fig4/{out_name}", formats=("pdf", "png"))
        out.append(paths)
    return out


def _save_panel_d_standalone(
    df_core: pd.DataFrame,
    df_pool: pd.DataFrame,
) -> List[str]:
    """D1 + D2 + D3 in one horizontal row at standalone resolution."""
    fig = plt.figure(figsize=(10.5, 3.0), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, wspace=0.40)
    ax_b = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])

    _draw_panel_d_bars(ax_b, df_core, df_pool)
    _draw_panel_d_amplitude(ax_a, df_pool)
    _draw_panel_d_significance(ax_p, df_pool)

    return save_figure(fig, "fig4/D_fragb_fragf_fraga_response", formats=("pdf", "png"))


def render_standalone_panels(
    fragments_df: pd.DataFrame,
    df_core: pd.DataFrame,
    df_pool: pd.DataFrame,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    out["A"] = _save_panel_a_standalone(fragments_df)
    out["A_legend"] = _save_panel_a_legend_standalone(fragments_df)
    out["B"] = _save_panel_b_standalone(fragments_df)
    out["C"] = _save_panel_c_standalone()
    sub_c = _save_panel_c_subpanels_standalone()
    for letter, paths in zip(["C1", "C2", "C3"], sub_c):
        out[letter] = paths
    out["D"] = _save_panel_d_standalone(df_core, df_pool)
    sub_d = _save_panel_d_subpanels_standalone(df_core, df_pool)
    panel_d_letters = ["D1", "D2", "D3", "D4_B", "D4_F", "D4_A"]
    for letter, paths in zip(panel_d_letters, sub_d):
        out[letter] = paths

    out["E_landscape"] = _save_per_ligand_landscapes(diff=False)
    out["F_diff_landscape"] = _save_per_ligand_landscapes(diff=True)

    out["G_landscape_grid"] = _save_landscape_grid_aligned_panelB()
    return out


def render_composite(
    fragments_df: pd.DataFrame,
    df_core: pd.DataFrame,
    df_pool: pd.DataFrame,
) -> plt.Figure:

    fig_w = 8.27
    fig_h = 10.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.55, 1.0, 1.0],
        hspace=0.28,
        wspace=0.40,
        left=0.05,
        right=0.98,
        top=0.965,
        bottom=0.04,
    )

    draw_panel_a(
        fig, gs[0, 0:3], fragments_df, title="KRAS modules · WT 8AZV", title_size=9.0
    )
    draw_panel_b(
        fig,
        gs[0, 3:6],
        fragments_df,
        layout="3+2",
        title="Frag-B predicted structure across ligand contexts",
        title_size=9.0,
    )
    draw_panel_c_diff_density(
        fig,
        gs[1, 0:6],
        title="Frag-B differential density · contrasts",
        title_size=9.0,
    )
    draw_panel_d(
        fig,
        gs[2, 0:6],
        df_core,
        df_pool,
        title="Cross-ligand response concentrates in Switch-II",
        title_size=9.0,
    )

    return fig


def main() -> int:
    apply_theme("main_double")
    OUT_FIG4_DIR.mkdir(parents=True, exist_ok=True)

    fragments_df = load_fragment_annotation()
    df_core = load_fragment_core_summary()
    df_pool = load_pool_summary()

    focus_views = render_panel_a_pymol_per_fragment(fragments_df)
    for frag, p in focus_views.items():
        print(f"[fig4] panel A focus {frag} -> {p}")

    panel_paths = render_standalone_panels(fragments_df, df_core, df_pool)
    for letter, paths in panel_paths.items():
        for p in paths:
            print(f"[fig4] panel {letter} -> {p}")

    fig = render_composite(fragments_df, df_core, df_pool)
    composite_paths = save_figure(fig, "fig4", formats=("pdf", "png"))
    for p in composite_paths:
        print(f"[fig4] composite -> {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

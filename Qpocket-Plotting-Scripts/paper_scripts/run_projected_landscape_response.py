"""Compute projected response distributions and basin occupancies."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

DEFAULT_RUN_DIR = os.environ.get(
    "QPOCKET_RUN_DIR", os.path.join(_PROJECT_ROOT, "inputs", "ensembles")
)
DEFAULT_OUT_DIR = os.path.join(_PROJECT_ROOT, "result", "landscape")
DEFAULT_TASKS_CSV = os.path.join(_PROJECT_ROOT, "inputs", "kras_tasks.csv")
DEFAULT_GROUPS_CSV = os.path.join(_PROJECT_ROOT, "inputs", "kras_analysis_groups.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("landscape_response")


SCORE_FIELDS = [
    "filter_energy",
    "full_energy",
    "score",
    "total_score",
    "energy",
    "hamiltonian_score",
    "objective",
    "qsad_score",
]
RMSD_FIELDS = [
    "kabsch_rmsd",
    "nokabsch_rmsd",
    "rmsd",
    "ca_rmsd",
    "backbone_rmsd",
    "pocket_rmsd",
]
CLASH_FIELDS = ["clash_score", "receptor_clash", "steric_clash"]
LIGAND_FIELDS_PREFER_RESNAME = [
    "ligand_resname",
    "ligand_family",
    "ligand",
    "ligand_name",
]


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_get(df: pd.DataFrame, col: Optional[str]) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.full(len(df), np.nan)
    return df[col].to_numpy(dtype=float)


_RE_MUT = re.compile(r"\b([A-Z][0-9]{1,3}[A-Z]|WT)\b")
_RE_FRAG = re.compile(r"Frag-([A-F])")
_RE_REFPDB = re.compile(r"^([0-9][A-Z0-9]{3})_", re.IGNORECASE)


def parse_case_id(case_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (ref_pdb, mutation, fragment_letter) parsed from case_id."""
    refm = _RE_REFPDB.match(case_id)
    ref_pdb = refm.group(1).upper() if refm else None
    mut_m = _RE_MUT.search(case_id)
    mutation = mut_m.group(1) if mut_m else None
    frag_m = _RE_FRAG.search(case_id)
    fragment = f"Frag-{frag_m.group(1)}" if frag_m else None
    return ref_pdb, mutation, fragment


@dataclass
class CaseRecord:
    case_id: str
    case_dir: str
    csv_path: str
    npz_path: str
    csv_source: str
    input_json_path: Optional[str]
    sequence: str
    ref_pdb: Optional[str]
    mutation: Optional[str]
    ligand_resname: Optional[str]
    ligand_family: Optional[str]
    pocket_module: Optional[str]
    fragment: Optional[str]
    n_residues_meta: Optional[int]
    n_rows: int
    coords_shape: Tuple[int, int, int]


def _load_metadata_table(tasks_csv: Optional[str]) -> pd.DataFrame:
    """Read case metadata from the specified or default task table."""
    if tasks_csv is None:
        if os.path.isfile(DEFAULT_TASKS_CSV):
            tasks_csv = DEFAULT_TASKS_CSV
            _LOG.info(f"Using default tasks CSV: {tasks_csv}")
        else:
            _LOG.warning(
                "Task table unavailable. Reading metadata from case input files and directory names."
            )
            return pd.DataFrame()
    df = pd.read_csv(tasks_csv)
    return df


def _peek_npz_shape(npz_path: str) -> Optional[Tuple[int, int, int]]:
    try:
        with np.load(npz_path) as z:
            if "coords" not in z.files:
                return None
            arr = z["coords"]
            return tuple(arr.shape)
    except Exception:
        return None


def discover_cases(
    run_dir: str,
    tasks_df: pd.DataFrame,
) -> Tuple[List[CaseRecord], List[Dict]]:
    """Find every case directory with usable samples. Return (records, skipped)."""
    skipped: List[Dict] = []
    records: List[CaseRecord] = []

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    meta_by_case: Dict[str, Dict[str, Any]] = {}
    if not tasks_df.empty and "case_id" in tasks_df.columns:
        for _, row in tasks_df.iterrows():
            meta_by_case[str(row["case_id"])] = row.to_dict()

    for name in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, name)
        if not os.path.isdir(sub):
            continue
        if name.startswith("_") or name in {"global_landscape"}:
            continue

        cand_csv = os.path.join(sub, "landscape", "landscape_candidates.csv")
        cand_npz = os.path.join(sub, "landscape", "landscape_coords.npz")
        csv_source = "landscape"
        if not (os.path.isfile(cand_csv) and os.path.isfile(cand_npz)):
            cand_csv = os.path.join(sub, "ensemble", "ensemble_candidates_index.csv")
            cand_npz = os.path.join(sub, "ensemble", "ensemble_candidates_coords.npz")
            csv_source = "ensemble"
        if not (os.path.isfile(cand_csv) and os.path.isfile(cand_npz)):
            skipped.append(
                {
                    "path": sub,
                    "reason": "no landscape/ or ensemble/ sample files",
                }
            )
            continue

        input_json = os.path.join(sub, "input.json")
        meta_json: Dict[str, Any] = {}
        if os.path.isfile(input_json):
            try:
                with open(input_json, "r") as fh:
                    meta_json = json.load(fh)
            except Exception:
                pass

        case_id_guess = meta_json.get("case_id", name)
        meta_csv = meta_by_case.get(case_id_guess, {})

        ref_pdb, mut_re, frag_re = parse_case_id(case_id_guess)
        ref_pdb_meta = meta_csv.get("ref_pdb") or meta_json.get("ref_pdb")
        if isinstance(ref_pdb_meta, str):
            ref_pdb_meta = ref_pdb_meta.split(".")[0].upper()
        ref_pdb_final = ref_pdb_meta or ref_pdb

        mutation_final = (
            meta_csv.get("mutation_group") or meta_json.get("mutation_group") or mut_re
        )
        ligand_resname = meta_csv.get("ligand_resname") or meta_json.get(
            "ligand_resname"
        )
        ligand_family = meta_csv.get("ligand_family") or meta_json.get("ligand_family")
        pocket_module = meta_csv.get("pocket_module") or meta_json.get("pocket_module")
        fragment_final = frag_re
        sequence = str(meta_json.get("sequence", ""))
        n_res_meta = meta_json.get("n_residues") or meta_csv.get("n_residues")
        try:
            n_res_meta = int(n_res_meta) if n_res_meta is not None else None
        except Exception:
            n_res_meta = None

        shape = _peek_npz_shape(cand_npz)
        if shape is None:
            skipped.append(
                {
                    "path": cand_npz,
                    "reason": "npz has no 'coords' array or failed to load",
                }
            )
            continue

        records.append(
            CaseRecord(
                case_id=case_id_guess,
                case_dir=sub,
                csv_path=cand_csv,
                npz_path=cand_npz,
                csv_source=csv_source,
                input_json_path=input_json if os.path.isfile(input_json) else None,
                sequence=sequence,
                ref_pdb=ref_pdb_final,
                mutation=mutation_final,
                ligand_resname=ligand_resname,
                ligand_family=ligand_family,
                pocket_module=pocket_module,
                fragment=fragment_final,
                n_residues_meta=n_res_meta,
                n_rows=int(shape[0]),
                coords_shape=shape,
            )
        )

    return records, skipped


def load_case_samples(
    case: CaseRecord,
    rng: np.random.Generator,
    max_per_case: int,
    top_ranked_only: bool,
) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Dict[str, Any]]:
    """Returns (coords subset (n, L, 3), aligned-row metadata frame, info dict)."""
    info: Dict[str, Any] = {"warnings": []}

    df = pd.read_csv(case.csv_path, low_memory=False)
    z = np.load(case.npz_path)
    coords = z["coords"]

    n_csv, n_npz = len(df), coords.shape[0]
    n_keep = min(n_csv, n_npz)
    if n_csv != n_npz:
        info["warnings"].append(
            f"csv rows ({n_csv}) != npz rows ({n_npz}); truncating to {n_keep}"
        )
        df = df.iloc[:n_keep].reset_index(drop=True)
        coords = coords[:n_keep]

    score_col = _first_present(df, SCORE_FIELDS)
    rmsd_col = _first_present(df, RMSD_FIELDS)
    clash_col = _first_present(df, CLASH_FIELDS)
    info["score_col"] = score_col
    info["rmsd_col"] = rmsd_col
    info["clash_col"] = clash_col

    finite_mask = np.isfinite(coords).all(axis=(1, 2))
    if not finite_mask.all():
        info["warnings"].append(
            f"{(~finite_mask).sum()} rows dropped: non-finite CA coords"
        )
        df = df.loc[finite_mask].reset_index(drop=True)
        coords = coords[finite_mask]

    if len(df) == 0:
        return None, None, info

    if len(df) > max_per_case:
        if top_ranked_only and score_col is not None:
            order = np.argsort(_safe_get(df, score_col))
            sel = order[:max_per_case]
            info["subsample_mode"] = f"top-ranked by {score_col}"
        elif top_ranked_only and score_col is None:
            sel = rng.choice(len(df), size=max_per_case, replace=False)
            info["warnings"].append(
                "--top-ranked-only requested but no score field present; falling back to random"
            )
            info["subsample_mode"] = "random (no score field)"
        else:
            sel = rng.choice(len(df), size=max_per_case, replace=False)
            info["subsample_mode"] = "random"
        sel = np.sort(sel)
        df = df.iloc[sel].reset_index(drop=True)
        coords = coords[sel]
    else:
        info["subsample_mode"] = "no subsample"

    meta = pd.DataFrame(
        {
            "case_id": case.case_id,
            "ref_pdb": case.ref_pdb,
            "mutation": case.mutation,
            "ligand": case.ligand_resname,
            "ligand_family": case.ligand_family,
            "fragment": case.fragment,
            "pocket_module": case.pocket_module,
            "sample_id": (
                df["candidate_id"].astype(str)
                if "candidate_id" in df.columns
                else [f"row{i}" for i in range(len(df))]
            ),
            "structure_path": case.npz_path,
            "coords_index": (
                df["coords_index"].to_numpy()
                if "coords_index" in df.columns
                else np.arange(len(df))
            ),
            "score": _safe_get(df, score_col),
            "rmsd": _safe_get(df, rmsd_col),
            "clash_score": _safe_get(df, clash_col),
            "rank": (
                np.argsort(np.argsort(_safe_get(df, score_col))) + 1
                if score_col is not None
                else np.full(len(df), np.nan)
            ),
            "source_file": case.csv_path,
            "parse_status": "ok",
        }
    )
    info["n_kept"] = len(df)
    return coords.astype(np.float32), meta, info


def ca_pairwise_distance_features(coords: np.ndarray) -> np.ndarray:
    """coords: (n, L, 3). Returns (n, L*(L-1)/2) upper-triangle distances."""
    n, L, _ = coords.shape
    iu, ju = np.triu_indices(L, k=1)
    diff = coords[:, iu, :] - coords[:, ju, :]
    return np.linalg.norm(diff, axis=2).astype(np.float64)


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    p /= p.sum()
    q = np.clip(q, eps, None)
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    return float(np.sqrt(0.5 * (_kl(p, m) + _kl(q, m))))


def _euclid(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _cos(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    return float(1.0 - np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + eps))


def pairwise_distance_table(
    occ_df: pd.DataFrame,
    key_col: str,
    basin_cols: List[str],
) -> pd.DataFrame:
    rows = []
    keys = occ_df[key_col].tolist()
    mat = occ_df[basin_cols].to_numpy(dtype=float)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            rows.append(
                {
                    "a": keys[i],
                    "b": keys[j],
                    "jensenshannon": _jsd(mat[i], mat[j]),
                    "euclid": _euclid(mat[i], mat[j]),
                    "cosine": _cos(mat[i], mat[j]),
                }
            )
    return pd.DataFrame(rows)


_DPI = 300
_FIGSIZE = (7.0, 5.5)
_MUT_COLORS = {
    "WT": "#4878d0",
    "G12D": "#ee854a",
    "G12C": "#6acc65",
    "G12V": "#d65f5f",
    "Q61H": "#956cb4",
    "Q61R": "#8c613c",
}
_LIG_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)


@dataclass
class FragmentResult:
    fragment: str
    n_cases: int
    n_samples: int
    common_L: int
    pca_var_pc1: float
    pca_var_pc2: float
    n_basins: int
    skipped: bool = False
    skip_reason: str = ""
    tables: Dict[str, str] = field(default_factory=dict)
    figures: Dict[str, str] = field(default_factory=dict)
    mutations: List[str] = field(default_factory=list)
    ligands: List[str] = field(default_factory=list)


def fragment_pipeline(
    fragment: str,
    coords_list: List[np.ndarray],
    meta_list: List[pd.DataFrame],
    out_dir: str,
    args,
    rng: np.random.Generator,
    skipped_records: List[Dict],
) -> FragmentResult:
    """Run the full per-fragment analysis. Returns FragmentResult."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    res = FragmentResult(
        fragment=fragment,
        n_cases=0,
        n_samples=0,
        common_L=0,
        pca_var_pc1=0.0,
        pca_var_pc2=0.0,
        n_basins=0,
    )

    if not coords_list:
        res.skipped = True
        res.skip_reason = "no cases"
        return res

    all_L = [c.shape[1] for c in coords_list]
    L_counter = pd.Series(all_L).value_counts()
    common_L = int(L_counter.idxmax())
    keep_mask = [L == common_L for L in all_L]
    if not all(keep_mask):
        for keep, c, m in zip(keep_mask, coords_list, meta_list):
            if not keep:
                skipped_records.append(
                    {
                        "path": m["source_file"].iloc[0] if len(m) else "?",
                        "reason": (
                            f"fragment={fragment} expected L={common_L} "
                            f"but case L={c.shape[1]}"
                        ),
                    }
                )
        coords_list = [c for k, c in zip(keep_mask, coords_list) if k]
        meta_list = [m for k, m in zip(keep_mask, meta_list) if k]

    coords_kept, meta_kept = [], []
    for c, m in zip(coords_list, meta_list):
        if len(m) >= args.min_samples_per_case:
            coords_kept.append(c)
            meta_kept.append(m)
        else:
            skipped_records.append(
                {
                    "path": m["source_file"].iloc[0] if len(m) else "?",
                    "reason": (
                        f"case has {len(m)} samples (< min_samples_per_case="
                        f"{args.min_samples_per_case})"
                    ),
                }
            )

    if not coords_kept:
        res.skipped = True
        res.skip_reason = "no cases passed min_samples_per_case"
        return res

    coords_all = np.concatenate(coords_kept, axis=0)
    meta_all = pd.concat(meta_kept, axis=0).reset_index(drop=True)
    n_total = len(meta_all)
    n_cases = meta_all["case_id"].nunique()

    if n_total < args.min_samples_per_fragment:
        res.skipped = True
        res.skip_reason = (
            f"only {n_total} samples (< min_samples_per_fragment="
            f"{args.min_samples_per_fragment})"
        )
        return res

    res.n_cases = n_cases
    res.n_samples = n_total
    res.common_L = common_L

    _LOG.info(f"  [{fragment}] cases={n_cases} samples={n_total} L={common_L}")

    X = ca_pairwise_distance_features(coords_all)
    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        skipped_records.append(
            {
                "path": fragment,
                "reason": f"{(~finite).sum()} rows dropped after feature build (non-finite)",
            }
        )
        X = X[finite]
        meta_all = meta_all.loc[finite].reset_index(drop=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = min(int(args.n_components), Xs.shape[1], Xs.shape[0] - 1)
    n_comp = max(2, n_comp)
    pca = PCA(n_components=n_comp, random_state=int(args.random_seed))
    Z = pca.fit_transform(Xs)
    res.pca_var_pc1 = float(pca.explained_variance_ratio_[0])
    res.pca_var_pc2 = float(pca.explained_variance_ratio_[1]) if n_comp >= 2 else 0.0

    meta_all["PC1"] = Z[:, 0]
    meta_all["PC2"] = Z[:, 1] if Z.shape[1] >= 2 else 0.0

    n_basins_eff = min(int(args.n_basins), max(2, len(meta_all) // 30))
    if n_basins_eff < 2:
        n_basins_eff = 2
    if n_basins_eff != int(args.n_basins):
        _LOG.info(
            f"  [{fragment}] auto-reduced n_basins {args.n_basins}→{n_basins_eff}"
        )
    gmm = GaussianMixture(
        n_components=n_basins_eff,
        covariance_type="full",
        random_state=int(args.random_seed),
        max_iter=300,
    )
    gmm.fit(Z[:, :2])
    raw_labels = gmm.predict(Z[:, :2])

    centers = gmm.means_
    order = np.argsort(centers[:, 0])
    relabel = {old: new for new, old in enumerate(order)}
    basin_idx = np.array([relabel[l] for l in raw_labels])
    centers_sorted = centers[order]
    meta_all["basin_id"] = [f"B{i+1}" for i in basin_idx]
    res.n_basins = n_basins_eff
    basin_cols = [f"B{i+1}" for i in range(n_basins_eff)]

    samples_path = os.path.join(out_dir, "tables", f"projected_samples_{fragment}.csv")
    meta_all.to_csv(samples_path, index=False)
    res.tables["projected_samples"] = samples_path

    bc_rows = []
    for i in range(n_basins_eff):
        m = meta_all["basin_id"] == f"B{i+1}"
        sub = meta_all.loc[m]
        row = {
            "fragment": fragment,
            "basin_id": f"B{i+1}",
            "center_PC1": float(centers_sorted[i, 0]),
            "center_PC2": float(centers_sorted[i, 1]),
            "n_samples": int(m.sum()),
            "global_occupancy": float(m.mean()),
            "mean_score": (
                float(sub["score"].mean())
                if sub["score"].notna().any()
                else float("nan")
            ),
            "mean_rmsd": (
                float(sub["rmsd"].mean()) if sub["rmsd"].notna().any() else float("nan")
            ),
            "mean_clash_score": (
                float(sub["clash_score"].mean())
                if sub["clash_score"].notna().any()
                else float("nan")
            ),
        }
        bc_rows.append(row)
    bc_df = pd.DataFrame(bc_rows)
    bc_path = os.path.join(out_dir, "tables", f"basin_centers_{fragment}.csv")
    bc_df.to_csv(bc_path, index=False)
    res.tables["basin_centers"] = bc_path

    case_groups = meta_all.groupby("case_id")
    occ_rows = []
    for cid, g in case_groups:
        n = len(g)
        counts = g["basin_id"].value_counts(normalize=True)
        row = {
            "case_id": cid,
            "ref_pdb": g["ref_pdb"].iloc[0],
            "mutation": g["mutation"].iloc[0],
            "ligand": g["ligand"].iloc[0],
            "fragment": fragment,
            "n_samples": n,
        }
        for b in basin_cols:
            row[b] = float(counts.get(b, 0.0))
        w = np.array([row[b] for b in basin_cols], dtype=float)
        w = w[w > 0]
        row["basin_entropy"] = float(-(w * np.log(w + 1e-15)).sum())
        occ_rows.append(row)
    occ_df = pd.DataFrame(occ_rows)
    occ_path = os.path.join(out_dir, "tables", f"case_basin_occupancy_{fragment}.csv")
    occ_df.to_csv(occ_path, index=False)
    res.tables["case_occupancy"] = occ_path

    res.mutations = sorted([str(x) for x in occ_df["mutation"].dropna().unique()])
    res.ligands = sorted([str(x) for x in occ_df["ligand"].dropna().unique()])

    def _agg(level: str) -> pd.DataFrame:
        rows = []
        for key, g in occ_df.groupby(level):
            row = {level: key, "n_cases": len(g)}
            for b in basin_cols:
                row[f"mean_{b}"] = float(g[b].mean())
                row[f"std_{b}"] = float(g[b].std(ddof=0))
            rows.append(row)
        return pd.DataFrame(rows)

    mut_occ = _agg("mutation")
    lig_occ = _agg("ligand")
    mut_occ_path = os.path.join(
        out_dir, "tables", f"mutation_basin_occupancy_{fragment}.csv"
    )
    lig_occ_path = os.path.join(
        out_dir, "tables", f"ligand_basin_occupancy_{fragment}.csv"
    )
    mut_occ.to_csv(mut_occ_path, index=False)
    lig_occ.to_csv(lig_occ_path, index=False)
    res.tables["mutation_occupancy"] = mut_occ_path
    res.tables["ligand_occupancy"] = lig_occ_path

    global_mean = occ_df[basin_cols].mean().to_numpy()

    def _delta(level: str) -> pd.DataFrame:
        rows = []
        for key, g in occ_df.groupby(level):
            mean = g[basin_cols].mean().to_numpy()
            d = mean - global_mean
            row = {level: key, "n_cases": len(g)}
            for i, b in enumerate(basin_cols):
                row[f"mean_{b}"] = float(mean[i])
                row[f"delta_{b}"] = float(d[i])
            rows.append(row)
        return pd.DataFrame(rows)

    mut_resp = _delta("mutation")
    lig_resp = _delta("ligand")
    mut_resp_path = os.path.join(
        out_dir, "tables", f"mutation_basin_response_{fragment}.csv"
    )
    lig_resp_path = os.path.join(
        out_dir, "tables", f"ligand_basin_response_{fragment}.csv"
    )
    mut_resp.to_csv(mut_resp_path, index=False)
    lig_resp.to_csv(lig_resp_path, index=False)
    res.tables["mutation_response"] = mut_resp_path
    res.tables["ligand_response"] = lig_resp_path

    case_dist = pairwise_distance_table(occ_df, "case_id", basin_cols)
    case_dist_path = os.path.join(
        out_dir, "tables", f"case_response_distance_{fragment}.csv"
    )
    case_dist.to_csv(case_dist_path, index=False)
    res.tables["case_distance"] = case_dist_path

    def _level_occ(level: str) -> pd.DataFrame:
        rows = []
        for key, g in occ_df.groupby(level):
            row = {level: key}
            mean = g[basin_cols].mean()
            for b in basin_cols:
                row[b] = float(mean[b])
            rows.append(row)
        return pd.DataFrame(rows)

    if not occ_df["mutation"].dropna().empty:
        mut_dist = pairwise_distance_table(
            _level_occ("mutation"), "mutation", basin_cols
        )
        if not mut_dist.empty:
            mut_dist_path = os.path.join(
                out_dir, "tables", f"mutation_response_distance_{fragment}.csv"
            )
            mut_dist.to_csv(mut_dist_path, index=False)
            res.tables["mutation_distance"] = mut_dist_path

    if not occ_df["ligand"].dropna().empty:
        lig_dist = pairwise_distance_table(_level_occ("ligand"), "ligand", basin_cols)
        if not lig_dist.empty:
            lig_dist_path = os.path.join(
                out_dir, "tables", f"ligand_response_distance_{fragment}.csv"
            )
            lig_dist.to_csv(lig_dist_path, index=False)
            res.tables["ligand_distance"] = lig_dist_path

    bs_rows = []
    n_boot = 100
    boot_seed = int(args.random_seed)
    for boot in range(n_boot):
        rb = np.random.default_rng(boot_seed + boot)
        boot_occ = []
        for cid, g in case_groups:
            if len(g) < 5:
                continue
            sel = rb.choice(len(g), size=max(5, int(0.7 * len(g))), replace=False)
            sub = g.iloc[sel]
            counts = sub["basin_id"].value_counts(normalize=True)
            row = {
                "case_id": cid,
                "mutation": sub["mutation"].iloc[0],
                "ligand": sub["ligand"].iloc[0],
            }
            for b in basin_cols:
                row[b] = float(counts.get(b, 0.0))
            boot_occ.append(row)
        boot_occ_df = pd.DataFrame(boot_occ)
        for level, col in [
            ("case", "case_id"),
            ("mutation", "mutation"),
            ("ligand", "ligand"),
        ]:
            grouped = (
                boot_occ_df.groupby(col)[basin_cols].mean()
                if level != "case"
                else boot_occ_df.set_index(col)[basin_cols]
            )
            for name, vec in grouped.iterrows():
                for b in basin_cols:
                    bs_rows.append(
                        {
                            "boot": boot,
                            "level": level,
                            "name": str(name),
                            "basin_id": b,
                            "occupancy": float(vec[b]),
                        }
                    )
    if bs_rows:
        bs_df = pd.DataFrame(bs_rows)
        agg = (
            bs_df.groupby(["level", "name", "basin_id"])["occupancy"]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg = agg.rename(columns={"mean": "mean_occupancy", "std": "std_occupancy"})
        agg["cv_occupancy"] = agg["std_occupancy"] / (agg["mean_occupancy"] + 1e-12)
        bs_path = os.path.join(
            out_dir, "tables", f"bootstrap_occupancy_stability_{fragment}.csv"
        )
        agg.to_csv(bs_path, index=False)
        res.tables["bootstrap"] = bs_path

    _make_figures(
        fragment=fragment,
        meta=meta_all,
        basin_cols=basin_cols,
        centers=centers_sorted,
        occ_df=occ_df,
        mut_resp=mut_resp,
        lig_resp=lig_resp,
        mut_dist_df=res.tables.get("mutation_distance"),
        lig_dist_df=res.tables.get("ligand_distance"),
        out_dir=out_dir,
        pca_var=(res.pca_var_pc1, res.pca_var_pc2),
        res=res,
    )

    return res


def _make_figures(
    *,
    fragment: str,
    meta: pd.DataFrame,
    basin_cols: List[str],
    centers: np.ndarray,
    occ_df: pd.DataFrame,
    mut_resp: pd.DataFrame,
    lig_resp: pd.DataFrame,
    mut_dist_df: Optional[str],
    lig_dist_df: Optional[str],
    out_dir: str,
    pca_var: Tuple[float, float],
    res: FragmentResult,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    pc_label = f"PC1 ({pca_var[0]*100:.1f}%)", f"PC2 ({pca_var[1]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for mut, sub in meta.groupby("mutation"):
        c = _MUT_COLORS.get(str(mut), "#888888")
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=8,
            alpha=0.35,
            c=c,
            label=str(mut),
            edgecolors="none",
        )
    for i, (cx, cy) in enumerate(centers):
        ax.scatter(
            cx,
            cy,
            s=140,
            marker="X",
            c="black",
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )
        ax.annotate(
            f"B{i+1}",
            (cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )
    ax.set_xlabel(pc_label[0])
    ax.set_ylabel(pc_label[1])
    ax.set_title(f"Projected information landscape — {fragment} (by mutation)")
    ax.legend(markerscale=2.5, loc="best", framealpha=0.9)
    p = os.path.join(fig_dir, f"projected_landscape_by_mutation_{fragment}.png")
    _save(fig, p)
    res.figures["by_mutation"] = p

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ligs_unique = sorted([str(x) for x in meta["ligand"].dropna().unique()])
    for k, lig in enumerate(ligs_unique):
        sub = meta[meta["ligand"] == lig]
        c = _LIG_PALETTE[k % len(_LIG_PALETTE)]
        ax.scatter(
            sub["PC1"], sub["PC2"], s=8, alpha=0.35, c=c, label=lig, edgecolors="none"
        )
    for i, (cx, cy) in enumerate(centers):
        ax.scatter(
            cx,
            cy,
            s=140,
            marker="X",
            c="black",
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )
        ax.annotate(
            f"B{i+1}",
            (cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )
    ax.set_xlabel(pc_label[0])
    ax.set_ylabel(pc_label[1])
    ax.set_title(f"Projected information landscape — {fragment} (by ligand)")
    ax.legend(markerscale=2.5, loc="best", framealpha=0.9, fontsize=9)
    p = os.path.join(fig_dir, f"projected_landscape_by_ligand_{fragment}.png")
    _save(fig, p)
    res.figures["by_ligand"] = p

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    cmap_basin = plt.cm.tab10
    for i, b in enumerate(basin_cols):
        sub = meta[meta["basin_id"] == b]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=8,
            alpha=0.45,
            c=[cmap_basin(i)],
            label=b,
            edgecolors="none",
        )
    for i, (cx, cy) in enumerate(centers):
        ax.scatter(
            cx,
            cy,
            s=160,
            marker="X",
            c="black",
            edgecolors="white",
            linewidths=1.6,
            zorder=10,
        )
        ax.annotate(
            f"B{i+1}",
            (cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )
    ax.set_xlabel(pc_label[0])
    ax.set_ylabel(pc_label[1])
    ax.set_title(f"Projected basins — {fragment} (GMM, K={len(basin_cols)})")
    ax.legend(markerscale=2.5, loc="best", framealpha=0.9)
    p = os.path.join(fig_dir, f"projected_basins_{fragment}.png")
    _save(fig, p)
    res.figures["basins"] = p

    M = occ_df.set_index("case_id")[basin_cols].to_numpy()
    h = max(4.5, 0.25 * len(occ_df))
    fig, ax = plt.subplots(figsize=(max(5.5, 0.55 * len(basin_cols) + 4), h))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.4, M.max()))
    ax.set_xticks(range(len(basin_cols)))
    ax.set_xticklabels(basin_cols)
    ax.set_yticks(range(len(occ_df)))
    ax.set_yticklabels(occ_df["case_id"])
    ax.set_xlabel("Projected basin")
    ax.set_title(f"Case × basin occupancy — {fragment}")
    fig.colorbar(im, ax=ax, label="occupancy")
    p = os.path.join(fig_dir, f"case_basin_occupancy_{fragment}.png")
    _save(fig, p)
    res.figures["case_occupancy"] = p

    if not mut_resp.empty:
        delta_cols = [c for c in mut_resp.columns if c.startswith("delta_")]
        M = mut_resp.set_index("mutation")[delta_cols].to_numpy()
        vmax = float(np.max(np.abs(M))) if M.size else 1e-3
        vmax = max(vmax, 1e-3)
        fig, ax = plt.subplots(
            figsize=(
                max(5.5, 0.55 * len(delta_cols) + 4),
                max(2.5, 0.45 * len(mut_resp) + 1.5),
            )
        )
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(delta_cols)))
        ax.set_xticklabels([c.replace("delta_", "") for c in delta_cols])
        ax.set_yticks(range(len(mut_resp)))
        ax.set_yticklabels(mut_resp["mutation"])
        ax.set_title(
            f"Mutation-associated projected basin occupancy response — {fragment}"
        )
        fig.colorbar(im, ax=ax, label="Δ occupancy")
        p = os.path.join(fig_dir, f"mutation_basin_response_{fragment}.png")
        _save(fig, p)
        res.figures["mutation_response"] = p

    if not lig_resp.empty:
        delta_cols = [c for c in lig_resp.columns if c.startswith("delta_")]
        M = lig_resp.set_index("ligand")[delta_cols].to_numpy()
        vmax = float(np.max(np.abs(M))) if M.size else 1e-3
        vmax = max(vmax, 1e-3)
        fig, ax = plt.subplots(
            figsize=(
                max(5.5, 0.55 * len(delta_cols) + 4),
                max(2.5, 0.45 * len(lig_resp) + 1.5),
            )
        )
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(delta_cols)))
        ax.set_xticklabels([c.replace("delta_", "") for c in delta_cols])
        ax.set_yticks(range(len(lig_resp)))
        ax.set_yticklabels(lig_resp["ligand"])
        ax.set_title(
            f"Ligand-associated projected basin occupancy response — {fragment}"
        )
        fig.colorbar(im, ax=ax, label="Δ occupancy")
        p = os.path.join(fig_dir, f"ligand_basin_response_{fragment}.png")
        _save(fig, p)
        res.figures["ligand_response"] = p

    pairs = _select_diff_pairs(meta)
    for level, a, b in pairs:
        sub_a = meta[meta[level] == a]
        sub_b = meta[meta[level] == b]
        if len(sub_a) < 50 or len(sub_b) < 50:
            continue
        x_min = meta["PC1"].min()
        x_max = meta["PC1"].max()
        y_min = meta["PC2"].min()
        y_max = meta["PC2"].max()
        nb = 60
        H_a, xe, ye = np.histogram2d(
            sub_a["PC1"],
            sub_a["PC2"],
            bins=nb,
            range=[[x_min, x_max], [y_min, y_max]],
            density=True,
        )
        H_b, _, _ = np.histogram2d(
            sub_b["PC1"],
            sub_b["PC2"],
            bins=nb,
            range=[[x_min, x_max], [y_min, y_max]],
            density=True,
        )
        try:
            from scipy.ndimage import gaussian_filter

            H_a = gaussian_filter(H_a, sigma=1.5)
            H_b = gaussian_filter(H_b, sigma=1.5)
        except Exception:
            pass
        diff = H_a.T - H_b.T
        vmax = float(np.max(np.abs(diff))) if diff.size else 1.0
        vmax = max(vmax, 1e-12)
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        im = ax.imshow(
            diff,
            origin="lower",
            cmap="RdBu_r",
            extent=[x_min, x_max, y_min, y_max],
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        for i, (cx, cy) in enumerate(centers):
            ax.scatter(
                cx,
                cy,
                s=120,
                marker="X",
                c="black",
                edgecolors="white",
                linewidths=1.4,
                zorder=10,
            )
            ax.annotate(
                f"B{i+1}",
                (cx, cy),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                weight="bold",
            )
        ax.set_xlabel(pc_label[0])
        ax.set_ylabel(pc_label[1])
        ax.set_title(f"ΔP(z) — {a} − {b} ({level}, {fragment})")
        fig.colorbar(im, ax=ax, label="density difference")
        safe_a = re.sub(r"[^A-Za-z0-9._-]+", "_", a)
        safe_b = re.sub(r"[^A-Za-z0-9._-]+", "_", b)
        p = os.path.join(
            fig_dir, f"differential_density_{safe_a}_vs_{safe_b}_{fragment}.png"
        )
        _save(fig, p)
        res.figures.setdefault("differential", []).append(p)

    def _dist_heatmap(level: str, fname_key: str):
        path_csv = res.tables.get(fname_key)
        if path_csv is None or not os.path.isfile(path_csv):
            return
        try:
            df = pd.read_csv(path_csv)
        except pd.errors.EmptyDataError:
            return
        if df.empty:
            return
        keys = sorted(set(df["a"]) | set(df["b"]))
        n = len(keys)
        if n < 2:
            return
        idx = {k: i for i, k in enumerate(keys)}
        M = np.zeros((n, n))
        for _, row in df.iterrows():
            i, j = idx[row["a"]], idx[row["b"]]
            M[i, j] = M[j, i] = row["jensenshannon"]
        fig, ax = plt.subplots(
            figsize=(max(4.5, 0.45 * n + 3.5), max(4.0, 0.45 * n + 3.0))
        )
        im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=max(M.max(), 1e-3))
        ax.set_xticks(range(n))
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(keys)
        ax.set_title(f"{level} response distance (JS) — {fragment}")
        fig.colorbar(im, ax=ax, label="JS distance")
        p = os.path.join(fig_dir, f"{level}_response_distance_{fragment}.png")
        _save(fig, p)
        res.figures[f"{level}_distance"] = p

    _dist_heatmap("mutation", "mutation_distance")
    _dist_heatmap("ligand", "ligand_distance")


def _select_diff_pairs(meta: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Pick up to 3 (level, a, b) tuples for differential density maps."""
    out: List[Tuple[str, str, str]] = []
    muts = meta["mutation"].dropna().value_counts()
    if "G12D" in muts.index and "G12C" in muts.index:
        out.append(("mutation", "G12D", "G12C"))
    elif len(muts) >= 2:
        a, b = muts.index[0], muts.index[1]
        out.append(("mutation", a, b))
    ligs = meta["ligand"].dropna().value_counts()
    if len(ligs) >= 2:
        out.append(("ligand", ligs.index[0], ligs.index[1]))
    if len(ligs) >= 3:
        out.append(("ligand", ligs.index[0], ligs.index[2]))
    return out[:3]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--tasks-csv", default=None)
    p.add_argument("--max-samples-per-case", type=int, default=500)
    p.add_argument("--projection", default="pca", choices=["pca"])
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--n-basins", type=int, default=5)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--feature-mode", default="ca_distance", choices=["ca_distance"])
    p.add_argument("--min-samples-per-case", type=int, default=10)
    p.add_argument("--min-samples-per-fragment", type=int, default=50)
    p.add_argument("--top-ranked-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    _LOG.info("Discovering cases …")
    tasks_df = _load_metadata_table(args.tasks_csv)
    records, skipped = discover_cases(args.run_dir, tasks_df)
    _LOG.info(f"Discovered {len(records)} cases (skipped {len(skipped)})")

    disc_rows = []
    for r in records:
        disc_rows.append(
            {
                "case_id": r.case_id,
                "case_dir": r.case_dir,
                "csv_source": r.csv_source,
                "n_rows": r.n_rows,
                "L": r.coords_shape[1],
                "ref_pdb": r.ref_pdb,
                "mutation": r.mutation,
                "ligand": r.ligand_resname,
                "ligand_family": r.ligand_family,
                "fragment": r.fragment,
                "pocket_module": r.pocket_module,
                "sequence": r.sequence,
            }
        )
    pd.DataFrame(disc_rows).to_csv(
        os.path.join(out_dir, "tables", "discovered_samples.csv"), index=False
    )

    if args.dry_run:
        _LOG.info("Dry run: stopping after discovery.")
        pd.DataFrame(skipped).to_csv(
            os.path.join(out_dir, "tables", "skipped_samples.csv"), index=False
        )
        return

    rng = np.random.default_rng(int(args.random_seed))
    by_frag: Dict[str, List[CaseRecord]] = {}
    for r in records:
        by_frag.setdefault(r.fragment or "UNKNOWN", []).append(r)

    skipped_records = list(skipped)
    frag_results: List[FragmentResult] = []

    for fragment in sorted(by_frag.keys()):
        _LOG.info(f"=== fragment {fragment} : {len(by_frag[fragment])} cases ===")
        coords_list, meta_list = [], []
        for case in by_frag[fragment]:
            _LOG.info(f"  loading {case.case_id} (n_rows={case.n_rows})")
            try:
                coords, meta, info = load_case_samples(
                    case,
                    rng,
                    max_per_case=args.max_samples_per_case,
                    top_ranked_only=bool(args.top_ranked_only),
                )
            except Exception as exc:
                _LOG.exception(f"    failed to load {case.case_id}: {exc}")
                skipped_records.append(
                    {"path": case.case_dir, "reason": f"load failed: {exc}"}
                )
                continue
            for w in info.get("warnings", []):
                _LOG.warning(f"    {case.case_id}: {w}")
            if coords is None:
                skipped_records.append(
                    {"path": case.case_dir, "reason": "no usable rows after filter"}
                )
                continue
            if len(meta) < args.min_samples_per_case:
                skipped_records.append(
                    {
                        "path": case.case_dir,
                        "reason": (
                            f"only {len(meta)} samples (< min_samples_per_case="
                            f"{args.min_samples_per_case})"
                        ),
                    }
                )
                continue
            coords_list.append(coords)
            meta_list.append(meta)

        try:
            res = fragment_pipeline(
                fragment,
                coords_list,
                meta_list,
                out_dir,
                args,
                rng,
                skipped_records,
            )
        except Exception as exc:
            _LOG.exception(f"fragment {fragment} pipeline failed: {exc}")
            res = FragmentResult(
                fragment=fragment,
                n_cases=len(coords_list),
                n_samples=sum(len(m) for m in meta_list),
                common_L=0,
                pca_var_pc1=0.0,
                pca_var_pc2=0.0,
                n_basins=0,
            )
            res.skipped = True
            res.skip_reason = f"pipeline error: {exc}"
        frag_results.append(res)

    pd.DataFrame(skipped_records).to_csv(
        os.path.join(out_dir, "tables", "skipped_samples.csv"), index=False
    )

    _LOG.info("Done.")


if __name__ == "__main__":
    main()

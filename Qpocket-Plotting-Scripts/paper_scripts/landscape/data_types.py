from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SampledConformation:
    """One unique backbone conformation recovered from quantum sampling."""

    bitstring_key: str
    full_bitstring: str
    sequence: str
    main_positions: np.ndarray
    energy: float
    energy_terms: Dict[str, float] = field(default_factory=dict)
    count: int = 1
    prob: float = 0.0
    saw_valid: bool = True


@dataclass
class LandscapeData:
    """Complete energy landscape dataset for one protein sequence."""

    protein_id: str
    sequence: str
    conformations: List[SampledConformation]

    pca_coords: np.ndarray
    pca_explained_variance: np.ndarray

    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray

    min_energy_idx: int

    median_energy: float
    funnel_score: float
    n_valid: int
    n_total_unique: int
    n_basins: int

    @property
    def energies(self) -> np.ndarray:
        return np.array([c.energy for c in self.conformations], dtype=float)

    @property
    def probs(self) -> np.ndarray:
        return np.array([c.prob for c in self.conformations], dtype=float)

    @property
    def counts(self) -> np.ndarray:
        return np.array([c.count for c in self.conformations], dtype=float)

    @property
    def min_energy_conformation(self) -> SampledConformation:
        return self.conformations[self.min_energy_idx]

    def to_summary_record(self) -> Dict[str, Any]:
        """Compact JSON-serialisable summary (no large arrays)."""
        e = self.energies
        return {
            "protein_id": self.protein_id,
            "sequence": self.sequence,
            "L": len(self.sequence),
            "n_valid": self.n_valid,
            "n_total_unique": self.n_total_unique,
            "min_energy": float(e.min()),
            "max_energy": float(e.max()),
            "mean_energy": float(e.mean()),
            "median_energy": float(self.median_energy),
            "std_energy": float(e.std()),
            "funnel_score": float(self.funnel_score),
            "n_basins": self.n_basins,
            "pca_var_pc1": float(self.pca_explained_variance[0]),
            "pca_var_pc2": float(self.pca_explained_variance[1]),
            "min_energy_bitstring": self.min_energy_conformation.bitstring_key,
        }

    def to_full_record(self) -> Dict[str, Any]:
        """Full record including per-conformation arrays for JSON export."""
        record = self.to_summary_record()
        record.update(
            {
                "energies": [float(x) for x in self.energies],
                "probs": [float(x) for x in self.probs],
                "counts": [int(x) for x in self.counts],
                "pca_pc1": [float(x) for x in self.pca_coords[:, 0]],
                "pca_pc2": [float(x) for x in self.pca_coords[:, 1]],
                "bitstring_keys": [c.bitstring_key for c in self.conformations],
                "energy_terms_list": [c.energy_terms for c in self.conformations],
            }
        )
        return record

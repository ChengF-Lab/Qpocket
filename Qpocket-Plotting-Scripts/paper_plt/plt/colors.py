from __future__ import annotations

from typing import Dict, Iterable, List, Optional


FRAGMENT_COLORS: Dict[str, str] = {
    "Frag-A": "#d62728",
    "Frag-B": "#1f4e79",
    "Frag-F": "#4a90c2",
    "Frag-C": "#2ca02c",
    "Frag-D": "#5a5a5a",
    "Frag-E": "#a8a8a8",
}

FRAGMENT_ORDER: List[str] = [
    "Frag-A",
    "Frag-B",
    "Frag-F",
    "Frag-C",
    "Frag-D",
    "Frag-E",
]


MODULE_COLORS: Dict[str, str] = {
    "P-loop_mutation_site": FRAGMENT_COLORS["Frag-A"],
    "Switch-II_ligand_contact": FRAGMENT_COLORS["Frag-B"],
    "Alpha3_allosteric": FRAGMENT_COLORS["Frag-C"],
    "Peripheral": FRAGMENT_COLORS["Frag-D"],
}

MODULE_ORDER: List[str] = [
    "P-loop_mutation_site",
    "Switch-II_ligand_contact",
    "Alpha3_allosteric",
    "Peripheral",
]

FRAGMENT_TO_MODULE: Dict[str, str] = {
    "Frag-A": "P-loop_mutation_site",
    "Frag-B": "Switch-II_ligand_contact",
    "Frag-F": "Switch-II_ligand_contact",
    "Frag-C": "Alpha3_allosteric",
    "Frag-D": "Peripheral",
    "Frag-E": "Peripheral",
}


MUTATION_COLORS: Dict[str, str] = {
    "WT": "#6e6e6e",
    "G12D": "#3d5a80",
    "G12C": "#e07a5f",
    "G12V": "#9d4edd",
}

MUTATION_ORDER: List[str] = ["WT", "G12D", "G12C", "G12V"]


LIGAND_COLORS: Dict[str, str] = {
    "OFU": "#0072b2",
    "6IC": "#e69f00",
    "7L8": "#009e73",
    "7IZ": "#cc79a7",
    "EZZ": "#56b4e9",
    "F0K": "#d55e00",
}

LIGAND_ORDER: List[str] = ["OFU", "6IC", "7L8", "7IZ", "EZZ", "F0K"]


NEUTRAL_REFERENCE: str = "#444444"

GRID_GREY: str = "#cccccc"
ANNOT_GREY: str = "#888888"

HIGHLIGHT: str = "#ffbf00"
GOOD: str = "#2ca02c"
BAD: str = "#d62728"


_FALLBACK_PALETTE: List[str] = [
    "#332288",
    "#117733",
    "#882255",
    "#88ccee",
    "#ddcc77",
    "#cc6677",
    "#aa4499",
    "#44aa99",
]


def _resolve(palette: Dict[str, str], key, default: Optional[str]) -> str:

    if key in palette:
        return palette[key]
    if default is not None:
        return default
    idx = abs(hash(str(key))) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[idx]


def color_for_fragment(frag: str, default: Optional[str] = None) -> str:
    """Return colour for a fragment id (``Frag-A`` ... ``Frag-F``)."""
    return _resolve(FRAGMENT_COLORS, frag, default)


def color_for_module(module: str, default: Optional[str] = None) -> str:
    """Return colour for a module id."""
    return _resolve(MODULE_COLORS, module, default)


def color_for_mutation(mutation: str, default: Optional[str] = None) -> str:
    """Return colour for a mutation tag (``WT``/``G12D``/...)."""
    return _resolve(MUTATION_COLORS, mutation, default)


def color_for_ligand(ligand: str, default: Optional[str] = None) -> str:
    """Return colour for a ligand id (3-letter PDB code)."""
    return _resolve(LIGAND_COLORS, ligand, default)


def palette_for(category: str, values: Iterable) -> Dict:
    """Build a colour mapping for an arbitrary iterable of values."""
    base_map = {
        "fragment": FRAGMENT_COLORS,
        "module": MODULE_COLORS,
        "mutation": MUTATION_COLORS,
        "ligand": LIGAND_COLORS,
    }.get(category)
    if base_map is None:
        raise ValueError(
            f"Unknown category {category!r}. "
            "Expected 'fragment'|'module'|'mutation'|'ligand'."
        )
    out: Dict = {}
    fb_idx = 0
    for v in values:
        if v in base_map:
            out[v] = base_map[v]
        else:
            out[v] = _FALLBACK_PALETTE[fb_idx % len(_FALLBACK_PALETTE)]
            fb_idx += 1
    return out

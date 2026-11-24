# Qpocket plotting scripts

Python drivers for the landscape and response analyses listed in SI Appendix, Table S49. The Qpocket implementation, input structures, task definitions, and published numerical results are available in [ChengF-Lab/Qpocket](https://github.com/ChengF-Lab/Qpocket). The [source-data directory](https://github.com/ChengF-Lab/Qpocket/tree/main/Qpocket_Source_Data) contains the published figure tables.

## Setup

Use Python 3.10 or later. From the extracted directory, run `python -m pip install -r requirements.txt`. The molecular structure panels also use PyMOL. Make the `pymol` executable available on PATH or set `PYMOL_BIN` to its executable path.

| Driver | Input and output |
| --- | --- |
| `paper_plt/fig3.py` | Reference PDB files, fragment definitions, and per-case coordinates and scores → pocket views and Frag-B landscape panels. |
| `paper_plt/fig4.py` | Structure files, annotation tables, pool response tables, and projected samples → structural views and response panels. |
| `paper_scripts/run_landscape_v2.py` | Per-case coordinates and filter energies → coordinate-PCA landscapes. |
| `paper_scripts/run_projected_landscape_response.py` | Per-case coordinates and condition labels → distance-feature projections, basin assignments, occupancy tables, and response plots. |
| `paper_scripts/render_fig4_landscape_grid.py` | Projected samples and basin centers → absolute and differential landscape grids. |
| `SI/scripts/make_si_landscapes.py` | Per-case coordinates and filter energies → SI galleries and projected sample tables. |
| `SI/scripts/fix_si.py` | Per-case coordinates and filter energies → per-fragment and per-case SI landscapes on a rectangular PCA grid. |
| `SI/scripts/remake_master.py` | Per-case coordinates and filter energies → overview matrix of SI landscapes. |

## Inputs and execution

For the coordinate-based drivers, set `QPOCKET_RUN_DIR` to the parent directory of the case folders, or place them in `inputs/ensembles/`. For each case, the sequence is read from `input.json`, candidate rows from `landscape/landscape_candidates.csv`, and Cα coordinates from `landscape/landscape_coords.npz` under the key `coords`. Coordinate arrays have shape (conformations, residues, 3) and must follow the candidate row order. The coordinate-PCA scripts read the `filter_energy` and `bitstring` columns. The Figure 3 source-data table contains the resulting 2-D projections.

For example, to generate per-case landscapes from a prepared ensemble directory:

```sh
python paper_scripts/run_landscape_v2.py --run-dir inputs/ensembles --out-dir figures/landscapes
```

To render a response grid from prepared projected-sample tables, set `QPOCKET_GRID_TABLES` to the directory containing `projected_samples_Frag-A.csv`, `projected_samples_Frag-B.csv`, and `projected_samples_Frag-F.csv`, with the corresponding `basin_centers_Frag-*.csv` files for basin overlays. The grid reads the per-sample coordinates, scores, basin assignments, and ligand labels.

```sh
QPOCKET_GRID_TABLES=result/landscape_2000/tables python paper_scripts/render_fig4_landscape_grid.py --out-dir figures/grid
```

For structural panels, place the reference PDB files in `kras_select_systems/` and the sequence definitions in `inputs/protein_sequence_summary.csv`. For Fig. 4, `QPOCKET_POCKET_DIR` selects the reconstructed pocket directory with `<PDB>/embedded_full_pocket.pdb` beneath it. The figure drivers give the filenames and relative paths for their additional panel-specific inputs.

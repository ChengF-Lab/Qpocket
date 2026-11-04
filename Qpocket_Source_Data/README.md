# Qpocket: quantum data and final results

This public companion package contains quantum scientific data and final
numeric results for Quantum-Guided Ensemble Generation for Accurate
Drug-Binding Pocket Prediction: Application to KRAS. Detailed execution
environments and intermediate structural/scoring outputs are excluded.

## Contents

| Folder | Contents |
| --- | --- |
| inputs | Nine reference PDB structures, 36 study-task definitions, analysis groups and sequences |
| hardware | 591 anonymous job rows, 108 workload summaries, 12,618 saved circuit records and 36 quantum case records |
| quantum_counts | First 20,000 stored count entries from each of 108 workloads, with original-order and truncation indexes |
| figure1 | Final 101-target RMSD table and eight-method summary, at reported precision |
| figure2 | Final KRAS RMSD table, nine-system docking summary and 81 tabulated SI pose scores |
| figure3 | Final projected-ensemble summary and 7,200 displayed projected points |
| figure4 | Final basin, occupancy, response and stability tables |
| sensitivity | Final metrics across 13 pool regimes and corresponding P/q summary tables |

MANIFEST.csv lists all other package members, their byte/row counts and SI
references. PROVENANCE.json records public scientific provenance and the
specific accompanying SI. The manuscript, SI PDF, figure images, software
environments and full raw-data archive are not duplicated in this ZIP.

## RMSD and docking: final results only

The RMSD files contain final numeric results. No candidate-level RMSD tables,
individual scoring records, comparator run records, coordinate examples or
generated 3D ensembles are included. The KRAS table retains its 36 case rows
and the original MEAN row; select explicit case IDs for per-case analysis.
One case, 6GJ6_G12D_Frag-B, lacks a complete native reference; its missing
native-referenced RMSD is not imputed.

Qpocket's min_kabsch_rmsd is native-referenced ensemble coverage over the
generated pool. It is not a native-independent single-prediction endpoint.
The sequence-window baselines use different input, atom-selection, alignment
and aggregation protocols. The final reported values retain those definitions.
General-benchmark exports retain the SI's published precision; -- means no
prediction. Other empty cells mean missing/unrecorded values.

Docking data are final aggregate results and the final SI S68 tabulated pose
scores. Nine poses within a receptor share one affinity observation; poses
are not independent biological systems. Superseded Vina outputs, complete
docking outputs, original logs and intermediate detail records are excluded.
No docking or RMSD values were recomputed for this export.

## Quantum data

hardware/jobs.csv has one row for each of 591 saved jobs.
hardware/workloads.csv has one row for each of 108 case/seed workloads.
The cohort has 36 cases and 103,366,656 shots, with 8,192 shots per circuit.
hardware/dispatch_records.jsonl preserves recorded backend-result metadata,
chunk planning, circuit summary statistics, transpilation summaries and
quantum configuration. hardware/case_records.jsonl preserves only molecular
input, prior and quantum configuration/summary records.

Job labels job_0001 through job_0591 are consistent archive-local pseudonyms.
They are not provider-issued IDs and cannot be submitted as Runtime job IDs.
The mapping to original provider IDs is not included. Workload IDs and random
seeds retain their scientific linkage across files. Account identifiers,
session/instance identifiers, absolute timestamps, personal directories,
machine/cluster names, scheduler records and full execution commands are
excluded. Durations and scientific parameters are retained where recorded.

workload_wall_seconds is a workload-level observation repeated for its jobs;
do not sum this column over jobs. estimated_quantum_seconds is a planning
estimate, not measured QPU usage. Missing provider timing/usage values are
not reconstructed. workload summaries retain recorded quantities; previously
reconstructed gate-count fields are not presented as hardware measurements.

Circuit names and ansatz labels are the labels stored by the historical run.
Current repository code must not be assumed to reconstruct the exact circuits
used in this cohort. Saved summaries contain statistical parameter summaries,
not the complete historical theta vectors or exact executed QASM. No
entangling-gate counts, calibrations or QPU billing times are inferred.

## Raw quantum count excerpts in source-file order

quantum_counts/workload_001.csv through workload_108.csv each contain the
first 20,000 saved bitstring/count entries from the corresponding workload.
The total is 2,160,000 retained count entries across all 108 workloads and
36 cases. No source circuit or count entry is sorted, shuffled or selected
by score. Each bitstring and its integer count is copied unchanged.

Columns:
- circuit_index_0based: the circuit's position in the original source list.
- entry_index_1based: the count entry's position within that circuit's stored mapping.
- bitstring: the original binary string, including leading zeroes.
- count: that string's original frequency in the complete source circuit.

Rows follow the raw file's circuit-list order and then the count entries'
stored textual order. The source files contain aggregate count dictionaries,
not individual-shot sequences. Therefore these are not claimed to be the
chronologically first 20,000 hardware shots. Counts are not expanded into
fabricated shot sequences. Keep bitstrings as strings when importing the CSV.

The 20,000-entry cutoff can partially retain a circuit's count table. Each
retained count is still the original full-circuit count, not a recomputed
frequency for a 20,000-shot experiment. The excerpts must not be treated as
complete source distributions or an unbiased subset. The sum of counts and
number of retained entries are separately recorded.

selection_index.csv identifies every excerpt by public workload/case/seed,
retained entries, original count mass, touched circuits, and truncation.
circuit_index.csv links the numeric circuit positions to recorded names and
provides original/retained per-circuit entry and count totals. The first
20,000 entries span the beginning of each file only; they do not represent
every circuit in that workload.

For each excerpt, selection_index.csv supplies an ordered SHA-256 digest.
For each row in file order concatenate circuit_index_0based, a TAB,
entry_index_1based, a TAB, bitstring, a TAB, count, and LF using ASCII.
Hash the concatenation using SHA-256. The archive was checked against the
source-generated digest for every excerpt, preserving both values and order.

Provider job IDs, personal paths and private execution metadata are excluded.
These excerpts supplement final result
tables; exact full-pipeline reruns require the complete original datasets.

## Other final tables

The Figure 3 points are 2D projections at reported precision, not 3D coordinates.
display_order_within_case is an export index, not an inferred candidate ID.
The revised PC-variance values follow final SI S69. Figure 4 tables retain
unrounded final values from the 2,000-per-case analysis. Mean RMSD values in
basin summaries are final aggregate metrics, not individual scoring records.

Sensitivity tables retain the final 13-regime metrics. Native-RMSD-enriched
pools are post-hoc sensitivity analyses. S82 q values are arithmetic BH
reductions of the saved full-precision P values within the 12 displayed pools
for each fragment. No new permutations or experiments were run. Regimes,
conformations, poses, seeds and cases are distinct sampling units.

Code and reproduction instructions:
https://github.com/qiqi-xingyi/Qpocket
The exact accompanying code commit and SI identity are in PROVENANCE.json.

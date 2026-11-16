# Archived v1 clip1 measurements

These lightweight tables derive from the completed v1 archive: seven training checkpoints, ten unique evaluation groups, 2,000 process-exit-verified formal rollouts. They contain **no corrected v2 results**. The public derivative removes private absolute paths and does not redistribute checkpoint tensors, robot/human data, raw case records or videos.

The legacy recipe trained against unclipped q01/q99-normalized action targets but sampled with DDIM `clip_sample=True`, range1. The resulting support mismatch is confirmed; its causal contribution to success rates was not isolated. Preserve this label when quoting the values.

| File | Contents |
|---|---|
| `groups.csv` | Exact terminal/once success rates, continuous reward proxy and checkpoint hashes |
| `training.csv` | Budgets, training counts and initialization/checkpoint hashes |
| `per_level.csv`, `per_task_level.csv` | Level and task-level measurements |
| `resource_accounting_totals.csv` | Assigned-GPU wall-time summary; no private machine/storage fields |
| `review.json` | Compact public derivative suitable for the static plotting entrypoint; inherits the archived verification claim |
| `provenance.json` | Original archive checksums, pinned dependencies, v1 core digests and export boundaries |

`success_at_end_rate` and `success_once_rate` are fractions, not percentages in CSV. `SubSR_released_code_proxy` is mean final normalized reward, not the paper's thresholded subtask success rate. Scratch is one identical 200-case cohort repeated in the three scale tables. One training seed per condition is available; no independent multi-seed uncertainty is claimed.

The public code can evolve to v2 while these tables remain v1. Opaque historical experiment hashes are preserved as provenance IDs; a different installation has different absolute paths and need not reproduce those IDs. Read the repository's English/Chinese README for method differences and current retraining status.

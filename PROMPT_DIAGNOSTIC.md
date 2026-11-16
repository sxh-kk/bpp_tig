# Fixed C15 paired-prompt diagnostic

`diagnose_bpp_imitator.py` is separate from all four formal evaluation regimes.
It accepts only a completed, explicitly non-pilot, human-prompt C15 45000-update
checkpoint. Its weights and training settings are never selected by this diagnostic.

The predeclared cohort is the five official Seen tasks at L0, reset seeds 100000
and 100001. Each scene runs twice: its correct human RGB clip, and the same episode
index from the next task in the official Seen task order (cyclic permutation).
Both conditions use the same reset/per-episode RNG seed, 500 controls, execution
horizon 12, native qpos18/RGB observations, physical action16, and rt-fast renderer.
The twenty cases and permutation are written to `diagnostic_plan.json`.

From the TIG repository root, after sourcing project storage.env:

```bash
CHECKPOINT="$BPP_ROOT/results/training/c15_human_cls_ddp8/final.pt"
OUTPUT="$BPP_ROOT/results/diagnostics/c15_paired_prompt"
CUDA_VISIBLE_DEVICES=0 "$BPP_RUNTIME_PYTHON" -m examples.baselines.bpp.diagnose_bpp_imitator \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT" --num-shards 16 --shard-index 0
```

Run shard indexes 0–15, using two workers per GPU over eight GPUs. All workers
share the same output directory. Sharding is stable across worker counts.
Each public CLI launches a supervised worker, owns its assigned cases until native
process exit, and records the actual return code. A completed rollout from a worker
that exits abnormally is invalidated. Verified cases can be reused; unverified cases
and their videos/first-query files are archived before retrying. Case locks prevent
concurrent workers from claiming the same rollout. Immutable manifests reject mixed
weights or source configurations. A locked aggregate prevents stale partial snapshots
from overwriting a complete aggregate.

After every worker exits, rebuild and validate the whole cohort:

```bash
CUDA_VISIBLE_DEVICES='' "$BPP_RUNTIME_PYTHON" -m examples.baselines.bpp.diagnose_bpp_imitator \
  --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT" --collect-only
```

`--plan-only` declares the cohort without opening the checkpoint or using a GPU.
Do not invoke the private worker flag directly.

Each case retains metrics, rollout.mp4, and first_query.npz. The latter contains the
first current RGB/qpos and raw/normalized predicted action chunk, with separate
current-observation and Torch RNG hashes. No extra model forward is introduced.
Paired action differences are reported only when both hashes match exactly.
The summary always has `diagnostic_only: true` and `formal_success_rate: null`.
All twenty rollouts must finish with verified exit 0 before condition-level
**diagnostic** rates are populated. Numerical action sensitivity is auxiliary.

A prompt-dependent output does not prove complete imitation or intent understanding.
A mismatched task may be incompatible with the scene. This test also cannot separate
human-domain adaptation from the difficulty of bimanual control. The fixed cohort and
checkpoint prevent this diagnostic from becoming a held-out model-selection loop.

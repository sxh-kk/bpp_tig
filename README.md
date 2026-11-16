# BPP Baseline

[简体中文](README_CN.md)

[BPP](https://github.com/real-stanford/behavior_prompting), adapted to The Imitator Game. The policy conditions on a human RGB demonstration and current robot RGB/qpos to predict dual-arm actions. Robot action trajectories provide training supervision; demonstrator actions, poses, depth, masks and task/level/goal labels are not prompt inputs.

## Scripts

- `download_bpp_imitator.py`, `prepare_bpp_imitator.py`: verify/download data and prepare paired manifests and train-only statistics.
- `train_bpp_imitator.py`: C15/C30/C45 pretraining, target-only Scratch, fine-tuning and resume.
- `eval_bpp_imitator.py`: online TIG evaluation.
- `run_bpp_imitator.py`, `monitor_bpp_imitator.py`: complete staged workflow, including frozen CLS caches, and read-only progress monitoring.
- `report_bpp_imitator.py`, `report_resources_bpp_imitator.py`, `plot_bpp_imitator.py`: verify records, account for resources and render results.

## Installation and execution context

Install this integration at `<TIG_ROOT>/examples/baselines/bpp`. Follow the [TIG environment instructions](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/README.md) and [BPP LIBERO instructions](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/docs/libero.md) for dependencies and external weights. Keep the BPP vendor checkout, weights, datasets, caches and outputs outside this integration repository.

| Dependency | Recorded revision |
|---|---|
| [TIG](https://github.com/imitator-game/The-Imitator-Game) | `3193957e3b0187d086e44d79e0e4af9845fbd4fd` |
| [Behavior Prompting](https://github.com/real-stanford/behavior_prompting) | `58125e682acceefef3d60c071212dc483a4db1c9` |
| IG-10K | `9a57b27edcb54823d8da7d2b77efa61c092ffa95` |

Run from the TIG repository root. The recorded installation uses Torch 2.7.1/CUDA 12.6 and external storage configured by `storage.env`; source it in each new shell. It defines `BPP_CHECKPOINT` and `BPP_RUNTIME_PYTHON`.

```bash
source $BPP_ROOT/storage.env
cd "$BPP_ROOT/repos/The-Imitator-Game"
BPP_BASELINE_DIR="$PWD/examples/baselines/bpp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BPP_RUN_ROOT="$BPP_ROOT/results/pipeline/bpp_human_minmax_v2"
```

## Configuration

| Setting | Value |
|---|---|
| Human prompt | One ZED RGB clip; 16 uniformly sampled, ordered frames including endpoints |
| Robot input | Current simulated ZED RGB and measured qpos18; history 2 |
| Action | Two arms × (7 absolute joint targets + 1 gripper command), action16; `pd_joint_pos` |
| Policy / vision | BPP prompting, Transformer and action diffusion; approximately 347M trainable parameters; frozen CLIP ViT CLS |
| Preprocessing | Fixed 224×224 + CLIP normalization; no training crop/color augmentation; online current RGB uses 95% center crop |
| Normalization | Train-only joint14 min/max, identity grippers at indices 7/15; state q01/q99 |
| Batch | 8 windows/GPU × 8 A800 = global64; BF16 |
| Optimizer | AdamW, betas `(0.9, 0.999)`, weight decay `1e-6`, gradient norm clip1 |
| Schedule | Cosine LR `1e-4 → 1e-6`; 100 warmup updates for Ck, 10 for target training |
| Perturbation / EMA | Epsilon input perturbation `0.1`; EMA decay upper bound `max_decay=0.9999`, no warmup; EMA for evaluation/fresh P+FT, raw state for resume |
| DDIM | 50 training noise indices; 16 inference denoising steps; `clip_sample=True`, range1 |
| Control | Predict 16, execute 12, replan; 500 controls at 20Hz, physics 100Hz |
| Simulator | Official L0–L3 tasks and success predicates; `physx_cpu`, `rt-fast` |

The released single-arm BPP-LIBERO initialization uses two cameras, EEF/finger state and action10. This integration uses one camera, qpos18 and action16: compatible weights are reused, while changed state/action projections are learned from TIG robot data. The BPP prompting/Transformer/diffusion core is retained. The TIG interface follows [Zero-WAM](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/zero_wam/README.md), with the [OpenVLA integration](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/openvla_oft/README.md) as a dual-arm interface reference.

## Data and training budget

Use the official [splits](https://github.com/imitator-game/The-Imitator-Game/tree/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/lerobot_dataset/config/exp_configs), [task mapping](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/lerobot_dataset/task_mapping.json) and [evaluation lists](https://github.com/imitator-game/The-Imitator-Game/tree/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/lerobot_dataset/eval/exp_list). Pair each robot query with an allowed same-task human video; frame-level human/robot alignment is not required. Numerical statistics are fitted only on the corresponding robot training split.

| Split | Robot episodes | Human prompt episodes | Robot query anchors | Optimizer updates |
|---|---:|---:|---:|---:|
| C15 | 3,000 | 750 | 1,033,711 | 45,000 |
| C30 | 6,000 | 1,500 | 2,124,030 | 90,000 |
| C45 | 9,000 | 2,250 eligible | 3,039,478 | 135,000 |
| Target-only / target FT | 200 | 50 | 60,246 | 3,000 |

C45 has 2,250 eligible human clips paired to the selected 45 robot tasks. Prepare the manifests/statistics with:

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split c15 --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split scratch --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
```

Use `--split c30` or `--split c45` for the other scales. `download_bpp_imitator.py` supports `--scale c15|c30|c45` and verifies pinned payloads. The training command below uses the recorded prepared manifests and frozen CLS caches; fresh installations can use the staged runner to build those caches.

## Training

The following is the recorded C15 configuration. For an independent reproduction, select a separate run root and set `--c15-output` to its `training/c15`.

```bash
C15_MANIFEST="$BPP_ROOT/cache/bpp_tig_training/manifests/c15-human-e13fef9fd040a9e112d162ef/manifest.json"
C15_FEATURES="$BPP_ROOT/cache/bpp_frozen_cls/manifests/c15-human-e13fef9fd040a9e112d162ef-45bb2a4dc86d04c8.complete.json"
TARGET_MANIFEST="$BPP_ROOT/cache/bpp_tig_training/manifests/scratch-human-3ceb62b3696d7655c99a3b17/manifest.json"
TARGET_FEATURES="$BPP_ROOT/cache/bpp_frozen_cls/manifests/scratch-human-3ceb62b3696d7655c99a3b17-45bb2a4dc86d04c8.complete.json"

: "${BPP_RUN_ROOT:?Set the run root before launching}"
"$BPP_RUNTIME_PYTHON" -m torch.distributed.run --standalone --nproc-per-node 8 \
  -m examples.baselines.bpp.train_bpp_imitator \
  --regime c15 --prompt-source human \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --checkpoint "$BPP_CHECKPOINT" --data-manifest "$C15_MANIFEST" \
  --feature-manifest "$C15_FEATURES" --fixed-vision-preprocessing \
  --output-dir "$BPP_RUN_ROOT/training/c15" \
  --batch-size 8 --num-workers 4 --max-updates 45000 --seed 1 --init-seed 1
```

C30/C45 use the corresponding manifests and 90,000/135,000 updates. Each Ck is independently initialized from the same external BPP-LIBERO weights and common new-head initialization; scales do not continue one another. Training/init seed is 1. Final budget checkpoints are used without held-out score selection.

- **Seen / Zero-shot:** evaluate the same Ck on five seen / five held-out target tasks.
- **Scratch:** target-only training from the common external BPP initialization, using the target manifests and `--regime scratch --max-updates 3000`; it is not full-network random initialization.
- **P+FT:** use the same target data/budget, `--regime finetune --pretraining-scale c15 --initialize-from "$BPP_RUN_ROOT/training/c15/final.pt"`; initialization uses Ck EMA with a fresh optimizer and target statistics.

Keep `--action-normalization train_minmax_joint14_gripper_identity_v2` for training, fine-tuning, resume and evaluation. Resume restores the same stage's raw model, optimizer/RNG and EMA state. The complete pipeline can be launched with:

```bash
"$BPP_ROOT/envs/downloader/bin/python" "$BPP_BASELINE_DIR/run_bpp_imitator.py" run \
  --run-root "$BPP_RUN_ROOT" --c15-output "$BPP_RUN_ROOT/training/c15" \
  --action-normalization train_minmax_joint14_gripper_identity_v2
```

## Evaluation

```bash
: "${BPP_RUN_ROOT:?Set the reviewed run root}"
CUDA_VISIBLE_DEVICES=0 "$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/eval_bpp_imitator.py" \
  --checkpoint "$BPP_RUN_ROOT/training/c15/final.pt" \
  --output-dir "$BPP_RUN_ROOT/evaluation/c15_zero_shot" \
  --regime zero_shot --pretraining-scale c15 \
  --reference-source human --training-prompt-source human \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --num-episodes 10 --max-episode-steps 500 --execution-horizon 12 \
  --shader rt-fast --amp-dtype bfloat16 --device cuda:0
```

Use `--regime seen` with the same Ck; Scratch uses `--regime scratch`; P+FT uses `--regime pretrain_finetune --pretraining-scale c15` and its fine-tuned checkpoint. Evaluation uses original task success predicates, L0–L3, reset seeds 100000–100009, ten episodes per task-level and 500 controls. Human reference episode is `seed-100000`, reused across levels. Parallel evaluation used two workers/GPU, 16 shards; videos are enabled by default.

To independently verify the completed records and checkpoint hashes:

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_bpp_imitator.py" \
  --run-root "$BPP_RUN_ROOT" --c15-output "$BPP_RUN_ROOT/training/c15" \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --output-dir "$BPP_ROOT/results/review/all_scales_minmax_v2" --verify-checkpoints
```

## Results

Seven training stages and ten unique evaluation groups are complete: 2,000/2,000 rollouts passed [independent verification](metrics/v2_minmax/review.json).

**Terminal success rate (%)**

| Scale | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 47.5 | 3.0 | 5.0 | 37.5 |
| C30 | 55.0 | 1.5 | 5.0 | 46.0 |
| C45 | 66.0 | 4.0 | 5.0 | 56.0 |
| Three-scale arithmetic mean | 56.17 | 2.83 | 5.00 | 46.50 |

Each cell has 200 rollouts (five tasks × four levels × ten seeds). **Scratch is the same 200-case cohort shared across scales.** The last row is an arithmetic mean, not another experiment. P+FT exceeds shared Scratch by 32.5/41.0/51.0 percentage points.

Terminal SR means the task still reports success at the end of 500 controls; Once SR means success at any point within those 500 controls. C15 Seen has 115/200 Once successes (57.5%) and 95/200 terminal successes (47.5%). Some official predicates latch earlier events, so terminal success need not imply a maintained final grasp. Auxiliary Once SR and `SubSR_released_code_proxy = mean(last_normalized_reward)` are in [groups.csv](metrics/v2_minmax/groups.csv); the reward proxy is a scalar, **not the paper's thresholded Sub-SR**. See [per-level results](metrics/v2_minmax/per_level.csv) and the [result figures](metrics/v2_minmax/plots/bpp_human_rgb_bimanual_all-scales_terminal_success.png).

Formal work totals **78.633338 assigned A800 GPU-hours** (training 68.699220 + evaluation 9.934119), within a 9.920175-hour first-to-last process window (about 9h55m). This measures assigned process wall intervals, not kernel utilization or FLOPs; details are in [resource notes](metrics/v2_minmax/resource_accounting/RESOURCE_NOTES.md).

## Notes

- Results cover this BPP-LIBERO RGB-only dual-arm adaptation; LIBERO-Gen, Laundry and Draw are untested. Interface conversion, frozen vision/preprocessing and target-statistic changes differ from the source recipe; compatible tensor shapes do not establish semantic preservation.
- One training seed, five task clusters, related levels and shared Scratch limit statistical claims. Each group has 50 unique human clips; Seen/Scratch/P+FT evaluation clips overlap their training prompt pools. There is no prompt ablation to establish causal prompt use or unseen-human generalization.
- Data pools and update budgets grow together. BPP query-window training differs from Zero-WAM episode/chunk training; equal updates do not imply equal samples, epochs or FLOPs. Zero-shot remains low and nonmonotonic. Physical clipping occurred on 288/1,000,000 control ticks (0.0288%); its performance impact is unisolated.

Detailed method and evidence limits are in [FINAL_REVIEW_CN.md](FINAL_REVIEW_CN.md); historical results and diagnostics are consolidated in [HISTORY.md](HISTORY.md). See [TRAINING_HEALTH.md](TRAINING_HEALTH.md) for the operational health policy; a sticky health-stopped stage must not be auto-resumed. Public exports contain metric summaries and plots; full case and operational records remain local.

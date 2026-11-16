# BPP baseline for The Imitator Game

[简体中文](README_CN.md)

[BPP / What Enables In-Context Behavior Prompting for Manipulation?](https://arxiv.org/abs/2606.30457), adapted to TIG as **BPP human-RGB bimanual adaptation**. The policy receives a human RGB demonstration and current robot observations; robot trajectories provide action supervision. Upstream implementation: [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting).

## Status and results

**The tables labeled archived v1 belong to archived v1 (`legacy_quantile`, DDIM clip1). They are not results of the corrected v2 recipe.** All seven training checkpoints and ten unique evaluation groups were independently verified: 2,000 completed rollouts, clean worker exits and checkpoint hashes. The authoritative record is v1 review.json (local run artifact: `results/review/all_scales_v1_clip1/review.json`), archived on 2026-11-16; full precision values are in [groups.csv](metrics/v1_clip1/groups.csv).

V1 used unclipped q01/q99-normalized action targets during training, but inherited a DDIM sampler with `clip_sample=True`, range 1. Its final normalized predictions are restricted to `[-1,1]`, so raw joint targets outside the training quantiles are not representable through that sampler. This is a confirmed training/sampling support mismatch. It is separate from robot-controller clipping; zero controller clip counts do not rule it out. The scores do not establish how much this mismatch caused any failure or the C45 decline.

The accepted v2 recipe combines three corrections: `train_minmax_joint14_gripper_identity_v2` for the 14 joint targets and identity grippers; source input perturbation `0.1`; and source EMA updates. State normalization and DDIM clip1 remain. V2 implementation and pre-retraining review are complete. **As of 2026-11-16 14:47 Asia/Shanghai, v2 C15 has completed all four regimes; C30 training is active and C45 is pending.** Old weights cannot be relabeled as v2 by swapping their statistics. Use separate v2 checkpoints, manifests and output roots.

### Corrected v2 — C15 complete

All four groups have 200 completed, clean-exit rollouts with actual EMA inference and passing case/manifest checks (800 total). C30/C45 results are pending; no three-scale v2 mean is available.

| Regime | Terminal SR (%) | Once SR (%) | Released-code reward proxy |
|---|---:|---:|---:|
| Seen | 47.5 | 57.5 | 0.7716 |
| Zero-shot | 3.0 | 3.0 | 0.1505 |
| Scratch | 5.0 | 5.0 | 0.2412 |
| P+FT | 37.5 | 38.0 | 0.6111 |

Seen changes from 112/200 to 95/200: 43 previous successes fail and 26 previous failures succeed. Both C15 training logs have no recorded gradient norm above1; this regression is not the old C45 gradient pathology. Min/max, epsilon perturbation and raw-to-EMA evaluation changed together, so existing evidence does not isolate their causal contributions. The reward proxy remains distinct from paper-defined Sub-SR. See the [C15 comparison and C45 handling analysis (Chinese)](metrics/v2_c15/analysis_CN.md).

EMA does not feed back into raw training gradients. A separate [training health gate](TRAINING_HEALTH.md) was deployed on 2026-11-16 at 16:04 Asia/Shanghai for C30/C45 and their fine-tuning stages. It reads existing logs, cooperatively stops sustained gradient anomalies and synchronously audits complete stage logs before admitting checkpoints. The in-flight training source, recipe and budget are unchanged; no additional model experiments or tests were run.

### Primary metric: terminal success rate (%) — archived v1

| Scale | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 56.0 | 1.0 | 2.0 | 40.0 |
| C30 | 37.5 | 1.0 | 2.0 | 50.5 |
| C45 | 4.5 | 5.0 | 2.0 | 6.0 |
| Three-scale mean | 32.67 | 2.33 | 2.00 | 32.17 |

Each scale/regime cell contains 200 rollouts: five base tasks × L0–L3 × ten resets. Seen uses the five official seen tasks; Zero-shot, Scratch and P+FT use the same five target tasks. Scratch is **one identical 200-case baseline reused across scales**, not three independent experiments. The last row is the arithmetic mean of the three displayed scale rates, not an additional independent evaluation.

`success_at_end_rate` is the released task success predicate in final info after 500 controls. Some task predicates already latch a historical event, so terminal SR does not necessarily imply a maintained final grasp or complete physical behavior.

### Auxiliary metric: success at least once (%) — archived v1

| Scale | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 67.0 | 2.0 | 2.0 | 44.5 |
| C30 | 49.5 | 1.0 | 2.0 | 52.0 |
| C45 | 7.5 | 5.5 | 2.0 | 6.0 |

### Auxiliary continuous reward proxy — archived v1

| Scale | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 0.8502 | 0.1344 | 0.1951 | 0.6058 |
| C30 | 0.7452 | 0.1636 | 0.1951 | 0.7104 |
| C45 | 0.3586 | 0.2708 | 0.1951 | 0.2751 |

`SubSR_released_code_proxy = mean(last_normalized_reward)` is the released continuous stage/reward implementation proxy, reported here as a scalar rather than percent. It is **not the paper's thresholded subtask-success rate**. Per-level and per-task-level values are in [per_level.csv](metrics/v1_clip1/per_level.csv) and [per_task_level.csv](metrics/v1_clip1/per_task_level.csv). [Training records](metrics/v1_clip1/training.csv) retain update counts, initialization lineage and checkpoint SHA256.

## Public checkout layout and pinned dependencies

This repository is an integration package. Install its root at **`<TIG_ROOT>/examples/baselines/bpp`**, so the entrypoint import paths resolve the surrounding TIG simulator and baseline utilities. Keep the TIG simulator, BPP vendor code, model weights and IG data separate from this repository.

| Dependency | Verified source revision |
|---|---|
| [The Imitator Game](https://github.com/imitator-game/The-Imitator-Game) | `3193957e3b0187d086e44d79e0e4af9845fbd4fd` |
| [Behavior Prompting](https://github.com/real-stanford/behavior_prompting) | `58125e682acceefef3d60c071212dc483a4db1c9` |
| IG-10K data | `9a57b27edcb54823d8da7d2b77efa61c092ffa95` |

The intended public release is `sxh-kk/bpp_tig`. For a fresh installation:

```bash
export BPP_ROOT=/absolute/path/to/external/bpp_storage
mkdir -p "$BPP_ROOT/repos"
export TIG_ROOT="$BPP_ROOT/repos/The-Imitator-Game"
git clone https://github.com/imitator-game/The-Imitator-Game.git "$TIG_ROOT"
git -C "$TIG_ROOT" checkout 3193957e3b0187d086e44d79e0e4af9845fbd4fd
git -C "$TIG_ROOT" submodule update --init --recursive

git clone --recurse-submodules https://github.com/real-stanford/behavior_prompting.git \
  "$BPP_ROOT/repos/behavior_prompting"
git -C "$BPP_ROOT/repos/behavior_prompting" checkout 58125e682acceefef3d60c071212dc483a4db1c9
git -C "$BPP_ROOT/repos/behavior_prompting" submodule update --init --recursive

# Adapter release target; clone after this public repository is published.
git clone https://github.com/sxh-kk/bpp_tig.git "$TIG_ROOT/examples/baselines/bpp"
cp "$TIG_ROOT/examples/baselines/bpp/storage.env.example" "$BPP_ROOT/storage.env"
# Customize storage.env, install the pinned upstream environments, then source it.
source "$BPP_ROOT/storage.env"
```

Set `BPP_RUNTIME_PYTHON` to your installed compatible runtime and create the external cache/tmp/assets directories described in `storage.env.example`. Follow the pinned TIG/BPP environment instructions and [BPP LIBERO weight instructions](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/docs/libero.md); weights and data are not in this Git repository. Existing run-root/reuse examples below describe the author's prepared installation, not ready-input files supplied by a fresh public clone.

The public export includes lightweight legacy v1 derivatives under `metrics/v1_clip1`, with private path fields removed. Original local archive/audit links below identify the author's retained records; raw checkpoints, videos, private manifests and large review payloads are not distributed. Existing upstream MIT notices are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.

## Entrypoints and execution context

This directory follows the layout of the TIG [Zero-WAM](../zero_wam/README.md), [Diffusion Policy](../diffusion_policy/README.md) and [OpenVLA](../openvla_oft/README.md) integrations.

| Entrypoint | Purpose |
|---|---|
| `download_bpp_imitator.py` | Pinned human-video download and verification |
| `prepare_bpp_imitator.py` | Paired robot query / human prompt manifests and train-only statistics |
| `train_bpp_imitator.py` | C15/C30/C45, target-only Scratch, target fine-tuning and resume |
| `eval_bpp_imitator.py` | Online TIG evaluation with owned-process exit checks |
| `run_bpp_imitator.py`, `monitor_bpp_imitator.py` | Staged orchestration and read-only monitoring |
| `report_bpp_imitator.py` | Independent case/checkpoint review |
| `report_resources_bpp_imitator.py` | Formal-run resource accounting |
| `plot_bpp_imitator.py` | Reviewed-result PNG/SVG/PDF figures and CSV provenance |

Run from the TIG repository root in the project environment. All weights, data, simulator assets, caches and results remain under `$BPP_ROOT`; source the environment in each new shell:

```bash
source $BPP_ROOT/storage.env
cd "$BPP_ROOT/repos/The-Imitator-Game"
BPP_BASELINE_DIR="$PWD/examples/baselines/bpp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
V2_RUN_ROOT="$BPP_ROOT/results/pipeline/bpp_human_minmax_v2"
```

`BPP_RUNTIME_PYTHON` selects the established Torch 2.7.1/CUDA 12.6 environment. Shared installation instructions are in the [TIG README](../../../README.md); use the upstream BPP dependencies already installed in this project. Canonical implementation is `examples.baselines.bpp.bpp`. The old `bpp_tig_adapter` package only forwards imports; it is not a second implementation. Entrypoints also support `python -m examples.baselines.bpp.<entrypoint>`.

## Model configuration and adaptation

| Item | TIG BPP adaptation |
|---|---|
| External initialization | Released single-arm BPP-LIBERO checkpoint, `$BPP_CHECKPOINT` |
| Demonstration input | One human ZED RGB clip; 16 uniformly sampled ordered frames including endpoints |
| Current robot input | Live simulated ZED RGB and measured qpos18; history 2 |
| State18 | Each Panda: 7 arm joints + 2 finger positions |
| Action16 | Each Panda: 7 absolute joint targets + 1 normalized gripper command; `pd_joint_pos` |
| Vision | One 224×224 RGB view; frozen CLIP ViT, cached per-frame CLS for training |
| Temporal prompt | Learned frame positions and current-conditioned Transformer attention; BPP prompting/diffusion core retained |
| Native trainable scope | Approximately 347M parameters; frozen visual backbone; ordinary gradient training |
| DDIM | 50 training noise indices; **16 inference denoising steps**, `clip_sample=True`, range 1 |
| Control rollout | Predict 16 actions, execute 12, replan; 500 controls at 20Hz, physics 100Hz |
| Environment | Official base task + L0–L3 flags, `physx_cpu`, `rt-fast`, original success predicates |
| v1 action statistics | Global train-only q01/q99, labels not clipped |
| Accepted v2 action statistics | 14 joints: global train-only min/max; grippers at slots 7/15: identity |
| State statistics in both | Existing global train-only q01/q99; not changed by the action correction |
| v2 input perturbation | `0.1` on forward-process noise; original epsilon target retained |
| v2 checkpoint weights | EMA for evaluation/fresh P+FT; raw model + EMA state for exact resume |

**From the original prompt to human video.** Original BPP uses robot-compatible sensorimotor demonstrations, including demonstrator observations, proprioception and actions; its iPhUMI demonstrations provide that interface. This experiment deliberately exposes only human RGB. Demonstrator poses, gripper states/actions, depth, masks, MANO and task/level/goal labels are not policy prompt inputs. Robot actions are supervised targets, not demonstration conditioning. The executing robot may still read its own current RGB and qpos. Its current image comes from the simulator reset/step rendering.

**From the LIBERO checkpoint to dual-arm control.** The selected LIBERO checkpoint uses two camera streams and end-effector/finger inputs with a 10-D pose/rotation6D/gripper action representation. TIG uses one RGB stream, qpos18 and action16. The adapter instantiates the native shapes, reuses compatible BPP tensors, maps compatible finger projections, and learns the new state/action projections and action channels from TIG robot data. Five current modalities and the 15,360-D policy condition width are retained, although their meanings change. Parameter-shape compatibility or a high reuse fraction does not prove semantic preservation. This uses the released OpenVLA integration as an interface-adaptation precedent; the BPP Transformer/diffusion method remains the policy. BPP also supports a native bimanual Laundry setting; the question here concerns transfer of the particular LIBERO checkpoint.

Training uses fixed resize224 + original CLIP normalization and frozen CLS caches, with source random crop augmentation disabled. Human frames remain an ordered sparse sequence, not an unordered global frame mean; actual use of temporal order is not established by configuration alone.

V1 omitted the source input perturbation and native-training EMA updates. V2 restores both under `training_recipe=bpp_source_epsilon_perturbation_ema_v2`: update EMA after every real optimizer step, using source decay `0.9999` and no EMA warmup; evaluate/fresh-P+FT from EMA; resume from raw parameters plus optimizer/RNG/EMA state. The released original checkpoint's `state_dicts.model` already contains EMA weights because the source workspace copies EMA into that key before saving. The omission concerns TIG training updates, not the source weight key.

Remaining training differences are explicit below. Source values come from the actual released LIBERO checkpoint configuration and source code; they are not an assertion about every BPP experiment.

| Training item | Released LIBERO BPP recipe | TIG native v1 / retained v2 setting |
|---|---|---|
| Learning rate | Config base `5e-5`; source workspace scales by process count | `1e-4` to `1e-6`, cosine |
| Batch | DataLoader batch100, accumulation1; global batch depends on launch | 8 windows/GPU × 8 = global64 |
| AdamW betas | `(0.95, 0.999)` | PyTorch defaults `(0.9, 0.999)` |
| Warmup | 5% of source schedule | 100 updates for Ck; 10 for target training |
| Gradient clipping | Norm0.3 | Norm1.0 |
| Decay | `1e-6`, module-specific groups including no-decay groups | `1e-6`, one group of trainable parameters |
| Visual backbone | Source encoder trainable, with its optimizer groups | Frozen; fixed CLS cache |
| Image transforms | Source random crop/resize recipe | Fixed224 preprocessing; no random crop |
| Duration / packing | Config15epochs, source task/window sampling | Fixed45k/90k/135k/3kupdates, native windows and valid-tail loss masks |

Training records `preclip_grad_norm_by_module_rank0` and `grad_clip_coefficient_rank0` in `training.jsonl` / `progress.json` at the existing first/periodic/final rank0 logging points. Unique trainable parameter names are grouped by their first four components to expose UNet blocks and observation-encoder branches; this adds no forward/backward pass or change to RNG, updates, recipe or budget.

Restoring EMA and perturbation closes identified omissions; their contribution to success and the effect of remaining choices have not been measured. V2 is a disclosed TIG adaptation, not a claim that every source hyperparameter is reproduced.

## Data and four evaluation regimes

Use IG-10K revision `9a57b27edcb54823d8da7d2b77efa61c092ffa95`, official [experiment splits](../lerobot_dataset/config/exp_configs), [task mapping](../lerobot_dataset/task_mapping.json), and [evaluation lists](../lerobot_dataset/eval/exp_list). Robot query trajectories supply current observations and target actions. Training randomly pairs them with an allowed same-task human video, without frame-level human/robot alignment.

| Split | Robot episodes | Human prompt episodes | Robot query anchors | Optimizer updates |
|---|---:|---:|---:|---:|
| C15 | 3,000 | 750 | 1,033,711 | 45,000 |
| C30 | 6,000 | 1,500 | 2,124,030 | 90,000 |
| C45 | 9,000 | 2,250 eligible | 3,039,478 | 135,000 |
| Target-only / target FT | 200 | 50 | 60,246 | 3,000 |

Ck robot training contains 50 episodes per task-level; target-only training contains ten. The corresponding human budgets are 50 or ten clips per base task. C45's official human file lists 48 repositories, of which 45 pair with selected robot query tasks. All numerical statistics are fitted globally to the current permitted robot training split; no evaluation/test trajectories fit them.

| Regime | Training and evaluation definition |
|---|---|
| Seen | Evaluate Ck on the five seen tasks; no evaluation-time updates |
| Zero-shot | Evaluate the same Ck on the five held-out target tasks; no target-task parameter updates |
| Scratch | Train only on target data from the common external BPP initialization; not random initialization of the full model |
| P+FT | Start from Ck and train on the same target data and 3,000-update budget as Scratch |

Each Ck is independently initialized from the common external LIBERO weights and common new-head initialization; C30/C45 do not continue a preceding scale. Scratch is trained/evaluated once. This gives **seven training runs and ten unique evaluation groups**, or 2,000 rollouts. Batch is 8/GPU × 8 A800 = 64 query windows, BF16; training/init seed is 1. Final budget checkpoints are fixed in advance, with no held-out score selection. P+FT starts a fresh optimizer and uses target train-only statistics; v2 starts its new run from Ck EMA. V2 resume restores the same stage raw model, optimizer/RNG and EMA state.

Evaluation uses reset seeds 100000–100009 and human reference episode `seed-100000`, shared across L0–L3. There are 50 unique prompt clips per group. In archived v1, all 50 Seen/Scratch/P+FT evaluation clips belong to that checkpoint's training prompt pool; Zero-shot clips are absent from Ck training. This is not a clip-disjoint or person-disjoint evaluation claim.

## Prepare data and train

Data and frozen CLS manifests already prepared for v1 can be reused by the action-normalization correction. Their saved statistics include train min/max; the explicit model mode selects those action bounds. The frozen CLS fingerprint depends on encoder weights, preprocessing, dtype/runtime, not action normalization. RGB/CLS regeneration is unnecessary; old weights still require a fresh v2 training run. For a fresh cache, the existing preparation commands are:

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split c15 --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split scratch --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
```

Use `--split c30` or `--split c45` for other scales. The human downloader accepts `--scale c15|c30|c45` and verifies pinned payloads; on this installation the three scales and target pool are already present. Preparation prints the actual manifest path.

The following documents the **active v2 C15 configuration**. `--action-normalization` is mandatory; the reviewed versioned root is `results/pipeline/bpp_human_minmax_v2`. Use an independent output root for another reproduction run.

```bash
C15_MANIFEST="$BPP_ROOT/cache/bpp_tig_training/manifests/c15-human-e13fef9fd040a9e112d162ef/manifest.json"
C15_FEATURES="$BPP_ROOT/cache/bpp_frozen_cls/manifests/c15-human-e13fef9fd040a9e112d162ef-45bb2a4dc86d04c8.complete.json"
TARGET_MANIFEST="$BPP_ROOT/cache/bpp_tig_training/manifests/scratch-human-3ceb62b3696d7655c99a3b17/manifest.json"
TARGET_FEATURES="$BPP_ROOT/cache/bpp_frozen_cls/manifests/scratch-human-3ceb62b3696d7655c99a3b17-45bb2a4dc86d04c8.complete.json"

: "${V2_RUN_ROOT:?Set the separately reviewed v2 run root before launching}"
"$BPP_RUNTIME_PYTHON" -m torch.distributed.run --standalone --nproc-per-node 8 \
  -m examples.baselines.bpp.train_bpp_imitator \
  --regime c15 --prompt-source human \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --checkpoint "$BPP_CHECKPOINT" --data-manifest "$C15_MANIFEST" \
  --feature-manifest "$C15_FEATURES" --fixed-vision-preprocessing \
  --output-dir "$V2_RUN_ROOT/training/c15" \
  --batch-size 8 --num-workers 4 --max-updates 45000 --seed 1 --init-seed 1
```

For Scratch use the target manifests, `--regime scratch --max-updates 3000`, the same v2 normalization mode and a separate target-only output directory. For P+FT use `--regime finetune --pretraining-scale c15 --initialize-from "$V2_RUN_ROOT/training/c15/final.pt"`, the target manifests and 3,000 updates. C30/C45 use their corresponding manifests and 90,000/135,000 updates. Every v2 Ck and Scratch starts from the common external LIBERO checkpoint; P+FT/resume must retain the same v2 mode. Legacy native checkpoints remain readable as `legacy_quantile`, but are rejected as v2 initialization/resume sources.

The v2 staged runner reserves `results/pipeline/bpp_human_minmax_v2`, with C15 in its own `training/c15`, then sequences C15, C30 and C45. Existing prepared-input evidence is reused from the v1 root; this does not reuse v1 trained checkpoints or scores. C15 has completed all four regimes and C30 training is active. The first command only declares a plan; the run/monitor commands document the active configuration, with monitor reading current progress:

```bash
"$BPP_ROOT/envs/downloader/bin/python" "$BPP_BASELINE_DIR/run_bpp_imitator.py" plan \
  --run-root "$V2_RUN_ROOT" --c15-output "$V2_RUN_ROOT/training/c15" \
  --reuse-prepared-inputs-from "$BPP_ROOT/results/pipeline/bpp_human" \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --output "$BPP_ROOT/results/protocols/bpp_human_minmax_v2_pipeline_plan.json"

"$BPP_ROOT/envs/downloader/bin/python" "$BPP_BASELINE_DIR/run_bpp_imitator.py" run \
  --run-root "$V2_RUN_ROOT" --c15-output "$V2_RUN_ROOT/training/c15" \
  --reuse-prepared-inputs-from "$BPP_ROOT/results/pipeline/bpp_human" \
  --action-normalization train_minmax_joint14_gripper_identity_v2
"$BPP_ROOT/envs/downloader/bin/python" "$BPP_BASELINE_DIR/monitor_bpp_imitator.py" \
  --run-root "$V2_RUN_ROOT" --c15-output "$V2_RUN_ROOT/training/c15" \
  --reuse-prepared-inputs-from "$BPP_ROOT/results/pipeline/bpp_human" \
  --action-normalization train_minmax_joint14_gripper_identity_v2
```

`--config` is optional. If changing `--run-root`, explicitly change `--c15-output` to the new root's `training/c15` as well. The re-use path validates existing manifests/CLS files rather than redownloading or encoding. Monitor is read-only; a failed owned run can be resumed with the same configuration and `--retry-failed`, preserving failed-attempt evidence.

## Evaluation and reporting

After a matching complete v2 checkpoint exists:

```bash
: "${V2_RUN_ROOT:?Set the reviewed v2 run root}"
CUDA_VISIBLE_DEVICES=0 "$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/eval_bpp_imitator.py" \
  --checkpoint "$V2_RUN_ROOT/training/c15/final.pt" \
  --output-dir "$V2_RUN_ROOT/evaluation/c15_zero_shot" \
  --regime zero_shot --pretraining-scale c15 \
  --reference-source human --training-prompt-source human \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --num-episodes 10 --max-episode-steps 500 --execution-horizon 12 \
  --shader rt-fast --amp-dtype bfloat16 --device cuda:0
```

Use `--regime seen` with the same Ck; Scratch uses `--regime scratch`; P+FT uses `--regime pretrain_finetune --pretraining-scale c15` and the target fine-tuned checkpoint. Video is enabled by default. Parallel evaluation uses stable shards with a single physical GPU visible to each worker and local `cuda:0`; formal v1 used two workers per GPU, 16 shards in total.

Evaluation requires the explicit v2 `--action-normalization` flag. The independent report now defaults to `results/pipeline/bpp_human_minmax_v2`, C15 at its `training/c15`, and output at `results/review/all_scales_minmax_v2`. It also accepts an explicit normalization mode and `--source-root` for archived-source verification. Reference v2 review commands are:

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_bpp_imitator.py" \
  --run-root "$V2_RUN_ROOT" --c15-output "$V2_RUN_ROOT/training/c15" \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --output-dir "$BPP_ROOT/results/review/all_scales_minmax_v2" --verify-checkpoints
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_resources_bpp_imitator.py" \
  --run-root "$V2_RUN_ROOT" --output "$BPP_ROOT/results/review/all_scales_minmax_v2/resource_accounting/formal_resource_accounting.json"
```

A valid training stage requires its exact update budget, non-pilot completion, exit0 and matching checkpoint/data/feature/initialization metadata. Evaluation requires all 200 unique scheduled cases and verified owned worker exits0; abnormal exits invalidate the affected evidence. Missing/error results are null, never zero. The independent report checks these records and recomputes end/once rates. Resource accounting is assigned-GPU wall time with overlap handling, not kernel utilization or energy; its storage inventory is not the entire project footprint.

## Archived v1 reproduction and figures

Raw checkpoints, videos and case records remain in `results/pipeline/bpp_human` (the initial C15 checkpoint is in `results/training/c15_human_cls_ddp8`). Cite the archived review (local run artifact: `results/review/all_scales_v1_clip1/review.json`). The source snapshot manifest (local run artifact: `results/review/source_snapshots/bpp_primary_clip1_v1_20261116T023106Z/source_snapshot_manifest.json`) preserves the exact v1 source/hashes. Current report code can recheck v1 by explicitly selecting the archived module source root and `legacy_quantile`, including the four training-source and four evaluation-source checks. Write the recheck to a separate directory, preserving the archive:

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_bpp_imitator.py" \
  --run-root "$BPP_ROOT/results/pipeline/bpp_human" \
  --c15-output "$BPP_ROOT/results/training/c15_human_cls_ddp8" \
  --source-root "$BPP_ROOT/results/review/source_snapshots/bpp_primary_clip1_v1_20261116T023106Z/bpp" \
  --action-normalization legacy_quantile \
  --output-dir "$BPP_ROOT/results/review/recheck_v1_legacy_clip1" --verify-checkpoints
```

This is a historical record recheck, not a new policy evaluation or full training rerun. Full v1 reruns require the archived source in an isolated matching TIG checkout; executing a snapshot entrypoint by path alone does not select that canonical implementation.

Figures can be regenerated directly from the archived validated review, without loading a model:

```bash
CUDA_VISIBLE_DEVICES='' "$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/plot_bpp_imitator.py" \
  --review "$BPP_ROOT/results/review/all_scales_v1_clip1/review.json" \
  --output-dir "$BPP_ROOT/results/review/figures/v1_clip1" --with-levels
```

The renderer exports 300dpi PNG, SVG/PDF, data CSV and input-review/source/version SHA metadata. It defaults to all three complete scales; `--scales c15` or `--scales c15 c30` explicitly labels a completed subset. Archived resource accounting (local run artifact: `results/review/all_scales_v1_clip1/resource_accounting/formal_resource_accounting.json`) retains the v1 coverage limitations.

## C45 v1 anomaly analysis

Existing C45 logs contain 676 records: 548 pre-clipping gradient norms exceed 1, median302.97, p90 approximately33,706 and maximum2.7995e9. Every recorded C15/C30 norm is below1. The audit's late-training loss is approximately13times C30, and C45 target fine-tuning late loss is approximately7.8times C30 fine-tuning on the same target input pipeline. These observations support an optimization-pathology concern; they do not identify its triggering layer or sample.

MugRack's expert joint targets are100% covered by each C15/C30/C45 output box, yet Seen successes are20/40,21/40,0/40. Thus outside-box expert targets alone cannot explain this task's C45 collapse. Static review of all9,000C45 episodes /3,039,478frames found no nonfinite labels or column/unit inconsistency; source EMA weight loading, recorded learning rates, actual optimizer-step budgets and four normalization buffers were also consistent. These checks do not prove every interface is lossless.

The clipping coefficient rescales gradients before AdamW; it is not the same-factor reduction in parameter updates, and does not show that parameters were frozen. Min/max, EMA and input perturbation are explicit v2 repairs, not demonstrated solutions to C45. The added module-gradient/clip-coefficient telemetry belongs to the main training log, not an additional ablation. Full source evidence: C45 failure analysis (local run artifact: `results/review/bpp_c45_failure_analysis_v1.json`).

Detailed evidence is available in the [C45 analysis (Chinese)](metrics/v1_clip1/c45_explanation/analysis_CN.md), with a [training/Seen diagnostic figure](metrics/v1_clip1/c45_explanation/c45_v1_training_and_seen.png). The figure uses existing v1 logs and results; it adds no model experiments.

## Interpretation and comparison limits

The task/supervision interface follows the released TIG baselines, but the recipes differ. Zero-WAM uses whole-episode/chunk training, relative joints, its own video encoder and batch4; this BPP uses sliding query windows, absolute joints and batch64. Equal update counts do not establish equal samples, epochs or FLOPs. Paper summary tables average scales; the arithmetic mean above supports that presentation but does not make this implementation or the current released Zero-WAM recipe a reproduction of the paper's original numerical table. This README does not assert a model ranking from those numbers.

One training seed per condition, five fixed task clusters, related levels, matched reset/reference cases and shared Scratch limit statistical claims. Successes establish behavior of the combined adapted system; they do not isolate human-domain learning, checkpoint-conversion loss or perfect bimanual coordination. Existing C15 prompt controls support prompt-dependent actions on a small Seen cohort, not unseen-human generalization or complete temporal imitation. The confirmed normalization mismatch motivates retraining, but v2 gains remain unmeasured.

Read the official-protocol alignment audit (local run artifact: `results/review/bpp_official_protocol_alignment_audit.json`), adaptation validity audit (local run artifact: `results/review/bpp_adaptation_validity_audit.json`) and C15/C30 capability audit (local run artifact: `results/review/bpp_capability_claims_audit.json`) for source evidence and scope. Historical prompt checks are documented in [PROMPT_DIAGNOSTIC.md](PROMPT_DIAGNOSTIC.md). They remain separate from formal SR; no additional diagnostic matrix is part of the current training/evaluation plan.

## Public export contents

This repository contains the adapter source and compact archived v1 metrics, not datasets, assets, caches, environments or trained checkpoints. The frozen v1 Python source is retained under `legacy/v1_clip1`. Local full audit files referenced in this document are generated run artifacts; the public compact review is under [metrics/v1_clip1](metrics/v1_clip1/README.md).

Copy `storage.env.example` to your run directory and configure `BPP_ROOT` before running commands. The `--reuse-prepared-inputs-from` relaunch command assumes an existing locally verified prepared-input run; a fresh clone does not supply that directory. Prepare the pinned TIG/BPP dependencies, data and feature manifests first using the documented entrypoints.

# BPP 基线

[English](README.md)

将 [BPP](https://github.com/real-stanford/behavior_prompting) 适配到 The Imitator Game。策略根据人类 RGB 示范和执行机器人当前 RGB/qpos 预测双臂动作，以机器人动作轨迹监督训练；示范者动作、位姿、深度、mask 和任务/等级/目标标签不作为 prompt 输入。

## 入口脚本

- `download_bpp_imitator.py`、`prepare_bpp_imitator.py`：下载校验数据，准备配对 manifest 与 train-only 统计。
- `train_bpp_imitator.py`：C15/C30/C45 预训练、目标任务 Scratch、微调和恢复。
- `eval_bpp_imitator.py`：TIG 在线评测。
- `run_bpp_imitator.py`、`monitor_bpp_imitator.py`：包含冻结 CLS 缓存的完整分阶段流程，以及只读进度监控。
- `report_bpp_imitator.py`、`report_resources_bpp_imitator.py`、`plot_bpp_imitator.py`：记录核验、资源记账与结果绘图。

## 安装与运行环境

将本集成包放在 `<TIG_ROOT>/examples/baselines/bpp`。依赖环境按 [TIG 环境说明](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/README.md) 和 [BPP LIBERO 文档](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/docs/libero.md) 安装，并准备外部权重。BPP vendor、权重、数据、缓存和输出保存在本集成仓库外。

| 依赖 | 记录的 revision |
|---|---|
| [TIG](https://github.com/imitator-game/The-Imitator-Game) | `3193957e3b0187d086e44d79e0e4af9845fbd4fd` |
| [Behavior Prompting](https://github.com/real-stanford/behavior_prompting) | `58125e682acceefef3d60c071212dc483a4db1c9` |
| IG-10K | `9a57b27edcb54823d8da7d2b77efa61c092ffa95` |

从 TIG 仓库根目录运行。本次环境为 Torch2.7.1/CUDA12.6，外部存储由 `storage.env` 配置，包含 `BPP_CHECKPOINT` 与 `BPP_RUNTIME_PYTHON`；每个新 shell 都先 source：

```bash
source $BPP_ROOT/storage.env
cd "$BPP_ROOT/repos/The-Imitator-Game"
BPP_BASELINE_DIR="$PWD/examples/baselines/bpp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BPP_RUN_ROOT="$BPP_ROOT/results/pipeline/bpp_human_minmax_v2"
```

## 配置

| 配置 | 取值 |
|---|---|
| 人类提示 | 一段 ZED RGB，均匀采样16帧并保留顺序和首尾帧 |
| 机器人输入 | 仿真当前 ZED RGB 与实测 qpos18，历史2步 |
| 动作 | 双臂各7个绝对关节目标+1个夹爪命令，共 action16；`pd_joint_pos` |
| 策略/视觉 | BPP prompting、Transformer 和动作扩散，约347M可训练参数；冻结 CLIP ViT CLS |
| 预处理 | 固定224×224与 CLIP normalization；训练无随机裁剪/颜色增强；在线当前 RGB 使用95% center crop |
| 归一化 | train-only joint14 min/max，夹爪索引7/15保持 identity；state 使用 q01/q99 |
| Batch | 每卡8个 window×8卡 A800，global64、BF16 |
| 优化器 | AdamW，betas `(0.9, 0.999)`、weight decay `1e-6`、梯度 norm clip1 |
| 学习率 | cosine `1e-4 → 1e-6`；Ck warmup100更新，目标训练10更新 |
| 扰动/EMA | epsilon 输入扰动 `0.1`；EMA decay 上限 `max_decay=0.9999`、无 warmup；评测/新 P+FT 用 EMA，resume 用 raw 状态 |
| DDIM | 50个训练噪声索引，推理去噪16步，`clip_sample=True`、range1 |
| 控制 | 预测16步、执行12步再规划；500次控制，20Hz，physics100Hz |
| 仿真 | 官方 L0–L3 任务与成功判据，`physx_cpu`、`rt-fast` |

外部单臂 BPP-LIBERO 权重原用双相机、EEF/手指状态与 action10；此处改为单相机、qpos18 和 action16，复用形状兼容参数，用 TIG 机器人数据学习变化的状态/动作投影，保留 BPP prompting/Transformer/扩散核心。TIG 接口参考 [Zero-WAM](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/zero_wam/README.md)，双臂接口适配参考 [OpenVLA 集成](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/openvla_oft/README.md)。

## 数据与训练预算

使用官方 [split](https://github.com/imitator-game/The-Imitator-Game/tree/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/lerobot_dataset/config/exp_configs)、[task mapping](https://github.com/imitator-game/The-Imitator-Game/blob/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/lerobot_dataset/task_mapping.json) 和 [评测环境列表](https://github.com/imitator-game/The-Imitator-Game/tree/3193957e3b0187d086e44d79e0e4af9845fbd4fd/examples/baselines/lerobot_dataset/eval/exp_list)。每个机器人 query 随机配对允许池中的同任务人类视频，不要求人类/机器人逐帧对齐。数值统计仅在对应机器人训练池拟合。

| split | 机器人 episode | 人类 prompt episode | query anchors | 更新数 |
|---|---:|---:|---:|---:|
| C15 | 3,000 | 750 | 1,033,711 | 45,000 |
| C30 | 6,000 | 1,500 | 2,124,030 | 90,000 |
| C45 | 9,000 | 2,250个配对有效示范 | 3,039,478 | 135,000 |
| Scratch / 目标 FT | 200 | 50 | 60,246 | 3,000 |

C45 有2,250段与所选45个机器人任务配对有效的人类视频。manifest 和统计准备示例：

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split c15 --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split scratch --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
```

其他规模改用 `--split c30` 或 `--split c45`。`download_bpp_imitator.py` 支持 `--scale c15|c30|c45` 并校验固定 payload。下文训练命令使用本次已准备的 manifest 和冻结 CLS 缓存；首次安装可使用分阶段编排器生成这些缓存。

## 训练

以下为实际 C15 配置。独立复现时选用另一 run-root，并把 `--c15-output` 同时设为其 `training/c15`。

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

C30/C45 使用对应 manifest，分别训练90,000/135,000更新。各 Ck 从相同外部 BPP-LIBERO 权重和共同新层初始化独立开始，不接着前一规模训练。训练/init seed均为1，使用预算结束的 final checkpoint，不按 held-out 分数选模。

- **Seen / Zero-shot**：同一 Ck 分别在5个已见/5个未见目标任务评测。
- **Scratch**：从共同外部 BPP 初始化开始只训练目标池，使用目标 manifest 和 `--regime scratch --max-updates 3000`，不是全网络随机初始化。
- **P+FT**：相同目标池和预算，使用 `--regime finetune --pretraining-scale c15 --initialize-from "$BPP_RUN_ROOT/training/c15/final.pt"`；从 Ck EMA 初始化，重建优化器并使用目标统计。

训练、微调、恢复和评测均保留 `--action-normalization train_minmax_joint14_gripper_identity_v2`。resume 恢复同阶段 raw model、优化器/RNG 和 EMA 状态。完整流程也可由编排器启动：

```bash
"$BPP_ROOT/envs/downloader/bin/python" "$BPP_BASELINE_DIR/run_bpp_imitator.py" run \
  --run-root "$BPP_RUN_ROOT" --c15-output "$BPP_RUN_ROOT/training/c15" \
  --action-normalization train_minmax_joint14_gripper_identity_v2
```

## 评测

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

Seen 用同一 Ck 和 `--regime seen`；Scratch 用 `--regime scratch`；P+FT 用 `--regime pretrain_finetune --pretraining-scale c15` 与对应微调权重。采用原任务成功判据、L0–L3、reset seeds100000–100009，每个 task-level 评测10回合、每回合500控制步。人类 reference episode 为 `seed-100000`，跨等级复用。并行评测每卡2个 worker，共16片，默认保存视频。

独立核验完整记录与 checkpoint 哈希：

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_bpp_imitator.py" \
  --run-root "$BPP_RUN_ROOT" --c15-output "$BPP_RUN_ROOT/training/c15" \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --output-dir "$BPP_ROOT/results/review/all_scales_minmax_v2" --verify-checkpoints
```

## 最终结果

七个训练阶段、十组唯一评测均完成，2,000/2,000个 rollout 通过 [独立核验](metrics/v2_minmax/review.json)。

**末态成功率（%）**

| 档位 | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 47.5 | 3.0 | 5.0 | 37.5 |
| C30 | 55.0 | 1.5 | 5.0 | 46.0 |
| C45 | 66.0 | 4.0 | 5.0 | 56.0 |
| 三档算术均值 | 56.17 | 2.83 | 5.00 | 46.50 |

每格200回合（5任务×4等级×10 seed）。**Scratch 是三档复用的同一组200个 case。** 均值行是算术均值，不是新增实验。P+FT 比共享 Scratch 高32.5/41.0/51.0个百分点。

末态 SR 表示500步结束时仍被任务判定成功，Once SR 表示这500步内任意时刻曾成功。C15 Seen 为115/200曾经成功（57.5%）、95/200末态成功（47.5%）。部分原判据会锁存此前事件，因此末态成功不必然表示最后仍保持抓取。辅助 Once 与 `SubSR_released_code_proxy = mean(last_normalized_reward)` 见 [groups.csv](metrics/v2_minmax/groups.csv)；reward proxy 是标量，**不等于论文阈值式 Sub-SR**。另见 [分等级结果](metrics/v2_minmax/per_level.csv) 和 [结果图](metrics/v2_minmax/plots/bpp_human_rgb_bimanual_all-scales_terminal_success.png)。

正式作业共 **78.633338 assigned A800 GPU-hours**（训练68.699220+评测9.934119），首尾正式进程窗口9.920175小时，约9小时55分。统计进程占用的 wall 区间，不是 kernel 利用率或 FLOPs，细节见 [资源说明](metrics/v2_minmax/resource_accounting/RESOURCE_NOTES.md)。

## 说明

- 成绩仅覆盖这份 BPP-LIBERO RGB-only 双臂适配，LIBERO-Gen、Laundry、Draw 未测。接口转换、冻结视觉/预处理与目标统计切换不同于源配方，形状兼容不等于语义无损。
- 只有一个训练 seed、5个固定任务簇，等级相关且 Scratch 共享；每组50段唯一人类视频，Seen/Scratch/P+FT 评测提示与训练池重叠。没有 prompt 消融来证明提示的因果作用或未见人类泛化。
- 数据池与更新预算同时变化，BPP query-window 与 Zero-WAM episode/chunk 训练不同，相同 updates 不代表样本、epoch 或 FLOPs 相同。Zero-shot 低且不单调；288/1,000,000个控制 tick 触发物理限幅（0.0288%），性能影响未隔离。

方法与证据边界详见 [FINAL_REVIEW_CN.md](FINAL_REVIEW_CN.md)，历史结果与诊断统一见 [HISTORY.md](HISTORY.md)。运行健康策略见 [TRAINING_HEALTH.md](TRAINING_HEALTH.md)，sticky health-stopped 阶段不得自动恢复。公开导出提供指标摘要和图表，完整逐例与运行记录保留在本地。

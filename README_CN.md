# BPP / The Imitator Game 基线

[English](README.md)

本目录将 [BPP：What Enables In-Context Behavior Prompting for Manipulation?](https://arxiv.org/abs/2606.30457) 适配到 TIG，版本名称为 **BPP human-RGB bimanual adaptation（人类 RGB 提示双臂适配）**。输入人类 RGB 示范与执行机器人的当前观测，用机器人动作轨迹监督训练。上游代码为 [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting)。

## 当前状态与指标

**标注为 v1 的表格是已封存的 v1（`legacy_quantile`、DDIM clip1）结果，不是修正后的 v2 成绩。** 七份训练 checkpoint、十组独立评测已完成独立核验，共2,000个正式 rollout，包含 worker 正常退出与权重哈希检查。权威归档为2026-11-16的 v1 review.json (local run artifact: `results/review/all_scales_v1_clip1/review.json`)，完整精度见 [groups.csv](metrics/v1_clip1/groups.csv)。

v1 将动作按训练 q01/q99 归一化，但训练目标没有裁剪；推理沿用 `clip_sample=True`、range1 的 DDIM，最终归一化输出受限于 `[-1,1]`，因此不能输出超出相应训练分位数的原始关节目标。**训练目标与采样支持范围不配套已经确认。** 这与控制器的物理关节限幅是两个环节，控制器 clip_count 为0不能排除此问题。该问题对成功率、尤其 C45 下降的具体贡献尚未被因果隔离。

已接受的 v2 配方包含三项修正：`train_minmax_joint14_gripper_identity_v2` 使14个关节目标使用**训练 min/max**、两维夹爪命令保持 identity；恢复源 input perturbation `0.1`；恢复源 EMA 更新。state18 的既有归一化与 DDIM clip1 保留。实现与重训前复核已完成，**截至北京时间2026-11-16 14:47，v2 C15 四口径已完成，C30 训练进行中，C45 尚未开始**。不能给旧权重换一份统计量就标成 v2；新训练使用独立 checkpoint、manifest 和输出目录。

### 修正版 v2——C15 四口径已完成

四组各200个正式回合，均通过case/manifest及worker正常退出核验，实际使用EMA推理，共800回合。C30/C45尚无新版成绩，不计算v2三档均值。

| 口径 | 末态成功率 SR（%） | Once 成功率（%） | 开源代码 reward proxy |
|---|---:|---:|---:|
| Seen | 47.5 | 57.5 | 0.7716 |
| Zero-shot | 3.0 | 3.0 | 0.1505 |
| Scratch | 5.0 | 5.0 | 0.2412 |
| P+FT | 37.5 | 38.0 | 0.6111 |

Seen 从112/200变为95/200：43例旧成功变失败，26例旧失败变成功。两版C15训练日志均没有记录到梯度范数大于1，不能套用旧C45的梯度异常解释。min/max、输入扰动和raw→EMA评测权重同时变化，现有证据无法分别归因。reward proxy仍不等同论文Sub-SR。详见 [C15下降与C45处理分析](metrics/v2_c15/analysis_CN.md)。

EMA不参与raw训练梯度反馈。已于北京时间2026-11-16 16:04部署独立的[训练健康门槛](TRAINING_HEALTH.md)，覆盖C30/C45及其微调：读取既有日志，持续异常时协调停止，阶段验收同步复核全段日志。在跑训练源码、配方和预算保持不变；未增加模型实验或测试。

### 主指标：末态成功率 SR（%）——v1 归档

| 档位 | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 56.0 | 1.0 | 2.0 | 40.0 |
| C30 | 37.5 | 1.0 | 2.0 | 50.5 |
| C45 | 4.5 | 5.0 | 2.0 | 6.0 |
| 三档算术均值 | 32.67 | 2.33 | 2.00 | 32.17 |

每个档位/口径单元格为5个基础任务×L0–L3×10次，共200个 rollout。Seen 使用官方5个已见任务，其余三口径使用同一组5个目标任务。**Scratch 是跨三档复用的同一组200回合，不是三个独立实验。** 最后一行是三档比率的算术均值，不是新增独立评测。

`success_at_end_rate` 取500次控制后的 final info 中原任务的成功布尔值。部分原任务判据本身会记住此前达成的事件，因此“末态 SR”不一定表示最后仍保持抓取或完整物理过程。

### 辅助指标：曾经成功率（%）——v1 归档

| 档位 | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 67.0 | 2.0 | 2.0 | 44.5 |
| C30 | 49.5 | 1.0 | 2.0 | 52.0 |
| C45 | 7.5 | 5.5 | 2.0 | 6.0 |

### 辅助指标：连续奖励代理——v1 归档

| 档位 | Seen | Zero-shot | Scratch | P+FT |
|---|---:|---:|---:|---:|
| C15 | 0.8502 | 0.1344 | 0.1951 | 0.6058 |
| C30 | 0.7452 | 0.1636 | 0.1951 | 0.7104 |
| C45 | 0.3586 | 0.2708 | 0.1951 | 0.2751 |

`SubSR_released_code_proxy = mean(last_normalized_reward)` 是发布代码的连续阶段/奖励实现代理，上表保留标量单位，**不能称为论文中按阈值统计的子任务成功率**。分等级与任务等级结果见 [per_level.csv](metrics/v1_clip1/per_level.csv)、[per_task_level.csv](metrics/v1_clip1/per_task_level.csv)；[training.csv](metrics/v1_clip1/training.csv) 保存更新数、初始化来源和 checkpoint SHA256。

## 公开仓库安装位置与固定依赖

本仓库是集成包，应将根目录放在 **`<TIG_ROOT>/examples/baselines/bpp`**，这样入口才能正确导入外层 TIG 仿真及基线工具。TIG 仿真源码、BPP vendor、权重和 IG 数据独立保存，不装进本仓库。

| 依赖 | 已核对的源码 revision |
|---|---|
| [The Imitator Game](https://github.com/imitator-game/The-Imitator-Game) | `3193957e3b0187d086e44d79e0e4af9845fbd4fd` |
| [Behavior Prompting](https://github.com/real-stanford/behavior_prompting) | `58125e682acceefef3d60c071212dc483a4db1c9` |
| IG-10K 数据 | `9a57b27edcb54823d8da7d2b77efa61c092ffa95` |

计划公开仓库为 `sxh-kk/bpp_tig`。首次安装示例：

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

将 `BPP_RUNTIME_PYTHON` 改为已安装且兼容的 runtime，并创建 `storage.env.example` 指定的外部 cache/tmp/assets 等目录。环境和权重按固定 TIG/BPP 上游说明及 [BPP LIBERO 文档](https://github.com/real-stanford/behavior_prompting/blob/58125e682acceefef3d60c071212dc483a4db1c9/docs/libero.md) 准备，Git 仓库不包含权重和数据。下文 run-root/reuse 示例对应作者已准备的本机环境，新克隆仓库不自带这些 ready-input 文件。

公开导出在 `metrics/v1_clip1` 提供已删除私有路径字段的轻量 v1 指标。下文原本机 archive/audit 链接说明作者保留的记录；原权重、视频、私有 manifest 和大型 review 文件不随仓库分发。现有上游 MIT 声明保留在 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 `licenses/`。

## 入口与运行环境

目录结构参考 TIG 官方 [Zero-WAM](../zero_wam/README.md)、[Diffusion Policy](../diffusion_policy/README.md)、[OpenVLA](../openvla_oft/README.md) 基线。

| 入口 | 用途 |
|---|---|
| `download_bpp_imitator.py` | 固定 revision 的人类视频下载与校验 |
| `prepare_bpp_imitator.py` | 机器人 query / 人类 prompt manifest 与 train-only 统计 |
| `train_bpp_imitator.py` | C15/C30/C45、Scratch、目标任务微调和恢复 |
| `eval_bpp_imitator.py` | TIG 在线评测及所属 worker 退出核验 |
| `run_bpp_imitator.py`、`monitor_bpp_imitator.py` | 分阶段编排与只读监控 |
| `report_bpp_imitator.py` | 独立逐案例/权重审查 |
| `report_resources_bpp_imitator.py` | 正式作业资源记账 |
| `plot_bpp_imitator.py` | 经核验结果的 PNG/SVG/PDF 图和 CSV 来源记录 |

从 TIG 仓库根目录运行，全部权重、数据、仿真资产、缓存和结果保存在 `$BPP_ROOT`。每个新 shell 先执行：

```bash
source $BPP_ROOT/storage.env
cd "$BPP_ROOT/repos/The-Imitator-Game"
BPP_BASELINE_DIR="$PWD/examples/baselines/bpp"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
V2_RUN_ROOT="$BPP_ROOT/results/pipeline/bpp_human_minmax_v2"
```

`BPP_RUNTIME_PYTHON` 指向现有 Torch2.7.1/CUDA12.6 环境。共享环境说明见 [TIG README](../../../README.md)，本项目已安装上游 BPP 所需依赖。唯一核心 namespace 是 `examples.baselines.bpp.bpp`；旧 `bpp_tig_adapter` 仅做 import 转发，不维护第二份实现。也支持 `python -m examples.baselines.bpp.<entrypoint>`。

## 模型配置与适配处理

| 项目 | TIG BPP 适配 |
|---|---|
| 外部初始化 | 已发布的单臂 BPP-LIBERO checkpoint，`$BPP_CHECKPOINT` |
| 示范输入 | 一段人类 ZED RGB；均匀取16帧，保留顺序及首尾帧 |
| 执行机器人输入 | 仿真当前 ZED RGB、实测 qpos18；历史2步 |
| state18 | 每臂7个关节位置+2个手指位置，两臂共18维 |
| action16 | 每臂7个绝对关节目标+1个归一化夹爪命令；`pd_joint_pos` |
| 视觉 | 单路224×224 RGB；冻结 CLIP ViT，训练使用逐帧 CLS 缓存 |
| 时序提示 | 学习式帧位置与当前观测条件下的 Transformer 注意力；保留 BPP 提示/扩散核心 |
| 可训练范围 | 约347M参数，视觉骨干冻结，其余按普通梯度训练 |
| DDIM | 50个训练噪声索引；**推理去噪16步**；`clip_sample=True`、range1 |
| 控制预算 | 每次预测16步、执行12步再规划；500次控制，control20Hz/physics100Hz |
| 仿真 | 官方基础任务+L0–L3 flags，`physx_cpu`、`rt-fast`、原成功判据 |
| v1 动作统计 | 当前训练池的全局 q01/q99，训练标签不裁剪 |
| 已接受的 v2 动作统计 | 14维关节：当前训练池全局 min/max；夹爪7/15维：identity |
| 两版 state 统计 | 保留当前训练池的全局 q01/q99，不随动作修正改变 |
| v2 input perturbation | 前向加噪过程使用0.1扰动，仍以原始 epsilon 为训练目标 |
| v2 权重使用 | 评测/新 P+FT 用 EMA；精确 resume 恢复 raw model+EMA 状态 |

**从原机器人传感运动示范到人类视频。** 原 BPP prompt 包含与机器人兼容的示范观测、proprioception 和动作；iPhUMI 人类示范提供这种接口。本实验只给人类 RGB，示范位姿、夹爪状态/动作、depth、mask、MANO 以及任务/等级/目标标签均不作为策略提示。机器人动作是训练监督，不是示范数值输入。执行机器人仍允许读取自身当前 RGB/qpos；当前图像由仿真 reset/step 渲染获得。

**从 LIBERO 单臂权重到双臂接口。** 所选 LIBERO checkpoint 原来有两路相机、末端/手指状态及10维 pose/rot6d/gripper 动作；TIG 改为一路 RGB、qpos18 和 action16。适配器先按新维度构建模型，再加载形状兼容的 BPP 参数，映射可兼容的手指投影，并用 TIG 机器人轨迹学习新的状态/动作投影和输出通道。保留5种当前观测模态及15,360维策略条件宽度，但模态语义发生改变。参数兼容或复用比例高不等于语义无损。这参考了 OpenVLA 的接口适配做法，策略仍是 BPP 的 Transformer/扩散方法。BPP 本身另有原生双臂 Laundry 场景，本项目需要评估的是这份 LIBERO checkpoint 的迁移。

训练采用固定 resize224+原 CLIP normalization 与冻结 CLS 缓存，关闭上游随机裁剪增强。人类帧仍保留稀疏时序，不是无序全局帧均值；配置能够表示顺序不等于已证明学会按时序模仿。

v1 漏接源 input perturbation 及 native 训练的 EMA 更新。v2 在 `training_recipe=bpp_source_epsilon_perturbation_ema_v2` 中恢复这两项：每次真实 optimizer step 后更新 EMA，源 decay为0.9999、不开 EMA warmup；评测与新 P+FT 使用 EMA；resume 恢复 raw 参数、优化器/RNG 和 EMA 状态。原始发布权重的 `state_dicts.model` 保存前已由源 workspace 拷入 EMA，因此问题在 TIG 训练漏更新 EMA，而不是取错原始权重键。

仍保留的配方差异如下。源值来自实际 LIBERO checkpoint 配置及源码，不代表所有 BPP 实验都使用同一组设置。

| 训练项目 | 发布 LIBERO BPP 配方 | TIG native v1 / v2 保留设置 |
|---|---|---|
| 学习率 | 配置基准5e-5，源 workspace 还按进程数放大 | 1e-4→1e-6，cosine |
| Batch | DataLoader batch100、accumulation1，全局 batch 取决于启动方式 | 每卡8个 window×8卡，global64 |
| AdamW betas | `(0.95, 0.999)` | PyTorch 默认 `(0.9, 0.999)` |
| Warmup | 源训练计划的5% | Ck为100更新，目标训练为10更新 |
| 梯度裁剪 | norm0.3 | norm1.0 |
| Weight decay | 1e-6，按模块分组，含不衰减组 | 1e-6，全部可训练参数为一个组 |
| 视觉骨干 | 源 encoder 可训练，使用自身优化器分组 | 冻结，使用固定 CLS 缓存 |
| 图像变换 | 源随机裁剪/resize配方 | 固定224，无随机裁剪 |
| 时长/组织 | 配置15epochs及源任务/window采样 | 固定45k/90k/135k/3k更新，native窗口与有效尾部mask |

训练在原有首步/周期/末步 rank0 记录点，将 `preclip_grad_norm_by_module_rank0` 和 `grad_clip_coefficient_rank0` 写入 `training.jsonl` / `progress.json`；按唯一可训练参数名的前四段分组，区分 UNet block 与观测编码器 branch，不增加 forward/backward，也不改变 RNG、更新、配方或预算。

EMA 与扰动修正闭合明确遗漏，但各自对成功率的贡献、其余设置的影响尚未测量。v2 是明确披露的 TIG 适配，不宣称逐项复现源超参数。

## 数据与四个测评口径

使用 IG-10K revision `9a57b27edcb54823d8da7d2b77efa61c092ffa95`、官方 [训练划分](../lerobot_dataset/config/exp_configs)、[task mapping](../lerobot_dataset/task_mapping.json) 和 [评测环境列表](../lerobot_dataset/eval/exp_list)。机器人 query 提供当前观测与目标动作；训练时从允许的人类池随机选同任务示范，不要求人类和机器人逐帧对齐。

| split | 机器人 episode | 人类 prompt episode | query anchors | 更新数 |
|---|---:|---:|---:|---:|
| C15 | 3,000 | 750 | 1,033,711 | 45,000 |
| C30 | 6,000 | 1,500 | 2,124,030 | 90,000 |
| C45 | 9,000 | 2,250个配对有效示范 | 3,039,478 | 135,000 |
| Scratch / 目标 FT | 200 | 50 | 60,246 | 3,000 |

Ck 机器人训练每个 task-level 取50条，目标任务训练每个 task-level 取10条；对应人类池每个基础任务取50/10条。C45 官方 human 文件列48个仓库，只有45个与选定机器人 query 配对。数值统计只对当前允许的机器人训练池全局拟合，不使用测试/评测轨迹。

| 口径 | 训练与评测定义 |
|---|---|
| Seen | 在官方5个已见任务评测 Ck，评测时不更新参数 |
| Zero-shot | 同一份 Ck 在5个未见目标任务评测，不用目标任务更新参数 |
| Scratch | 从共同外部 BPP 初始化开始，只训练目标数据；不是整个模型随机初始化 |
| P+FT | 从 Ck 开始，使用与 Scratch 相同的目标数据和3,000更新预算 |

各 Ck 均从共同外部 LIBERO 权重和共同新层初始化独立开始，C30/C45 不接着前一档继续训练。Scratch 只训练/评测一次，所以共**7个训练阶段、10组唯一评测、2,000回合**。8卡 A800、每卡8个 query window、global batch64、BF16，训练/init seed均为1。使用预定预算结束的 final checkpoint，不按 held-out SR 选模。P+FT 重建优化器并使用目标训练池统计，v2 新微调从 Ck EMA 开始；v2 resume 恢复同阶段 raw model、优化器/RNG 和 EMA 状态。

每个 task-level 评测 seed100000–100009，对应人类 reference episode `seed-100000`，同任务跨 L0–L3 共用示范。每组有50个唯一人类片段。v1 的 Seen/Scratch/P+FT 这50个评测片段均在相应训练 prompt 池内，Zero-shot 片段不在 Ck 池内；不能称为“未见片段”或“跨人”测试。

## 数据准备与训练

v1 已准备的数据与冻结 CLS manifest 可供动作归一化修正复用；其中已保存训练 min/max，由明确的模型 mode 选择动作边界。冻结 CLS 的 fingerprint 依赖 encoder 权重、预处理、dtype/runtime，不依赖动作归一化。不需要重制 RGB/CLS，旧权重仍须通过新的 v2 训练产生对应模型。如需首次准备，原有命令为：

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split c15 --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/prepare_bpp_imitator.py" \
  --split scratch --prompt-source human --workers 8 \
  --cache-root "$BPP_ROOT/cache/bpp_tig_training"
```

其他档位改用 `--split c30` / `--split c45`。人类下载入口支持 `--scale c15|c30|c45` 并校验固定 revision；本机三档和目标池已经齐备。prepare 命令输出实际 manifest 路径。

以下为**正在运行的 v2 C15 配置**。`--action-normalization` 必须显式传入；已复核的版本化根目录是 `results/pipeline/bpp_human_minmax_v2`。另行复现时使用独立输出目录。

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

Scratch 改用目标 manifest、`--regime scratch --max-updates 3000`，保持 v2 normalization 并使用独立输出目录。P+FT 用 `--regime finetune --pretraining-scale c15 --initialize-from "$V2_RUN_ROOT/training/c15/final.pt"`、目标 manifest 和3,000更新；C30/C45 分别使用对应 manifest 与90,000/135,000更新。每个 v2 Ck 及 Scratch 从共同外部 LIBERO 权重开始，P+FT/resume 必须保持相同 v2 mode。无新字段的历史 native checkpoint 仍按 `legacy_quantile` 可读，但不能作为 v2 初始化/恢复来源。

v2 编排器使用独立 `results/pipeline/bpp_human_minmax_v2`，C15 保存于该根目录的 `training/c15`，然后依次完成 C15/C30/C45 四口径。复用 v1 的已准备输入证据，不复用其已训练权重或成绩。C15 四口径已完成并核验，C30 训练进行中。第一条只生成计划，run/monitor 展示当前配置，其中 monitor 读取当前进度：

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

`--config` 可选。若自定义 `--run-root`，必须同时将 `--c15-output` 改成新根目录的 `training/c15`。复用入口校验已有 manifest/CLS，不重新下载或编码。monitor 只读；所属作业失败后可按原配置加 `--retry-failed` 恢复，并保留失败记录。

## 评测与独立汇总

对应 v2 完整 checkpoint 产生后：

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

Seen 用同一 Ck 和 `--regime seen`；Scratch 用 `--regime scratch`；P+FT 用 `--regime pretrain_finetune --pretraining-scale c15` 和目标微调权重。默认保存视频。并行评测使用固定分片，每个 worker 只看见一张物理 GPU，进程内用 `cuda:0`；v1 正式评测每卡2个 worker，共16片。

评测必须显式传入 v2 `--action-normalization`。独立 report 的默认 run-root 已是 `results/pipeline/bpp_human_minmax_v2`，C15 位于其 `training/c15`，默认输出为 `results/review/all_scales_minmax_v2`；也可显式指定归一化 mode 和用于历史源码核验的 `--source-root`。v2 汇总示例：

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_bpp_imitator.py" \
  --run-root "$V2_RUN_ROOT" --c15-output "$V2_RUN_ROOT/training/c15" \
  --action-normalization train_minmax_joint14_gripper_identity_v2 \
  --output-dir "$BPP_ROOT/results/review/all_scales_minmax_v2" --verify-checkpoints
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_resources_bpp_imitator.py" \
  --run-root "$V2_RUN_ROOT" --output "$BPP_ROOT/results/review/all_scales_minmax_v2/resource_accounting/formal_resource_accounting.json"
```

完整训练同时要求精确预算、非 pilot、正常完成、进程 exit0，以及权重/数据/特征/初始化 metadata 一致。评测要求200个唯一计划 case、所属 worker 均被监督并 exit0；异常退出会使相应证据无效。缺失或错误指标为 null，不填0。独立 report 重算末态/曾经成功率。资源记账是处理同卡重叠后的 assigned-GPU wall 时间，不是 GPU kernel 利用率或能耗；其存储清单也不是项目全部占用。

## v1 复现来源与绘图

原权重、视频和 case 仍在 `results/pipeline/bpp_human`，初始 C15 权重在 `results/training/c15_human_cls_ddp8`。引用 归档审查 (local run artifact: `results/review/all_scales_v1_clip1/review.json`)；源码快照 manifest (local run artifact: `results/review/source_snapshots/bpp_primary_clip1_v1_20261116T023106Z/source_snapshot_manifest.json`) 保存原 v1 源码/哈希。当前 report 可通过显式指定 archived module source-root 与 `legacy_quantile` 重审 v1，同时核对四个训练源码及四个评测源码；输出另存，不覆盖归档：

```bash
"$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/report_bpp_imitator.py" \
  --run-root "$BPP_ROOT/results/pipeline/bpp_human" \
  --c15-output "$BPP_ROOT/results/training/c15_human_cls_ddp8" \
  --source-root "$BPP_ROOT/results/review/source_snapshots/bpp_primary_clip1_v1_20261116T023106Z/bpp" \
  --action-normalization legacy_quantile \
  --output-dir "$BPP_ROOT/results/review/recheck_v1_legacy_clip1" --verify-checkpoints
```

这只是历史记录复核，不是新的策略测评或完整重训。完整重跑 v1 仍需在独立、匹配的 TIG checkout 使用归档源码；仅按路径运行快照入口不会自动选择那份 canonical 实现。

可直接从已核验归档结果生成图，无需加载模型：

```bash
CUDA_VISIBLE_DEVICES='' "$BPP_RUNTIME_PYTHON" "$BPP_BASELINE_DIR/plot_bpp_imitator.py" \
  --review "$BPP_ROOT/results/review/all_scales_v1_clip1/review.json" \
  --output-dir "$BPP_ROOT/results/review/figures/v1_clip1" --with-levels
```

输出300dpi PNG、SVG/PDF、数据 CSV 与输入 review/源码/版本 SHA metadata。默认要求三档完整；`--scales c15`、`--scales c15 c30` 可显式生成标明范围的阶段图。v1 资源归档 (local run artifact: `results/review/all_scales_v1_clip1/resource_accounting/formal_resource_accounting.json`) 保留了资源覆盖范围说明。

## C45 v1 异常分析

C45 现有日志共676条，其中548条裁剪前梯度范数大于1，中位302.97、p90约33,706、最大2.7995e9；C15/C30全部已记录范数小于1。审查中的训练末段 loss 约为 C30 的13倍；同一目标输入流程下，C45 微调末段 loss 约为 C30 微调的7.8倍。这是明显的优化异常证据，但触发层或样本尚未定位。

MugRack 专家关节目标在 C15/C30/C45 输出 box 中均100%覆盖，Seen 成功数却为20/40、21/40、0/40，因此不能仅用“专家目标超出限幅范围”解释该任务的 C45 下滑。全部9,000条 C45 episode、3,039,478帧数值数据已静态核对，未发现非有限标签或列序/单位异常；源 EMA 权重读取、实际学习率、optimizer step预算和四个归一化 buffer 也一致。这不等于证明所有接口完全无损。

clip 系数是 AdamW 之前的梯度缩放系数，不等于参数更新按相同比例缩小，不能据此说“参数被冻结”。min/max、EMA、input perturbation 是明确的 v2 修正，但尚未证明解决 C45。新增分模块梯度/裁剪倍率属于主训练日志，不是额外消融。完整依据见 C45 异常分析 (local run artifact: `results/review/bpp_c45_failure_analysis_v1.json`)。

进一步说明见 [C45 详细分析](metrics/v1_clip1/c45_explanation/analysis_CN.md) 与 [训练梯度及同任务结果图](metrics/v1_clip1/c45_explanation/c45_v1_training_and_seen.png)。图表仅整理现有 v1 日志和成绩，没有追加模型实验。

## 与官方测评的对应及局限

任务、监督与仿真接口参考 TIG 已发布基线，但各方法配方不同。Zero-WAM 使用整 episode/chunk、相对关节、自己的视频编码器和 batch4；本 BPP 使用滑动 query window、绝对关节和 batch64。更新数相同不能解释为样本、epoch 或 FLOPs 相同。论文汇总表对三档取均值；上表提供这种算术均值，但它不使本实现或当前发布的 Zero-WAM 配方成为论文原始数值表的精确复现，也不据此宣称模型排名。

每种条件只有一个训练 seed，只有5个固定任务簇；等级相关、跨策略 case 配对、Scratch 跨档复用。成功测量的是整体适配系统，不能单独识别人类域学习、接口转换损失或完整双臂协作能力。已有 C15 小样本提示控制支持该 Seen 队列上的提示依赖，不能证明未见人类泛化或完整时序模仿。归一化范围问题支持另立 v2 重训，但其收益仍需实际测量。

源码证据与范围见 官方协议对应审查 (local run artifact: `results/review/bpp_official_protocol_alignment_audit.json`)、适配有效性审查 (local run artifact: `results/review/bpp_adaptation_validity_audit.json`)、C15/C30 能力审查 (local run artifact: `results/review/bpp_capability_claims_audit.json`)。历史提示检查见 [PROMPT_DIAGNOSTIC.md](PROMPT_DIAGNOSTIC.md)，不进入正式 SR；当前训练/评测安排不追加诊断矩阵。

## 公开仓库内容

本仓库包含适配源码和旧版 v1 轻量指标，不包含数据、仿真资产、缓存、环境或训练 checkpoint。v1 的冻结 Python 源码保留在 `legacy/v1_clip1`。文档中的完整本机审计属于运行产物；公开轻量摘要见 [metrics/v1_clip1](metrics/v1_clip1/README.md)。

先将 `storage.env.example` 复制到运行目录并设置 `BPP_ROOT`。含 `--reuse-prepared-inputs-from` 的重启命令要求本地已有通过核验的准备结果；新克隆并不自带这个目录，需先按入口准备固定版本的 TIG/BPP 依赖、数据和特征清单。

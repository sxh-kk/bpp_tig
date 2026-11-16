# Training health gate / 训练健康门槛

**Completed: all four scoped stages admitted; guardian ended normally. Deployment verified: 2026-11-16 16:04 Asia/Shanghai.** The live C30 producer was retained, all four frozen training-source hashes matched, and the guardian heartbeat/worker identity were verified. Deployment evidence remains in `results/preflight/bpp_training_health_activation.json`.

**本轮已结束：四个受控阶段均通过准入，guardian正常结束。部署核验时间：北京时间2026-11-16 16:04。** 当前C30进程保持原样，四份冻结训练源码hash一致，已核验守护进程心跳和worker身份。部署证据保存在 `results/preflight/bpp_training_health_activation.json`。

## Scope and unchanged recipe / 范围与既定配方

The gate covers only these stages under `results/pipeline/bpp_human_minmax_v2`:

门槛只覆盖 `results/pipeline/bpp_human_minmax_v2` 下的以下阶段：

| Stage / 阶段 | Budget / 预算 |
|---|---:|
| `training/c30` | 90,000 updates |
| `training/c30_finetune` | 3,000 updates |
| `training/c45` | 135,000 updates |
| `training/c45_finetune` | 3,000 updates |

Continue the accepted training recipe: train-only joint14 min/max, gripper identity, source epsilon perturbation `0.1`, EMA, the existing query-state quantile statistics and DDIM clip1. The guard does not change the learning rate, optimizer, batch size, update budget, normalization, EMA schedule, sampler or model architecture. It adds no forward/backward pass and no model experiment. No additional tests, training comparisons, ablations or evaluations are requested by this change.

继续已接受的训练配方：joint14 使用训练集 min/max、夹爪 identity、源 epsilon perturbation `0.1`、EMA，并保留既有 query-state 分位数统计与 DDIM clip1。守护程序不改学习率、优化器、batch、更新预算、归一化、EMA 调度、采样器或模型架构，不增加 forward/backward 或模型实验。本次不增加测试、训练对照、消融或评测。

The policy is independent of held-out success rate. It neither selects a checkpoint by SR nor treats a health pass as evidence of task capability. C15 and shared Scratch had already completed before deployment; historical measurements are described in HISTORY.md.

策略不使用 held-out 成功率，不按 SR 选 checkpoint；健康门槛通过也不等于具备任务能力。C15与共享Scratch在部署前已完成；历史测评统一见HISTORY.md。

## Signals and stopping rules / 观测量与停止条件

Use the existing rank0 `training.jsonl` records. `grad_norm_rank0` is the total gradient norm **before** clipping, returned by `clip_grad_norm_`; it is not the norm after the configured clip1. `loss` is the existing logged training loss. Per-module preclip norms and the clip coefficient are retained for diagnosis, but they are not additional decision thresholds.

读取现有 rank0 `training.jsonl`。`grad_norm_rank0` 是 `clip_grad_norm_` 返回的**裁剪前**总梯度范数，不是 clip1 后范数。`loss` 使用既有训练日志值。分模块裁剪前范数与裁剪系数保留用于诊断，不另设停止阈值。

A window contains 10 logged observations at the existing 200-update cadence and must span at least 1,800 updates, measured as its last step minus its first step. An incomplete window does not satisfy a persistence rule. The two windows in rule B are adjacent and do not overlap; each must span at least 1,800 updates.

一个窗口包含既有每200更新记录的10条观测，且末步减首步至少为1,800更新。窗口不足不能满足持续性规则。规则 B 的前后窗口相邻且不重叠，各自都须覆盖至少1,800更新。

| Rule / 规则 | Stop condition / 停止条件 |
|---|---|
| A: sustained extreme norm / 持续极端范数 | At least 8 of the current 10 records have `grad_norm_rank0 >= 100` / 当前10条中至少8条范数 `>= 100` |
| B: sustained high norm and stalled loss / 持续高范数且 loss 停滞 | At least 8 of the current 10 records have norm `>= 10`, **and** current-window median loss `>= 0.98 ×` previous-window median loss / 当前10条中至少8条范数 `>= 10`，**且**当前窗口 loss 中位数 `>= 0.98 ×` 前窗口中位数 |

Rule A does not require a loss plateau. Rule B requires both persistent high norms and no more than a 2% reduction in the median loss, including unchanged or increasing loss. The existing trainer's NaN/Inf checks remain in force. An isolated finite spike, including a single norm above 1, does not trigger either persistence rule.

规则 A 不要求 loss 停滞。规则 B 同时要求持续高范数和 loss 中位数下降不超过2%，也包括不变或上升。训练器已有 NaN/Inf 检查继续生效。单次有限尖峰，包括单次 norm 大于1，不触发上述持续性规则。

These are engineering stop thresholds, not a calibrated detector of every optimization failure. With clip1, preclip norms of 10 and 100 imply clip coefficients of approximately 0.1 and 0.01. The 8/10 requirement and update span distinguish persistence from one logged spike. They do not imply that AdamW parameter updates shrink by the same factors. The background and historical optimization diagnostics are consolidated in HISTORY.md; the gate does not identify a root cause. The thresholds are not tuned against SR.

这是工程停止阈值，不是能检出全部优化失败的已校准检测器。clip1 下，裁剪前范数10与100对应约0.1与0.01的裁剪系数；8/10及更新跨度要求用于区分持续异常与单次记录尖峰。不能据此声称 AdamW 参数更新按相同比例缩小。门槛背景与历史优化诊断统一见HISTORY.md；门槛本身不定位根因。阈值不按 SR 调整。

Sparse 200-update logs can miss events between records. A pass means only that the available log did not meet the specified rejection conditions and passed admission checks. It does not certify stability or explain why a trigger occurred.

每200更新的稀疏日志可能遗漏记录间的事件。通过只表示现有日志未满足指定拒绝条件且满足验收要求，不保证稳定，也不解释触发根因。

Windows are stage-local. In an uninterrupted 3,000-update fine-tuning stage, the first-step and 200-update logging schedule yields only 16 records. Rule B needs 20 records and cannot become eligible within that budget; rule A can still apply. This limitation does not authorize extending the budget, increasing logging frequency or borrowing observations from another stage.

窗口只使用本阶段观测。未中断的3,000更新微调按首步及每200更新记录，共只有16条；规则 B 需要20条，无法在该预算内满足资格，规则 A 仍可生效。不能因此增加预算、修改日志频率或借用其他阶段的观测。

## Runtime stop and checkpoint handling / 运行时停止与权重处理

The watcher polls every 5 seconds. On a trigger it writes a sticky stage stop marker and sends `SIGTERM` to the **verified rank0 training worker**. PID identity, process command and stage output must match before any signal. It does not initially terminate the torchrun launcher or pipeline controller, and does not use `killpg`.

守护程序每5秒轮询。触发后先写阶段的持久停止标记，再向**身份已验证的 rank0 训练 worker** 发送 `SIGTERM`；发送前必须确认 PID 身份、进程命令与阶段输出目录匹配。不在初次停止时终止 torchrun launcher 或流水线 controller，不使用 `killpg`。

The existing trainer propagates the stop request to all eight ranks with `all_reduce`, then saves `latest.pt` and `interrupted.pt` through its existing coordinated save path. That path preserves model/raw parameters, optimizer, scheduler, EMA, per-rank RNG and training position. This is a graceful-stop attempt, not a guarantee if the worker or distributed communication fails. If the verified launcher has still not exited after 300 seconds, the guardian may send it `SIGTERM` as a fallback; Elastic termination can prevent an in-progress save from completing. Preserve original logs and any prior checkpoints in either case, and verify the actual artifacts and process exits.

既有训练器通过 `all_reduce` 将停止请求传播到全部8个 rank，再沿原有协调保存路径写出 `latest.pt` 与 `interrupted.pt`，保留 raw 参数、optimizer、scheduler、EMA、逐 rank RNG 和训练位置。这是优雅停止尝试；若 worker 或分布式通信故障，不能保证保存成功。300秒后经验证的 launcher 仍未退出时，守护程序可对其发送 `SIGTERM` 兜底；Elastic 终止可能使正在进行的保存未完成。两种情况下都须保留原日志和此前 checkpoint，并核验实际产物及进程退出情况。

A sticky `training_health_stop.json` continues to reject admission even if later observations appear normal or a `final.pt` exists. Do not automatically remove the marker, restart a stopped stage or change the recipe. Stop the pipeline and review the evidence. An interrupted checkpoint is a recovery/diagnostic artifact; it is not the completed fixed-budget model and must not enter P+FT or formal evaluation as such.

`training_health_stop.json` 为持久标记；即使后续观测恢复或出现 `final.pt`，仍拒绝放行。不自动删除标记、重启异常阶段或修改配方；停止流水线，等待证据 review。中断权重是恢复/诊断产物，不能冒充预定预算完成的模型进入 P+FT 或正式评测。

## Completion admission / 完成验收

Before a covered stage becomes a valid P+FT initialization or formal evaluation checkpoint, apply the same policy to its full training log, not only the most recent window. Validate the final logged step, expected sampling coverage, fixed update budget, completion/checkpoint identity and clean worker exit through the existing pipeline checks. A heartbeat alone is not admission evidence. A historical triggering window or sticky stop marker blocks downstream work even if the last window looks healthy. Missing or invalid evidence must be reported as an admission failure rather than silently accepted.

覆盖阶段在成为有效 P+FT 初始化或正式评测权重前，须将同一策略应用到完整训练日志，不能只看末窗口。沿既有流水线检查末条日志步数、应有采样覆盖、固定更新预算、completion/checkpoint 身份与 worker 正常退出；heartbeat 本身不足以验收。即使末窗口健康，历史触发窗口或持久停止标记仍阻止下游工作。缺失或无效的证据应明确拒绝验收，不能默许放行。

Final admission binds SHA256 identities for the complete `training.jsonl`, `training_manifest.json`, `completion.json` and `process_status.json`, together with the policy identity. It checks that these originals remain unchanged during admission, and rejects an incomplete trailing log line, missing scheduled records, wrong output/regime, incomplete formal budget or nonzero worker exit. The pipeline records a successful final admission in stage-local `training_health_admission.json`, or a rejection in `training_health_rejection.json`. Stored status or admission records never replace a fresh check of the originals.

最终验收将完整 `training.jsonl`、`training_manifest.json`、`completion.json`、`process_status.json` 的 SHA256 与策略身份绑定，并确认验收期间原件没有变化。日志尾行未完整写入、应有记录缺失、输出目录/训练口径不符、正式预算未完成或 worker 非零退出都应拒绝。流水线把最终通过记录写入阶段内 `training_health_admission.json`，拒绝证据写入 `training_health_rejection.json`；既存状态或验收记录不能替代重新核验原件。

## Commands and status artifacts / 命令与状态产物

Use Python >=3.10 on Linux with both `os.pidfd_open` and `signal.pidfd_send_signal`. On this host the project Python builds lack the former, so the CPU-only guardian uses `/usr/bin/python3`; training continues in its original environment.

需要具备上述两个pidfd接口的Linux Python>=3.10。本机项目Python缺少 `os.pidfd_open`，因此仅CPU守护进程使用 `/usr/bin/python3`；训练仍使用原环境。

The following CLI is the deployed integration interface. Start commands are for the guardian only and do not launch additional model experiments.

以下 CLI 为已部署的集成接口。启动命令仅部署运行门槛与守护程序，不启动额外模型实验。

Run `arm` before `watch`. It creates `training_health_policy.json` and the independent `training_health_required.json` sentinel at the run root. The sentinel pins `policy_sha256`, the four existing training-source hashes and monitoring-source hashes. If the required sentinel exists, a missing, invalid or identity-mismatched policy rejects admission instead of silently disabling the gate. Arming also refuses a source-hash mismatch against the in-flight C30 training manifest; it never rewrites that manifest.

先执行 `arm`，再执行 `watch`。`arm` 在运行根目录创建 `training_health_policy.json` 与独立 `training_health_required.json` 哨兵；哨兵固定 `policy_sha256`、四份既有训练源码 hash 和监控源码 hash。required 哨兵存在时，策略缺失、无效或身份不匹配均拒绝验收，不会静默关闭门槛。启用时若源码 hash 与在跑 C30 的训练 manifest 不符，也会拒绝；不会改写该 manifest。

```bash
source $BPP_ROOT/storage.env
cd "$BPP_ROOT/repos/The-Imitator-Game"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BPP_HEALTH_PYTHON=/usr/bin/python3
BPP_RUN_ROOT="$BPP_ROOT/results/pipeline/bpp_human_minmax_v2"

# Pin the policy and original-source provenance first / 先固定策略与原源码来源。
"$BPP_HEALTH_PYTHON" -m examples.baselines.bpp.bpp.training_health_guard arm \
  --run-root "$BPP_RUN_ROOT"

# Run in the project's existing supervised/background execution context.
# 在项目既有的监督/后台执行环境中运行。
"$BPP_HEALTH_PYTHON" -m examples.baselines.bpp.bpp.training_health_guard watch \
  --run-root "$BPP_RUN_ROOT" --poll-seconds 5
```

```bash
# Inspect guardian status / 查看守护状态。
"$BPP_HEALTH_PYTHON" -m examples.baselines.bpp.bpp.training_health_guard status \
  --run-root "$BPP_RUN_ROOT"

# Assess an active stage's available log; completion is not required.
# 评估在跑阶段的已有日志；不要求阶段完成。
"$BPP_HEALTH_PYTHON" -m examples.baselines.bpp.bpp.training_health assess \
  --output "$BPP_RUN_ROOT/training/c30"

# Check final admission using the original full-log evidence.
# 用完整原始证据执行最终验收。
"$BPP_HEALTH_PYTHON" -m examples.baselines.bpp.bpp.training_health check \
  --output "$BPP_RUN_ROOT/training/c30"
```

`assess` and `check` are read-only CLI commands and print JSON; the pipeline/guardian writes the durable admission records. An active-stage assessment does not grant final admission. `not_scoped` is not a health pass; an unarmed run with neither a policy nor a required sentinel has no policy to enforce.

`assess` 与 `check` CLI 只读并输出 JSON，持久验收记录由流水线/守护程序写出。在跑阶段评估不等于最终放行。`not_scoped` 不代表健康通过；未启用且没有策略或 required 哨兵的运行没有可执行策略。

| Location / 位置 | Artifact / 产物 | Purpose / 用途 |
|---|---|---|
| Run root / 运行根目录 | `training_health_policy.json` | Versioned policy / 策略版本 |
| Run root / 运行根目录 | `training_health_required.json` | Required-policy sentinel and pinned source provenance / 必需策略哨兵与固定源码来源 |
| Run root / 运行根目录 | `training_health_guard_status.json` | Guardian state / 守护状态 |
| Run root / 运行根目录 | `training_health_guard_events.jsonl` | Guardian events and stop attempts / 事件与停止尝试 |
| Run root / 运行根目录 | `training_health_guard.lock` | Single-guardian coordination / 单守护进程协调 |
| Each covered stage / 每个覆盖阶段 | `training_health_status.json` | Stage observations and decision / 阶段观测与判定 |
| Each triggered stage / 每个触发阶段 | `training_health_stop.json` | Sticky rejection evidence / 持久拒绝证据 |
| Each admitted final stage / 每个最终通过阶段 | `training_health_admission.json` | Final pass bound to original artifact hashes / 与原件 hash 绑定的最终通过证据 |
| Each rejected stage / 每个验收拒绝阶段 | `training_health_rejection.json` | Pipeline rejection evidence / 流水线拒绝证据 |

Read these artifacts together with `training.jsonl`, `progress.json`, `completion.json`, `process_status.json`, the training manifest and actual checkpoint metadata. Retain every failed attempt and error; a normal-looking metric record never overrides abnormal worker exit.

上述文件须与 `training.jsonl`、`progress.json`、`completion.json`、`process_status.json`、训练 manifest 及实际 checkpoint 元数据共同审查。保留所有失败尝试与错误；看似正常的指标记录不能覆盖 worker 异常退出。

## Separate provenance / 独立来源记录

The health policy and watcher are standard-library sidecars, in `bpp/training_health.py` and `bpp/training_health_guard.py`. Preserve the running trainer's source hashes and training experiment identity. Do not rewrite its existing manifest to claim that a later-added guardian was part of the original numerical recipe. Record the policy version/source hashes and deployment evidence separately from trainer/model/data/cache provenance. The main-agent review must confirm the actual source hashes, policy/status artifacts, PID checks, stop behavior and admission integration before this document is marked deployed.

健康策略与守护程序为只使用标准库的独立模块，位于 `bpp/training_health.py` 与 `bpp/training_health_guard.py`。保留在跑训练器的源码 hash 和实验身份；不改写既有 manifest，不能把后来加入的守护程序写成原数值配方的一部分。策略版本/源码 hash 与部署证据应独立于 trainer/model/data/cache 的来源记录保存。本文标记为已部署前，主代理必须核验实际源码 hash、策略/状态产物、PID 校验、停止行为与验收接入。

## Completed run / 本轮已完成

The run finished measurements at 2026-11-16 22:34 Asia/Shanghai. C30/C45 and both target fine-tuning stages passed complete-log admission. The guardian ended with `pipeline_finished`, which is a normal terminal state. A temporary supervisor interruption was recovered without restarting C45; `orphan_exit_observer.py` recorded its actual supervisor reap exit status 0. See `FINAL_REVIEW_CN.md` and the final audit for the preserved evidence. Long-running supervisors should run in an independent session or managed service, with persistent log files.

本轮于北京时间2026-11-16 22:34完成测量，四个受控训练阶段均通过全日志准入。guardian 的 `pipeline_finished` 是正常结束。监督进程曾中断，恢复时没有重启C45；独立退出观察器记录到原训练的真实退出码0。完整证据和解释边界见 `FINAL_REVIEW_CN.md`。长时监督进程应使用独立会话或托管服务，并将日志写入持久目录。

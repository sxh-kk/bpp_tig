"""Independent case-level review of all three BPP/TIG scales.

The default report remains useful while jobs run and imports no Torch. A complete
review additionally requires --verify-checkpoints; a green pipeline state alone
cannot establish success rates or checkpoint lineage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

SCALES = ("c15", "c30", "c45")
REGIMES = ("seen", "zero_shot", "scratch", "pretrain_finetune")
BUDGETS = {"c15": 45000, "c30": 90000, "c45": 135000, "scratch": 3000, "finetune": 3000}
REVISION = "9a57b27edcb54823d8da7d2b77efa61c092ffa95"
ROOT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))


V2_ACTION_NORMALIZATION = "train_minmax_joint14_gripper_identity_v2"
LEGACY_ACTION_NORMALIZATION = "legacy_quantile"
V2_RECIPE = {"action_normalization": V2_ACTION_NORMALIZATION,
             "training_recipe": "bpp_source_epsilon_perturbation_ema_v2",
             "input_perturbation": 0.1, "use_ema": True,
             "evaluation_weights": "ema", "resume_weights": "raw",
             "ema_config": {"decay": 0.9999, "min_decay": 0.0,
                            "update_after_step": 0, "use_ema_warmup": False,
                            "inv_gamma": 1.0, "power": 0.75, "foreach": False}}
V2_SAMPLING = {"scheduler_class": "DDIMScheduler", "clip_sample": True,
               "clip_sample_range": 1.0, "thresholding": False,
               "num_inference_steps": 16, "num_train_timesteps": 50,
               "prediction_type": "epsilon"}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metric_rows(rows, expected_count):
    complete = (len(rows) == expected_count and all(
        row.get("status") == "completed" and row.get("process_exit_verified") is True
        and row.get("process_returncode") == 0 and row.get("is_pilot") is False
        and type(row.get("success_at_end")) is bool and type(row.get("success_once")) is bool
        and isinstance(row.get("last_normalized_reward"), (int, float))
        and math.isfinite(row["last_normalized_reward"]) for row in rows))
    return {"planned_episodes": expected_count, "observed_episodes": len(rows), "valid": complete,
            "success_at_end_rate": sum(row["success_at_end"] for row in rows) / expected_count if complete else None,
            "success_once_rate": sum(row["success_once"] for row in rows) / expected_count if complete else None,
            "SubSR_released_code_proxy": sum(row["last_normalized_reward"] for row in rows) / expected_count if complete else None,
            "mean_wall_seconds": sum(row.get("wall_seconds", 0) for row in rows) / expected_count if complete else None,
            "mean_clipped_control_steps": sum(row.get("clipped_control_steps", 0) for row in rows) / expected_count if complete else None}


def expected_tasks(tig_repo, regime):
    filename = "seen_5tasks_env_list.txt" if regime == "seen" else "unseen_5tasks_env_list.txt"
    path = Path(tig_repo) / "examples/baselines/lerobot_dataset/eval/exp_list" / filename
    tasks = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(tasks) != 20 or len(set(tasks)) != 20 or any(not re.fullmatch(r"L[0-3]_.+", task) for task in tasks):
        raise ValueError("Official task list must contain 20 unique L0-L3 variants")
    return tasks


def inspect_evaluation(directory, scale, regime, tig_repo, *, source_root=None, action_normalization=None):
    directory = Path(directory)
    result = {"scale": scale, "regime": regime, "directory": str(directory), "status": "pending", "issues": [], **metric_rows([], 200)}
    manifest = read(directory / "evaluation_manifest.json")
    if not manifest:
        return result, []
    issues = result["issues"]
    if action_normalization == V2_ACTION_NORMALIZATION:
        for key, expected in V2_RECIPE.items():
            if manifest.get("training_metadata", {}).get(key) != expected:
                issues.append(f"evaluation training recipe differs: {key}")
        if manifest.get("action_normalization") != V2_ACTION_NORMALIZATION:
            issues.append("actual inference action normalization differs")
        if manifest.get("loaded_weight_selection") != "ema":
            issues.append("actual EMA weight loading is unverified")
        if manifest.get("loaded_ema_optimization_step") != manifest.get("training_metadata", {}).get("train_step"):
            issues.append("actual loaded EMA step differs")
        for key, expected in V2_SAMPLING.items():
            if manifest.get("inference_sampling", {}).get(key) != expected:
                issues.append(f"actual inference sampler differs: {key}")
        sampling = manifest.get("inference_sampling", {})
        saved = sampling.get("checkpoint_scheduler_config", {})
        resolved = sampling.get("resolved_scheduler_config", {})
        if not saved:
            issues.append("saved checkpoint sampler configuration is missing")
        for key, expected in saved.items():
            if resolved.get(key) != expected:
                issues.append(f"sampler differs from checkpoint configuration: {key}")
    elif action_normalization == LEGACY_ACTION_NORMALIZATION and manifest.get("action_normalization", LEGACY_ACTION_NORMALIZATION) != LEGACY_ACTION_NORMALIZATION:
        issues.append("historical review requested for a non-legacy evaluation")
    experiment = manifest.get("experiment_identity")
    if identity({key: value for key, value in manifest.items() if key != "experiment_identity"}) != experiment:
        issues.append("evaluation manifest identity is invalid")
    tasks = expected_tasks(tig_repo, regime)
    required = {"regime": regime, "tasks": tasks, "seeds": list(range(100000, 100010)),
                "reference_source": "human", "training_prompt_source": "human", "is_pilot": False,
                "control_mode": "pd_joint_pos", "action_dimension": 16, "max_episode_steps": 500,
                "execution_horizon": 12, "shader": "rt-fast", "dataset_revision": REVISION,
                "prompt_modalities": ["RGB"], "prompt_frames": 16}
    for key, value in required.items():
        if manifest.get(key) != value:
            issues.append(f"evaluation setting differs: {key}")
    if manifest.get("pretraining_scale") != (None if regime == "scratch" else scale):
        issues.append("evaluation pretraining scale differs")
    source_root = Path(source_root or Path(__file__).parent)
    for filename, sha in manifest.get("source_sha256", {}).items():
        path = source_root / filename
        if not path.is_file() or digest(path) != sha:
            issues.append(f"evaluation implementation changed: {filename}")
    if set(manifest.get("source_sha256", {})) != {"native_eval.py", "native_model.py", "data.py", "training_data.py"}:
        issues.append("evaluation implementation provenance is incomplete")
    mapping = read(Path(tig_repo) / "examples/baselines/lerobot_dataset/task_mapping.json")
    human = {task: row["human_task_id"] for row in mapping["task_mappings"] for task in row["sim_task_id"]}
    cases = [(task, episode) for task in tasks for episode in range(10)]
    claim_files = sorted(directory.glob("shard_*_of_*.json"))
    shard_counts = set()
    covered = set()
    for path in claim_files:
        match = re.fullmatch(r"shard_(\d+)_of_(\d+)\.json", path.name)
        if match is None:
            continue
        index, count = map(int, match.groups())
        shard_counts.add(count)
        claim = read(path)
        process = read(directory / ("process_status_" + path.name), {})
        expected = cases[index::count] if count > 0 and 0 <= index < count else []
        actual = [tuple(case) for case in claim.get("cases", [])]
        if actual != expected or covered.intersection(actual) or claim.get("experiment_identity") != experiment:
            issues.append(f"worker coverage differs: {path.name}")
        covered.update(actual)
        if process.get("status") != "completed" or process.get("returncode") != 0 or process.get("owns_evaluation_attempt") is not True or process.get("attempt_id") != claim.get("attempt_id") or process.get("experiment_identity") != experiment:
            issues.append(f"worker exit unverified: {path.name}")
    if len(shard_counts) != 1 or len(claim_files) != next(iter(shard_counts), -1) or covered != set(cases):
        issues.append("complete unique shard coverage has not been verified")
    rows = []
    for task, episode in cases:
        path = directory / "human" / task / f"seed_{100000+episode}" / "metrics.json"
        row = read(path)
        if row is None:
            continue
        expected_case = {"experiment_identity": experiment, "task_id": task, "seed": 100000+episode,
                         "reference": row.get("reference", {}), "reference_source": "human"}
        if row.get("case_id") != identity(expected_case) or row.get("reference_source") != "human" or row.get("max_episode_steps") != 500:
            issues.append(f"case hash or source/horizon differs: {task}/{episode}")
        if row.get("experiment_identity") != experiment or row.get("task_id") != task or row.get("seed") != 100000+episode:
            issues.append(f"case identity differs: {task}/{episode}")
        reference = row.get("reference", {})
        if reference.get("dataset_id") != human.get(task) or reference.get("episode_index") != episode or reference.get("condition") != "human_video_only" or reference.get("prompt_modalities") != ["RGB"] or reference.get("prompt_frames") != 16 or reference.get("dataset_revision") != REVISION:
            issues.append(f"human reference differs: {task}/{episode}")
        level = int(task[1])
        flags = {"L1": level == 1, "L2": level == 2, "L3": level == 3, "mirror_robot_pose": False}
        if row.get("actual_environment_flags") != flags:
            issues.append(f"environment level flags differ: {task}/{episode}")
        if row.get("executed_control_steps") != 500:
            issues.append(f"case did not execute 500 controls: {task}/{episode}")
        rows.append(row)
    metrics = metric_rows(rows, 200)
    if not metrics["valid"]:
        issues.append("200 completed, finite, non-pilot cases with clean worker exits are required")
    summary = read(directory / "summary.json", {})
    for key, value in {"experiment_identity": experiment, "planned_episode_count": 200, "completed_episode_count": 200,
                       "error_episode_count": 0, "is_pilot": False, "incomplete_or_errors": False, "process_exit_pending": False}.items():
        if summary.get(key) != value:
            issues.append(f"saved summary provenance differs: {key}")
    for key in ("success_at_end_rate", "success_once_rate"):
        if summary.get(key) != metrics[key]:
            issues.append(f"saved summary differs from cases: {key}")
    result.update(metrics, status="complete" if not issues else "incomplete_or_invalid",
                  experiment_identity=experiment, checkpoint=manifest.get("checkpoint"),
                  checkpoint_sha256=manifest.get("checkpoint_sha256"), shard_count=next(iter(shard_counts), None))
    if issues:
        result["valid"] = False
        for key in ("success_at_end_rate", "success_once_rate", "SubSR_released_code_proxy"):
            result[key] = None
    return result, rows


def inspect_training(output, expected_regime, scale, *, verify_checkpoint=False, expected_external=None, expected_head=None, source_root=None, action_normalization=None):
    output = Path(output)
    result = {"directory": str(output), "regime": expected_regime, "scale": scale, "status": "pending", "issues": []}
    manifest = read(output / "training_manifest.json")
    if not manifest:
        return result
    issues = result["issues"]
    expected_scale = None if expected_regime == "scratch" else scale
    expected = {"regime": expected_regime, "pretraining_scale": expected_scale, "pilot": False, "prompt_source": "human",
                "world_size": 8, "global_batch": 64, "max_updates": BUDGETS[expected_regime], "fixed_vision_preprocessing": True, "seed": 1, "init_seed": 1}
    if action_normalization == V2_ACTION_NORMALIZATION:
        expected.update(V2_RECIPE)
        if expected_regime == "finetune":
            expected["initialize_from_weights"] = "ema"
    elif action_normalization == LEGACY_ACTION_NORMALIZATION and manifest.get("action_normalization", LEGACY_ACTION_NORMALIZATION) != LEGACY_ACTION_NORMALIZATION:
        issues.append("historical review requested for a non-legacy training run")
    unsigned = {key: value for key, value in manifest.items() if key != "experiment_identity"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != manifest.get("experiment_identity"):
        issues.append("training manifest identity is invalid")
    for key, value in expected.items():
        if manifest.get(key) != value:
            issues.append(f"training setting differs: {key}")
    if expected_external and manifest.get("external_checkpoint_sha256") != expected_external:
        issues.append("external initialization differs between stages")
    data_path = Path(manifest.get("data_manifest", ""))
    feature_path = Path(manifest.get("feature_manifest", ""))
    data = read(data_path, {})
    expected_split = "scratch" if expected_regime in ("scratch", "finetune") else scale
    counts = {"c15": (3000, 750), "c30": (6000, 1500), "c45": (9000, 2250), "scratch": (200, 50)}
    query_count, prompt_count = counts[expected_split]
    if data.get("request", {}).get("split") != expected_split or data.get("request", {}).get("prompt_source") != "human" or data.get("query_episode_count") != query_count or data.get("prompt_episode_count") != prompt_count:
        issues.append("training data split or episode budget differs")
    for path, key in ((data_path, "data_manifest_sha256"), (feature_path, "feature_manifest_sha256")):
        if not path.is_file() or digest(path) != manifest.get(key):
            issues.append(f"training input manifest changed: {key}")
    for filename, key in (("train.py", "trainer_sha256"), ("native_model.py", "native_model_sha256"), ("training_data.py", "training_data_sha256"), ("feature_cache.py", "feature_cache_source_sha256")):
        source_path = Path(source_root or Path(__file__).parent) / filename
        if digest(source_path) != manifest.get(key):
            issues.append(f"training implementation changed: {filename}")
    if expected_regime != "finetune" and (manifest.get("initialize_from") is not None or manifest.get("initialize_from_metadata") is not None):
        issues.append("pretraining/Scratch did not use independent common external initialization")
    completion = read(output / "completion.json", {})
    process = read(output / "process_status.json", {})
    completed = completion.get("status") == "completed" and completion.get("pilot") is False and completion.get("completed_updates") == BUDGETS[expected_regime] and completion.get("target_updates") == BUDGETS[expected_regime] and completion.get("experiment_identity") == manifest.get("experiment_identity")
    clean_exit = process.get("status") == "completed" and process.get("returncode", process.get("exit_code")) == 0
    if not completed or not clean_exit:
        result.update(status="running_or_incomplete", progress=read(output / "progress.json"), process=process)
        return result
    final = output / "final.pt"
    if not final.is_file():
        issues.append("final checkpoint is missing")
    initialization = read(output / "model_initialization.json", {})
    if verify_checkpoint and final.is_file():
        import torch
        payload = torch.load(final, map_location="cpu", weights_only=False, mmap=True)
        metadata = payload.get("metadata", {})
        for key in (*expected, "experiment_identity", "external_checkpoint_sha256", "initialize_from_sha256"):
            if metadata.get(key) != manifest.get(key):
                issues.append(f"final checkpoint differs: {key}")
        if metadata.get("train_step") != BUDGETS[expected_regime] or metadata.get("completed") is not True:
            issues.append("final checkpoint budget is incomplete")
        head = initialization.get("init_head_sha256")
        if not head or payload.get("init_head_sha256") != head or metadata.get("init_head_sha256") != head or (expected_head and head != expected_head):
            issues.append("new-head initialization differs from its checkpoint/common BPP baseline")
        if payload.get("normalizer_stats", {}).get("source", {}).get("manifest_hash") != data.get("manifest_identity"):
            issues.append("checkpoint normalizer does not identify its training manifest")
        if action_normalization == V2_ACTION_NORMALIZATION:
            issues.extend(v2_checkpoint_issues(payload, metadata, data))
        result["checkpoint_sha256"] = digest(final)
    observed_wall = None
    try:
        observed_wall = (datetime.fromisoformat(process["finished_utc"]) - datetime.fromisoformat(process["started_utc"])).total_seconds()
    except (KeyError, ValueError):
        issues.append("training process timestamps are missing")
    result.update(status="complete" if not issues and verify_checkpoint else "checkpoint_review_pending" if not issues else "invalid",
                  checkpoint=str(final), completed_updates=completion.get("completed_updates"),
                  global_batch=manifest.get("global_batch"), query_windows_seen=completion.get("completed_updates", 0) * manifest.get("global_batch", 0),
                  optimizer_loop_wall_seconds=completion.get("wall_seconds"), process_observed_wall_seconds=observed_wall,
                  gpu_hours=observed_wall * manifest.get("world_size", 0) / 3600 if observed_wall is not None else None,
                  gpu_hours_scope="8-GPU reservation over supervised training process wall time; feature-cache preparation and evaluation excluded",
                  experiment_identity=manifest.get("experiment_identity"),
                  external_checkpoint_sha256=manifest.get("external_checkpoint_sha256"),
                  initialize_from=manifest.get("initialize_from"), initialize_from_sha256=manifest.get("initialize_from_sha256"),
                  init_head_sha256=initialization.get("init_head_sha256"), trainable_parameters=initialization.get("trainable_parameters"),
                  query_anchors=data.get("length"), query_episodes=data.get("query_episode_count"), prompt_episodes=data.get("prompt_episode_count"))
    return result


def v2_checkpoint_issues(payload, metadata, data):
    """Inspect saved raw buffers and EMA headers without constructing a policy."""
    import torch
    issues = []
    for location, record in (("payload", payload), ("config", payload.get("config", {}))):
        for key, expected in V2_RECIPE.items():
            if record.get(key) != expected:
                issues.append(f"checkpoint {location} v2 recipe differs: {key}")
    if payload.get("state_dict_weight_type") != "raw":
        issues.append("checkpoint raw training weights are not identified")
    stats = payload.get("normalizer_stats", {})
    normalizer_ref = data.get("normalizer", {})
    normalizer_path = Path(normalizer_ref.get("path", ""))
    if not normalizer_path.is_file() or digest(normalizer_path) != normalizer_ref.get("sha256"):
        issues.append("original train-only normalizer statistics changed")
    elif read(normalizer_path) != stats:
        issues.append("checkpoint statistics differ from the original train-only statistics")
    try:
        lower, upper = stats["action"]["min"], stats["action"]["max"]
        if len(lower) != 16 or len(upper) != 16:
            raise ValueError("action statistics must have 16 dimensions")
        scale, offset = [], []
        for i, (lo, hi) in enumerate(zip(lower, upper)):
            if i in (7, 15):
                scale.append(1.0); offset.append(0.0)
            else:
                span = float(hi)-float(lo)
                if not math.isfinite(span) or span < 0:
                    raise ValueError("invalid joint-action range")
                a = 1.0 if span < 1e-6 else 2.0/span
                scale.append(a);offset.append(-float(lo) if span < 1e-6 else -1.0-float(lo)*a)
        raw = payload["state_dict"]
        for key, values in (("action_scale", scale), ("action_offset", offset)):
            actual = raw[key].detach().cpu()
            expected = torch.tensor(values, dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                issues.append(f"saved effective min/max or gripper identity buffer differs: {key}")
        bounds = stats.get("selected_bounds", {}).get("state")
        if bounds is None:
            mode = "q01" if stats.get("mode") == "quantile" else "min"
            high_key = "q99" if stats.get("mode") == "quantile" else "max"
            bounds = {"min": stats["state"][mode], "max": stats["state"][high_key]}
        if len(bounds["min"]) != 18 or len(bounds["max"]) != 18:
            raise ValueError("state statistics must have 18 dimensions")
        state_scale, state_offset = [], []
        for lo, hi in zip(bounds["min"], bounds["max"]):
            span = float(hi)-float(lo)
            if not math.isfinite(span) or span < 0:
                raise ValueError("invalid state range")
            a = 1.0 if span < 1e-6 else 2.0/span
            state_scale.append(a);state_offset.append(-float(lo) if span < 1e-6 else -1.0-float(lo)*a)
        for key, values in (("state_scale", state_scale), ("state_offset", state_offset)):
            actual = raw[key].detach().cpu()
            expected = torch.tensor(values, dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                issues.append(f"saved state normalizer buffer differs: {key}")
        ema = payload.get("ema", {})
        names, shapes = ema.get("parameter_names", []), ema.get("parameter_shapes", [])
        state = ema.get("state_dict", {})
        shadow = state.get("shadow_params", [])
        if ema.get("schema") != "bpp_tig_ema_v2" or not names or len(names) != len(set(names)) or len(names) != len(shapes) or len(names) != len(shadow):
            raise ValueError("EMA schema or unique parameter coverage is incomplete")
        for name, shape, tensor in zip(names, shapes, shadow):
            if name not in raw or list(raw[name].shape) != list(shape) or list(tensor.shape) != list(shape):
                raise ValueError(f"EMA parameter mapping differs: {name}")
        if sum(t.numel() for t in shadow) != payload.get("parameter_audit", {}).get("unique_parameters"):
            issues.append("EMA does not cover all unique model parameters")
        if state.get("optimization_step") != metadata.get("train_step"):
            issues.append("saved EMA update count differs from actual training updates")
        for key, expected in V2_RECIPE["ema_config"].items():
            if key != "foreach" and state.get(key) != expected:
                issues.append(f"saved actual EMA schedule differs: {key}")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        issues.append(f"v2 buffer/EMA inspection failed: {exc}")
    return issues


def build_report(run_root, tig_repo, c15_output, *, verify_checkpoints=False, source_root=None, action_normalization=V2_ACTION_NORMALIZATION):
    run_root, tig_repo = Path(run_root), Path(tig_repo)
    groups, tasks_rows, levels_rows, training = [], [], [], []
    common = read(Path(c15_output) / "training_manifest.json", {}).get("external_checkpoint_sha256")
    common_head = read(Path(c15_output) / "model_initialization.json", {}).get("init_head_sha256")
    training_by_key = {}
    for key in ("c15", "c30", "c45", "scratch", "c15_finetune", "c30_finetune", "c45_finetune"):
        regime = "finetune" if key.endswith("_finetune") else key
        scale = key.split("_")[0] if key != "scratch" else None
        output = Path(c15_output) if key == "c15" else run_root / "training" / key
        row = inspect_training(output, regime, scale, verify_checkpoint=verify_checkpoints, expected_external=common, expected_head=common_head, source_root=source_root, action_normalization=action_normalization)
        training.append(row)
        training_by_key[key] = row
    for scale in SCALES:
        ft, origin = training_by_key[scale+"_finetune"], training_by_key[scale]
        if ft.get("status") == "complete" and (origin.get("status") != "complete" or ft.get("initialize_from") != origin.get("checkpoint") or ft.get("initialize_from_sha256") != origin.get("checkpoint_sha256")):
            ft["issues"].append("P+FT source checkpoint lineage differs from the verified scale checkpoint")
            ft["status"] = "invalid"
    for scale in SCALES:
        for regime in REGIMES:
            name = "shared_scratch" if regime == "scratch" else f"{scale}_{regime}"
            key = "scratch" if regime == "scratch" else scale+"_finetune" if regime == "pretrain_finetune" else scale
            group, rows = inspect_evaluation(run_root / "evaluation" / name, scale, regime, tig_repo, source_root=source_root, action_normalization=action_normalization)
            group["shared_scratch_baseline"] = regime == "scratch"
            train = training_by_key[key]
            if group["status"] == "complete" and (train["status"] != "complete" or group.get("checkpoint_sha256") != train.get("checkpoint_sha256")):
                group["issues"].append("matching final training checkpoint has not passed independent review")
                group.update(status="checkpoint_review_pending", valid=False, success_at_end_rate=None, success_once_rate=None, SubSR_released_code_proxy=None)
            groups.append(group)
            for task in sorted({row["task_id"] for row in rows}):
                metrics = metric_rows([row for row in rows if row["task_id"] == task], 10)
                if group["status"] != "complete":
                    metrics.update(valid=False, success_at_end_rate=None, success_once_rate=None, SubSR_released_code_proxy=None)
                tasks_rows.append(dict(scale=scale, regime=regime, task_id=task, **metrics))
            for level in range(4):
                metrics = metric_rows([row for row in rows if row["task_id"].startswith(f"L{level}_")], 50)
                if group["status"] != "complete":
                    metrics.update(valid=False, success_at_end_rate=None, success_once_rate=None, SubSR_released_code_proxy=None)
                levels_rows.append(dict(scale=scale, regime=regime, level=f"L{level}", **metrics))
    unique_groups = {row["directory"]: row for row in groups}
    issues = [f"{row['scale']}/{row['regime']}: {issue}" for row in groups for issue in row["issues"]]
    issues.extend(f"training {row['scale']}/{row['regime']}: {issue}" for row in training for issue in row["issues"])
    all_complete = all(row["status"] == "complete" for row in training) and all(row["status"] == "complete" for row in groups) and not issues
    return {"schema": "bpp_tig_case_level_review_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if all_complete else "incomplete", "checkpoint_verification_requested": verify_checkpoints,
            "action_normalization": action_normalization,
            "training_recipe": V2_RECIPE["training_recipe"] if action_normalization == V2_ACTION_NORMALIZATION else "legacy_v1_quantile_clip1",
            "reviewed_source_root": str(Path(source_root or Path(__file__).parent)),
            "unique_evaluation_groups": len(unique_groups), "expected_unique_rollouts": 2000,
            "verified_unique_rollouts": sum(row.get("observed_episodes", 0) for row in unique_groups.values() if row["status"] == "complete"),
            "primary_metric": "success_at_end_rate", "secondary_success_metric": "success_once_rate",
            "reward_metric": "SubSR_released_code_proxy = mean(last_normalized_reward)",
            "reward_caveat": "Released RewardTracker averages continuous phase peaks. This is not asserted equal to the paper's thresholded stage-completion definition.",
            "protocol_caveats": ["BPP human-RGB bimanual adaptation, with about 347M trainable parameters and frozen vision; not a native frozen BPP or LoRA-only measurement.",
                                 "Equal optimizer update counts do not establish equal episode epochs, sample units or FLOPs with Zero-WAM.",
                                 "Scratch is one common external-BPP initialized target-only baseline reused in all three scale tables.",
                                 "Success alone does not isolate human-video understanding and bimanual control, or establish that the policy uses its prompt."],
            "training": training, "groups": groups, "per_task_level": tasks_rows, "per_level": levels_rows,
            "training_gpu_hours": sum(row.get("gpu_hours") or 0 for row in training), "issues": issues,
            "pipeline_state": str(run_root / "pipeline_state.json"), "overall_human_review_still_required": True}


def write_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else [
        "scale", "regime", "task_id", "planned_episodes", "observed_episodes", "valid",
        "success_at_end_rate", "success_once_rate", "SubSR_released_code_proxy"]
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/pipeline/bpp_human_minmax_v2")
    parser.add_argument("--tig-repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--c15-output", type=Path, default=ROOT / "results/pipeline/bpp_human_minmax_v2/training/c15")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/review/all_scales_minmax_v2")
    parser.add_argument("--source-root", type=Path, help="Explicit archived module source root for historical reviews")
    parser.add_argument("--action-normalization", choices=(V2_ACTION_NORMALIZATION, LEGACY_ACTION_NORMALIZATION), default=V2_ACTION_NORMALIZATION)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    result = build_report(args.run_root, args.tig_repo, args.c15_output, verify_checkpoints=args.verify_checkpoints, source_root=args.source_root, action_normalization=args.action_normalization)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "review.json", result)
    for key in ("groups", "per_task_level", "per_level", "training"):
        write_csv(args.output_dir / (key+".csv"), result[key])
    print(json.dumps({"status": result["status"], "verified_unique_rollouts": result["verified_unique_rollouts"], "expected_unique_rollouts": 2000, "review": str(args.output_dir / "review.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

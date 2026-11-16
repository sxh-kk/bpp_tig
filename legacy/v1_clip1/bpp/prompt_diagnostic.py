"""Fixed C15 Seen/L0 paired human-prompt diagnostic; never formal SR or model selection.

Five official Seen tasks, reset seeds 100000/100001, and correct versus cyclically
mismatched human RGB prompts yield exactly twenty rollouts. Only the prompt changes.
Native policy, observation, controller, rollout budget and scoring are reused.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import ExitStack
from dataclasses import dataclass
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from .data import DATASET_REVISION, file_sha256, read_reference_rgb, select_reference

# Preserve the validated native-library lifetime/order BEFORE model construction
# or Torch seeding. Do not move these imports into a helper or worker function.
from examples.baselines.diffusion_policy.diffusion_policy.make_env import make_eval_envs as _official_make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper as _official_flatten_wrapper
from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import configure_dual_task_level as _official_configure_level
from . import native_eval as evaluation

SCHEMA = "bpp_tig_c15_paired_prompt_diagnostic_v1"
CONDITIONS = ("correct", "cyclic_mismatch")
SEEDS = (100000, 100001)
LIMITS = [
    "Numerical or behavioral prompt sensitivity does not prove complete imitation or intent understanding.",
    "This diagnostic cannot separately identify human-domain and bimanual-control learning difficulty.",
    "A wrong-task video can be incompatible with the unchanged scene; its performance drop has that limitation.",
    "The final fixed-budget checkpoint is used once; diagnostic results do not select weights or training settings.",
]


@dataclass(frozen=True)
class DiagnosticConfig:
    checkpoint: Path
    output_dir: Path
    dataset_root: Path = evaluation.EvalConfig.dataset_root
    tig_repo: Path = evaluation.EvalConfig.tig_repo
    bpp_repo: Path = evaluation.EvalConfig.bpp_repo
    device: str = "cuda:0"
    num_shards: int = 1
    shard_index: int = 0


def _read(path):
    return json.loads(Path(path).read_text())


def _immutable_json(path, record):
    """Concurrent identical declarations are allowed; different experiments fail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            canonical = evaluation._jsonable(record)
            if path.exists() and _read(path) != canonical:
                raise ValueError(f"Existing diagnostic declaration differs: {path}")
            evaluation._write_json(path, canonical)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def build_plan(tig_repo):
    path, full = evaluation.official_tasks(Path(tig_repo), "seen")
    tasks = tuple(task for task in full if task.startswith("L0_"))
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("Require exactly five official Seen L0 tasks")
    mismatch = {task: tasks[(index + 1) % len(tasks)] for index, task in enumerate(tasks)}
    cases = []
    for task in tasks:
        for seed in SEEDS:
            for condition in CONDITIONS:
                cases.append({"task_id": task, "seed": seed, "episode_index": seed - 100000,
                    "condition": condition, "prompt_task_id": task if condition == "correct" else mismatch[task]})
    return {"schema": SCHEMA, "diagnostic_only": True, "formal_success_rate": None,
        "checkpoint_selection": "Completed non-pilot C15 human-prompt 45000-update final checkpoint; no score-based selection",
        "official_seen_environment_list": str(path.resolve()),
        "official_seen_environment_list_sha256": file_sha256(path),
        "task_mapping_sha256": file_sha256(Path(tig_repo) / "examples/baselines/lerobot_dataset/task_mapping.json"),
        "task_order": tasks, "mismatch_mapping": mismatch, "seeds": SEEDS, "conditions": CONDITIONS,
        "max_episode_steps": 500, "execution_horizon": 12, "prediction_horizon": 16,
        "control_hz": 20, "sim_hz": 100, "prompt_source": "human RGB video only",
        "prompt_sampling": "16 uniformly sampled RGB frames, including endpoints; prompt episode=seed-100000",
        "current_observation": "ZED RGB + measured dual-arm qpos18, history2",
        "action": "physical-unit dual-arm pd_joint_pos action16, original clipping limits",
        "shader": "rt-fast", "amp_dtype": "bfloat16", "ignore_terminations": True,
        "initial_randomness": "Same reset seed and per-episode Python/NumPy/Torch seed in both conditions",
        "first_query_check": "Pair raw/normalized predicted actions only when current-observation and Torch RNG hashes match exactly",
        "formal_metric_exclusion": "All outputs remain diagnostic; they never enter Seen/Zero-shot/Scratch/P+FT main scores",
        "interpretation_limits": LIMITS, "case_count": len(cases), "cases": cases}


def shard_cases(plan, num_shards, shard_index):
    if not 1 <= num_shards <= plan["case_count"] or not 0 <= shard_index < num_shards:
        raise ValueError("Require 1..20 shards and 0 <= shard-index < num-shards")
    return [case for index, case in enumerate(plan["cases"]) if index % num_shards == shard_index]


def case_config(config, case):
    return evaluation.EvalConfig(config.checkpoint, config.output_dir / case["condition"], "seen",
        dataset_root=config.dataset_root, tig_repo=config.tig_repo, bpp_repo=config.bpp_repo,
        reference_source="human", training_prompt_source="human", pretraining_scale="c15",
        device=config.device, shader="rt-fast", amp_dtype="bfloat16", inference_process="shared",
        max_episode_steps=500, execution_horizon=12, capture_video=True, allow_pilot=True)


def case_path(config, case):
    return evaluation.case_path(case_config(config, case), case["task_id"], case["seed"])


def _case_lock_name(case):
    return f"{case['condition']}__{case['task_id']}__{case['seed']}.lock"


def _verified(record, identity):
    return (record.get("experiment_identity") == identity and record.get("status") == "completed"
        and record.get("process_exit_verified") is True and record.get("process_returncode") == 0
        and record.get("diagnostic_only") is True and record.get("executed_control_steps") == 500)


def _archive_unverified(path, identity):
    if not path.exists():
        return None
    previous = _read(path)
    if previous.get("experiment_identity") != identity:
        raise ValueError(f"Existing diagnostic case identity differs: {path}")
    if _verified(previous, identity):
        return previous
    archive = path.parent / "previous_attempts" / uuid.uuid4().hex
    archive.mkdir(parents=True, exist_ok=True)
    path.replace(archive / "metrics.json")
    # Keep the failed attempt's video and first-query evidence before retrying.
    for name in ("first_query.npz", "rollout.mp4"):
        artifact = path.parent / name
        if artifact.exists():
            artifact.replace(archive / name)
            if name == "first_query.npz" and previous.get("first_query"):
                previous["first_query"]["path"] = str((archive / name).resolve())
            if name == "rollout.mp4":
                previous["video_path"] = str((archive / name).resolve())
    evaluation._write_json(archive / "metrics.json", previous)
    return None


def _array_identity(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(evaluation._numpy(value))
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


class FirstQueryRecorder:
    """Transparent inference proxy saving the first query, without extra inference."""
    def __init__(self, model, path, device):
        self.model, self.path, self.device = model, Path(path), device
        self.first_query = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict_action(self, current, prompt_features):
        import torch
        rng = None
        if self.first_query is None:
            rng = {"cpu": torch.get_rng_state().numpy()}
            if self.device.startswith("cuda"):
                rng["cuda"] = torch.cuda.get_rng_state(self.device).cpu().numpy()
            rng = _array_identity(rng)
        predicted = self.model.predict_action(current, prompt_features=prompt_features)
        if self.first_query is None:
            raw = np.ascontiguousarray(evaluation._numpy(predicted), dtype=np.float32)
            if raw.shape != (1, 16, 16) or not np.isfinite(raw).all():
                raise ValueError("First diagnostic prediction must be finite [1,16,16]")
            scale = evaluation._numpy(self.model.action_scale).astype(np.float32)
            offset = evaluation._numpy(self.model.action_offset).astype(np.float32)
            if scale.shape != (16,) or offset.shape != (16,):
                raise ValueError("Expected the checkpoint's native action16 normalization buffers")
            arrays = {"action_raw": raw, "action_normalized": raw * scale + offset,
                "agentview_rgb": evaluation._numpy(current["agentview_rgb"]),
                "qpos_gripper_states": evaluation._numpy(current["qpos_gripper_states"])}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            self.first_query = {"path": str(self.path.resolve()), "sha256": file_sha256(self.path),
                "current_observation_sha256": _array_identity(current), "torch_rng_sha256": rng,
                "prediction_shape": list(raw.shape), "prediction_before_controller_clipping": True}
        return predicted


def validate_checkpoint(metadata, config):
    """Reuse the formal training gate while keeping this fixed subset diagnostic."""
    _, tasks = evaluation.official_tasks(config.tig_repo, "seen")
    cfg = evaluation.EvalConfig(config.checkpoint, config.output_dir, "seen", tig_repo=config.tig_repo)
    reasons = evaluation.checkpoint_reasons(metadata, cfg, tasks, tasks)
    if reasons:
        raise ValueError("Diagnostic needs the completed formal C15 checkpoint: " + "; ".join(reasons))


def run_worker(config, attempt_id):
    import torch
    import dill
    from .native_model import NativeBPPModel, SCHEMA as MODEL_SCHEMA
    from .training_data import uniform_prompt_indices

    plan = build_plan(config.tig_repo)
    cases = shard_cases(plan, config.num_shards, config.shard_index)
    _immutable_json(config.output_dir / "diagnostic_plan.json", plan)
    payload = torch.load(config.checkpoint, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("Expected a TIG-native BPP checkpoint")
    validate_checkpoint(payload.get("metadata", {}), config)
    protocol = {"schema": SCHEMA, "plan_sha256": evaluation._identity(plan), "plan": plan,
        "checkpoint": str(config.checkpoint.resolve()), "checkpoint_sha256": file_sha256(config.checkpoint),
        "dataset_revision": DATASET_REVISION, "dataset_root": str(config.dataset_root.resolve()),
        "training_metadata": payload["metadata"], "normalizer_source": payload["normalizer_stats"].get("source"),
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in
            ("prompt_diagnostic.py", "native_eval.py", "native_model.py", "data.py", "training_data.py")},
        "torch_version": torch.__version__, "diagnostic_only": True, "formal_success_rate": None}
    del payload
    identity = evaluation._identity(protocol)
    protocol["experiment_identity"] = identity
    _immutable_json(config.output_dir / "diagnostic_manifest.json", protocol)
    shard_name = f"shard_{config.shard_index:03d}_of_{config.num_shards:03d}"
    evaluation._write_json(config.output_dir / f"{shard_name}.json",
        {"experiment_identity": identity, "attempt_id": attempt_id, "cases": cases, "device": config.device})
    model = NativeBPPModel.from_training_checkpoint(config.checkpoint, device=config.device,
        bpp_repo=config.bpp_repo, amp_dtype="bfloat16")
    model.eval().requires_grad_(False)
    cache = {}
    failed = False
    try:
        for case in cases:
            path = case_path(config, case)
            prior = _archive_unverified(path, identity)
            if prior is not None:
                print(json.dumps({**case, "status": "reused_exit_verified"}), flush=True)
                continue
            record = {"experiment_identity": identity, "diagnostic_only": True, **case,
                "status": "error", "success_at_end": None, "success_once": None,
                "last_normalized_reward": None, "process_exit_verified": False,
                "diagnostic_attempt_id": attempt_id}
            proxy = None
            evaluation._write_json(path.parent / "attempt_claim.json",
                {"experiment_identity": identity, "diagnostic_attempt_id": attempt_id, **case})
            try:
                ref = select_reference(config.dataset_root, config.tig_repo / "examples/baselines/lerobot_dataset/task_mapping.json",
                    case["prompt_task_id"], "human_video_only", case["episode_index"])
                key = ref.dataset_id, ref.episode_index
                if key not in cache:
                    pixels = read_reference_rgb(ref)
                    indices = uniform_prompt_indices(len(pixels), 16)
                    sampled = np.ascontiguousarray(pixels[indices].transpose(0, 3, 1, 2))[None]
                    reference_manifest = ref.manifest()
                    reference_manifest.update(sampled_frame_indices=indices.tolist(), decoded_frames=len(pixels),
                        prompt_frames=16, prompt_modalities=["RGB"], sampled_rgb_sha256=_array_identity({"rgb": sampled}))
                    cache[key] = {"rgb": torch.as_tensor(sampled), "manifest": reference_manifest, "features": None}
                    del pixels, sampled
                entry = cache[key]
                def get_features(entry=entry):
                    if entry["features"] is None:
                        entry["features"] = model.encode_prompt(entry["rgb"])
                        del entry["rgb"]
                    return entry["features"]
                proxy = FirstQueryRecorder(model, path.parent / "first_query.npz", config.device)
                record = evaluation.run_episode(case_config(config, case), proxy, case["task_id"], case["seed"],
                    ref, entry["manifest"], get_features, identity)
                record.update(diagnostic_only=True, condition=case["condition"], prompt_task_id=case["prompt_task_id"],
                    episode_index=case["episode_index"], diagnostic_attempt_id=attempt_id,
                    process_exit_verified=False, first_query=proxy.first_query,
                    video_path=str((path.parent / "rollout.mp4").resolve()),
                    paired_case_id=evaluation._identity({"task_id": case["task_id"], "seed": case["seed"]}))
            except Exception as exc:
                record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            finally:
                evaluation._write_json(path, record)
                proxy = None
            failed |= record.get("status") != "completed" or record.get("executed_control_steps") != 500
            print(json.dumps({**case, "status": record["status"], "steps": record.get("executed_control_steps"),
                "diagnostic_only": True}), flush=True)
    finally:
        cache.clear()
        if "entry" in locals(): entry.clear()
        if "get_features" in locals(): del get_features
        model.policy.obs_encoder.reset()
        del model
        gc.collect()
        if config.device.startswith("cuda"):
            torch.cuda.synchronize(config.device); torch.cuda.empty_cache()
        gc.collect()
    return 2 if failed else 0


def record_process_exit(config, cases, attempt_id, returncode, wall_seconds):
    """A worker's abnormal native exit invalidates all of its new case evidence."""
    for case in cases:
        path = case_path(config, case)
        claim_path = path.parent / "attempt_claim.json"
        claim = _read(claim_path) if claim_path.exists() else {}
        record = _read(path) if path.exists() else None
        owns_case = (claim.get("diagnostic_attempt_id") == attempt_id
            or (record or {}).get("diagnostic_attempt_id") == attempt_id)
        if not owns_case:
            continue
        if record is None:
            record = {"experiment_identity": claim.get("experiment_identity"), **case,
                "diagnostic_only": True, "diagnostic_attempt_id": attempt_id,
                "status": "error", "error_type": "MissingDiagnosticCase",
                "error": "Worker claimed the case but did not finish its record",
                "success_at_end": None, "success_once": None, "last_normalized_reward": None}
        record["diagnostic_attempt_id"] = attempt_id
        if returncode != 0:
            record["rollout_metrics_before_process_error"] = {key: record.get(key) for key in
                ("status", "success_at_end", "success_once", "last_normalized_reward")}
            record.update(status="error", error_type="DiagnosticWorkerProcessExit",
                error=f"Diagnostic worker exited with code {returncode}", success_at_end=None,
                success_once=None, last_normalized_reward=None, process_exit_verified=False,
                process_returncode=returncode)
        else:
            record.update(process_exit_verified=True, process_returncode=0)
        evaluation._write_json(path, record)
    name = f"shard_{config.shard_index:03d}_of_{config.num_shards:03d}"
    evaluation._write_json(config.output_dir / f"process_status_{name}.json",
        {"attempt_id": attempt_id, "cases": cases, "process_returncode": returncode,
         "process_exit_verified": returncode == 0, "wall_seconds": wall_seconds, "diagnostic_only": True})
    return aggregate_results(config)


def aggregate_results(config):
    """Serialize snapshots so a stale partial aggregate cannot overwrite a final one."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "diagnostic_summary.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _aggregate_results_locked(config)
        except Exception as exc:
            evaluation._write_json(config.output_dir / "diagnostic_summary.json", {
                "schema": SCHEMA, "diagnostic_only": True, "formal_success_rate": None,
                "incomplete_or_errors": True, "error_type": type(exc).__name__, "error": str(exc),
                "conditions": {condition: {"diagnostic_success_at_end_rate": None,
                    "diagnostic_success_once_rate": None,
                    "diagnostic_mean_final_normalized_reward": None} for condition in CONDITIONS}})
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _aggregate_results_locked(config):
    """Only complete, identity-matching, exit-verified cohorts receive diagnostic rates."""
    manifest_path = config.output_dir / "diagnostic_manifest.json"
    if not manifest_path.exists():
        return {"diagnostic_only": True, "formal_success_rate": None, "incomplete_or_errors": True,
            "error": "Diagnostic manifest absent; worker initialization may have failed"}
    manifest = _read(manifest_path)
    identity, plan = manifest["experiment_identity"], manifest["plan"]
    records = {}
    pending, errors = [], []
    for case in plan["cases"]:
        path = case_path(config, case)
        record = _read(path) if path.exists() else None
        key = case["task_id"], case["seed"], case["condition"]
        if record is not None and record.get("experiment_identity") != identity:
            raise ValueError(f"Aggregate diagnostic identity differs: {path}")
        if record is not None and record.get("status") == "completed" and any(record.get(key) != value for key, value in case.items()):
            raise ValueError(f"Aggregate diagnostic case labels differ: {path}")
        if record is None or (record.get("status") == "completed" and not record.get("process_exit_verified")):
            pending.append(case)
        elif not _verified(record, identity):
            errors.append(case)
        else:
            records[key] = record
    valid = len(records) == plan["case_count"] and not pending and not errors
    conditions = {}
    for condition in CONDITIONS:
        selected = [record for (task, seed, label), record in records.items() if label == condition]
        conditions[condition] = {"planned_rollouts": 10, "exit_verified_rollouts": len(selected),
            "diagnostic_success_at_end_rate": sum(bool(row["success_at_end"]) for row in selected) / 10 if valid else None,
            "diagnostic_success_once_rate": sum(bool(row["success_once"]) for row in selected) / 10 if valid else None,
            "diagnostic_mean_final_normalized_reward": sum(row["last_normalized_reward"] for row in selected) / 10 if valid else None}
    pairs = []
    if valid:
        for task in plan["task_order"]:
            for seed in plan["seeds"]:
                correct, mismatch = [records[(task, seed, condition)] for condition in CONDITIONS]
                pair = {"task_id": task, "seed": seed,
                    "success_at_end_correct_minus_mismatch": int(correct["success_at_end"]) - int(mismatch["success_at_end"])}
                first, second = correct.get("first_query"), mismatch.get("first_query")
                reasons = []
                if not first or not second:
                    reasons.append("first_query_metadata_missing")
                else:
                    if first["current_observation_sha256"] != second["current_observation_sha256"]:
                        reasons.append("current_observation_sha256_mismatch")
                    if first["torch_rng_sha256"] != second["torch_rng_sha256"]:
                        reasons.append("torch_rng_sha256_mismatch")
                comparable = not reasons
                pair.update(first_query_same_observation_and_rng=comparable,
                    first_query_comparison_unavailable_reasons=reasons,
                    action_raw_rms_difference=None, action_raw_rms_difference_per_dimension=None,
                    action_normalized_rms_difference=None, action_normalized_rms_difference_per_dimension=None)
                if comparable:
                    arrays = []
                    for entry in (first, second):
                        path = Path(entry["path"])
                        if file_sha256(path) != entry["sha256"]:
                            raise ValueError(f"First-query artifact changed: {path}")
                        with np.load(path, allow_pickle=False) as value:
                            arrays.append({key: value[key].astype(np.float64) for key in ("action_raw", "action_normalized")})
                    for key in ("action_raw", "action_normalized"):
                        diff = arrays[0][key] - arrays[1][key]
                        pair[key + "_rms_difference"] = float(np.sqrt(np.mean(diff ** 2)))
                        pair[key + "_rms_difference_per_dimension"] = np.sqrt(np.mean(diff ** 2, axis=(0, 1))).tolist()
                pairs.append(pair)
    summary = {"schema": SCHEMA, "experiment_identity": identity, "diagnostic_only": True,
        "formal_success_rate": None, "planned_rollouts": plan["case_count"], "exit_verified_rollouts": len(records),
        "pending_cases": pending, "error_cases": errors, "incomplete_or_errors": not valid,
        "conditions": conditions, "pairs": pairs, "interpretation_limits": LIMITS}
    evaluation._write_json(config.output_dir / "diagnostic_summary.json", summary)
    return summary


def supervise(config, cli_values):
    plan = build_plan(config.tig_repo)
    cases = shard_cases(plan, config.num_shards, config.shard_index)
    _immutable_json(config.output_dir / "diagnostic_plan.json", plan)
    attempt = uuid.uuid4().hex
    started = time.monotonic()
    with ExitStack() as stack:
        # Own all assigned cases until AFTER native interpreter exit has been checked.
        # This also prevents a concurrent worker-count change from reusing pending evidence.
        for case in sorted(cases, key=_case_lock_name):
            path = config.output_dir / "case_locks" / _case_lock_name(case)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another diagnostic worker owns this case: {case}") from exc
        env = dict(os.environ, BPP_PROMPT_DIAGNOSTIC_ATTEMPT=attempt)
        done = subprocess.run([sys.executable, "-X", "faulthandler", "-m",
            "examples.baselines.bpp.bpp.prompt_diagnostic", *cli_values, "--_worker"], env=env)
        summary = record_process_exit(config, cases, attempt, done.returncode, time.monotonic() - started)
    manifest_path = config.output_dir / "diagnostic_manifest.json"
    identity = _read(manifest_path).get("experiment_identity") if manifest_path.exists() else None
    own_cases_complete = bool(identity) and all(case_path(config, case).exists()
        and _verified(_read(case_path(config, case)), identity) for case in cases)
    print(json.dumps({"process_returncode": done.returncode, "diagnostic_only": True,
        "assigned_cases_complete": own_cases_complete,
        "cohort_complete": not summary["incomplete_or_errors"]}), flush=True)
    return 0 if done.returncode == 0 and own_cases_complete else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DiagnosticConfig.dataset_root)
    parser.add_argument("--tig-repo", type=Path, default=DiagnosticConfig.tig_repo)
    parser.add_argument("--bpp-repo", type=Path, default=DiagnosticConfig.bpp_repo)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true", help="Declare the fixed cohort without reading weights or using a GPU")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild the exit-verified aggregate without loading weights")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    values = vars(parser.parse_args(argv))
    flags = {name: values.pop(name) for name in ("plan_only", "collect_only", "_worker")}
    return DiagnosticConfig(**values), flags


def main(argv=None):
    cli_values = list(sys.argv[1:] if argv is None else argv)
    config, flags = parse_args(cli_values)
    if flags["plan_only"]:
        plan = build_plan(config.tig_repo)
        shard_cases(plan, config.num_shards, config.shard_index)
        _immutable_json(config.output_dir / "diagnostic_plan.json", plan)
        print(json.dumps(plan, ensure_ascii=False), flush=True)
        return 0
    if flags["collect_only"]:
        summary = aggregate_results(config)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 2 if summary["incomplete_or_errors"] else 0
    if flags["_worker"]:
        attempt = os.environ.get("BPP_PROMPT_DIAGNOSTIC_ATTEMPT")
        if not attempt:
            raise ValueError("Use the supervised diagnostic entrypoint; worker attempt ID is missing")
        atexit.register(evaluation._runtime_marker, "prompt_diagnostic_python_atexit")
        result = run_worker(config, attempt)
        evaluation._runtime_marker("prompt_diagnostic_worker_main_return")
        return result
    return supervise(config, cli_values)


if __name__ == "__main__":
    raise SystemExit(main())

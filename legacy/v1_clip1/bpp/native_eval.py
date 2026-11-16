"""TIG evaluation of trained dual-arm BPP with RGB-only demonstration prompts.

Human RGB is the main reference source; robot RGB is an explicit extra condition.
Policy inputs contain only executing-robot ZED RGB and measured qpos18. Simulator
success/reward information is used for scoring after actions have been selected.
"""
from __future__ import annotations

import argparse
import atexit
import gc
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import types
import uuid
from typing import Any

import numpy as np

from .data import (DATASET_REVISION, ReferenceVideo, file_sha256,
                   read_reference_rgb, resize_rgb, select_reference, split_task_level)

# Match the released TIG evaluators: keep these as module-level imports.
from examples.baselines.diffusion_policy.diffusion_policy.make_env import make_eval_envs as _official_make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper as _official_flatten_wrapper
from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import configure_dual_task_level as _official_configure_level

REGIMES = {"seen": "c15", "zero_shot": "c15", "scratch": "scratch",
           "pretrain_finetune": "finetune"}
ARMS = ("panda_wristcam-0", "panda_wristcam-1")
CURRENT_KEYS = {"agentview_rgb", "qpos_gripper_states"}
SEED_ORIGIN = 100000


def _runtime_marker(stage):
    print(json.dumps({"native_eval_runtime": stage, "pid": os.getpid()}),
          file=sys.stderr, flush=True)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _identity(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def prepare_environment_runtime(config):
    """Use the official evaluator's module-level native-library import lifetime."""
    return _official_make_eval_envs, _official_flatten_wrapper, _official_configure_level


def official_tasks(tig_repo: Path, regime: str) -> tuple[Path, tuple[str, ...]]:
    filename = "seen_5tasks_env_list.txt" if regime == "seen" else "unseen_5tasks_env_list.txt"
    path = tig_repo / "examples/baselines/lerobot_dataset/eval/exp_list" / filename
    tasks = tuple(s.strip() for s in path.read_text().splitlines()
                  if s.strip() and not s.lstrip().startswith("#"))
    if len(tasks) != 20 or len(set(tasks)) != 20:
        raise ValueError(f"Expected the official 20 unique L0-L3 variants: {path}")
    for task in tasks:
        split_task_level(task)
    return path, tasks


def evaluation_cases(tasks, num_episodes=10, num_shards=1, shard_index=0):
    """Stable task-major sharding; seed/reference indexes never depend on workers."""
    if num_episodes <= 0 or num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid episode count or shard selection")
    cases = [(task, episode) for task in tasks for episode in range(num_episodes)]
    return [case for index, case in enumerate(cases) if index % num_shards == shard_index]


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path
    output_dir: Path
    regime: str
    dataset_root: Path = Path("/gaozt-test1/sxh/bpp/data/IG-10K-Dataset")
    tig_repo: Path = Path("/gaozt-test1/sxh/bpp/repos/The-Imitator-Game")
    bpp_repo: Path = Path("/gaozt-test1/sxh/bpp/repos/behavior_prompting")
    tasks: tuple[str, ...] | None = None
    reference_source: str = "human"
    training_prompt_source: str = "human"
    pretraining_scale: str = "c15"
    num_episodes: int = 10
    max_episode_steps: int = 500
    execution_horizon: int = 12
    device: str = "cuda:0"
    shader: str = "rt-fast"
    amp_dtype: str = "bfloat16"
    inference_process: str = "shared"
    capture_video: bool = True
    num_shards: int = 1
    shard_index: int = 0
    allow_pilot: bool = False

    @property
    def task_mapping(self):
        return self.tig_repo / "examples/baselines/lerobot_dataset/task_mapping.json"


def checkpoint_reasons(metadata, config, full_tasks, selected_tasks):
    """Require a complete, matching training regime before reporting main scores."""
    reasons = []
    expected = (config.pretraining_scale if config.regime in ("seen", "zero_shot")
                else REGIMES[config.regime])
    if metadata.get("regime") != expected:
        reasons.append(f"training regime {metadata.get('regime')!r} differs from {expected!r}")
    if metadata.get("pilot") is not False:
        reasons.append("checkpoint is pilot or lacks an explicit non-pilot label")
    if metadata.get("completed") is not True:
        reasons.append("training checkpoint is incomplete")
    if config.regime == "pretrain_finetune" and metadata.get("pretraining_scale") != config.pretraining_scale:
        reasons.append(f"fine-tune pretraining scale {metadata.get('pretraining_scale')!r} differs from "
                       f"{config.pretraining_scale!r}")
    updates = {"c15": 45000, "c30": 90000, "c45": 135000}.get(expected, 3000)
    if metadata.get("train_step") != updates or metadata.get("max_updates") != updates:
        reasons.append(f"checkpoint does not contain the declared {updates}-update budget")
    if metadata.get("prompt_source") != config.training_prompt_source:
        reasons.append(f"training prompt source {metadata.get('prompt_source')!r} differs from "
                       f"{config.training_prompt_source!r}")
    if config.regime == "pretrain_finetune":
        source = metadata.get("initialize_from_metadata") or {}
        source_updates = {"c15": 45000, "c30": 90000, "c45": 135000}[config.pretraining_scale]
        valid_source = (
            source.get("regime") == config.pretraining_scale
            and source.get("pretraining_scale") == config.pretraining_scale
            and source.get("train_step") == source_updates
            and source.get("max_updates") == source_updates
            and source.get("completed") is True and source.get("pilot") is False
            and source.get("prompt_source") == config.training_prompt_source
        )
        digest = metadata.get("initialize_from_sha256", "")
        valid_digest = isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
        if not valid_source or not valid_digest:
            reasons.append("fine-tune source must retain the matching complete Ck budget, prompt source and checkpoint SHA256")
    if tuple(selected_tasks) != tuple(full_tasks):
        reasons.append("only a subset of the official task variants was selected")
    if config.num_episodes != 10 or config.max_episode_steps != 500 or config.execution_horizon != 12:
        reasons.append("episode count or evaluation horizon differs from the declared protocol")
    return reasons


def native_observation(raw_observation, env):
    """Whitelist actual ZED RGB and the two Panda robots' measured qpos only."""
    rgb = np.array(resize_rgb(raw_observation["sensor_data"]["zed2i"]["rgb"], 224),
                   dtype=np.uint8, copy=True, order="C")
    measured = []
    for uid in ARMS:
        value = _numpy(env.unwrapped.agent.agents_dict[uid].robot.get_qpos())
        if value.shape == (1, 9):
            value = value[0]
        if value.shape != (9,) or not np.isfinite(value).all():
            raise ValueError(f"Expected measured Panda qpos9 for {uid}, got {value.shape}")
        measured.append(value.astype(np.float32, copy=True))
    result = {"agentview_rgb": np.ascontiguousarray(rgb.transpose(2, 0, 1)),
              "qpos_gripper_states": np.concatenate(measured)}
    return result, rgb


def current_history(history):
    values = list(history)
    if len(values) != 2 or any(set(value) != CURRENT_KEYS for value in values):
        raise ValueError("Current observations must contain exactly RGB and measured qpos18, history2")
    return {key: np.stack([value[key] for value in values])[None] for key in sorted(CURRENT_KEYS)}


def joint_actions(prediction, env):
    """Use released action units: raw arm joint targets plus controller gripper input."""
    value = _numpy(prediction).astype(np.float32)
    if value.shape != (16,) or not np.isfinite(value).all():
        raise ValueError(f"Expected one finite physical-unit action16, got {value.shape}")
    low = np.concatenate([_numpy(env.action_space[uid].low).reshape(-1) for uid in ARMS])
    high = np.concatenate([_numpy(env.action_space[uid].high).reshape(-1) for uid in ARMS])
    if low.shape != (16,) or high.shape != (16,):
        raise ValueError("Unexpected native dual-Panda joint action space")
    clipped = np.clip(value, low, high).astype(np.float32)
    counts = (clipped != value).astype(np.int64)
    return {uid: clipped[8*i:8*(i+1)][None] for i, uid in enumerate(ARMS)}, counts


class OfficialEnvironment:
    """Expose only ZED RGB while retaining official CPU/vector wrapper lifetime."""
    def __init__(self, vector):
        self.vector = vector
        self.unwrapped = vector.envs[0].unwrapped
        self.action_space = vector.action_space
        self.last_info = {}

    @staticmethod
    def _one(value):
        if isinstance(value, np.ndarray) and value.dtype == object:
            return value.reshape(-1)[0]
        return value

    @staticmethod
    def _observation(observation):
        rgb = _numpy(observation["rgb"])
        while rgb.ndim > 3:
            rgb = rgb[0] if rgb.ndim == 5 else rgb[-1]
        rgb = np.array(resize_rgb(rgb), copy=True)
        return {"sensor_data": {"zed2i": {"rgb": rgb}}}

    def reset(self, seed):
        observation, info = self.vector.reset(seed=[seed])
        self.last_info = info
        return self._observation(observation), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.vector.step(actions)
        # SyncVectorEnv may already reset the raw BaseEnv on the truncation step.
        if "final_info" in info:
            final_info = self._one(info["final_info"])
            final_observation = self._one(info.get("final_observation"))
            if isinstance(final_info, dict):
                info = final_info
            if isinstance(final_observation, dict):
                observation = final_observation
        self.last_info = info
        return self._observation(observation), reward, terminated, truncated, info

    def final_evaluation(self):
        return self.last_info

    def close(self):
        self.vector.close()


def make_native_environment(config: EvalConfig, task_id: str):
    """Use the exact CPU wrapper stack and base+level dispatch of TIG baselines."""
    make_eval_envs, FlattenRGBDObservationWrapper, configure_dual_task_level = prepare_environment_runtime(config)
    for name in ("MANI_SKILL_L1", "MANI_SKILL_L2", "MANI_SKILL_L3"):
        os.environ.pop(name, None)
    level, base = split_task_level(task_id)
    configure_dual_task_level(level, mirror_robot_pose=False)
    vector = make_eval_envs(base, 1, "physx_cpu", {
        "render_backend": config.device if config.device.startswith("cuda") else "gpu",
        "control_mode": "pd_joint_pos", "obs_mode": "rgb", "render_mode": "sensors",
        "reward_mode": "normalized_dense", "max_episode_steps": config.max_episode_steps,
        "enhanced_determinism": True, "hi_res": False, "wrist_sensor": False,
        "sensor_configs": {"width": 224, "height": 224, "shader_pack": config.shader},
        "human_render_camera_configs": {"shader_pack": config.shader},
    }, {"obs_horizon": 2}, None, [FlattenRGBDObservationWrapper], l_level=level)
    return OfficialEnvironment(vector)


def _inference_worker(connection, checkpoint, bpp_repo, device, amp_dtype):
    """Keep BPP CUDA kernels and SAPIEN graphics in separate native runtimes."""
    try:
        import torch
        from .native_model import NativeBPPModel
        torch.set_num_threads(1)
        model = NativeBPPModel.from_training_checkpoint(checkpoint, device=device,
                    bpp_repo=bpp_repo, amp_dtype=amp_dtype)
        model.eval().requires_grad_(False)
        prompts = {}
        connection.send({"ok": True, "value": {"pid": os.getpid()}})
        while True:
            request = connection.recv()
            try:
                operation = request["operation"]
                if operation == "close":
                    connection.send({"ok": True, "value": None})
                    break
                if operation == "reset":
                    model.policy.obs_encoder.reset()
                    result = None
                elif operation == "seed":
                    seed = request["seed"]
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if device.startswith("cuda"):
                        torch.cuda.manual_seed_all(seed)
                    result = None
                elif operation == "encode":
                    prompt_id = len(prompts)
                    prompts[prompt_id] = model.encode_prompt(torch.as_tensor(request["rgb"]))
                    result = {"remote_prompt_id": prompt_id}
                elif operation == "predict":
                    prompt = prompts[request["prompt_id"]]
                    result = model.predict_action(request["current"], prompt_features=prompt).cpu().numpy()
                else:
                    raise ValueError(f"Unknown inference operation: {operation}")
                connection.send({"ok": True, "value": result})
            except Exception:
                connection.send({"ok": False, "error": traceback.format_exc()})
        prompts.clear()
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        gc.collect()
        _runtime_marker("inference_worker_clean_return")
    except Exception:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        connection.close()


class IsolatedBPPModel:
    """RAM-only Pipe requests contain RGB/qpos arrays and return action arrays."""
    def __init__(self, config):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_inference_worker, args=(child,
            str(config.checkpoint), str(config.bpp_repo), config.device, config.amp_dtype))
        self.process.start()
        child.close()
        self._receive()
        self.policy = types.SimpleNamespace(obs_encoder=types.SimpleNamespace(reset=self.reset))

    def _receive(self):
        if not self.connection.poll(180):
            raise TimeoutError("Timed out waiting for the isolated BPP inference worker")
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError("Isolated BPP worker: " + response.get("error", "unknown error"))
        return response.get("value")

    def _request(self, operation, **payload):
        if not self.process.is_alive():
            raise RuntimeError(f"BPP inference worker exited: {self.process.exitcode}")
        self.connection.send(dict(operation=operation, **payload))
        return self._receive()

    def eval(self):
        return self

    def requires_grad_(self, value):
        if value:
            raise ValueError("Evaluation inference worker cannot enable training")
        return self

    def set_seed(self, seed):
        return self._request("seed", seed=seed)

    def reset(self):
        return self._request("reset")

    def encode_prompt(self, prompt_rgb):
        return self._request("encode", rgb=_numpy(prompt_rgb))

    def predict_action(self, current, prompt_features):
        if set(current) != CURRENT_KEYS or set(prompt_features) != {"remote_prompt_id"}:
            raise ValueError("Unexpected inference-worker input; require RGB/qpos and an opaque prompt id")
        return self._request("predict", current={k: _numpy(v) for k, v in current.items()},
                             prompt_id=prompt_features["remote_prompt_id"])

    def close(self):
        if self.process.is_alive():
            try:
                self._request("close")
            finally:
                self.process.join(30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
            raise RuntimeError("BPP inference worker failed to close cleanly")
        self.connection.close()
        if self.process.exitcode != 0:
            raise RuntimeError(f"BPP inference worker exited with code {self.process.exitcode}")
        return {"pid": self.process.pid, "returncode": self.process.exitcode}


class RGBRecorder:
    def __init__(self, path):
        import av
        self.av = av
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=20)
        self.stream.width = self.stream.height = 224
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "24"}

    def append(self, rgb):
        for packet in self.stream.encode(self.av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.container.mux(packet)

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


def case_path(config, task, seed):
    return config.output_dir / config.reference_source / task / f"seed_{seed}" / "metrics.json"


def run_episode(config, model, task, seed, reference, reference_manifest, prompt_features,
                identity, *, env_factory=make_native_environment):
    """Errors are persisted explicitly and cannot become failures or successes."""
    case = {"experiment_identity": identity, "task_id": task, "seed": seed,
            "reference": reference_manifest, "reference_source": config.reference_source}
    case_id = _identity(case)
    path = case_path(config, task, seed)
    if path.exists():
        old = json.loads(path.read_text())
        retry_reference_error = (
            old.get("status") == "error" and old.get("case_id") is None
            and old.get("experiment_identity") == identity
            and old.get("task_id") == task and old.get("seed") == seed
            and old.get("reference_source") == config.reference_source
        )
        if old.get("case_id") != case_id and not retry_reference_error:
            raise ValueError(f"Existing episode identity differs: {path}")
        if old.get("status") == "completed":
            return old
    record = dict(case, case_id=case_id, status="running", success_at_end=None,
                  success_once=None, last_normalized_reward=None,
                  is_pilot=config.allow_pilot, max_episode_steps=config.max_episode_steps)
    _write_json(path, record)
    env = recorder = None
    steps = queries = clipped_steps = 0
    clip_counts = np.zeros(16, dtype=np.int64)
    success_once = False
    final_success = False
    reward = None
    query_seconds = []
    phase_peaks = {}
    started = time.monotonic()
    try:
        if env_factory is make_native_environment:
            prepare_environment_runtime(config)
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if config.device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        model.eval()
        if hasattr(model, "set_seed"):
            model.set_seed(seed)
        model.policy.obs_encoder.reset()
        env = env_factory(config, task)
        record["actual_environment_class"] = type(env.unwrapped).__module__ + "." + type(env.unwrapped).__qualname__
        record["environment_dispatch"] = "official base Gym ID + MANI_SKILL L-level flags"
        if isinstance(env, OfficialEnvironment):
            from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
            record["actual_environment_flags"] = {
                "L1": L0_L3_utils.is_l1_enabled(), "L2": L0_L3_utils.is_l2_enabled(),
                "L3": L0_L3_utils.is_l3_enabled(),
                "mirror_robot_pose": L0_L3_utils.is_lr_mirror_robot_pose_enabled(),
            }
            module = sys.modules.get(type(env.unwrapped).__module__)
            filename = getattr(module, "__file__", None)
            if filename:
                record["actual_environment_source"] = str(Path(filename).resolve())
                record["actual_environment_source_sha256"] = file_sha256(filename)
        if int(env.unwrapped.control_freq) != 20 or int(env.unwrapped.sim_freq) != 100:
            raise ValueError("Expected TIG sim100Hz/control20Hz")
        raw, info = env.reset(seed=seed)
        # Start the graphics/interop runtime before the first vision CUDA forward.
        if callable(prompt_features):
            prompt_features = prompt_features()
        first, rgb = native_observation(raw, env)
        history = deque((first, first), maxlen=2)
        queue = deque()
        if config.capture_video:
            recorder = RGBRecorder(path.parent / "rollout.mp4")
            recorder.append(rgb)
        for _ in range(config.max_episode_steps):
            if not queue:
                then = time.monotonic()
                with torch.inference_mode():
                    prediction = _numpy(model.predict_action(current_history(history),
                                                              prompt_features=prompt_features))
                if prediction.shape != (1, 16, 16) or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid native BPP predicted action chunk: {prediction.shape}")
                queue.extend(prediction[0, :config.execution_horizon])
                queries += 1
                query_seconds.append(time.monotonic() - then)
            action, clipped = joint_actions(queue.popleft(), env)
            clip_counts += clipped
            clipped_steps += int(clipped.any())
            raw, reward, terminated, truncated, info = env.step(action)
            steps += 1
            final_success = bool(_numpy(info.get("success", False)).reshape(-1)[0])
            success_once |= final_success
            for key, value in info.items():
                if key.startswith("peak_r_"):
                    phase_peaks[key] = float(_numpy(value).reshape(-1)[0])
            frame, rgb = native_observation(raw, env)
            history.append(frame)
            if recorder:
                recorder.append(rgb)
            # Match official ignore_terminations=True; keep going after success.
            if bool(_numpy(truncated).any()):
                break
        final_info = (env.final_evaluation() if hasattr(env, "final_evaluation")
                      else env.unwrapped.evaluate())
        final_success = bool(_numpy(final_info.get("success", final_success)).reshape(-1)[0])
        success_once |= final_success
        for key, value in final_info.items():
            if key.startswith("peak_r_"):
                phase_peaks[key] = float(_numpy(value).reshape(-1)[0])
        record.update(status="completed", success_at_end=final_success,
                      success_once=success_once,
                      last_normalized_reward=float(_numpy(reward).reshape(-1)[0]),
                      phase_peaks=phase_peaks)
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), success_at_end=None,
                      success_once=None, last_normalized_reward=None)
    finally:
        for item, label in ((recorder, "video_close"), (env, "environment_close")):
            if item is not None:
                try:
                    if label == "environment_close":
                        # Release observation/NumPy aliases while their native scene is alive.
                        raw = info = first = frame = rgb = final_info = value = None
                        prediction = action = None
                        if "history" in locals():
                            history.clear()
                        if "queue" in locals():
                            queue.clear()
                        gc.collect()
                    _runtime_marker(label + "_begin")
                    item.close()
                    _runtime_marker(label + "_end")
                except Exception as exc:
                    record.update(status="error", success_at_end=None, success_once=None,
                                  last_normalized_reward=None, **{label + "_error": str(exc)})
        env = recorder = item = None
        gc.collect()
        try:
            model.policy.obs_encoder.reset()
        except Exception as exc:
            record.update(status="error", success_at_end=None, success_once=None,
                          last_normalized_reward=None, model_reset_error=str(exc))
        record.update(executed_control_steps=steps, policy_queries=queries,
                      simulated_seconds=steps / 20, wall_seconds=time.monotonic() - started,
                      query_seconds=query_seconds, clip_counts_per_action_dimension=clip_counts,
                      clipped_control_steps=clipped_steps)
        _write_json(path, record)
    return record


def summarize(records, expected_count, *, is_pilot):
    completed = [r for r in records if r.get("status") == "completed"]
    errors = [r for r in records if r.get("status") != "completed"]
    all_valid = len(completed) == expected_count and not errors
    rate = (sum(bool(r["success_at_end"]) for r in completed) / expected_count
            if all_valid and expected_count else None)
    return {"planned_episode_count": expected_count, "completed_episode_count": len(completed),
            "error_episode_count": len(errors), "is_pilot": is_pilot,
            "success_at_end_rate": rate if not is_pilot else None,
            "diagnostic_success_at_end_rate": rate,
            "success_once_rate": (sum(bool(r["success_once"]) for r in completed) / expected_count
                                  if all_valid and expected_count and not is_pilot else None),
            "incomplete_or_errors": not all_valid}


def _revision(repo):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config: EvalConfig):
    if config.regime not in REGIMES or config.reference_source not in ("human", "robot"):
        raise ValueError("Unknown evaluation regime or reference source")
    if config.training_prompt_source not in ("human", "robot"):
        raise ValueError("Unknown training prompt source")
    if config.pretraining_scale not in ("c15", "c30", "c45"):
        raise ValueError("Unknown pretraining scale")
    if not 1 <= config.execution_horizon <= 16 or config.max_episode_steps <= 0:
        raise ValueError("Execution horizon must be 1..16 and episode horizon positive")
    split_path, full_tasks = official_tasks(config.tig_repo, config.regime)
    tasks = full_tasks if config.tasks is None else tuple(t for t in full_tasks if t in config.tasks)
    if not tasks or (config.tasks is not None and (len(set(config.tasks)) != len(config.tasks) or set(config.tasks) != set(tasks))):
        raise ValueError("Selected tasks must be a unique subset of the official regime list")
    cases = evaluation_cases(tasks, config.num_episodes, config.num_shards, config.shard_index)
    prepare_environment_runtime(config)
    import dill
    import torch
    from .native_model import NativeBPPModel, SCHEMA
    from .training_data import uniform_prompt_indices
    payload = torch.load(config.checkpoint, map_location="cpu", pickle_module=dill,
                         weights_only=False, mmap=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Expected a trained TIG-native BPP checkpoint")
    metadata = payload.get("metadata", {})
    reasons = checkpoint_reasons(metadata, config, full_tasks, tasks)
    if reasons and not config.allow_pilot:
        raise ValueError("Formal evaluation rejected: " + "; ".join(reasons) +
                         ". Use --allow-pilot only for explicitly excluded diagnostics.")
    details = {"training_metadata": metadata, "normalizer_source": payload["normalizer_stats"].get("source"),
               "parameter_audit": payload.get("parameter_audit"),
               "init_head_sha256": payload.get("init_head_sha256"),
               "native_model_config": payload.get("config")}
    del payload
    sources = {name: file_sha256(Path(__file__).with_name(name)) for name in
               ("native_eval.py", "native_model.py", "data.py", "training_data.py")}
    protocol = {
        "schema": "bpp_tig_native_evaluation_v1", "regime": config.regime,
        "pretraining_scale": None if config.regime == "scratch" else config.pretraining_scale,
        "scratch_reuse": "same external initialization and target data; shared across C15/C30/C45",
        "reference_source": config.reference_source,
        "training_prompt_source": metadata.get("prompt_source", "unspecified_legacy_pilot"),
        "expected_training_prompt_source": config.training_prompt_source,
        "checkpoint": str(config.checkpoint.resolve()), "checkpoint_sha256": file_sha256(config.checkpoint),
        "tasks": tasks, "seeds": list(range(SEED_ORIGIN, SEED_ORIGIN + config.num_episodes)),
        "official_environment_list": str(split_path), "official_environment_list_sha256": file_sha256(split_path),
        "task_mapping_sha256": file_sha256(config.task_mapping), "dataset_revision": DATASET_REVISION,
        "dataset_root": str(config.dataset_root.resolve()),
        "reference": "episode i with reset seed 100000+i; same L0/base reference across L0-L3",
        "prompt_modalities": ["RGB"], "prompt_frames": 16, "prompt_sampling": "uniform including endpoints",
        "prompt_encoding_initialization": "after first simulator reset/render, then cached across matching references",
        "current_observations": {"zed2i_rgb": [2, 3, 224, 224], "measured_qpos": [2, 18]},
        "forbidden_policy_inputs": ["demonstrator actions", "demonstrator proprioception", "task extra state", "success", "reward"],
        "control_mode": "pd_joint_pos", "action_dimension": 16,
        "action_units": "released joint targets; each Panda arm7 raw radians + normalized mimic-gripper command",
        "prediction_horizon": 16, "execution_horizon": config.execution_horizon,
        "max_episode_steps": config.max_episode_steps, "control_hz": 20, "sim_hz": 100,
        "ignore_terminations": True, "mirror_robot_pose": False,
        "environment_dispatch": "official Zero-WAM base Gym ID + level flag; independent L3 classes are not used",
        "environment_wrappers": ["FlattenRGBDObservationWrapper", "FrameStack(2)",
                                 "CPUGymWrapper(ignore_terminations=True,record_metrics=True)", "SyncVectorEnv"],
        "environment_source_namespace": "examples.baselines.diffusion_policy.diffusion_policy.make_env.make_eval_envs",
        "native_runtime_import_order": "official environment/ManiSkill libraries before Torch seeding and model construction",
        "terminal_scoring": "final_info before vector autoreset; never evaluate reset BaseEnv as terminal state",
        "reward_mode": "normalized_dense", "reward_use": "scoring only; returned step reward",
        "shader": config.shader,
        "shader_difference_from_zero_wam": config.shader != "rt-fast",
        "shader_note": "rt-fast matches the released Zero-WAM evaluation renderer; minimal is an explicit diagnostic alternative",
        "wrist_camera": False, "sim_backend": "physx_cpu", "capture_video": config.capture_video,
        "amp_dtype": config.amp_dtype, "source_sha256": sources,
        "inference_process": config.inference_process,
        "inference_transport": "RAM-only Pipe: current RGB/qpos arrays, cached prompt id, action16 arrays",
        "tig_revision": _revision(config.tig_repo), "bpp_revision": _revision(config.bpp_repo),
        "torch_version": torch.__version__, "is_pilot": config.allow_pilot,
        "diagnostic_reasons": reasons, **details,
    }
    identity = _identity(protocol)
    protocol["experiment_identity"] = identity
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "evaluation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("experiment_identity") != identity:
        raise ValueError("Output directory belongs to a different evaluation configuration")
    _write_json(manifest_path, protocol)
    shard_name = f"shard_{config.shard_index:03d}_of_{config.num_shards:03d}"
    _write_json(config.output_dir / (shard_name + ".json"), {
        "experiment_identity": identity, "device": config.device, "num_shards": config.num_shards,
        "shard_index": config.shard_index, "cases": cases,
        "attempt_id": os.environ.get("BPP_NATIVE_EVAL_ATTEMPT")})
    if config.inference_process == "isolated":
        model = IsolatedBPPModel(config)
    elif config.inference_process == "shared":
        model = NativeBPPModel.from_training_checkpoint(config.checkpoint, device=config.device,
                    bpp_repo=config.bpp_repo, amp_dtype=config.amp_dtype)
    else:
        raise ValueError("Unknown inference process isolation setting")
    model.eval().requires_grad_(False)
    cache, records = {}, []
    for task, episode in cases:
        seed = SEED_ORIGIN + episode
        try:
            condition = "human_video_only" if config.reference_source == "human" else "robot_video_only"
            reference = select_reference(config.dataset_root, config.task_mapping, task, condition, episode)
            key = (reference.dataset_id, reference.episode_index)
            if key not in cache:
                start = time.monotonic()
                reference_manifest = reference.manifest()
                rgb = read_reference_rgb(reference)
                indices = uniform_prompt_indices(len(rgb), 16)
                prompt_rgb = torch.as_tensor(np.ascontiguousarray(rgb[indices].transpose(0, 3, 1, 2)))[None]
                reference_manifest.update(sampled_frame_indices=indices.tolist(),
                    decoded_frames=len(rgb), prompt_frames=16, prompt_modalities=["RGB"],
                    prompt_prepare_seconds=time.monotonic() - start)
                # Delay vision CUDA kernels until SAPIEN has rendered its first reset.
                cache[key] = {"manifest": reference_manifest, "prompt_rgb": prompt_rgb,
                              "features": None}
            entry = cache[key]
            manifest = entry["manifest"]
            def get_features(entry=entry):
                if entry["features"] is None:
                    encoding_start = time.monotonic()
                    entry["features"] = model.encode_prompt(entry["prompt_rgb"])
                    entry["manifest"]["prompt_prepare_seconds"] += time.monotonic() - encoding_start
                    del entry["prompt_rgb"]
                return entry["features"]
            stable_manifest = {k: v for k, v in manifest.items() if k != "prompt_prepare_seconds"}
            record = run_episode(config, model, task, seed, reference, stable_manifest,
                                 get_features, identity)
            record["prompt_prepare_seconds_cached_reference"] = manifest["prompt_prepare_seconds"]
            _write_json(case_path(config, task, seed), record)
        except Exception as exc:
            if isinstance(exc, ValueError) and "Existing episode identity differs" in str(exc):
                raise
            record = {"experiment_identity": identity, "task_id": task, "seed": seed,
                      "reference_source": config.reference_source, "status": "error",
                      "error_type": type(exc).__name__, "error": str(exc),
                      "traceback": traceback.format_exc(), "success_at_end": None,
                      "success_once": None, "last_normalized_reward": None,
                      "is_pilot": config.allow_pilot}
            _write_json(case_path(config, task, seed), record)
        records.append(record)
        summary = dict(summarize(records, len(cases), is_pilot=config.allow_pilot),
                       experiment_identity=identity, regime=config.regime,
                       reference_source=config.reference_source, results=records)
        _write_json(config.output_dir / ("summary_" + shard_name + ".json"), summary)
        print(json.dumps({"task": task, "seed": seed, "status": record["status"],
                          "steps": record.get("executed_control_steps"),
                          "is_pilot": config.allow_pilot}, ensure_ascii=False), flush=True)
    summary = dict(summarize(records, len(cases), is_pilot=config.allow_pilot),
                   experiment_identity=identity, regime=config.regime,
                   reference_source=config.reference_source, results=records)
    _write_json(config.output_dir / ("summary_" + shard_name + ".json"), summary)
    aggregate_results(config, identity, tasks)
    # Release tensors and native objects before interpreter/global-module teardown.
    _runtime_marker("explicit_cleanup_begin")
    cache.clear()
    if "entry" in locals():
        entry.clear()
    if "get_features" in locals():
        del get_features
    if "features" in locals():
        del features
    if "prompt_rgb" in locals():
        del prompt_rgb
    if isinstance(model, IsolatedBPPModel):
        process_result = model.close()
        _write_json(config.output_dir / ("inference_process_" + shard_name + ".json"), process_result)
    del model
    gc.collect()
    if config.device.startswith("cuda"):
        torch.cuda.synchronize(config.device)
        torch.cuda.empty_cache()
    gc.collect()
    _runtime_marker("explicit_cleanup_end")
    return summary


def aggregate_results(config, identity, tasks):
    """Publish a whole-regime score only after every planned case has completed."""
    import fcntl
    lock_path = config.output_dir / ".summary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = []
        for task, episode in evaluation_cases(tasks, config.num_episodes):
            path = case_path(config, task, SEED_ORIGIN + episode)
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("experiment_identity") != identity:
                    raise ValueError(f"Aggregate episode identity differs: {path}")
                records.append(record)
        summary = dict(summarize(records, len(tasks) * config.num_episodes,
                                 is_pilot=config.allow_pilot),
                       experiment_identity=identity, regime=config.regime,
                       reference_source=config.reference_source, results=records,
                       scope="all planned task variants and seeds")
        pending = any(r.get("status") == "completed" and r.get("process_exit_verified") is not True
                      for r in records)
        summary["process_exit_pending"] = pending
        if pending:
            summary["success_at_end_rate"] = summary["success_once_rate"] = None
        _write_json(config.output_dir / "summary.json", summary)
        return summary


def record_process_exit(config, returncode, attempt_id, wall_seconds):
    """A supervisor catches native crashes that Python exception handlers cannot."""
    name = f"shard_{config.shard_index:03d}_of_{config.num_shards:03d}"
    shard_file = config.output_dir / (name + ".json")
    status = {"attempt_id": attempt_id, "returncode": returncode,
              "wall_seconds": wall_seconds, "device": config.device,
              "status": "completed" if returncode == 0 else "process_error"}
    shard = json.loads(shard_file.read_text()) if shard_file.exists() else {}
    # Invalid CLI/configuration must never invalidate an older experiment.
    owns_attempt = shard.get("attempt_id") == attempt_id
    status["owns_evaluation_attempt"] = owns_attempt
    if owns_attempt:
        identity = shard["experiment_identity"]
        status["experiment_identity"] = identity
        if returncode != 0:
            for task, episode in shard["cases"]:
                path = case_path(config, task, SEED_ORIGIN + episode)
                record = json.loads(path.read_text()) if path.exists() else {
                    "experiment_identity": identity, "task_id": task,
                    "seed": SEED_ORIGIN + episode,
                    "reference_source": config.reference_source,
                    "is_pilot": config.allow_pilot}
                if record.get("experiment_identity") != identity:
                    raise ValueError(f"Process-error episode identity differs: {path}")
                record["rollout_metrics_before_process_error"] = {
                    key: record.get(key) for key in
                    ("status", "success_at_end", "success_once", "last_normalized_reward")}
                record.update(status="error", error_type="NativeProcessExit",
                              error=f"Evaluation worker exited with code {returncode}",
                              process_returncode=returncode, process_exit_verified=False, success_at_end=None,
                              success_once=None, last_normalized_reward=None)
                _write_json(path, record)
        else:
            for task, episode in shard["cases"]:
                path = case_path(config, task, SEED_ORIGIN + episode)
                if path.exists():
                    record = json.loads(path.read_text())
                    if record.get("experiment_identity") != identity:
                        raise ValueError(f"Process-success episode identity differs: {path}")
                    record.update(process_exit_verified=True, process_returncode=0)
                    _write_json(path, record)
        manifest = json.loads((config.output_dir / "evaluation_manifest.json").read_text())
        all_tasks = manifest["tasks"]
        summary = aggregate_results(config, identity, all_tasks)
        _write_json(config.output_dir / ("summary_" + name + ".json"), dict(
            summary, scope="all planned cases after worker process exit",
            worker_returncode=returncode))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / ("process_status_" + name + ".json"), status)
    return status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regime", choices=tuple(REGIMES), required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--reference-source", choices=("human", "robot"), default="human")
    parser.add_argument("--training-prompt-source", choices=("human", "robot"), default="human")
    parser.add_argument("--pretraining-scale", choices=("c15", "c30", "c45"), default="c15")
    parser.add_argument("--dataset-root", type=Path, default=EvalConfig.dataset_root)
    parser.add_argument("--tig-repo", type=Path, default=EvalConfig.tig_repo)
    parser.add_argument("--bpp-repo", type=Path, default=EvalConfig.bpp_repo)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shader", choices=("minimal", "default", "rt-fast"), default="rt-fast")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--inference-process", choices=("isolated", "shared"), default="shared")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--no-video", dest="capture_video", action="store_false")
    values = vars(parser.parse_args(argv))
    values["tasks"] = None if values["tasks"] is None else tuple(values["tasks"])
    return EvalConfig(**values)


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    worker_flag = "--_native-worker"
    is_worker = worker_flag in values
    if is_worker:
        values.remove(worker_flag)
    config = parse_args(values)
    if is_worker:
        atexit.register(_runtime_marker, "python_atexit")
        result = run(config)
        _runtime_marker("worker_main_return")
        return 2 if result["incomplete_or_errors"] else 0
    attempt_id = uuid.uuid4().hex
    child_env = dict(os.environ, BPP_NATIVE_EVAL_ATTEMPT=attempt_id)
    started = time.monotonic()
    completed = subprocess.run([sys.executable, "-X", "faulthandler", "-m",
                                "examples.baselines.bpp.bpp.native_eval", *values, worker_flag], env=child_env)
    record_process_exit(config, completed.returncode, attempt_id, time.monotonic() - started)
    return 0 if completed.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

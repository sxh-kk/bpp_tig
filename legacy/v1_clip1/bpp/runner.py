"""Frozen BPP-LIBERO on TIG: robot RGB reference, then human RGB reference.

The executing robot is the task's left Panda, with the other arm held fixed.
The task, success predicates and reset distributions are the original TIG ones.
Current wrist RGB is an explicitly reported observation extension; no
demonstrator states or actions enter the policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .data import (
    CONDITIONS, DATASET_REVISION, REFERENCE_SEED_ORIGIN, REFERENCE_SELECTIONS,
    ReferenceVideo, file_sha256, history_batch, read_reference_rgb,
    reference_episode_for_seed, resize_rgb, select_reference, split_task_level,
)


ACTIVE_ARM = "panda_wristcam-0"
WRIST_CAMERA = "panda_wristcam_0_hand_camera"
CURRENT_KEYS = ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states")


@dataclass(frozen=True)
class RunConfig:
    checkpoint: Path
    output_dir: Path
    dataset_root: Path = Path("/gaozt-test1/sxh/bpp/data/IG-10K-Dataset")
    tig_repo: Path = Path("/gaozt-test1/sxh/bpp/repos/The-Imitator-Game")
    bpp_repo: Path = Path("/gaozt-test1/sxh/bpp/repos/behavior_prompting")
    tasks: tuple[str, ...] = ("L0_TwoRobotPickRemoteControl-v1",)
    seeds: tuple[int, ...] = (100000,)
    conditions: tuple[str, ...] = ("robot_video_only",)
    reference_selection: str = "per_seed"
    reference_episode: int = 0
    max_episode_steps: int = 500
    execution_horizon: int = 12
    device: str = "cuda:0"
    shader: str = "minimal"
    capture_video: bool = True
    prompt_hz: float = 1.0
    amp_dtype: str = "bfloat16"
    dataset_revision: str = DATASET_REVISION
    protocol_path: Path = Path(
        "/gaozt-test1/sxh/bpp/results/protocols/bpp_video_only_robot_vs_human.json"
    )

    @property
    def task_mapping(self) -> Path:
        return self.tig_repo / "examples/baselines/lerobot_dataset/task_mapping.json"

    def validate(self) -> None:
        if not self.tasks or not self.seeds:
            raise ValueError("Provide at least one task and evaluation seed")
        if len(set(self.tasks)) != len(self.tasks) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Task aliases and seeds must be unique")
        if any(seed < 0 or seed >= 2**32 for seed in self.seeds):
            raise ValueError("Evaluation seeds must be in [0, 2**32)")
        for seed in self.seeds:
            reference_episode_for_seed(
                seed, selection=self.reference_selection,
                reference_episode=self.reference_episode,
            )
        if self.conditions not in (CONDITIONS, CONDITIONS[:1], CONDITIONS[1:]):
            raise ValueError("Conditions must be robot first, human second")
        for task in self.tasks:
            _, base = split_task_level(task)
            if base != "TwoRobotPickRemoteControl-v1":
                raise ValueError(
                    "The audited single-arm runner currently supports only "
                    "TwoRobotPickRemoteControl-v1; audit a new task's active arm first"
                )
        if min(self.max_episode_steps, self.execution_horizon) <= 0:
            raise ValueError("Episode and execution horizons must be positive")
        if not np.isfinite(self.prompt_hz) or self.prompt_hz <= 0:
            raise ValueError("prompt_hz must be positive")
        protocol = json.loads(self.protocol_path.read_text())
        if protocol.get("training") is not False:
            raise ValueError("The confirmed video-only, no-training protocol is required")
        ids = tuple(item["id"] for item in protocol["conditions"])
        if ids != CONDITIONS:
            raise ValueError(f"Protocol condition order differs: {ids}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _identity(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = as_numpy(value)
    if result.shape == (1, *shape):
        result = result[0]
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite executing-robot value {shape}, got {result.shape}")
    return result.astype(np.float64, copy=True)


def executing_arm_states(env: Any) -> dict[str, Any]:
    """Read only the executing robots' own kinematic state, never task state."""
    from .control import ArmState
    agents = env.unwrapped.agent.agents_dict
    result = {}
    for uid in ("panda_wristcam-0", "panda_wristcam-1"):
        agent = agents[uid]
        qpos = _single(agent.robot.get_qpos(), (9,))
        result[uid] = ArmState(
            tcp_pose_world=_single(agent.tcp.pose.raw_pose, (7,)),
            base_pose_world=_single(agent.robot.pose.raw_pose, (7,)),
            finger_qpos=qpos[7:9].copy(),
        )
    return result


def observation_frame(
    raw_observation: dict[str, Any], states: dict[str, Any], control: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Whitelist the two RGB views and selected-arm proprioception."""
    sensor_data = raw_observation["sensor_data"]
    external = resize_rgb(sensor_data["zed2i"]["rgb"])
    wrist = resize_rgb(sensor_data[WRIST_CAMERA]["rgb"])
    proprio = control.proprioception(states[ACTIVE_ARM])
    result = {
        "agentview_rgb": external.transpose(2, 0, 1).astype(np.float32) / 255,
        "eye_in_hand_rgb": wrist.transpose(2, 0, 1).astype(np.float32) / 255,
        "ee_pos": np.asarray(proprio["ee_pos"], dtype=np.float32).reshape(3),
        "ee_ori": np.asarray(proprio["ee_ori"], dtype=np.float32).reshape(6),
        "gripper_states": np.asarray(proprio["gripper_states"], dtype=np.float32).reshape(2),
    }
    if set(result) != set(CURRENT_KEYS):
        raise AssertionError("Unexpected policy observation modality")
    return result, np.concatenate((external, wrist), axis=1)


def patch_absolute_panda_controller() -> None:
    """Supply finite Box bounds for the existing, unnormalized EEF controller."""
    from mani_skill.agents.robots.panda.panda import Panda
    if getattr(Panda, "_bpp_tig_finite_ee_bounds", False):
        return
    original = Panda._controller_configs.fget

    def bounded(instance):
        result = original(instance)
        arm = result["pd_ee_pose"]["arm"]
        arm.pos_lower, arm.pos_upper = [-2.0] * 3, [2.0] * 3
        arm.rot_lower, arm.rot_upper = [-float(np.pi)] * 3, [float(np.pi)] * 3
        arm.normalize_action = False
        arm.use_delta = False
        return result

    Panda._controller_configs = property(bounded)
    Panda._bpp_tig_finite_ee_bounds = True


def make_environment(config: RunConfig, task_id: str) -> Any:
    """Use the vendored simulator directly, without baseline training imports."""
    tig_repo = str(config.tig_repo.resolve())
    if tig_repo not in sys.path:
        sys.path.insert(0, tig_repo)
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import configure_dual_task_level

    patch_absolute_panda_controller()
    level, base = split_task_level(task_id)
    configure_dual_task_level(level, mirror_robot_pose=False)
    gym_id = base if level != "L3" else base.removesuffix("-v1") + "L3-v1"
    return gym.make(
        gym_id, num_envs=1, sim_backend="physx_cpu",
        render_backend=config.device if config.device.startswith("cuda") else "gpu",
        control_mode="pd_ee_pose", obs_mode="rgb", render_mode="sensors",
        reconfiguration_freq=1, max_episode_steps=config.max_episode_steps,
        enhanced_determinism=True,
        hi_res=False, wrist_sensor=True,
        sensor_configs={"width": 224, "height": 224, "shader_pack": config.shader},
        human_render_camera_configs={"shader_pack": config.shader},
    )


class RGBRecorder:
    """An execution-only ZED + active-wrist video at the real control Hz."""
    def __init__(self, path: Path, fps: int = 20):
        import av
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = 448, 224
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "24"}
        self.av = av
        self.closed = False

    def append(self, rgb: np.ndarray) -> None:
        frame = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.closed:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.closed = True


def _case_path(config: RunConfig, condition: str, task_id: str, seed: int) -> Path:
    return config.output_dir / condition / task_id / f"seed_{seed}" / "metrics.json"


def run_episode(
    config: RunConfig, model: Any, reference: ReferenceVideo,
    prompt_frames: np.ndarray, condition: str, task_id: str, seed: int,
    experiment_identity: str, *, env_factory: Callable = make_environment,
    control_factory: Callable | None = None,
) -> dict[str, Any]:
    """One independent case. Exceptions remain explicit, never count as success."""
    case = dict(
        experiment_identity=experiment_identity, condition=condition,
        task_id=task_id, seed=int(seed), reference=reference.manifest(),
        reference_selection=config.reference_selection,
        reference_episode_offset=config.reference_episode,
    )
    case_id = _identity(case)
    path = _case_path(config, condition, task_id, seed)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("experiment_identity") != experiment_identity:
            raise ValueError(f"Existing experiment identity differs: {path}")
        if existing.get("status") == "completed":
            if existing.get("case_id") != case_id:
                raise ValueError(f"Existing episode identity differs: {path}")
            return existing
    record = dict(**case, case_id=case_id, status="running", training=False)
    _write_json(path, record)
    env = recorder = None
    started = time.monotonic()
    steps = queries = 0
    success_once = False
    final_success = False
    phase_peaks: dict[str, float] = {}
    control_diagnostics = []
    try:
        if control_factory is None:
            from .control import ControlAdapter
            control_factory = ControlAdapter
        control = control_factory(active_arm=ACTIVE_ARM)
        random.seed(seed)
        np.random.seed(seed)
        model.reset(seed=seed)
        prompt_metadata = model.prepare_video(prompt_frames, fps=reference.fps)
        record["prompt"] = prompt_metadata
        env = env_factory(config, task_id)
        if int(env.unwrapped.control_freq) != 20:
            raise ValueError("This BPP-LIBERO adapter fixes the executing robot control rate at 20 Hz")
        raw_obs, info = env.reset(seed=seed)
        states = executing_arm_states(env)
        control.reset(states)
        first, video_frame = observation_frame(raw_obs, states, control)
        history = deque((first, first), maxlen=2)
        action_queue: deque[np.ndarray] = deque()
        if config.capture_video:
            recorder = RGBRecorder(path.parent / "rollout.mp4", fps=20)
            recorder.append(video_frame)
        for _ in range(config.max_episode_steps):
            if not action_queue:
                prediction = np.asarray(model.get_action(history_batch(history)), dtype=np.float32)
                if prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[2] != 10:
                    raise ValueError(f"Invalid BPP-LIBERO action chunk: {prediction.shape}")
                if not np.isfinite(prediction).all() or prediction.shape[1] < 1:
                    raise ValueError("Empty or non-finite BPP action chunk")
                action_queue.extend(prediction[0, :min(config.execution_horizon, prediction.shape[1])])
                queries += 1
            step = control.step(action_queue.popleft(), states)
            actions = {
                uid: np.asarray(action, dtype=np.float32)[None]
                for uid, action in step.actions.items()
            }
            raw_obs, _, terminated, truncated, info = env.step(actions)
            steps += 1
            current_success = bool(as_numpy(info.get("success", False)).reshape(-1)[0])
            success_once = success_once or current_success
            final_success = current_success
            for key, value in info.items():
                if key.startswith("peak_r_"):
                    phase_peaks[key] = float(as_numpy(value).reshape(-1)[0])
            control_diagnostics.append(_jsonable(step.diagnostics))
            states = executing_arm_states(env)
            frame, video_frame = observation_frame(raw_obs, states, control)
            history.append(frame)  # Every real control step, including inside a chunk.
            if recorder is not None:
                recorder.append(video_frame)
            # Match TIG ignore_terminations evaluation: run to horizon/truncation.
            if bool(as_numpy(truncated).any()):
                break
        # Reward phase trackers can update after BaseEnv builds step info.
        # Read evaluation only for scoring, never as model conditioning.
        final_info = env.unwrapped.evaluate() if hasattr(env.unwrapped, "evaluate") else info
        final_success = bool(as_numpy(final_info.get("success", final_success)).reshape(-1)[0])
        success_once = success_once or final_success
        for key, value in final_info.items():
            if key.startswith("peak_r_"):
                phase_peaks[key] = float(as_numpy(value).reshape(-1)[0])
        record.update(
            status="completed", success_at_end=final_success, success_once=success_once,
            executed_control_steps=steps, policy_queries=queries,
            simulated_seconds=steps / 20.0, phase_peaks=phase_peaks,
            control_diagnostics=control_diagnostics,
        )
    except Exception as exc:
        record.update(
            status="error", error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), executed_control_steps=steps,
            policy_queries=queries,
        )
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                record.update(status="error", video_error=str(exc))
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                record["environment_close_error"] = str(exc)
        try:
            model.reset()
        except Exception as exc:
            record.update(status="error", model_reset_error=str(exc))
        record["wall_seconds"] = time.monotonic() - started
        _write_json(path, record)
    return record


def run(config: RunConfig) -> dict[str, Any]:
    config.validate()
    reference_episodes = {
        seed: reference_episode_for_seed(
            seed, selection=config.reference_selection,
            reference_episode=config.reference_episode,
        )
        for seed in config.seeds
    }
    expected_assets = Path(os.environ.get("MS_ASSET_DIR", "")) / "data" / "robotwin"
    if not os.environ.get("MS_ASSET_DIR"):
        raise RuntimeError("Source /gaozt-test1/sxh/bpp/storage.env before evaluation")
    from .model import FrozenBPPModel
    model = FrozenBPPModel(
        config.checkpoint, device=config.device, bpp_repo=config.bpp_repo,
        amp_dtype=config.amp_dtype, prompt_hz=config.prompt_hz,
    )
    from .control import ControlAdapter
    bridge = ControlAdapter(active_arm=ACTIVE_ARM)
    source_root = Path(__file__).resolve().parent
    constants = _jsonable(asdict(config))
    for key in ("conditions", "output_dir"):
        constants.pop(key)
    constants.update(
        checkpoint_sha256=file_sha256(config.checkpoint),
        tig_commit=_git_revision(config.tig_repo), bpp_commit=_git_revision(config.bpp_repo),
        active_arm=ACTIVE_ARM, inactive_arm="panda_wristcam-1", inactive_arm_behavior="hold initial TCP and gripper",
        control_mode="pd_ee_pose", control_hz=20, physics_hz=100,
        render_backend=config.device if config.device.startswith("cuda") else "gpu",
        enhanced_determinism=True,
        controller_position_guard_base_m=[[-2, -2, -2], [2, 2, 2]],
        control_bridge={
            "policy_from_world": bridge.policy_from_world,
            "tcp_to_libero_eef": bridge.tcp_to_libero_eef,
            "osc_output_max": bridge.osc_output_max,
            "workspace_bounds": bridge.workspace_bounds,
            "current_proprioception_frame": (
                "policy_from_world @ TIG world TCP @ nominal TCP-to-LIBERO gripsite"
            ),
            "world_alignment_note": (
                "Identity policy_from_world does not calibrate TIG's table/base "
                "coordinates to the LIBERO training scene"
            ),
            "gripper_bridge": (
                "Directional Panda position targets; not MuJoCo's integrated "
                "gripper actuator/torque dynamics"
            ),
        },
        adapter_sources_sha256={
            name: file_sha256(source_root / name)
            for name in ("data.py", "runner.py", "model.py", "control.py")
        },
        mirror_robot_pose=False,
        current_observation_keys=list(CURRENT_KEYS),
        current_cameras=["zed2i", WRIST_CAMERA],
        current_camera_extension=True,
        current_camera_note="Added active-arm wrist RGB to TIG's ZED stream for BPP-LIBERO; identical in both conditions",
        prompt_modalities=["RGB"], prompt_camera="zed2i",
        robot_reference_level="L0", task_labels_supplied_to_policy=False,
        reference_schedule={
            "selection": config.reference_selection,
            "reference_episode_offset": config.reference_episode,
            "seed_origin": REFERENCE_SEED_ORIGIN if config.reference_selection == "per_seed" else None,
            "rule": (
                "reference_episode + (seed - 100000); no modulo or list reindexing"
                if config.reference_selection == "per_seed" else "reference_episode"
            ),
            "seed_to_reference_episode": reference_episodes,
            "shared_across_task_levels_and_conditions": True,
            "missing_reference_behavior": "error; never substitute another episode",
        },
        demonstration_numerical_features_loaded=False, training=False,
        asset_root=str(expected_assets),
        model_info=getattr(model, "model_info", {}),
        scoring_primary="success_at_end", scoring_secondary="success_once",
    )
    identity = _identity(constants)
    manifest = dict(
        schema="bpp_tig_video_only_v1", experiment_identity=identity,
        condition_order=list(CONDITIONS), constants=constants,
    )
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("experiment_identity") != identity:
            raise ValueError(f"Output directory belongs to another experiment: {config.output_dir}")
    _write_json(manifest_path, manifest)
    rows = []
    for condition in config.conditions:
        if condition == "human_video_only":
            missing = []
            for task in config.tasks:
                for seed in config.seeds:
                    path = _case_path(config, "robot_video_only", task, seed)
                    record = json.loads(path.read_text()) if path.is_file() else {}
                    if record.get("status") != "completed" or record.get("experiment_identity") != identity:
                        missing.append((task, seed))
            if missing:
                raise RuntimeError(f"Complete matching robot-video cases before human-video cases: {missing}")
        references: dict[tuple[str, int], tuple[ReferenceVideo, np.ndarray]] = {}
        for task in config.tasks:
            _, base = split_task_level(task)
            for seed in config.seeds:
                episode = reference_episodes[seed]
                reference_key = (base, episode)
                try:
                    if reference_key not in references:
                        reference = select_reference(
                            config.dataset_root, config.task_mapping, task, condition,
                            episode, dataset_revision=config.dataset_revision,
                        )
                        references[reference_key] = (reference, read_reference_rgb(reference))
                except Exception as exc:
                    row = dict(
                        experiment_identity=identity, condition=condition,
                        task_id=task, seed=seed, status="error", training=False,
                        reference_selection=config.reference_selection,
                        reference_episode=episode,
                        phase="reference_loading", error_type=type(exc).__name__,
                        error=str(exc), traceback=traceback.format_exc(),
                    )
                    path = _case_path(config, condition, task, seed)
                    if path.is_file():
                        previous = json.loads(path.read_text())
                        if previous.get("status") == "completed":
                            # Never overwrite an immutable completed result.
                            raise RuntimeError(f"Cannot reload reference for completed case: {path}") from exc
                    _write_json(path, row)
                    rows.append(row)
                    print(json.dumps({
                        "condition": condition, "task": task, "seed": seed,
                        "reference_episode": episode, "status": "error",
                        "phase": "reference_loading", "error": str(exc),
                    }), flush=True)
                    continue
                reference, frames = references[reference_key]
                row = run_episode(config, model, reference, frames, condition, task, seed, identity)
                rows.append(row)
                print(json.dumps({
                    "condition": condition, "task": task, "seed": seed,
                    "reference_episode": episode,
                    "status": row["status"], "success_at_end": row.get("success_at_end"),
                    "success_once": row.get("success_once"), "error": row.get("error"),
                }), flush=True)
    completed = [row for row in rows if row["status"] == "completed"]
    groups = {}
    for condition in config.conditions:
        selected = [row for row in completed if row["condition"] == condition]
        errors = [row for row in rows if row["condition"] == condition and row["status"] != "completed"]
        groups[condition] = dict(
            completed=len(selected), errors=len(errors),
            planned=len(config.tasks) * len(config.seeds),
            success_at_end=(
                float(np.mean([row["success_at_end"] for row in selected]))
                if selected and not errors else None
            ),
            success_once=(
                float(np.mean([row["success_once"] for row in selected]))
                if selected and not errors else None
            ),
            success_at_end_completed_only=(
                float(np.mean([row["success_at_end"] for row in selected])) if selected else None
            ),
            complete=not errors and len(selected) == len(config.tasks) * len(config.seeds),
        )
    summary = dict(
        experiment_identity=identity, variant="BPP video-only inference variant",
        training=False, conditions=groups, completed=len(completed), errors=len(rows) - len(completed),
    )
    _write_json(config.output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=RunConfig.dataset_root)
    parser.add_argument("--tig-repo", type=Path, default=RunConfig.tig_repo)
    parser.add_argument("--bpp-repo", type=Path, default=RunConfig.bpp_repo)
    parser.add_argument("--tasks", nargs="+", default=list(RunConfig.tasks))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(RunConfig.seeds))
    parser.add_argument("--condition", choices=(*CONDITIONS, "both"), default=CONDITIONS[0])
    parser.add_argument(
        "--reference-selection", choices=REFERENCE_SELECTIONS, default="per_seed",
        help="per_seed: offset + seed - 100000; fixed: one episode for every seed",
    )
    parser.add_argument(
        "--reference-episode", type=int, default=0,
        help="Non-negative reference episode offset (per_seed) or episode index (fixed)",
    )
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--execution-horizon", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shader", default="minimal")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--prompt-hz", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--protocol-path", type=Path, default=RunConfig.protocol_path)
    values = vars(parser.parse_args(argv))
    condition = values.pop("condition")
    values["conditions"] = CONDITIONS if condition == "both" else (condition,)
    values["capture_video"] = not values.pop("no_video")
    for key in ("tasks", "seeds"):
        values[key] = tuple(values[key])
    return RunConfig(**values)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2), flush=True)
    return int(summary["errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

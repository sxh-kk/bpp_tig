"""Audit the released Seen/held-out base+level environments without a BPP model.

Each case performs a seeded reset, reads real ZED RGB, and executes one joint
hold control step. This checks assets/rendering/controllers, not policy success.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image

# Preserve the official evaluator's native-library import order and lifetime.
from . import native_eval as evaluation
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils


def flags():
    return {"L1": L0_L3_utils.is_l1_enabled(), "L2": L0_L3_utils.is_l2_enabled(),
            "L3": L0_L3_utils.is_l3_enabled(),
            "mirror_robot_pose": L0_L3_utils.is_lr_mirror_robot_pose_enabled()}


def audit_case(config, task, seed):
    import torch
    start = time.monotonic()
    record = {"task_id": task, "seed": seed, "status": "running", "policy_loaded": False,
              "control_mode": "pd_joint_pos", "shader": config.shader,
              "dispatch": "official base Gym ID + level flags", "control_steps": 0}
    path = config.output_dir / task / "metrics.json"
    evaluation._write_json(path, record)
    env = None
    raw = info = current = rgb = action = prediction = after = None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = evaluation.make_native_environment(config, task)
        base = env.unwrapped
        level, gym_id = evaluation.split_task_level(task)
        record.update(actual_environment_class=type(base).__module__ + "." + type(base).__qualname__,
                      gym_id=gym_id, actual_flags=flags(),
                      sim_hz=int(base.sim_freq), control_hz=int(base.control_freq))
        assert record["sim_hz"] == 100 and record["control_hz"] == 20
        expected = {"L1": level == "L1", "L2": level == "L2", "L3": level == "L3",
                    "mirror_robot_pose": False}
        assert record["actual_flags"] == expected, (record["actual_flags"], expected)
        assert ".dual_tasks_l3." not in type(base).__module__
        module = sys.modules[type(base).__module__]
        record["task_source"] = str(Path(module.__file__).resolve())
        record["task_source_sha256"] = evaluation.file_sha256(module.__file__)
        record["model_attributes"] = {key: value for key, value in vars(base).items()
            if ("modelname" in key or "model_id" in key or key.endswith("_mass"))
            and isinstance(value, (str, int, float, bool, np.integer, np.floating))}
        raw, info = env.reset(seed=seed)
        current, rgb = evaluation.native_observation(raw, env)
        assert current["qpos_gripper_states"].shape == (18,)
        assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
        assert float(rgb.std()) > 1.0
        Image.fromarray(rgb).save(path.parent / "initial_rgb.png")
        record["initial_rgb"] = {"shape": list(rgb.shape), "std": float(rgb.std()),
                                  "sha256": __import__("hashlib").sha256(rgb.tobytes()).hexdigest()}
        q = current["qpos_gripper_states"]
        prediction = np.concatenate((q[:7], [1.], q[9:16], [1.])).astype(np.float32)
        action, clip_counts = evaluation.joint_actions(prediction, env)
        raw, reward, terminated, truncated, info = env.step(action)
        record["control_steps"] = 1
        after, rgb = evaluation.native_observation(raw, env)
        assert np.isfinite(after["qpos_gripper_states"]).all() and float(rgb.std()) > 1.0
        Image.fromarray(rgb).save(path.parent / "hold_step_rgb.png")
        record.update(status="passed", final_rgb_std=float(rgb.std()),
                      clip_counts_per_dimension=clip_counts.tolist(),
                      flags_after_step=flags(), wrist_camera=False)
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    finally:
        raw = info = current = rgb = action = prediction = after = None
        gc.collect()
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                record.update(status="error", close_error=str(exc))
        env = None
        gc.collect()
        record["wall_seconds"] = time.monotonic() - start
        evaluation._write_json(path, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args(argv)
    config = evaluation.EvalConfig(Path("unused-audit-checkpoint"), args.output_dir, "zero_shot",
        device=args.device, shader=args.shader, capture_video=False, max_episode_steps=500,
        allow_pilot=True)
    _, seen = evaluation.official_tasks(config.tig_repo, "seen")
    _, heldout = evaluation.official_tasks(config.tig_repo, "zero_shot")
    tasks = seen + heldout
    assert len(tasks) == len(set(tasks)) == 40
    import torch
    torch.set_num_threads(1)
    manifest = {"schema": "bpp_tig_base_level_asset_audit_v1", "training": False,
        "policy_loaded": False, "tasks": list(tasks), "seed": args.seed,
        "shader": args.shader, "sim_backend": "physx_cpu", "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable, "torch": torch.__version__,
        "native_eval_sha256": evaluation.file_sha256(evaluation.__file__),
        "audit_source_sha256": evaluation.file_sha256(__file__),
        "tig_revision": evaluation._revision(config.tig_repo), "rows": []}
    path = args.output_dir / "audit.json"
    for task in tasks:
        row = audit_case(config, task, args.seed)
        manifest["rows"].append(row)
        evaluation._write_json(path, manifest)
        print(json.dumps({key: row.get(key) for key in ("task_id", "status", "control_steps", "wall_seconds")}), flush=True)
    manifest["passed"] = sum(row["status"] == "passed" for row in manifest["rows"])
    manifest["errors"] = 40 - manifest["passed"]
    manifest["all_scene_checks_passed_before_process_exit"] = manifest["passed"] == 40
    evaluation._write_json(path, manifest)
    return 0 if manifest["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

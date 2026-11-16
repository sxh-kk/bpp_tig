"""Safety of native evaluation observations, scores and reproducible dispatch."""
from collections import deque
from dataclasses import replace
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from examples.baselines.bpp.bpp.report import V2_RECIPE
from examples.baselines.bpp.bpp import native_eval as evaluation

TMP = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp")) / "tmp"
TASK = "L0_TwoRobotPickRemoteControl-v1"


class FakeEnvironment:
    sim_freq, control_freq = 100, 20

    def __init__(self, config, task):
        self.unwrapped = self
        self.step_index = 0
        self.horizon = config.max_episode_steps
        self.actions = []
        self.closed = False
        self.agent = types.SimpleNamespace(agents_dict={
            uid: types.SimpleNamespace(robot=types.SimpleNamespace(
                get_qpos=lambda arm=i: np.full((1, 9), arm + self.step_index / 10, np.float32)))
            for i, uid in enumerate(evaluation.ARMS)
        })
        self.action_space = {
            uid: types.SimpleNamespace(low=np.full(8, -1.), high=np.full(8, 1.))
            for uid in evaluation.ARMS
        }

    def observation(self):
        return {"sensor_data": {"zed2i": {"rgb": np.full((1, 224, 224, 3), self.step_index, np.uint8)}},
                "extra": {"goal_pose": np.full(18, 9001.)},
                "states": np.full((1, 18), 9001.), "success": True}

    def reset(self, seed):
        self.seed = seed
        return self.observation(), self.evaluate()

    def evaluate(self):
        return {"success": np.array([self.step_index == 1])}

    def step(self, action):
        self.actions.append(action)
        self.step_index += 1
        return self.observation(), np.array([self.step_index / 10]), np.array([True]), \
            np.array([self.step_index == self.horizon]), self.evaluate()

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.inputs = []
        self.resets = 0
        def reset():
            self.resets += 1
        self.policy = types.SimpleNamespace(obs_encoder=types.SimpleNamespace(reset=reset))

    def eval(self):
        return self

    def predict_action(self, current, prompt_features):
        if self.fail:
            raise RuntimeError("injected inference failure")
        self.inputs.append(current)
        result = np.zeros((1, 16, 16), np.float32)
        result[:, :, 0] = 4.0
        return result


class NativeEvaluationTests(unittest.TestCase):
    def test_observation_whitelist_and_history_use_only_measured_robot_state(self):
        config = evaluation.EvalConfig(Path("fake.pt"), Path("fake"), "zero_shot")
        env = FakeEnvironment(config, TASK)
        first, _ = evaluation.native_observation(env.observation(), env)
        self.assertEqual(set(first), evaluation.CURRENT_KEYS)
        np.testing.assert_array_equal(first["qpos_gripper_states"], [0.] * 9 + [1.] * 9)
        env.step_index = 1
        second, _ = evaluation.native_observation(env.observation(), env)
        current = evaluation.current_history(deque([first, second]))
        self.assertEqual(current["agentview_rgb"].shape, (1, 2, 3, 224, 224))
        self.assertEqual(current["agentview_rgb"].dtype, np.uint8)
        self.assertFalse(np.any(current["qpos_gripper_states"] == 9001))
        with self.assertRaises(ValueError):
            evaluation.current_history([dict(first, privileged=np.zeros(2)), second])

    def test_shards_are_disjoint_and_preserve_episode_reference_index(self):
        tasks = ["L0_TwoRobotPickRemoteControl-v1", "L1_TwoRobotPickRemoteControl-v1"]
        whole = evaluation.evaluation_cases(tasks, 10)
        shards = [evaluation.evaluation_cases(tasks, 10, 3, rank) for rank in range(3)]
        flattened = [case for shard in shards for case in shard]
        self.assertEqual(set(flattened), set(whole))
        self.assertEqual(len(flattened), len(set(flattened)))
        for shard in shards:
            for task, index in shard:
                self.assertEqual(evaluation.SEED_ORIGIN + index, 100000 + index)
        with self.assertRaises(ValueError):
            evaluation.evaluation_cases(tasks, 10, 2, 2)

    def test_closed_loop_records_lost_success_and_controller_clipping(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            config = evaluation.EvalConfig(Path("fake.pt"), Path(folder), "zero_shot",
                max_episode_steps=3, execution_horizon=2, capture_video=False, device="cpu", allow_pilot=True)
            model = FakeModel()
            envs = []
            def create(cfg, task):
                env = FakeEnvironment(cfg, task)
                envs.append(env)
                return env
            record = evaluation.run_episode(config, model, TASK, 100000, None,
                {"episode_index": 0}, {"tokens": "opaque"}, "identity", env_factory=create)
            self.assertEqual(record["status"], "completed")
            self.assertFalse(record["success_at_end"])
            self.assertTrue(record["success_once"])
            self.assertEqual(record["executed_control_steps"], 3)
            self.assertEqual(record["policy_queries"], 2)
            self.assertEqual(record["clipped_control_steps"], 3)
            self.assertEqual(record["clip_counts_per_action_dimension"][0], 3)
            np.testing.assert_allclose(model.inputs[1]["qpos_gripper_states"][0, :, 0], [.1, .2])
            self.assertTrue(envs[0].closed)
            self.assertEqual(envs[0].actions[0][evaluation.ARMS[0]].shape, (1, 8))
            self.assertEqual(envs[0].actions[0][evaluation.ARMS[0]][0, 0], 1.)
            self.assertAlmostEqual(record["last_normalized_reward"], .3)
            self.assertIsNone(evaluation.summarize([record], 1, is_pilot=True)["success_at_end_rate"])
            previous_queries = len(model.inputs)
            reused = evaluation.run_episode(config, model, TASK, 100000, None,
                {"episode_index": 0}, {"tokens": "opaque"}, "identity", env_factory=create)
            self.assertEqual(reused["case_id"], record["case_id"])
            self.assertEqual(len(model.inputs), previous_queries)
            with self.assertRaises(ValueError):
                evaluation.run_episode(config, model, TASK, 100000, None,
                    {"episode_index": 1}, {}, "identity", env_factory=create)

    def test_errors_invalidate_main_success_rate_and_persist_nulls(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            config = evaluation.EvalConfig(Path("fake.pt"), Path(folder), "zero_shot",
                max_episode_steps=3, capture_video=False, device="cpu", allow_pilot=True)
            record = evaluation.run_episode(config, FakeModel(fail=True), TASK, 100000, None,
                {"episode_index": 0}, {}, "identity", env_factory=FakeEnvironment)
            self.assertEqual(record["status"], "error")
            self.assertIsNone(record["success_at_end"])
            self.assertIsNone(record["success_once"])
            stored = json.loads(evaluation.case_path(config, TASK, 100000).read_text())
            self.assertIsNone(stored["last_normalized_reward"])
            summary = evaluation.summarize([record], 1, is_pilot=False)
            self.assertIsNone(summary["success_at_end_rate"])
            self.assertTrue(summary["incomplete_or_errors"])
            success = {"status": "completed", "success_at_end": True, "success_once": True}
            self.assertIsNone(evaluation.summarize([success], 2, is_pilot=False)["success_at_end_rate"])
            self.assertEqual(evaluation.summarize([success], 1, is_pilot=False)["success_at_end_rate"], 1.)

    def test_official_facade_uses_final_info_and_final_rgb_after_autoreset(self):
        reset_rgb = np.zeros((1, 1, 224, 224, 3), np.uint8)
        final_rgb = np.full((1, 224, 224, 3), 123, np.uint8)
        base = types.SimpleNamespace(evaluate=lambda: {"success": False})
        def step(actions):
            final_info = {"success": True, "episode": {"success_at_end": True}}
            info = {"final_info": np.array([final_info], dtype=object),
                    "final_observation": np.array([{"rgb": final_rgb}], dtype=object)}
            return {"rgb": reset_rgb}, np.array([1.]), np.array([False]), np.array([True]), info
        vector = types.SimpleNamespace(envs=[types.SimpleNamespace(unwrapped=base)],
            action_space={}, step=step)
        facade = evaluation.OfficialEnvironment(vector)
        raw, reward, terminated, truncated, info = facade.step({})
        self.assertTrue(info["success"])
        self.assertFalse(facade.unwrapped.evaluate()["success"])
        self.assertTrue(facade.final_evaluation()["success"])
        self.assertEqual(int(raw["sensor_data"]["zed2i"]["rgb"][0, 0, 0]), 123)
        self.assertFalse(np.shares_memory(raw["sensor_data"]["zed2i"]["rgb"], final_rgb))

    def test_native_process_crash_preserves_rollout_evidence_but_invalidates_score(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            config = evaluation.EvalConfig(Path("fake.pt"), Path(folder), "zero_shot",
                tasks=(TASK,), num_episodes=1, allow_pilot=True)
            path = evaluation.case_path(config, TASK, 100000)
            evaluation._write_json(path, {
                "experiment_identity": "identity", "task_id": TASK, "seed": 100000,
                "status": "completed", "success_at_end": True, "success_once": True,
                "last_normalized_reward": 1., "reference_source": "human"})
            evaluation._write_json(Path(folder) / "evaluation_manifest.json", {"tasks": [TASK]})
            evaluation._write_json(Path(folder) / "shard_000_of_001.json", {
                "attempt_id": "current", "experiment_identity": "identity", "cases": [[TASK, 0]]})
            pending = evaluation.aggregate_results(config, "identity", [TASK])
            self.assertTrue(pending["process_exit_pending"])
            evaluation.record_process_exit(config, -11, "current", 2.)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "error")
            self.assertIsNone(record["success_at_end"])
            self.assertTrue(record["rollout_metrics_before_process_error"]["success_at_end"])
            self.assertEqual(record["process_returncode"], -11)
            summary = json.loads((Path(folder) / "summary.json").read_text())
            self.assertIsNone(summary["success_at_end_rate"])

    def test_concurrent_processes_and_threads_atomically_share_manifest(self):
        writer = """
import sys
from concurrent.futures import ThreadPoolExecutor
from examples.baselines.bpp.bpp.native_eval import _write_json
path, tag = sys.argv[1:]
def write(index):
    _write_json(path, {"experiment_identity": "same-experiment", "writer": tag,
                      "iteration": index, "payload": tag * 8192})
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(write, range(100)))
"""
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            path = Path(folder) / "evaluation_manifest.json"
            processes = [subprocess.Popen([sys.executable, "-c", writer, str(path), str(i)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(2)]
            reads = 0
            deadline = time.monotonic() + 90
            try:
                while any(p.poll() is None for p in processes):
                    self.assertLess(time.monotonic(), deadline, "Concurrent writers timed out")
                    if path.exists():
                        record = json.loads(path.read_text())
                        self.assertEqual(record["experiment_identity"], "same-experiment")
                        self.assertEqual(record["payload"], record["writer"] * 8192)
                        reads += 1
                    time.sleep(.001)
                for process in processes:
                    _, error = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 0, error.decode())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.communicate(timeout=5)
            self.assertGreater(reads, 0)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])
            final = json.loads(path.read_text())
            self.assertEqual(final["payload"], final["writer"] * 8192)

    def test_formal_checkpoint_contract_rejects_wrong_regime_prompt_and_pilot(self):
        config = evaluation.EvalConfig(Path("fake.pt"), Path("fake"), "zero_shot")
        metadata = {**V2_RECIPE, "regime": "c15", "pilot": False, "completed": True,
                    "max_updates": 45000, "train_step": 45000, "prompt_source": "human"}
        self.assertEqual(evaluation.checkpoint_reasons(metadata, config, [TASK], [TASK]), [])
        for change in ({"regime": "scratch"}, {"pilot": True}, {"completed": False},
                       {"prompt_source": "robot"}, {"train_step": 20}):
            self.assertTrue(evaluation.checkpoint_reasons(dict(metadata, **change), config, [TASK], [TASK]))
        self.assertTrue(evaluation.checkpoint_reasons(metadata, replace(config, num_episodes=1), [TASK], [TASK]))
        self.assertEqual(evaluation.checkpoint_reasons(dict(metadata, regime="c45", train_step=135000, max_updates=135000),
            replace(config, pretraining_scale="c45"), [TASK], [TASK]), [])
        ft = dict(metadata, regime="finetune", train_step=3000, max_updates=3000,
                  pretraining_scale="c30", initialize_from_sha256="a" * 64, initialize_from_weights="ema",
                  initialize_from_metadata=dict(metadata, regime="c30", pretraining_scale="c30",
                                                train_step=90000, max_updates=90000))
        self.assertTrue(evaluation.checkpoint_reasons(ft,
            replace(config, regime="pretrain_finetune", pretraining_scale="c15"), [TASK], [TASK]))
        self.assertEqual(evaluation.checkpoint_reasons(ft,
            replace(config, regime="pretrain_finetune", pretraining_scale="c30"), [TASK], [TASK]), [])
        for incorrect in ({"max_updates": 45000}, {"completed": False},
                          {"prompt_source": "robot"}, {"pretraining_scale": "c15"}):
            bad_source = dict(ft["initialize_from_metadata"], **incorrect)
            self.assertTrue(evaluation.checkpoint_reasons(dict(ft, initialize_from_metadata=bad_source),
                replace(config, regime="pretrain_finetune", pretraining_scale="c30"), [TASK], [TASK]))


if __name__ == "__main__":
    unittest.main()

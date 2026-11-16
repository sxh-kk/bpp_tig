"""CPU fixtures for fixed pairing, inference transparency and process-exit gating."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from examples.baselines.bpp.bpp import prompt_diagnostic as diagnostic

TMP = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp")) / "tmp"
TIG = Path(os.environ.get("BPP_REPOS_DIR", "/gaozt-test1/sxh/bpp/repos")) / "The-Imitator-Game"


class FakePredictor:
    action_scale = np.full(16, 2., dtype=np.float32)
    action_offset = np.full(16, -.5, dtype=np.float32)

    def __init__(self):
        self.calls = 0

    def predict_action(self, current, prompt_features):
        self.calls += 1
        return torch.randn((1, 16, 16)) + prompt_features


class PromptDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TMP)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = diagnostic.DiagnosticConfig(Path("future_final.pt"), self.output, tig_repo=TIG, device="cpu")
        self.plan = diagnostic.build_plan(TIG)

    def populate_complete(self):
        identity = "fixture-identity"
        diagnostic.evaluation._write_json(self.output / "diagnostic_manifest.json",
            {"experiment_identity": identity, "plan": self.plan})
        for case in self.plan["cases"]:
            record = {"experiment_identity": identity, **case, "diagnostic_only": True,
                "status": "completed", "success_at_end": case["condition"] == "correct", "success_once": True,
                "last_normalized_reward": .7, "process_exit_verified": True, "process_returncode": 0,
                "executed_control_steps": 500, "diagnostic_attempt_id": "previous-valid"}
            diagnostic.evaluation._write_json(diagnostic.case_path(self.config, case), record)
        return identity

    def test_fixed_cohort_cycle_and_stable_disjoint_shards(self):
        self.assertEqual(self.plan["case_count"], 20)
        self.assertEqual(len(self.plan["task_order"]), 5)
        self.assertTrue(all(task.startswith("L0_") for task in self.plan["task_order"]))
        mapping = self.plan["mismatch_mapping"]
        self.assertEqual(set(mapping), set(mapping.values()))
        self.assertTrue(all(task != other for task, other in mapping.items()))
        expected = {json.dumps(row, sort_keys=True) for row in self.plan["cases"]}
        for count in (1, 8, 16, 20):
            rows = [case for rank in range(count) for case in diagnostic.shard_cases(self.plan, count, rank)]
            self.assertEqual(len(rows), 20)
            self.assertEqual({json.dumps(row, sort_keys=True) for row in rows}, expected)
        for case in self.plan["cases"]:
            self.assertEqual(case["episode_index"], case["seed"] - 100000)
        with self.assertRaises(ValueError): diagnostic.shard_cases(self.plan, 21, 0)
        with self.assertRaises(ValueError): diagnostic.shard_cases(self.plan, 8, 8)

    def test_declaration_is_immutable_and_checkpoint_gate_requires_completed_c15(self):
        path = self.output / "plan.json"
        diagnostic._immutable_json(path, self.plan)
        diagnostic._immutable_json(path, self.plan)
        with self.assertRaises(ValueError): diagnostic._immutable_json(path, dict(self.plan, case_count=10))
        good = {"regime": "c15", "pilot": False, "completed": True, "max_updates": 45000,
                "train_step": 45000, "prompt_source": "human"}
        diagnostic.validate_checkpoint(good, self.config)
        for changes in ({"pilot": True}, {"train_step": 20}, {"regime": "scratch"}, {"prompt_source": "robot"}):
            with self.assertRaises(ValueError): diagnostic.validate_checkpoint(dict(good, **changes), self.config)

    def test_first_query_is_transparent_and_pairing_uses_equal_noise(self):
        current = {"agentview_rgb": np.zeros((1, 2, 3, 224, 224), np.uint8),
                   "qpos_gripper_states": np.zeros((1, 2, 18), np.float32)}
        captures, predictions = [], []
        for index, prompt in enumerate((0., 1.)):
            inner = FakePredictor()
            proxy = diagnostic.FirstQueryRecorder(inner, self.output / f"first_{index}.npz", "cpu")
            torch.manual_seed(42)
            result = proxy.predict_action(current, prompt_features=prompt)
            self.assertEqual(inner.calls, 1)
            self.assertIsInstance(result, torch.Tensor)
            predictions.append(result)
            captures.append(proxy.first_query)
            with np.load(proxy.first_query["path"], allow_pickle=False) as value:
                np.testing.assert_array_equal(value["action_raw"], result.numpy())
                np.testing.assert_allclose(value["action_normalized"], result.numpy() * 2 - .5)
        self.assertEqual(captures[0]["torch_rng_sha256"], captures[1]["torch_rng_sha256"])
        self.assertEqual(captures[0]["current_observation_sha256"], captures[1]["current_observation_sha256"])
        np.testing.assert_allclose((predictions[1] - predictions[0]).numpy(), 1., atol=3e-7)

    def test_nonzero_native_exit_preserves_evidence_and_invalidates_entire_cohort(self):
        self.populate_complete()
        case = self.plan["cases"][0]
        path = diagnostic.case_path(self.config, case)
        record = diagnostic._read(path)
        record["diagnostic_attempt_id"] = "this-attempt"
        record["process_exit_verified"] = False
        diagnostic.evaluation._write_json(path, record)
        pending = diagnostic.aggregate_results(self.config)
        self.assertTrue(pending["incomplete_or_errors"])
        result = diagnostic.record_process_exit(self.config, [case], "this-attempt", -11, 2.)
        self.assertTrue(result["incomplete_or_errors"])
        self.assertIsNone(result["formal_success_rate"])
        for condition in diagnostic.CONDITIONS:
            self.assertIsNone(result["conditions"][condition]["diagnostic_success_at_end_rate"])
        record = diagnostic._read(path)
        self.assertIsNone(record["success_at_end"])
        self.assertTrue(record["rollout_metrics_before_process_error"]["success_at_end"])
        self.assertEqual(record["process_returncode"], -11)
        self.assertEqual(record["status"], "error")

    def test_unverified_completed_case_is_archived_instead_of_trusted_on_resume(self):
        identity = self.populate_complete()
        case = self.plan["cases"][0]
        path = diagnostic.case_path(self.config, case)
        self.assertIsNotNone(diagnostic._archive_unverified(path, identity))
        record = diagnostic._read(path)
        record["process_exit_verified"] = False
        query = path.parent / "first_query.npz"
        video = path.parent / "rollout.mp4"
        query.write_bytes(b"first-query-fixture")
        video.write_bytes(b"video-fixture")
        record["first_query"] = {"path": str(query), "sha256": diagnostic.file_sha256(query)}
        diagnostic.evaluation._write_json(path, record)
        self.assertIsNone(diagnostic._archive_unverified(path, identity))
        self.assertFalse(path.exists())
        archives = list((path.parent / "previous_attempts").glob("*/metrics.json"))
        self.assertEqual(len(archives), 1)
        archived = diagnostic._read(archives[0])
        self.assertEqual(archived["status"], "completed")
        self.assertFalse(query.exists())
        self.assertFalse(video.exists())
        self.assertEqual(Path(archived["first_query"]["path"]).read_bytes(), b"first-query-fixture")
        self.assertEqual(Path(archived["video_path"]).read_bytes(), b"video-fixture")

    def test_claim_catches_native_crash_before_result_annotation(self):
        self.populate_complete()
        case = self.plan["cases"][0]
        path = diagnostic.case_path(self.config, case)
        record = diagnostic._read(path)
        record.pop("diagnostic_attempt_id")
        record["process_exit_verified"] = False
        diagnostic.evaluation._write_json(path, record)
        diagnostic.evaluation._write_json(path.parent / "attempt_claim.json",
            {"experiment_identity": "fixture-identity", "diagnostic_attempt_id": "claimed-attempt", **case})
        result = diagnostic.record_process_exit(self.config, [case], "claimed-attempt", -11, 1.)
        self.assertTrue(result["incomplete_or_errors"])
        self.assertEqual(diagnostic._read(path)["process_returncode"], -11)
        self.assertIsNone(diagnostic._read(path)["success_at_end"])

    def test_first_action_difference_requires_exact_observation_and_rng_pair(self):
        self.populate_complete()
        for case in self.plan["cases"][:2]:
            path = diagnostic.case_path(self.config, case)
            record = diagnostic._read(path)
            value = 0. if case["condition"] == "correct" else 1.
            array_path = path.parent / "first_query.npz"
            np.savez_compressed(array_path, action_raw=np.full((1, 16, 16), value),
                                action_normalized=np.full((1, 16, 16), value * 2))
            record["first_query"] = {"path": str(array_path), "sha256": diagnostic.file_sha256(array_path),
                "current_observation_sha256": "same-obs", "torch_rng_sha256": "same-rng"}
            diagnostic.evaluation._write_json(path, record)
        result = diagnostic.aggregate_results(self.config)
        self.assertFalse(result["incomplete_or_errors"])
        self.assertIsNone(result["formal_success_rate"])
        self.assertEqual(result["conditions"]["correct"]["diagnostic_success_at_end_rate"], 1.)
        pair = result["pairs"][0]
        self.assertTrue(pair["first_query_same_observation_and_rng"])
        self.assertEqual(pair["action_raw_rms_difference"], 1.)
        self.assertEqual(pair["action_normalized_rms_difference"], 2.)
        path = diagnostic.case_path(self.config, self.plan["cases"][1])
        record = diagnostic._read(path)
        record["first_query"]["torch_rng_sha256"] = "different-rng"
        diagnostic.evaluation._write_json(path, record)
        pair = diagnostic.aggregate_results(self.config)["pairs"][0]
        self.assertFalse(pair["first_query_same_observation_and_rng"])
        self.assertIsNone(pair["action_raw_rms_difference"])
        self.assertIsNone(pair["action_normalized_rms_difference"])
        self.assertEqual(pair["first_query_comparison_unavailable_reasons"], ["torch_rng_sha256_mismatch"])
        record["first_query"]["torch_rng_sha256"] = "same-rng"
        diagnostic.evaluation._write_json(path, record)
        Path(record["first_query"]["path"]).write_bytes(b"corrupted-artifact")
        with self.assertRaises(ValueError): diagnostic.aggregate_results(self.config)
        invalid = diagnostic._read(self.output / "diagnostic_summary.json")
        self.assertTrue(invalid["incomplete_or_errors"])
        self.assertIsNone(invalid["conditions"]["correct"]["diagnostic_success_at_end_rate"])

    def test_supervisor_records_worker_failure_before_model_load(self):
        with patch.object(diagnostic.subprocess, "run", return_value=types.SimpleNamespace(returncode=-11)) as run:
            code = diagnostic.supervise(self.config, ["--checkpoint", "future_final.pt", "--output-dir", str(self.output)])
        self.assertEqual(code, 2)
        self.assertIn("--_worker", run.call_args.args[0])
        status = diagnostic._read(self.output / "process_status_shard_000_of_001.json")
        self.assertEqual(status["process_returncode"], -11)
        self.assertFalse(status["process_exit_verified"])


if __name__ == "__main__":
    unittest.main()

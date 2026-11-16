"""CPU contracts for actual v2 inference identity and independent EMA/buffer review."""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from examples.baselines.bpp.bpp import report

BASE = Path(__file__).resolve().parents[1]


def native_symbols():
    tree = ast.parse((BASE / "bpp/native_eval.py").read_text())
    functions = {"model_runtime_descriptor", "runtime_reasons", "checkpoint_reasons"}
    constants = {"ACTION_NORMALIZATION", "TRAINING_RECIPE", "EXPECTED_SAMPLING"}
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) and node.name in functions or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    ns = {"_jsonable": lambda value: value,
          "REGIMES": {"seen": "c15", "zero_shot": "c15", "scratch": "scratch", "pretrain_finetune": "finetune"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BASE / "bpp/native_eval.py"), "exec"), ns)
    return ns


class V2EvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.ns = native_symbols()
        self.config = SimpleNamespace(action_normalization=report.V2_ACTION_NORMALIZATION,
            pretraining_scale="c15", regime="seen", training_prompt_source="human",
            num_episodes=10, max_episode_steps=500, execution_horizon=12)
        self.metadata = dict(report.V2_RECIPE, regime="c15", prompt_source="human",
            pilot=False, completed=True, train_step=45000, max_updates=45000)
        scheduler_type = type("DDIMScheduler", (), {})
        scheduler = scheduler_type()
        scheduler.config = dict(clip_sample=True, clip_sample_range=1.0, thresholding=False,
                                num_train_timesteps=50, prediction_type="epsilon", beta_schedule="squaredcos_cap_v2")
        self.model = SimpleNamespace(policy=SimpleNamespace(noise_scheduler=scheduler, num_inference_steps=16),
            model_config={"noise_scheduler": dict(scheduler.config)},
            action_normalization=report.V2_ACTION_NORMALIZATION,
            loaded_weight_selection="ema", loaded_ema_optimization_step=45000)

    def test_actual_runtime_and_isolated_reply_have_identical_descriptor(self):
        actual = self.ns["model_runtime_descriptor"](self.model)
        actor = SimpleNamespace(runtime_descriptor=copy.deepcopy(actual))
        self.assertEqual(actual, self.ns["model_runtime_descriptor"](actor))
        self.assertEqual([], self.ns["runtime_reasons"](actual, self.metadata, self.config))
        changed = copy.deepcopy(actual)
        changed["loaded_weight_selection"] = "raw"
        self.assertNotEqual(report.identity(actual), report.identity(changed))

    def test_actual_raw_weights_or_sampling_override_cannot_be_hidden_by_metadata(self):
        self.model.loaded_weight_selection = "raw"
        self.model.policy.noise_scheduler.config["clip_sample"] = False
        actual = self.ns["model_runtime_descriptor"](self.model)
        reasons = self.ns["runtime_reasons"](actual, self.metadata, self.config)
        self.assertTrue(any("EMA weights" in value for value in reasons))
        self.assertTrue(any("clip_sample" in value for value in reasons))

    def test_formal_checkpoint_requires_source_recipe_and_complete_updates(self):
        tasks = ("L0_TwoRobotFixture-v1",)
        self.assertEqual([], self.ns["checkpoint_reasons"](self.metadata, self.config, tasks, tasks))
        changed = dict(self.metadata, input_perturbation=0.0)
        self.assertTrue(any("input_perturbation" in value for value in self.ns["checkpoint_reasons"](changed, self.config, tasks, tasks)))
        self.model.loaded_ema_optimization_step = 44999
        self.assertTrue(self.ns["runtime_reasons"](self.ns["model_runtime_descriptor"](self.model), self.metadata, self.config))


class V2SavedCheckpointReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        stats = {"source": {"manifest_hash": "fixture"}, "mode": "quantile",
            "state": {"q01": [-1.0]*18, "q99": [1.0]*18}, "action": {
            "min": [-1.0 if i in (7, 15) else -2.0 for i in range(16)],
            "max": [1.0 if i in (7, 15) else 2.0 for i in range(16)]}}
        path = Path(self.temp.name) / "normalizer.json"
        path.write_text(json.dumps(stats))
        self.data = {"normalizer": {"path": str(path), "sha256": report.digest(path)}}
        self.metadata = {"train_step": 45000}
        self.payload = dict(report.V2_RECIPE,
            config=copy.deepcopy(report.V2_RECIPE), state_dict_weight_type="raw",
            normalizer_stats=stats, parameter_audit={"unique_parameters": 2},
            state_dict={"action_scale": torch.tensor([1.0 if i in (7, 15) else .5 for i in range(16)]),
                        "action_offset": torch.zeros(16), "state_scale": torch.ones(18),
                        "state_offset": torch.zeros(18), "policy.weight": torch.tensor([2.0, 3.0])},
            ema={"schema": "bpp_tig_ema_v2", "parameter_names": ["policy.weight"],
                 "parameter_shapes": [[2]], "state_dict": dict(report.V2_RECIPE["ema_config"],
                     optimization_step=45000, shadow_params=[torch.tensor([1.9, 2.9])])})

    def test_valid_minmax_identity_buffers_and_complete_ema(self):
        self.assertEqual([], report.v2_checkpoint_issues(self.payload, self.metadata, self.data))

    def test_tagged_v2_with_quantile_buffers_is_rejected(self):
        self.payload["state_dict"]["action_scale"][0] = 1.0
        issues = report.v2_checkpoint_issues(self.payload, self.metadata, self.data)
        self.assertTrue(any("action_scale" in issue for issue in issues))

    def test_corrupt_state_buffer_is_rejected(self):
        self.payload["state_dict"]["state_offset"][0] = 0.25
        self.assertTrue(any("state_offset" in issue for issue in report.v2_checkpoint_issues(self.payload, self.metadata, self.data)))

    def test_incomplete_or_wrong_step_ema_is_rejected(self):
        self.payload["ema"]["state_dict"]["optimization_step"] = 44999
        issues = report.v2_checkpoint_issues(self.payload, self.metadata, self.data)
        self.assertTrue(any("update count" in issue for issue in issues))
        self.payload["ema"]["state_dict"]["shadow_params"] = []
        self.assertTrue(any("EMA schema" in issue for issue in report.v2_checkpoint_issues(self.payload, self.metadata, self.data)))


if __name__ == "__main__":
    unittest.main()

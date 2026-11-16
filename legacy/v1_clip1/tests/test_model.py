"""Protocol and cache tests with a tiny network; no checkpoint download needed."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from examples.baselines.bpp.bpp.model import FrozenBPPModel, install_video_only_tokenizer, select_prompt_frame_indices


class FrameSamplingTests(unittest.TestCase):
    def test_sampling_uses_video_time_and_keeps_end(self):
        result = select_prompt_frame_indices(90, 30.0, 1.0, 50)
        np.testing.assert_array_equal(result, [0, 30, 60, 89])

    def test_long_clip_covers_start_and_end_under_position_limit(self):
        result = select_prompt_frame_indices(9000, 30.0, 1.0, 50)
        self.assertEqual(len(result), 50)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[-1], 8999)
        self.assertTrue(np.all(np.diff(result) > 0))

    def test_single_frame_is_valid(self):
        np.testing.assert_array_equal(select_prompt_frame_indices(1, 30, 1, 50), [0])

    def test_invalid_fps_and_empty_reference_fail(self):
        for args in [(0, 30, 1, 50), (3, 0, 1, 50), (3, float("nan"), 1, 50), (3, 30, 0, 50)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                select_prompt_frame_indices(*args)


def tiny_policy(torch, *, positional_embeddings=False):
    nn = torch.nn

    class ForbiddenAction(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))

        def forward(self, *_args, **_kwargs):
            raise AssertionError("A demonstration action was encoded")

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.calls = 0

        def get_obs_names(self, **_kwargs):
            return ["eye_in_hand_rgb", "agentview_rgb", "ee_ori", "ee_pos", "gripper_states"]

        def forward(self, obs, *, rgb_keys, low_dim_keys, is_prompt):
            if set(obs) != {"agentview_rgb"} or rgb_keys != ["agentview_rgb"] or low_dim_keys:
                raise AssertionError("Prompt proprioception or a second prompt view reached the encoder")
            if not is_prompt:
                raise AssertionError("Expected prompt-only image encoding")
            self.calls += 1
            mean = obs["agentview_rgb"].mean(dim=(-1, -2, -3))
            return mean[:, :, None, None] * self.weight, {}

    class Pool(nn.Module):
        def forward(self, value):
            return value.mean(dim=1)

    class Tokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.concat_prompt_and_receding = False
            self.merge_prompt_tokens = "obs_and_action"
            self.prompt_rgb_keys = ["agentview_rgb", "eye_in_hand_rgb"]
            self.prompt_low_dim_keys = ["ee_pos", "ee_ori", "gripper_states"]
            self.current_obs_keys = ["agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states"]
            self.n_emb = 2
            self.prompt_obs_encoder = Vision()
            self.prompt_attention_pool = Pool()
            self.action_proj = ForbiddenAction()
            self.pool_modality_pos_embed = (
                nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(6, 2))
                if positional_embeddings else None
            )
            self.current_calls = 0

        def forward(self, obs):
            if "prompt" in obs:
                raise AssertionError("The original sensorimotor prompt path was used")
            if set(obs) != set(self.current_obs_keys):
                raise AssertionError("The current observation slots changed")
            self.current_calls += 1
            return "original queries", "original current slots"

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_encoder = Tokenizer()
            self.prompt_pos_emb = nn.Parameter(torch.zeros(1, 50, 2))
            self.prompt_current_obs_pos_emb = nn.Parameter(torch.zeros(1, 10, 2))
            self.prompt_tokens_cache = None
            self.prompt_mask_cache = None
            self.prompt_calls = 0

        def prompt(self, prompt):
            self.prompt_tokens_cache, self.prompt_mask_cache = self.obs_encoder({"prompt": prompt})
            self.prompt_calls += 1

        def reset(self):
            self.prompt_tokens_cache = None
            self.prompt_mask_cache = None

    class PromptNormalizer:
        def normalize(self, obs):
            if set(obs) != {"agentview_rgb"}:
                raise AssertionError("Only RGB may be normalized as demonstration input")
            return obs

    class Normalizer:
        def get_prompt_normalizer(self):
            return PromptNormalizer()

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_encoder = Encoder()
            self.normalizer = Normalizer()
            self.action_horizon = 16
            self.action_dim = 10
            self.num_inference_steps = 16

        def predict_action(self, current):
            if self.training or any(p.requires_grad for p in self.parameters()):
                raise AssertionError("Policy parameters were not frozen in eval mode")
            if not torch.is_inference_mode_enabled():
                raise AssertionError("Inference mode is required")
            if self.obs_encoder.prompt_tokens_cache is None:
                raise AssertionError("The video was not cached")
            self.obs_encoder.obs_encoder(current)
            batch = current["ee_pos"].shape[0]
            return {"action": torch.zeros(batch, 16, 10)}

        def reset(self):
            self.obs_encoder.reset()

    return Policy()


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required for tiny-network protocol tests")
class VideoOnlyModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory(dir="/gaozt-test1/sxh/bpp/tmp")
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "tiny.ckpt"
        self.checkpoint.write_bytes(b"test fixture; no pretrained weights")
        self.shape = {"obs": {
            "agentview_rgb": {"type": "rgb", "shape": [3, 8, 8], "horizon": 2},
            "eye_in_hand_rgb": {"type": "rgb", "shape": [3, 8, 8], "horizon": 2},
            "ee_pos": {"type": "low_dim", "shape": [3], "horizon": 2},
            "ee_ori": {"type": "low_dim", "shape": [6], "horizon": 2},
            "gripper_states": {"type": "low_dim", "shape": [2], "horizon": 2},
        }}

    def make_model(self):
        policy = tiny_policy(self.torch)
        with mock.patch("examples.baselines.bpp.bpp.model._load_frozen_policy", return_value=(policy, self.shape, [])):
            return FrozenBPPModel(self.checkpoint, device="cpu", amp_dtype="float32")

    def current(self, batch=1):
        return {key: np.zeros((batch, spec["horizon"], *spec["shape"]), dtype=np.float32)
                for key, spec in self.shape["obs"].items()}

    def test_only_rgb_reaches_prompt_and_cache_is_reused_for_predictions(self):
        model = self.make_model()
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        metadata = model.prepare_video(np.full((65, 12, 10, 3), 127, dtype=np.uint8), fps=30)
        first = model.get_action(self.current(batch=2))
        second = model.get_action(self.current(batch=2))
        self.assertEqual(first.shape, (2, 16, 10))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(model.tokenizer.prompt_obs_encoder.calls, 1)
        self.assertEqual(model.policy.obs_encoder.prompt_calls, 1)
        self.assertEqual(model.tokenizer.current_calls, 2)
        self.assertEqual(metadata["demo_action_tokens"], 0)
        self.assertEqual(metadata["demo_proprioception_tokens"], 0)
        after = model.policy.state_dict()
        self.assertEqual(set(before), set(after))
        for key, value in before.items():
            self.assertTrue(self.torch.equal(value, after[key]), key)
        self.assertEqual(tuple(model.policy.obs_encoder.prompt_current_obs_pos_emb.shape), (1, 10, 2))

    def test_position_embedding_selects_original_agentview_row(self):
        policy = tiny_policy(self.torch, positional_embeddings=True)
        tokenizer = policy.obs_encoder.obs_encoder
        before = {k: v.clone() for k, v in tokenizer.state_dict().items()}
        install_video_only_tokenizer(tokenizer)
        frames = self.torch.zeros(1, 3, 3, 8, 8)
        with self.torch.inference_mode():
            tokens, _ = tokenizer({"prompt": {"obs": {"agentview_rgb": frames}, "metadata": {}}})
        self.assertEqual(tuple(tokens.shape), (1, 3, 2))
        self.assertTrue(self.torch.equal(tokens[0, 0], tokenizer.pool_modality_pos_embed[1]))
        for key, value in before.items():
            self.assertTrue(self.torch.equal(value, tokenizer.state_dict()[key]), key)

    def test_sensorimotor_prompt_is_rejected_before_any_encoder_call(self):
        tokenizer = tiny_policy(self.torch).obs_encoder.obs_encoder
        install_video_only_tokenizer(tokenizer)
        rgb = self.torch.zeros(1, 2, 3, 8, 8)
        cases = [
            {"obs": {"agentview_rgb": rgb}, "action": self.torch.zeros(1, 2, 20, 10)},
            {"obs": {"agentview_rgb": rgb, "ee_pos": self.torch.zeros(1, 2, 3)}},
            {"obs": {"agentview_rgb": rgb, "eye_in_hand_rgb": rgb}},
        ]
        for prompt in cases:
            with self.subTest(keys=list(prompt)), self.assertRaises(ValueError):
                tokenizer({"prompt": prompt})
        self.assertEqual(tokenizer.prompt_obs_encoder.calls, 0)

    def test_reset_invalidates_prompt_and_requires_explicit_history(self):
        model = self.make_model()
        model.prepare_video(np.zeros((1, 8, 8, 3), dtype=np.uint8), fps=30)
        invalid = self.current()
        invalid["ee_pos"] = invalid["ee_pos"][:, 0]
        with self.assertRaises(ValueError):
            model.get_action(invalid)
        model.reset()
        self.assertIsNone(model.policy.obs_encoder.prompt_tokens_cache)
        with self.assertRaises(RuntimeError):
            model.get_action(self.current())

    def test_dictionary_reference_and_extra_current_demo_keys_are_rejected(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model.prepare_video({"rgb": np.zeros((1, 8, 8, 3)), "qpos": np.zeros((1, 18))}, fps=30)
        model.prepare_video(np.zeros((1, 8, 8, 3), dtype=np.uint8), fps=30)
        current = self.current()
        current["prompt"] = {}
        with self.assertRaises(ValueError):
            model.get_action(current)


if __name__ == "__main__":
    unittest.main()

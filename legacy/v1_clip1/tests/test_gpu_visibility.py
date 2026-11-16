"""CPU-only contracts for eval resource mapping and unchanged protocol identity."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from examples.baselines.bpp import eval_bpp_imitator
from examples.baselines.bpp.bpp.gpu_visibility import (
    configure_eval_gpu_visibility, select_gpu_visibility,
)

BASELINE = Path(__file__).resolve().parents[1]
TIG_ROOT = BASELINE.parents[2]
MASK = "0,1,2,3,4,5,6,7"


class GPUVisibilityTests(unittest.TestCase):
    def test_every_worker_keeps_the_same_physical_gpu(self):
        for mask in (MASK, "7,3,5,1,0,6,2,4"):
            for index, token in enumerate(mask.split(",")):
                with self.subTest(mask=mask, index=index):
                    plan = select_gpu_visibility(["--device", f"cuda:{index}", "--shard-index", "11"], mask)
                    self.assertEqual(plan.effective_visible_devices, token)
                    self.assertEqual(plan.arguments, ("--device", "cuda:0", "--shard-index", "11"))
                    self.assertEqual(plan.selected_physical_token, token)
                    self.assertEqual(plan.original_device, f"cuda:{index}")

    def test_uuid_and_mig_tokens_are_opaque_and_joined_option_is_preserved(self):
        for tokens in (("GPU-deadbeef-a1", "GPU-01234567-b2"),
                       ("MIG-GPU-deadbeef/1/0", "MIG-GPU-deadbeef/2/0"),
                       ("MIG-1234-opaque", "MIG-5678-opaque")):
            plan = select_gpu_visibility(["--device=cuda:1"], ", ".join(tokens))
            self.assertEqual(plan.arguments, ("--device=cuda:0",))
            self.assertEqual(plan.effective_visible_devices, tokens[1])

    def test_cpu_help_implicit_device_and_unknown_masks_are_untouched(self):
        fixtures = [(["--device", "cpu"], MASK), (["--device=cuda:3", "--help"], MASK),
                    (["-h", "--device", "cuda:2"], MASK), (["--regime", "seen"], MASK),
                    (["--device", "cuda"], MASK), (["--device", "cuda:3"], None),
                    (["--device", "cuda:3"], "0,1"), (["--device", "cuda:1"], "0,-1,2"),
                    (["--device", "cuda:1"], "0,,2"), (["--device", "cuda:1"], "0,0,2"),
                    (["--device", "cuda:1"], "0..7"), (["--device", "cuda:0"], "")]
        for arguments, mask in fixtures:
            with self.subTest(arguments=arguments, mask=mask):
                plan = select_gpu_visibility(arguments, mask)
                self.assertEqual(plan.arguments, tuple(arguments))
                self.assertEqual(plan.effective_visible_devices, mask)
                self.assertFalse(plan.changed)

    def test_repeated_device_uses_the_effective_last_value(self):
        arguments = ["--device", "cuda:7", "--device=cuda:2"]
        plan = select_gpu_visibility(arguments, MASK)
        self.assertEqual(plan.original_device, "cuda:2")
        self.assertEqual(plan.effective_visible_devices, "2")
        self.assertEqual(plan.arguments, ("--device", "cuda:7", "--device=cuda:0"))
        final_cpu = ["--device=cuda:7", "--device", "cpu"]
        self.assertEqual(select_gpu_visibility(final_cpu, MASK).arguments, tuple(final_cpu))

    def test_initialized_or_unknown_cuda_state_is_not_remapped(self):
        for modules, reason in (
            ({"torch": SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))}, "cuda_already_initialized"),
            ({"torch": SimpleNamespace()}, "cuda_initialization_state_unknown"),
            ({"sapien.core": object()}, "cuda_consumer_already_loaded"),
        ):
            with self.subTest(reason=reason):
                env, stream = {"CUDA_VISIBLE_DEVICES": MASK}, io.StringIO()
                arguments = configure_eval_gpu_visibility(["--device", "cuda:4"], environ=env,
                                                          modules=modules, stderr=stream)
                self.assertEqual(arguments, ["--device", "cuda:4"])
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], MASK)
                self.assertEqual(json.loads(stream.getvalue())["bpp_eval_gpu_visibility"]["reason"], reason)

    def test_uninitialized_torch_can_remap_and_records_original_mapping(self):
        env, stream = {"CUDA_VISIBLE_DEVICES": "GPU-aa,GPU-bb"}, io.StringIO()
        cuda = SimpleNamespace(is_initialized=lambda: False)
        result = configure_eval_gpu_visibility(["--device=cuda:1"], environ=env,
                                              modules={"torch.cuda": cuda}, stderr=stream)
        self.assertEqual(result, ["--device=cuda:0"])
        record = json.loads(stream.getvalue())["bpp_eval_gpu_visibility"]
        self.assertEqual(record["original_device"], "cuda:1")
        self.assertEqual(record["effective_device"], "cuda:0")
        self.assertEqual(record["original_visible_devices"], "GPU-aa,GPU-bb")
        self.assertEqual(record["effective_visible_devices"], "GPU-bb")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-bb")

    def test_visibility_is_selected_before_dispatch_import_and_remains_inherited(self):
        runtime = ModuleType("examples.baselines.bpp.bpp.runtime")
        calls = []

        def dispatch(command, arguments):
            calls.append((command, arguments, os.environ["CUDA_VISIBLE_DEVICES"]))
            return 17

        runtime.dispatch = dispatch
        with patch.dict(sys.modules, {runtime.__name__: runtime}), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": MASK}), \
                patch("examples.baselines.bpp.bpp.gpu_visibility._loaded_cuda_runtime_reason", return_value=None), \
                patch("sys.stderr", new_callable=io.StringIO):
            result = eval_bpp_imitator.main(["--device", "cuda:6", "--regime", "zero_shot"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "6")
        self.assertEqual(result, 17)
        self.assertEqual(calls, [("eval", ["--device", "cuda:0", "--regime", "zero_shot"], "6")])

    def test_cpu_and_help_configure_have_no_environment_or_log_side_effect(self):
        for arguments in (["--device", "cpu"], ["--help", "--device=cuda:7"]):
            env, stream = {"CUDA_VISIBLE_DEVICES": MASK}, io.StringIO()
            result = configure_eval_gpu_visibility(arguments, environ=env, modules={}, stderr=stream)
            self.assertEqual(result, arguments)
            self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": MASK})
            self.assertEqual(stream.getvalue(), "")

    def test_helper_and_launcher_imports_do_not_import_torch_or_sapien(self):
        code = """
import json, sys
from examples.baselines.bpp.bpp import gpu_visibility
from examples.baselines.bpp import eval_bpp_imitator
print(json.dumps([name for name in sys.modules if name.split('.')[0] in ('torch', 'sapien')]))
"""
        env = dict(os.environ, PYTHONPATH=str(TIG_ROOT), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_real_cli_help_does_not_select_a_gpu(self):
        env = dict(os.environ, PYTHONPATH=str(TIG_ROOT), CUDA_VISIBLE_DEVICES="-1",
                   PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(BASELINE / "eval_bpp_imitator.py"),
                                 "--device=cuda:7", "--help"], env=env,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertNotIn("bpp_eval_gpu_visibility", result.stderr)

    def test_frozen_protocol_identity_has_no_resource_mapping_dependency(self):
        tree = ast.parse((BASELINE / "bpp/native_eval.py").read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        assignments = {target.id: node.value for node in run.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        protocol, sources, details = (assignments[name] for name in ("protocol", "sources", "details"))
        config_fields = {node.attr for node in ast.walk(protocol) if isinstance(node, ast.Attribute)
                         and isinstance(node.value, ast.Name) and node.value.id == "config"}
        self.assertNotIn("device", config_fields)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", ast.unparse(protocol))
        self.assertNotIn("config", {node.id for node in ast.walk(details) if isinstance(node, ast.Name)})
        names = ast.literal_eval(sources.generators[0].iter)
        self.assertEqual(names, ("native_eval.py", "native_model.py", "data.py", "training_data.py"))
        # Evaluate the actual frozen protocol expression twice, without importing
        # native_eval/Torch/SAPIEN; only the runtime device/mask changes.
        config = SimpleNamespace(**{name: "fixture" for name in config_fields})
        config.checkpoint, config.dataset_root = Path("checkpoint.pt"), Path("dataset")
        config.regime, config.num_episodes = "seen", 10
        config.execution_horizon, config.max_episode_steps = 12, 500
        config.allow_pilot, config.capture_video = False, True
        namespace = {"config": config, "metadata": {"prompt_source": "human"}, "tasks": ("L0_fixture",),
                     "SEED_ORIGIN": 100000, "split_path": Path("official.txt"),
                     "file_sha256": lambda _: "fixture_sha", "DATASET_REVISION": "fixed_revision",
                     "sources": {name: hashlib.sha256((BASELINE / "bpp" / name).read_bytes()).hexdigest() for name in names},
                     "_revision": lambda _: "fixed_head", "torch": SimpleNamespace(__version__="2.7.1"),
                     "reasons": [], "details": {"training_metadata": {"prompt_source": "human"}}}
        expression = compile(ast.Expression(protocol), str(BASELINE / "bpp/native_eval.py"), "eval")
        identities = []
        for device, mask in (("cuda:7", MASK), ("cuda:0", "7")):
            config.device = device
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}):
                identities.append(hashlib.sha256(json.dumps(eval(expression, namespace), sort_keys=True).encode()).hexdigest())
        self.assertEqual(identities[0], identities[1])


if __name__ == "__main__":
    unittest.main()

"""Canonical/legacy identity and CLI process-boundary contracts."""
from __future__ import annotations

import importlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.baselines.bpp.bpp import native_eval


CANONICAL = "examples.baselines.bpp.bpp"
BASELINE = Path(__file__).resolve().parents[1]
TIG_ROOT = BASELINE.parents[2]
PROJECT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
LEGACY_ROOT = PROJECT / "repos/bpp_tig_adapter"
MODULES = ("data", "model", "control", "runner", "native_model", "training_data", "native_eval", "feature_cache", "train")


class CanonicalCompatibilityTests(unittest.TestCase):
    def clean_environment(self):
        env = dict(os.environ)
        env.pop("BPP_ADAPTER_ROOT", None)
        env["PYTHONPATH"] = str(TIG_ROOT)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        return env

    def test_baseline_entries_run_without_external_adapter_path(self):
        for entry in ("prepare", "train", "eval"):
            with self.subTest(entry=entry):
                result = subprocess.run(
                    [sys.executable, str(BASELINE / f"{entry}_bpp_imitator.py"), "--help"],
                    env=self.clean_environment(), cwd=PROJECT,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    @unittest.skipUnless((LEGACY_ROOT / "bpp_tig/__init__.py").is_file(), "Optional legacy facade checkout unavailable")
    def test_legacy_imports_are_the_identical_canonical_module_objects(self):
        with patch.object(sys, "path", [str(LEGACY_ROOT), *sys.path]):
            for name in MODULES:
                with self.subTest(module=name):
                    legacy = importlib.import_module(f"bpp_tig.{name}")
                    canonical = importlib.import_module(f"{CANONICAL}.{name}")
                    self.assertIs(legacy, canonical)
                    self.assertEqual(legacy.__name__, f"{CANONICAL}.{name}")
                    self.assertEqual(Path(legacy.__file__).resolve(), BASELINE / "bpp" / f"{name}.py")

    @unittest.skipUnless((LEGACY_ROOT / "bpp_tig/__init__.py").is_file(), "Optional legacy facade checkout unavailable")
    def test_legacy_module_clis_forward_to_canonical_main(self):
        env = self.clean_environment()
        env["PYTHONPATH"] = os.pathsep.join((str(LEGACY_ROOT), str(TIG_ROOT)))
        for module, arguments in (
            ("train", ["--help"]), ("native_eval", ["--help"]),
            ("training_data", ["prepare", "--help"]),
            ("feature_cache", ["--help"]), ("runner", ["--help"]),
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", f"bpp_tig.{module}", *arguments],
                    env=env, cwd=PROJECT, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_supervisor_worker_launch_uses_canonical_module_in_a_clean_child(self):
        original_run = subprocess.run
        seen = []

        def run_help(command, *, env):
            seen.append(command)
            self.assertEqual(command[4], f"{CANONICAL}.native_eval")
            child_env = dict(env)
            child_env.pop("BPP_ADAPTER_ROOT", None)
            child_env["PYTHONPATH"] = str(TIG_ROOT)
            # Exercise the actual child module loader/parser, without a model,
            # simulator or rollout. The supervisor command is asserted above.
            completed = original_run(
                [*command[:5], "--help"], env=child_env, cwd=PROJECT,
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return completed

        (PROJECT / "tmp").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bpp-canonical-supervisor-", dir=PROJECT / "tmp") as folder:
            with patch.object(native_eval.subprocess, "run", side_effect=run_help), patch.object(native_eval, "record_process_exit") as record:
                result = native_eval.main([
                    "--checkpoint", str(Path(folder) / "unused.pt"),
                    "--output-dir", folder, "--regime", "zero_shot",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(len(seen), 1)
            record.assert_called_once()

    def test_spawn_can_resolve_a_canonical_evaluator_function(self):
        self.assertEqual(native_eval._runtime_marker.__module__, f"{CANONICAL}.native_eval")
        process = multiprocessing.get_context("spawn").Process(
            target=native_eval._runtime_marker, args=("canonical_spawn_contract",),
        )
        process.start()
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail("Canonical spawn target did not exit")
        self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
    unittest.main()

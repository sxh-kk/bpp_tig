"""Resource accounting regressions; no model, CUDA or simulator imports."""
import importlib
import os
from pathlib import Path
import tempfile
import time
import unittest

resources = importlib.import_module("examples.baselines.bpp.bpp.resource_accounting")


class ResourceAccountingTests(unittest.TestCase):
    def write(self, path, **value):
        resources.write_json(path, value)

    def worker(self, directory, index, start, end, command, code=0):
        self.write(directory / f"worker_{index:03d}.json", pid=999999999,
                   command=command, started_utc=resources.utc(start),
                   finished_utc=resources.utc(end) if end is not None else None,
                   status="running" if end is None else "completed" if code == 0 else "process_error",
                   returncode=code if end is not None else None)

    def entry(self, kind, directory, output, commands, **extra):
        return dict(kind=kind, output_dir=str(output), status="complete", attempts=[
            dict(directory=str(directory), commands=commands)], **extra)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=resources.ROOT / "tmp")

    def test_two_eval_workers_and_failed_retry_share_physical_gpu(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            command = ["python", "eval", "--device", "cuda:3"]
            first, retry = root / "first", root / "retry"
            self.worker(first, 0, 100, 110, command)
            self.worker(first, 1, 102, 112, command, code=2)
            self.worker(retry, 0, 120, 125, command)
            step = self.entry("eval", first, root / "output", [command, command])
            step["attempts"].append(dict(directory=str(retry), commands=[command]))
            result = resources.account({"steps": {"eval": step}}, snapshot=200)
            self.assertAlmostEqual(result["recorded_assigned_gpu_wall_hours"] * 3600, 17)
            self.assertEqual(result["physical_gpus"][3]["assigned_wall_seconds"], 17)
            self.assertEqual(result["stages"][0]["failed_closed_process_records"], 1)

    def test_global_union_prevents_cross_stage_double_charge(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            train_command = ["python", "-m", "torch.distributed.run", "--nproc-per-node=8"]
            train = root / "training"
            self.write(train / "process_status.json", command=train_command,
                       started_utc=resources.utc(100), finished_utc=resources.utc(110), status="completed")
            evaluation = root / "eval_attempt"
            eval_command = ["python", "eval", "--device", "cuda:0"]
            self.worker(evaluation, 0, 105, 115, eval_command)
            state = {"steps": {"train": dict(kind="train", output_dir=str(train)),
                               "eval": self.entry("eval", evaluation, root / "eval_output", [eval_command])}}
            result = resources.account(state, snapshot=200)
            self.assertAlmostEqual(result["recorded_assigned_gpu_wall_hours"] * 3600, 85)
            self.assertAlmostEqual(sum(x["assigned_gpu_wall_hours"] for x in result["stages"]) * 3600, 90)

    def test_scratch_is_shared_once_but_feature_stages_are_distinct(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            state = {"steps": {}}
            for scale in ("c15", "c30", "c45"):
                state["steps"][scale + ".scratch"] = dict(kind="train", regime="scratch", output_dir=str(root / "scratch"))
            for scale in ("c30", "c45"):
                state["steps"][scale + ".features"] = dict(kind="features", scale=scale, output_dir=str(root / "features"))
            groups = resources.stage_groups(state)
            self.assertEqual(len(groups), 3)
            self.assertEqual(sorted(len(group["aliases"]) for group in groups), [1, 1, 3])

    def test_cpu_stage_cannot_charge_gpu_even_with_device_argument(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            command = ["python", "prepare", "--device", "cuda:5"]
            self.worker(root / "attempt", 0, 100, 140, command)
            state = {"steps": {"prepare": self.entry("prepare", root / "attempt", root / "output", [command])}}
            result = resources.account(state, snapshot=200)
            self.assertEqual(result["recorded_assigned_gpu_wall_hours"], 0)
            self.assertEqual(result["stages"][0]["recorded_process_wall_union_seconds"], 40)

    def test_unmatched_live_pid_cannot_charge_unfinished_status(self):
        snapshot = time.time()
        actual = [x.decode() for x in Path("/proc/self/cmdline").read_bytes().split(b"\0") if x]
        tail = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        boot = next(int(x.split()[1]) for x in Path("/proc/stat").read_text().splitlines() if x.startswith("btime "))
        born = boot + int(tail[19]) / os.sysconf("SC_CLK_TCK")
        status = dict(pid=os.getpid(), command=actual, started_utc=resources.utc(born))
        self.assertTrue(resources.matching_active(status))
        status["command"] = ["not-the-current-command"]
        self.assertFalse(resources.matching_active(status))
        value, reason = resources.interval(status, snapshot)
        self.assertIsNone(value)
        self.assertIn("matching_live_pid", reason)

    def test_reservation_window_includes_cpu_preparation_and_idle_gaps(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            train = root / "train"
            self.write(train / "process_status.json", command=["python", "--nproc-per-node", "8"],
                       started_utc=resources.utc(100), finished_utc=resources.utc(110))
            command = ["python", "prepare"]
            self.worker(root / "cpu", 0, 120, 140, command)
            state = {"steps": {"train.c15": dict(kind="train", output_dir=str(train)),
                "prepare": self.entry("prepare", root / "cpu", root / "output", [command])}}
            result = resources.account(state, snapshot=200)
            self.assertEqual(result["formal_pipeline_elapsed_wall"]["seconds"], 40)
            self.assertAlmostEqual(result["recorded_assigned_gpu_wall_hours"] * 3600, 80)
            self.assertAlmostEqual(result["hypothetical_continuous_eight_gpu_reservation_hours"] * 3600, 320)

    def test_bad_and_missing_attempt_statuses_are_explicitly_excluded(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            attempt = root / "attempt"
            attempt.mkdir()
            bad = attempt / "worker_000.json"
            bad.write_text('{"started_utc":')
            command = ["python", "eval", "--device", "cuda:2"]
            state = {"steps": {"eval": self.entry("eval", attempt, root / "output", [command, command])}}
            result = resources.account(state, snapshot=200)
            reasons = {Path(row["source"]).name: row["reason"] for row in result["excluded_records"]}
            self.assertEqual(reasons["worker_000.json"], "invalid_json")
            self.assertEqual(reasons["worker_001.json"], "missing_file")
            self.assertEqual(result["recorded_assigned_gpu_wall_hours"], 0)

    def test_missing_required_manifests_are_not_future_placeholders(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            required = root / "current_data.json"
            current_features = root / "current_features.json"
            future = root / "future_features.json"
            state = {"steps": {"merge.c30": dict(kind="merge", scale="c30", status="complete")},
                     "artifacts": {"c30.data": str(required), "c30.features": str(current_features),
                                   "c45.features": str(future)}}
            result = resources.storage(state, project_root=root)
            self.assertEqual(result["missing_references_count"], 2)
            self.assertEqual(set(result["missing_references_sample"]), {str(required), str(current_features)})
            self.assertEqual({row["source"] for row in result["expected_manifest_issues"]},
                             {str(required), str(current_features)})
            self.assertEqual([row["source"] for row in result["not_yet_expected"]], [str(future)])

    def test_repeated_scale_artifacts_and_hardlinks_count_storage_once(self):
        with self.temporary() as temporary:
            root = Path(temporary)
            cached = root / "cache/bpp_tig_training/episode.npy"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"cache")
            alias = cached.with_name("alias.npy")
            os.link(cached, alias)
            manifests = []
            for scale in ("c15", "c30", "c45"):
                manifest = root / scale / "manifest.json"
                self.write(manifest, query_episodes=[{"rgb": {"path": str(cached)}, "states": {"path": str(alias)}}])
                manifests.append(manifest)
            state = {"steps": {}, "artifacts": {scale + ".data": str(path)
                for scale, path in zip(("c15", "c30", "c45"), manifests)}}
            result = resources.storage(state, project_root=root)
            category = next(x for x in result["categories"] if x["category"] == "rgb_numeric_cache")
            self.assertEqual(category["file_count"], 1)
            self.assertEqual(category["logical_bytes"], 5)


if __name__ == "__main__":
    unittest.main()

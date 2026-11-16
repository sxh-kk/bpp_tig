"""Pinned inventory, content verification and real local HTTP resume contracts."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from examples.baselines.bpp.bpp import download_data as data


PROJECT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
TIG = Path(__file__).resolve().parents[4]
ENTRY = Path(__file__).resolve().parents[1] / "download_bpp_imitator.py"


def spec_for(filename, payload, *, lfs=True):
    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
    return data.FileSpec(filename, len(payload), hashlib.sha256(payload).hexdigest() if lfs else None, "0" * 40 if lfs else blob)


class DownloadContracts(unittest.TestCase):
    def setUp(self):
        (PROJECT / "tmp").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="bpp-download-tests-", dir=PROJECT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data"
        self.payload = b"Pinned TIG content.0123456789" * 17
        self.spec = spec_for("imitator_human_v1/human_H4/videos/observation.images.zed2i/chunk-000/file-000.mp4", self.payload)

    def http_server(self, *, ignore_range=False, disconnect_once=False, wrong_range=False):
        owner = self
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                header = self.headers.get("Range")
                requests.append(header)
                offset = int(header.removeprefix("bytes=").removesuffix("-")) if header and not ignore_range else 0
                if header and not ignore_range:
                    self.send_response(206)
                    start = offset + 1 if wrong_range else offset
                    self.send_header("Content-Range", f"bytes {start}-{len(owner.payload)-1}/{len(owner.payload)}")
                else:
                    self.send_response(200)
                body = owner.payload[offset:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if disconnect_once and len(requests) == 1:
                    self.wfile.write(body[:23])
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/fixture", requests

    def partial(self, payload):
        path = self.dataset / (self.spec.filename + ".partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_real_http_resume_and_server_ignoring_range_produce_exact_content(self):
        for ignore in (False, True):
            with self.subTest(ignore_range=ignore):
                destination = self.dataset / self.spec.filename
                if destination.exists():
                    destination.unlink()
                partial = self.partial(self.payload[:19])
                url, requests = self.http_server(ignore_range=ignore)
                record = data.download_file(self.spec, self.dataset, url=url, retry_delay=0)
                self.assertEqual(requests[0], "bytes=19-")
                self.assertEqual(destination.read_bytes(), self.payload)
                self.assertFalse(partial.exists())
                self.assertEqual(record["original_partial_bytes"], 19)
                self.assertEqual(record["actual_sha256"], self.spec.sha256)

    def test_real_interrupted_response_retries_from_persisted_bytes(self):
        url, requests = self.http_server(disconnect_once=True)
        record = data.download_file(self.spec, self.dataset, url=url, retries=3, retry_delay=0)
        self.assertEqual(requests, [None, "bytes=23-"])
        self.assertEqual(record["attempts"], 2)
        self.assertEqual((self.dataset / self.spec.filename).read_bytes(), self.payload)

    def test_wrong_content_range_cannot_be_spliced_into_partial(self):
        partial = self.partial(self.payload[:19])
        url, requests = self.http_server(wrong_range=True)
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            data.download_file(self.spec, self.dataset, url=url, retries=2, retry_delay=0)
        self.assertEqual(len(requests), 2)
        self.assertEqual(partial.read_bytes(), self.payload[:19])
        self.assertFalse((self.dataset / self.spec.filename).exists())

    def test_existing_files_require_content_hash_and_git_blob_uses_git_header(self):
        for lfs in (True, False):
            with self.subTest(lfs=lfs):
                name = f"imitator_human_v1/human_H4/meta/{'info.json' if lfs else 'episodes/chunk-000/file-000.parquet'}"
                spec = spec_for(name, self.payload, lfs=lfs)
                path = self.dataset / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.payload)
                with patch.object(data, "urlopen", side_effect=AssertionError("Verified content used HTTP")):
                    record = data.download_file(spec, self.dataset, offline=True)
                self.assertTrue(record["reused_existing"])
                self.assertEqual(record["actual_git_blob_sha1"], hashlib.sha1(f"blob {len(self.payload)}\0".encode() + self.payload).hexdigest())
                path.write_bytes(b"x" * len(self.payload))
                with self.assertRaises(FileNotFoundError):
                    data.download_file(spec, self.dataset, offline=True)

    def test_default_proxy_retry_budget_is_forty(self):
        attempts = []
        with patch.object(data, "urlopen", side_effect=OSError("fixture proxy interruption")) as request, patch.object(data.time, "sleep"):
            with self.assertRaisesRegex(OSError, "proxy interruption"):
                data.download_file(self.spec, self.dataset, on_retry=lambda attempt, exc: attempts.append(attempt))
        self.assertEqual(request.call_count, 40)
        self.assertEqual(attempts, list(range(1, 41)))

    def test_offline_scale_verification_never_counts_partial_stat_as_verified(self):
        folders, _ = data.selected_folders(TIG, "c15")
        files = []
        payload = b"small source fixture"
        for folder in folders:
            for suffix in ("meta/info.json", "meta/episodes/chunk-000/file-000.parquet", "videos/observation.images.zed2i/chunk-000/file-000.mp4"):
                spec = spec_for(f"imitator_human_v1/{folder['repo_id']}/{suffix}", payload)
                files.append(spec)
                path = self.dataset / spec.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
        bad = files[-1]
        (self.dataset / bad.filename).unlink()
        partial = self.dataset / (bad.filename + ".partial")
        partial.write_bytes(b"x" * len(payload))  # Full size still is not verified content.
        source = self.root / "source.json"
        source.write_text(json.dumps(dict(repo_id=data.REPO_ID, revision=data.REVISION, inventory_complete=True, files=[data.asdict(item) for item in files])))
        config = data.DownloadConfig("c15", dataset_root=self.dataset, manifest_root=self.root / "logs", lock_root=self.root / "locks", source_manifests=(source,), offline=True)
        with patch.object(data, "urlopen", side_effect=AssertionError("Offline verifier used HTTP")):
            result = data.run_download(config)
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["verified_files"], len(files) - 1)
            self.assertEqual(result["verified_bytes"], (len(files) - 1) * len(payload))
            self.assertEqual(result["partial_unverified_bytes"], len(payload))
            partial.write_bytes(payload)
            complete = data.run_download(config)
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["verified_files"], len(files))
        self.assertFalse(partial.exists())
        verification = json.loads((self.root / "logs/c15/verification.json").read_text())
        self.assertEqual(len(verification["files"]), 60)
        self.assertTrue((self.root / "logs/c15/history").is_dir())

    def test_entry_help_and_plan_do_not_import_torch_or_access_network(self):
        blocker = '''import importlib.abc, runpy, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch': raise AssertionError('Downloader imported Torch')
sys.meta_path.insert(0, NoTorch())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        env = dict(os.environ, PYTHONPATH=str(TIG), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", blocker, str(ENTRY)], env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--scale", result.stdout)
        with patch.object(data, "list_folder", side_effect=AssertionError("Plan used HF API")), patch.object(data, "urlopen", side_effect=AssertionError("Plan downloaded bytes")):
            for scale, count in (("c15", 20), ("c30", 35), ("c45", 53)):
                plan = data.run_download(data.DownloadConfig(scale, dataset_root=self.dataset, manifest_root=self.root / "plans", lock_root=self.root / "locks", plan_only=True))
                self.assertEqual(plan["selected_folder_count"], count)
                self.assertFalse(plan["downloads_started"])
                self.assertTrue((self.root / "plans" / scale / "plan.json").is_file())

    def test_previous_scale_inventory_is_reused_and_wrong_revision_is_rejected(self):
        folders, _ = data.selected_folders(TIG, "c15")
        records = []
        for folder in folders:
            for suffix in ("meta/info.json", "meta/episodes/chunk-000/file-000.parquet", "videos/observation.images.zed2i/chunk-000/file-000.mp4"):
                records.append(data.asdict(spec_for(f"imitator_human_v1/{folder['repo_id']}/{suffix}", b"fixture")))
        previous = self.root / "manifests/c15/source.json"
        previous.parent.mkdir(parents=True)
        previous.write_text(json.dumps(dict(repo_id=data.REPO_ID, revision=data.REVISION, inventory_complete=True, files=records)))
        config = data.DownloadConfig("c30", dataset_root=self.dataset, manifest_root=self.root / "manifests", lock_root=self.root / "locks", source_manifests=(previous,), plan_only=True)
        with patch.object(data, "list_folder", side_effect=AssertionError("Plan accessed HF")):
            plan = data.run_download(config)
        self.assertEqual(len(plan["files"]), 60)
        self.assertEqual(len(plan["missing_inventory_folders"]), 15)
        self.assertIn(str(previous), plan["source_inventories"])
        broken = json.loads(previous.read_text())
        broken["revision"] = "different-revision"
        previous.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, "pinned dataset revision"):
            data.run_download(config)


class SelectionContracts(unittest.TestCase):
    def test_official_nested_download_sets_include_target_and_all_c45_configured_extras(self):
        pools = {}
        for scale, counts in (("c15", (20, 800)), ("c30", (35, 1550)), ("c45", (53, 2450))):
            folders, configs = data.selected_folders(TIG, scale)
            self.assertEqual((len(folders), sum(len(row["episode_ids"]) for row in folders)), counts)
            self.assertEqual(len(configs), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in configs))
            targets = {row["repo_id"] for row in folders if "target" in row["splits"]}
            self.assertEqual(targets, {"human_H4", "human_H13", "human_H20", "human_H21", "human_H30"})
            pools[scale] = {row["repo_id"] for row in folders}
        self.assertLess(pools["c15"], pools["c30"])
        self.assertLess(pools["c30"], pools["c45"])
        self.assertTrue({"human_H25", "human_H37", "human_H38"}.issubset(pools["c45"]))
        for suffix in ("videos/observation.images.zed2i_depth/chunk-000/file-000.mp4", "videos/observation.images.cam1/chunk-000/file-000.mp4", "data/chunk-000/file-000.parquet", "meta/stats.json", "../../outside.mp4"):
            self.assertFalse(data.selected_path(f"imitator_human_v1/human_H4/{suffix}", pools["c45"]))


if __name__ == "__main__":
    unittest.main()

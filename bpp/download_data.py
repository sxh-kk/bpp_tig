"""Pinned human RGB downloads for one TIG BPP scale, without importing Torch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ID = "imitator-game/IG-10K-Dataset"
REVISION = "9a57b27edcb54823d8da7d2b77efa61c092ffa95"
SCALE_CONFIGS = {"c15": "human_train_config_15.json", "c30": "human_train_config_30.json", "c45": "human_train_config_45.json"}
TARGET_CONFIG = "human_test_config_unseen.json"
RGB_DIRECTORY = "videos/observation.images.zed2i/"
PROJECT_ROOT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
TIG_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "bpp_tig_pinned_human_download_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def selected_folders(tig_repo: Path, scale: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Configured scale human pool UNION target5, including C45's three extras."""
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale {scale!r}")
    directory = Path(tig_repo) / "examples/baselines/lerobot_dataset/config/exp_configs"
    folders: dict[str, dict[str, Any]] = {}
    configs = []
    for label, filename in ((scale, SCALE_CONFIGS[scale]), ("target", TARGET_CONFIG)):
        path = directory / filename
        configs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for row in json.loads(path.read_text()):
            repo = row["repo_id"]
            if not re.fullmatch(r"human_H\d+", repo):
                raise ValueError(f"Invalid official human repository ID: {repo}")
            first, stop = map(int, row["train"].split(":"))
            if first < 0 or stop <= first:
                raise ValueError(f"Invalid train range in {path}: {row['train']}")
            record = folders.setdefault(repo, {"repo_id": repo, "splits": [], "episode_ids": []})
            record["splits"].append(label)
            record["episode_ids"] = sorted(set(record["episode_ids"]) | set(range(first, stop)))
    return [folders[name] for name in sorted(folders)], configs


def selected_path(filename: str, folder_names: set[str]) -> bool:
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or "\\" in filename or len(path.parts) < 4:
        return False
    if path.parts[0] != "imitator_human_v1" or path.parts[1] not in folder_names:
        return False
    relative = "/".join(path.parts[2:])
    return (
        relative == "meta/info.json"
        or re.fullmatch(r"meta/episodes/chunk-\d+/file-\d+\.parquet", relative) is not None
        or re.fullmatch(r"videos/observation\.images\.zed2i/chunk-\d+/file-\d+\.mp4", relative) is not None
    )


@dataclass(frozen=True)
class FileSpec:
    filename: str
    size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None

    def __post_init__(self):
        path = PurePosixPath(self.filename)
        folder = path.parts[1] if len(path.parts) > 1 else ""
        if not re.fullmatch(r"human_H\d+", folder) or not selected_path(self.filename, {folder}):
            raise ValueError(f"Unapproved download path: {self.filename}")
        if self.size < 0 or (not self.sha256 and not self.git_blob_sha1):
            raise ValueError(f"File must have a size and expected content hash: {self.filename}")
        for value, length in ((self.sha256, 64), (self.git_blob_sha1, 40)):
            if value is not None and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
                raise ValueError(f"Invalid source hash for {self.filename}")

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> "FileSpec":
        return cls(str(row["filename"]), int(row["size"]), row.get("sha256"), row.get("git_blob_sha1"))


def content_hashes(path: Path) -> dict[str, Any]:
    before = path.stat()
    sha256, blob = hashlib.sha256(), hashlib.sha1()
    blob.update(f"blob {before.st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            sha256.update(block)
            blob.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise OSError(f"File changed during verification: {path}")
    return {"size": after.st_size, "sha256": sha256.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def verify_file(path: Path, spec: FileSpec) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size != spec.size:
        return None
    actual = content_hashes(path)
    # For LFS entries, blob_id hashes the POINTER, not the downloaded payload.
    key = "sha256" if spec.sha256 else "git_blob_sha1"
    if actual[key] != getattr(spec, key):
        return None
    return actual


def resolve_url(spec: FileSpec) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/datasets/{REPO_ID}/resolve/{REVISION}/{quote(spec.filename, safe='/')}"


@contextmanager
def file_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def download_file(
    spec: FileSpec, dataset_root: Path, *, url: str | None = None,
    retries: int = 40, timeout: float = 45., retry_delay: float = 3., offline: bool = False,
    on_progress: Callable[[int], None] | None = None,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> dict[str, Any]:
    """Freshly hash existing content; resume .partial by validated HTTP Range."""
    if retries < 1 or timeout <= 0 or retry_delay < 0:
        raise ValueError("Invalid retry/timeout settings")
    destination = Path(dataset_root) / spec.filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    original_partial = partial.stat().st_size if partial.is_file() else 0
    for attempt in range(1, retries + 1):
        try:
            actual = verify_file(destination, spec)
            reused = actual is not None
            if actual is None and partial.is_file() and partial.stat().st_size == spec.size:
                actual = verify_file(partial, spec)
                if actual is not None:
                    partial.replace(destination)
            if actual is not None:
                return dict(
                    **asdict(spec), path=str(destination.resolve()),
                    actual_sha256=actual["sha256"], actual_git_blob_sha1=actual["git_blob_sha1"],
                    verification_algorithm="sha256" if spec.sha256 else "git_blob_sha1",
                    actual_hash=actual["sha256" if spec.sha256 else "git_blob_sha1"],
                    reused_existing=reused, original_partial_bytes=original_partial,
                    verified_utc=utc_now(), attempts=attempt,
                )
            if offline:
                raise FileNotFoundError(f"Offline: no fully verified file for {spec.filename}")
            offset = partial.stat().st_size if partial.is_file() else 0
            if offset >= spec.size:
                partial.unlink()  # Full-size corrupt or oversized partial is not reusable.
                offset = 0
            headers = {"Accept-Encoding": "identity", "User-Agent": "TIG-BPP-pinned-downloader/1", "Cache-Control": "no-cache"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(url or resolve_url(spec), headers=headers)
            with urlopen(request, timeout=timeout) as response:
                status = response.status
                append = False
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if match is None or int(match[1]) != offset or int(match[3]) != spec.size or int(match[2]) >= spec.size:
                        raise ValueError(f"Invalid HTTP Content-Range for {spec.filename}")
                    append = offset > 0
                elif status != 200:
                    raise OSError(f"Unexpected HTTP status {status}")
                if not append:
                    offset = 0  # HTTP200 ignored Range: replace, never append a full response.
                with partial.open("ab" if append else "wb") as handle:
                    if on_progress:
                        on_progress(offset)
                    while True:
                        block = response.read(1 << 20)
                        if not block:
                            break
                        if offset + len(block) > spec.size:
                            raise ValueError(f"Response exceeds pinned size for {spec.filename}")
                        handle.write(block)
                        offset += len(block)
                        if on_progress:
                            on_progress(offset)
            if partial.stat().st_size != spec.size:
                raise OSError(f"Short HTTP download: {partial.stat().st_size}/{spec.size} for {spec.filename}")
            if verify_file(partial, spec) is None:
                partial.unlink()
                raise ValueError(f"Downloaded content hash mismatch: {spec.filename}")
            partial.replace(destination)
            # Next iteration performs fresh final-path verification; no stat-only success.
            actual = verify_file(destination, spec)
            if actual is None:
                raise ValueError(f"Final-path verification failed: {spec.filename}")
            return dict(
                **asdict(spec), path=str(destination.resolve()),
                actual_sha256=actual["sha256"], actual_git_blob_sha1=actual["git_blob_sha1"],
                verification_algorithm="sha256" if spec.sha256 else "git_blob_sha1",
                actual_hash=actual["sha256" if spec.sha256 else "git_blob_sha1"],
                reused_existing=False, original_partial_bytes=original_partial,
                verified_utc=utc_now(), attempts=attempt,
            )
        except Exception as exc:
            if on_retry:
                on_retry(attempt, exc)
            if offline or attempt == retries:
                raise
            if isinstance(exc, HTTPError) and exc.code == 416 and partial.exists():
                partial.unlink()
            time.sleep(min(30., retry_delay * attempt))
    raise RuntimeError("Unreachable retry state")


def load_inventories(paths: list[Path], folders: list[dict[str, Any]]) -> tuple[list[FileSpec], set[str], list[str]]:
    allowed = {row["repo_id"] for row in folders}
    files: dict[str, FileSpec] = {}
    covered, origins = set(), []
    for path in paths:
        if not path.is_file():
            continue
        source = json.loads(path.read_text())
        if source.get("repo_id") != REPO_ID or source.get("revision") != REVISION:
            raise ValueError(f"Source inventory is not the pinned dataset revision: {path}")
        if source.get("inventory_complete") is not True:
            continue
        origins.append(str(path.resolve()))
        local = []
        for row in source.get("files", []):
            if not selected_path(row["filename"], allowed):
                continue
            spec = FileSpec.from_record(row)
            if spec.filename in files and files[spec.filename] != spec:
                raise ValueError(f"Conflicting pinned inventory records: {spec.filename}")
            files[spec.filename] = spec
            local.append(spec)
        for folder in allowed:
            prefix = f"imitator_human_v1/{folder}/"
            relative = [item.filename[len(prefix):] for item in local if item.filename.startswith(prefix)]
            if "meta/info.json" in relative and any(value.startswith("meta/episodes/") for value in relative) and any(value.startswith(RGB_DIRECTORY) for value in relative):
                covered.add(folder)
    return sorted(files.values(), key=lambda item: item.filename), covered, origins


def list_folder(folder: str, retries: int, retry_delay: float, on_retry=None) -> list[FileSpec]:
    # Lazy optional dependency: help, plans, reuse and HTTP fixtures need no HF/Torch import.
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    for attempt in range(1, retries + 1):
        try:
            items = []
            for item in HfApi(token=False).list_repo_tree(REPO_ID, repo_type="dataset", revision=REVISION, path_in_repo=f"imitator_human_v1/{folder}", recursive=True):
                if not isinstance(item, RepoFile) or not selected_path(item.path, {folder}):
                    continue
                lfs = item.lfs
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                items.append(FileSpec(item.path, int(item.size), sha, item.blob_id))
            names = [item.filename for item in items]
            if not any(name.endswith("meta/info.json") for name in names) or not any("/meta/episodes/" in name for name in names) or not any(f"/{RGB_DIRECTORY}" in name for name in names):
                raise ValueError(f"Incomplete required RGB/metadata inventory: {folder}")
            return items
        except Exception as exc:
            if on_retry:
                on_retry(attempt, exc)
            if attempt == retries:
                raise
            time.sleep(min(30., retry_delay * attempt))
    raise RuntimeError("Unreachable discovery retry state")


@dataclass(frozen=True)
class DownloadConfig:
    scale: str
    dataset_root: Path = PROJECT_ROOT / "data/IG-10K-Dataset"
    tig_repo: Path = TIG_ROOT
    manifest_root: Path = PROJECT_ROOT / "logs/bpp_downloads"
    lock_root: Path = PROJECT_ROOT / "cache/bpp_download_locks"
    source_manifests: tuple[Path, ...] = ()
    workers: int = 6
    retries: int = 40
    timeout: float = 45.
    retry_delay: float = 3.
    offline: bool = False
    plan_only: bool = False


def run_download(config: DownloadConfig) -> dict[str, Any]:
    if config.workers < 1 or config.retries < 1:
        raise ValueError("workers and retries must be positive")
    for path in (config.dataset_root, config.manifest_root, config.lock_root):
        if not Path(path).resolve().is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError(f"All project downloads/logs/locks must stay below {PROJECT_ROOT}")
    folders, configs = selected_folders(config.tig_repo, config.scale)
    directory = config.manifest_root / config.scale
    inputs = [directory / "source.json", *(config.manifest_root / scale / "source.json" for scale in SCALE_CONFIGS), *config.source_manifests]
    if not config.source_manifests:
        inputs.extend(PROJECT_ROOT / "logs" / name for name in ("human_c15_target_source.json", "human_c30_c45_source.json"))
    files, covered, origins = load_inventories(list(dict.fromkeys(inputs)), folders)
    plan = dict(schema=SCHEMA, repo_id=REPO_ID, revision=REVISION, scale=config.scale, folders=folders, config_sources=configs,
                selected_folder_count=len(folders), selected_episode_references=sum(len(row["episode_ids"]) for row in folders),
                inventory_complete=len(covered) == len(folders), source_inventories=origins,
                missing_inventory_folders=sorted({row["repo_id"] for row in folders} - covered),
                files=[asdict(item) for item in files], plan_only=config.plan_only, downloads_started=False)
    if config.plan_only:
        atomic_json(directory / "plan.json", plan)
        return plan
    with file_lock(directory / "run.lock"):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        for name in ("source.json", "progress.json", "verification.json"):
            path = directory / name
            if path.is_file():
                archived = directory / "history" / f"{run_id}_{name}"
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, archived)
        started = time.monotonic()
        lock = threading.RLock()
        verified, errors, attempt_errors, active, partial_sizes = {}, [], [], {}, {}
        inventory_active: set[str] = set()
        state = dict(schema=SCHEMA, scale=config.scale, repo_id=REPO_ID, revision=REVISION, status="discovering",
                     selected_folder_count=len(folders), expected_files=None, expected_bytes=None,
                     completed_inventory_folders=len(covered),
                     verified_files=0, verified_bytes=0, partial_unverified_bytes=0, files=[], errors=errors)
        last_update = [0.]

        def snapshot(force=False):
            with lock:
                now = time.monotonic()
                if not force and now - last_update[0] < 2.:
                    return
                state.update(verified_files=len(verified), verified_bytes=sum(row["size"] for row in verified.values()),
                             partial_unverified_bytes=sum(partial_sizes.values()), active_files=sorted(active),
                             active_inventory_folders=sorted(inventory_active),
                             elapsed_seconds=now-started, updated_utc=utc_now())
                atomic_json(directory / "progress.json", {key: value for key, value in state.items() if key != "files"})
                atomic_json(directory / "verification.json", dict(
                    schema=SCHEMA, scale=config.scale, repo_id=REPO_ID, revision=REVISION, status=state["status"],
                    verified_files=len(verified), verified_bytes=state["verified_bytes"],
                    files=[verified[name] for name in sorted(verified)], errors=errors, attempt_errors=attempt_errors,
                ))
                last_update[0] = now

        snapshot(True)
        try:
            missing = plan["missing_inventory_folders"]
            if missing and config.offline:
                raise FileNotFoundError(f"Offline inventory missing folders: {missing}")
            if missing:
                def discover(folder):
                    with lock:
                        inventory_active.add(folder)

                    def retry_inventory(attempt, exc):
                        with lock:
                            attempt_errors.append(dict(phase="inventory", folder=folder, attempt=attempt, error_type=type(exc).__name__, error=str(exc), utc=utc_now()))
                        snapshot(True)

                    try:
                        items = list_folder(folder, config.retries, config.retry_delay, retry_inventory)
                        with lock:
                            state["completed_inventory_folders"] += 1
                        return items
                    finally:
                        with lock:
                            inventory_active.discard(folder)
                        snapshot(True)

                with ThreadPoolExecutor(max_workers=config.workers) as pool:
                    found = list(pool.map(discover, missing))
                merged = {item.filename: item for item in files}
                for items in found:
                    merged.update((item.filename, item) for item in items)
                files = sorted(merged.values(), key=lambda item: item.filename)
            source = dict(plan, inventory_complete=True, missing_inventory_folders=[], plan_only=False, files=[asdict(item) for item in files])
            atomic_json(directory / "source.json", source)
            state.update(status="downloading_and_verifying", expected_files=len(files), expected_bytes=sum(item.size for item in files))
            for item in files:
                partial = config.dataset_root / (item.filename + ".partial")
                if partial.is_file():
                    partial_sizes[item.filename] = partial.stat().st_size
            snapshot(True)

            def transfer(spec):
                with lock:
                    active[spec.filename] = 0

                def update(offset):
                    with lock:
                        active[spec.filename] = offset
                        partial_sizes[spec.filename] = offset
                    snapshot()

                def retry(attempt, exc):
                    with lock:
                        attempt_errors.append(dict(filename=spec.filename, attempt=attempt, error_type=type(exc).__name__, error=str(exc), utc=utc_now()))
                    snapshot(True)

                try:
                    key = hashlib.sha256(f"{REPO_ID}:{REVISION}:{spec.filename}".encode()).hexdigest()
                    with file_lock(config.lock_root / f"{key}.lock"):
                        record = download_file(spec, config.dataset_root, retries=config.retries, timeout=config.timeout, retry_delay=config.retry_delay, offline=config.offline, on_progress=update, on_retry=retry)
                    with lock:
                        verified[spec.filename] = record
                    return record
                except Exception as exc:
                    with lock:
                        errors.append(dict(filename=spec.filename, error_type=type(exc).__name__, error=str(exc)))
                    return None
                finally:
                    with lock:
                        active.pop(spec.filename, None)
                        partial = config.dataset_root / (spec.filename + ".partial")
                        if partial.is_file():
                            partial_sizes[spec.filename] = partial.stat().st_size
                        else:
                            partial_sizes.pop(spec.filename, None)
                    snapshot(True)

            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                for future in as_completed([pool.submit(transfer, spec) for spec in sorted(files, key=lambda item: item.size)]):
                    future.result()
            state["status"] = "complete" if not errors and len(verified) == len(files) else "error"
        except Exception as exc:
            state["status"] = "error"
            errors.append(dict(error_type=type(exc).__name__, error=str(exc)))
        finally:
            snapshot(True)
        return {key: value for key, value in state.items() if key != "files"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=tuple(SCALE_CONFIGS), required=True)
    parser.add_argument("--dataset-root", type=Path, default=DownloadConfig.dataset_root)
    parser.add_argument("--tig-repo", type=Path, default=TIG_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DownloadConfig.manifest_root)
    parser.add_argument("--lock-root", type=Path, default=DownloadConfig.lock_root)
    parser.add_argument("--source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=45.)
    parser.add_argument("--offline", action="store_true", help="Freshly verify existing/full partial files without HTTP")
    parser.add_argument("--plan-only", action="store_true", help="Read local inventory and print the selection; never access the network")
    values = vars(parser.parse_args(argv))
    values["source_manifests"] = tuple(values.pop("source_manifest"))
    result = run_download(DownloadConfig(**values))
    if result.get("plan_only"):
        summary = {key: result[key] for key in ("scale", "revision", "selected_folder_count", "selected_episode_references", "inventory_complete", "missing_inventory_folders", "downloads_started")}
        summary.update(status="planned", plan=str(values["manifest_root"] / values["scale"] / "plan.json"))
    else:
        summary = result
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.get("plan_only") or result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

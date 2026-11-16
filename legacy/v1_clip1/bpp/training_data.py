"""Train-only TIG robot supervision with human RGB prompts by default.

This module is independent of the frozen LIBERO evaluator. Preparation reads
the published Parquet columns directly and writes uint8 RGB NPY files. Dataset
instances keep query numeric arrays in RAM and open RGB caches as read-only
mmaps. Human prompts persist only16 uniformly selected frames per episode.
Robot L0 prompts remain an explicitly selected supplementary condition.
No HF/LeRobot runtime, simulator, model or training process is started here.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import threading
from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset

from .data import DATASET_REVISION, RGB_KEY, ReferenceVideo, file_sha256, read_reference_rgb, split_task_level


STATE_KEY = "observation.qpos_gripper_states"
ACTION_KEY = "action.qpos_gripper_actions"
CACHE_SCHEMA = "bpp_tig_native_training_cache_v1"
NORMALIZER_SCHEMA = "bpp_tig_native_normalizer_v1"
OBSERVATION_HORIZON = 2
ACTION_HORIZON = 16
PROMPT_FRAMES = 16
SPLIT_FILES = {
    "c15": "sim_train_config_15.json", "c30": "sim_train_config_30.json",
    "c45": "sim_train_config_45.json", "scratch": "sim_test_config_unseen.json",
}
HUMAN_SPLIT_FILES = {
    "c15": "human_train_config_15.json", "c30": "human_train_config_30.json",
    "c45": "human_train_config_45.json", "scratch": "human_test_config_unseen.json",
}
PROMPT_SOURCES = ("human", "robot")
PROJECT_ROOT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
LOGGER = logging.getLogger(__name__)
_VIDEO_PREFIX = f"videos/{RGB_KEY}/"
_EPISODE_COLUMNS = (
    "episode_index", "length", "dataset_from_index", "dataset_to_index",
    "data/chunk_index", "data/file_index",
    *(_VIDEO_PREFIX + field for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")),
)
_RGB_EPISODE_COLUMNS = ("episode_index", "length", *(_VIDEO_PREFIX + field for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")))
_NUMERIC_COLUMNS = ("episode_index", "frame_index", "index", STATE_KEY, ACTION_KEY)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _file_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _train_ids(value: str) -> list[int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected official train range start:stop, got {value!r}")
    first, stop = map(int, parts)
    if first < 0 or stop <= first:
        raise ValueError(f"Invalid official train range {value!r}")
    return list(range(first, stop))


def uniform_prompt_indices(length: int, count: int = PROMPT_FRAMES) -> np.ndarray:
    """Fixed-size RGB sampling, including both endpoints (repeats if short)."""
    if length <= 0 or count < 2:
        raise ValueError("Prompt needs a nonempty episode and at least two samples")
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def read_uniform_prompt_rgb(
    reference: ReferenceVideo, size: int = 224, count: int = PROMPT_FRAMES,
) -> np.ndarray:
    """Stream one episode while retaining only the requested RGB frames.

    No whole-episode image array or numerical demonstration data is loaded.
    All source frame positions are checked, so seek losses cannot silently
    shift the sampled indices; an incomplete seek retries from the beginning.
    """
    import av
    indices = uniform_prompt_indices(reference.length, count)
    selected = set(indices.tolist())
    stop = reference.from_timestamp + reference.length / reference.fps
    tolerance = .25 / reference.fps
    if abs(reference.to_timestamp - stop) > 1.25 / reference.fps:
        raise ValueError("Prompt video duration and episode length disagree")

    def decode(*, seek: bool) -> np.ndarray:
        values = {}
        observed = 0
        with av.open(str(reference.video_path)) as container:
            stream = container.streams.video[0]
            if seek and reference.from_timestamp > 0:
                seek_time = max(0., reference.from_timestamp - 1.)
                container.seek(int(seek_time / float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is None:
                    raise ValueError("Prompt video frame has no timestamp")
                if frame.time < reference.from_timestamp - tolerance:
                    continue
                if frame.time >= stop - tolerance:
                    break
                expected = reference.from_timestamp + observed / reference.fps
                if abs(frame.time - expected) > tolerance:
                    raise ValueError(f"Prompt frame timestamp gap at index {observed}: {frame.time} vs {expected}")
                if observed in selected:
                    values[observed] = frame.reformat(width=size, height=size, format="rgb24").to_ndarray()
                observed += 1
                if observed >= reference.length:
                    break
        if observed != reference.length or set(values) != selected:
            raise ValueError(f"Short prompt RGB episode: {observed}/{reference.length}")
        return np.ascontiguousarray(np.stack([values[int(index)] for index in indices]))

    errors = []
    for seek in ((True, False) if reference.from_timestamp > 0 else (False,)):
        try:
            return decode(seek=seek)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(
        f"Cannot sample exact prompt RGB for {reference.dataset_id}:{reference.episode_index}, "
        f"range=[{reference.from_timestamp}, {stop}): {'; '.join(errors)}"
    )


def _selections(
    tig_repo: Path, split: str, tasks: Iterable[str] | None, episodes: Iterable[int] | None,
    prompt_source: str,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if split not in SPLIT_FILES:
        raise ValueError(f"Unsupported training split: {split}")
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(f"Unsupported prompt source: {prompt_source}")
    path = tig_repo / "examples/baselines/lerobot_dataset/config/exp_configs" / SPLIT_FILES[split]
    rows = json.loads(path.read_text())
    official = {row["repo_id"]: _train_ids(row["train"]) for row in rows}
    official_ranges = {row["repo_id"]: row["train"] for row in rows}
    if len(official) != len(rows):
        raise ValueError("Official split contains duplicate repositories")
    chosen_tasks = set(official) if tasks is None else set(tasks)
    if not chosen_tasks or not chosen_tasks.issubset(official):
        raise ValueError(f"Tasks absent from {split}: {sorted(chosen_tasks - set(official))}")
    episode_filter = None if episodes is None else set(int(value) for value in episodes)
    if episode_filter is not None and not episode_filter:
        raise ValueError("Episode selection cannot be empty")
    resources: dict[str, Any] = {"prompt_source": prompt_source, "task_pairings": {}}
    human_official = {}
    task_humans = {}
    if prompt_source == "human":
        human_path = path.parent / HUMAN_SPLIT_FILES[split]
        human_rows = json.loads(human_path.read_text())
        human_official = {row["repo_id"]: _train_ids(row["train"]) for row in human_rows}
        if len(human_official) != len(human_rows):
            raise ValueError("Official human split contains duplicate repositories")
        mapping_path = tig_repo / "examples/baselines/lerobot_dataset/task_mapping.json"
        for mapping in json.loads(mapping_path.read_text())["task_mappings"]:
            for robot_task in mapping.get("sim_task_id", []):
                human = mapping["human_task_id"]
                if robot_task in task_humans and task_humans[robot_task] != human:
                    raise ValueError(f"Ambiguous human mapping for {robot_task}")
                task_humans[robot_task] = human
        resources.update(
            prompt_split_path=str(human_path.resolve()), prompt_split_sha256=file_sha256(human_path),
            task_mapping_path=str(mapping_path.resolve()), task_mapping_sha256=file_sha256(mapping_path),
        )
    queries, prompts = [], {}
    for repo_id in sorted(chosen_tasks):
        _, base = split_task_level(repo_id)
        ids = official[repo_id]
        if episode_filter is not None:
            if not episode_filter.issubset(ids):
                raise ValueError(f"Requested episodes expand the official train split: {repo_id}")
            ids = sorted(episode_filter)
        queries.append({"repo_id": repo_id, "episode_ids": ids, "official_train": official_ranges[repo_id]})
        prompt_repo = f"L0_{base}" if prompt_source == "robot" else task_humans.get(repo_id)
        prompt_official = official if prompt_source == "robot" else human_official
        if prompt_repo not in prompt_official:
            raise ValueError(f"Paired {prompt_source} prompt task is absent from this training split: {prompt_repo}")
        prompt_ids = prompt_official[prompt_repo]
        if episode_filter is not None:
            if not episode_filter.issubset(prompt_ids):
                raise ValueError(f"Requested prompt episodes expand the train split: {prompt_repo}")
            prompt_ids = sorted(episode_filter)
        if prompt_source == "robot" and any(not (set(prompt_ids) - {episode}) for episode in ids):
            raise ValueError("Training needs another allowed L0 episode for every query episode")
        resources["task_pairings"][repo_id] = prompt_repo
        if prompt_repo in prompts and prompts[prompt_repo]["task_base"] != base:
            raise ValueError(f"Human prompt repository maps to multiple selected base tasks: {prompt_repo}")
        prompts[prompt_repo] = {"repo_id": prompt_repo, "task_base": base, "episode_ids": prompt_ids}
    return path, queries, [prompts[key] for key in sorted(prompts)], resources


def _repository_records(
    root: Path, repo_id: str, ids: set[int], query_ids: set[int], *, task_base: str | None = None,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    repo = root / repo_id
    info_path = repo / "meta/info.json"
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    if features.get(RGB_KEY, {}).get("dtype") != "video":
        raise ValueError(f"No released ZED RGB video in {info_path}")
    if query_ids and any(features.get(key, {}).get("shape") != [width] for key, width in ((STATE_KEY, 18), (ACTION_KEY, 16))):
        raise ValueError(f"Expected native state18/action16 in {info_path}")
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid released video FPS: {info_path}")
    found = {}
    for metadata in sorted((repo / "meta/episodes").glob("chunk-*/*.parquet")):
        parquet = pq.ParquetFile(metadata)
        allowed_columns = _EPISODE_COLUMNS if query_ids else _RGB_EPISODE_COLUMNS
        columns = [key for key in allowed_columns if key in parquet.schema_arrow.names]
        # Statistics, demo actions and qpos fields are excluded from this read.
        for row in parquet.read(columns=columns).to_pylist():
            episode = int(row["episode_index"])
            if episode not in ids:
                continue
            if episode in found:
                raise ValueError(f"Duplicate episode metadata: {repo_id}:{episode}")
            length = int(row["length"])
            if length <= 0:
                raise ValueError(f"Empty training episode: {repo_id}:{episode}")
            replacements = dict(
                video_key=RGB_KEY, episode_index=episode,
                episode_chunk=episode // int(info.get("chunks_size", 1000)),
                chunk_index=int(row.get(_VIDEO_PREFIX + "chunk_index", 0)),
                file_index=int(row.get(_VIDEO_PREFIX + "file_index", 0)),
            )
            start = float(row.get(_VIDEO_PREFIX + "from_timestamp", 0.))
            record = dict(
                repo_id=repo_id, task_base=task_base or split_task_level(repo_id)[1], episode_index=episode,
                length=length, fps=fps, info_path=str(info_path.resolve()),
                episode_metadata_path=str(metadata.resolve()),
                video_path=str((repo / info["video_path"].format(**replacements)).resolve()),
                from_timestamp=start,
                to_timestamp=float(row.get(_VIDEO_PREFIX + "to_timestamp", start + length / fps)),
            )
            if episode in query_ids:
                replacements.update(chunk_index=int(row.get("data/chunk_index", 0)), file_index=int(row.get("data/file_index", 0)))
                record["data_path"] = str((repo / info["data_path"].format(**replacements)).resolve())
                record["dataset_from_index"] = int(row["dataset_from_index"])
                record["dataset_to_index"] = int(row["dataset_to_index"])
            found[episode] = record
    if set(found) != ids:
        raise IndexError(f"Missing allowed episodes in {repo_id}: {sorted(ids - set(found))}")
    return [found[index] for index in sorted(found)]


def _array_artifact(path: Path, expected_shape: tuple[int, ...], dtype: str, producer, *, verify_hashes: bool) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    with _file_lock(path.with_suffix(".lock")):
        reusable = False
        if path.is_file() and sidecar.is_file():
            description = json.loads(sidecar.read_text())
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                reusable = (
                    value.shape == expected_shape and value.dtype == np.dtype(dtype)
                    and path.stat().st_size == description["size_bytes"]
                    and description.get("path") == str(path.resolve())
                    and description.get("shape") == list(expected_shape)
                    and description.get("dtype") == str(np.dtype(dtype))
                    and description.get("mtime_ns", path.stat().st_mtime_ns) == path.stat().st_mtime_ns
                )
                if reusable and verify_hashes:
                    reusable = file_sha256(path) == description["sha256"]
                del value
            except (OSError, ValueError, KeyError):
                reusable = False
        if not reusable:
            value = np.asarray(producer())
            if value.shape != expected_shape or value.dtype != np.dtype(dtype):
                raise ValueError(f"Cache producer returned {value.shape}/{value.dtype}, expected {expected_shape}/{dtype}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError("Non-finite robot training values")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, value, allow_pickle=False)
            temporary.replace(path)
            description = dict(path=str(path.resolve()), shape=list(expected_shape), dtype=str(np.dtype(dtype)), size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=file_sha256(path))
            _write_json(sidecar, description)
        return description


def _numeric_episodes(records: list[dict[str, Any]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import pyarrow.parquet as pq
    by_path: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if "data_path" in record:
            by_path.setdefault(record["data_path"], []).append(record)
    result = {}
    for path, requested in by_path.items():
        ids = [record["episode_index"] for record in requested]
        table = pq.read_table(path, columns=list(_NUMERIC_COLUMNS), filters=[("episode_index", "in", ids)]).to_pydict()
        episodes = np.asarray(table["episode_index"], dtype=np.int64)
        frames = np.asarray(table["frame_index"], dtype=np.int64)
        indices = np.asarray(table["index"], dtype=np.int64)
        states = np.asarray(table[STATE_KEY], dtype=np.float32)
        actions = np.asarray(table[ACTION_KEY], dtype=np.float32)
        for record in requested:
            episode = record["episode_index"]
            selected = np.flatnonzero(episodes == episode)
            selected = selected[np.argsort(frames[selected])]
            length = record["length"]
            if len(selected) != length or not np.array_equal(frames[selected], np.arange(length)):
                raise ValueError(f"Noncontiguous or mismatched training frames: {path}:{episode}")
            expected = np.arange(record["dataset_from_index"], record["dataset_to_index"])
            if not np.array_equal(indices[selected], expected):
                raise ValueError(f"Episode global bounds disagree with numeric rows: {path}:{episode}")
            if states[selected].shape != (length, 18) or actions[selected].shape != (length, 16):
                raise ValueError("Released robot dimensions differ from state18/action16")
            result[episode] = (states[selected].copy(), actions[selected].copy())
    return result


def _fit_normalizer(records: list[dict[str, Any]], source: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in ("minmax", "quantile"):
        raise ValueError(f"Unknown normalizer mode: {mode}")
    result: dict[str, Any] = dict(schema=NORMALIZER_SCHEMA, mode=mode, source=source, selected_bounds={})
    for field, artifact_key, width in (("state", "states", 18), ("action", "actions", 16)):
        values = np.concatenate([np.load(record[artifact_key]["path"], allow_pickle=False) for record in records], axis=0)
        if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError(f"Invalid train-only {field} values")
        quantiles = np.quantile(values, [0.01, 0.99], axis=0)
        stats = dict(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist(), q01=quantiles[0].tolist(), q99=quantiles[1].tolist(), count=int(len(values)))
        result[field] = stats
        result["selected_bounds"][field] = dict(min=stats["min" if mode == "minmax" else "q01"], max=stats["max" if mode == "minmax" else "q99"])
    result["transform"] = "linear to [-1,1], no clipping; constant dimensions map training constant to zero"
    return result


def _validate_artifacts(manifest: dict[str, Any], *, verify_hashes: bool) -> None:
    seen = set()
    for record in manifest["query_episodes"] + manifest["prompt_episodes"]:
        for key in ("rgb", "states", "actions"):
            artifact = record.get(key)
            if artifact is None or artifact["path"] in seen:
                continue
            seen.add(artifact["path"])
            path = Path(artifact["path"])
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(value.shape) != artifact["shape"] or str(value.dtype) != artifact["dtype"] or path.stat().st_size != artifact["size_bytes"]:
                raise ValueError(f"Training cache shape/dtype/size changed: {path}")
            if verify_hashes and file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"Training cache checksum changed: {path}")
            if artifact.get("mtime_ns", path.stat().st_mtime_ns) != path.stat().st_mtime_ns:
                raise ValueError(f"Training cache modification time changed: {path}")
            del value


def prepare_training_data(
    dataset_root: Path, tig_repo: Path, cache_root: Path, split: str = "c15", workers: int = 8, *,
    prompt_source: str = "human",
    tasks: Iterable[str] | None = None, episodes: Iterable[int] | None = None,
    normalizer_mode: str = "quantile", image_size: int = 224, prompt_frames: int = PROMPT_FRAMES,
    verify_hashes: bool = False,
) -> Path:
    """Prepare an explicit split and return its immutable manifest path.

    ``tasks``/``episodes`` make declared pilot subsets; they cannot expand the
    official train ranges. Robot query C15 RGB uses about156 GB (reused across
    conditions); human prompts add only16 RGB frames per episode, about1.8 GB.
    C15/C30/C45 artifacts share paths by source video/episode content; only
    selected robot query pairings enter each scale's human training pool.
    Only rank0 should call preparation before a distributed training barrier.
    """
    dataset_root, tig_repo, cache_root = map(lambda value: Path(value).resolve(), (dataset_root, tig_repo, cache_root))
    if not cache_root.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"Project caches must stay below {PROJECT_ROOT}")
    if workers < 1 or image_size < 1 or prompt_frames != PROMPT_FRAMES:
        raise ValueError("workers/image_size must be positive; this native protocol fixes prompt_frames=16")
    if normalizer_mode not in ("minmax", "quantile"):
        raise ValueError("normalizer_mode must be minmax or quantile")
    sim_root = dataset_root / "imitator_sim_v1_zed2i" if (dataset_root / "imitator_sim_v1_zed2i").is_dir() else dataset_root
    human_root = sim_root.parent / "imitator_human_v1"
    split_path, query_pools, prompt_pools, prompt_selection = _selections(tig_repo, split, tasks, episodes, prompt_source)
    query_ids = {row["repo_id"]: set(row["episode_ids"]) for row in query_pools}
    prompt_ids = {row["repo_id"]: set(row["episode_ids"]) for row in prompt_pools}
    prompt_bases = {row["repo_id"]: row["task_base"] for row in prompt_pools}
    repositories = {}
    for repo_id in sorted(set(query_ids) | set(prompt_ids)):
        human_prompt = prompt_source == "human" and repo_id in prompt_ids
        records = _repository_records(
            human_root if human_prompt else sim_root, repo_id,
            query_ids.get(repo_id, set()) | prompt_ids.get(repo_id, set()), query_ids.get(repo_id, set()),
            task_base=prompt_bases[repo_id] if human_prompt else None,
        )
        for record in records:
            record["source_kind"] = "human" if human_prompt else "robot"
        repositories[repo_id] = records
    query_source_paths = {split_path.resolve()}
    prompt_source_paths = {
        Path(prompt_selection[key]) for key in ("prompt_split_path", "task_mapping_path") if key in prompt_selection
    }
    for repo_id, records in repositories.items():
        for record in records:
            for key in ("info_path", "episode_metadata_path", "video_path", "data_path"):
                if key in record:
                    if record["episode_index"] in query_ids.get(repo_id, set()):
                        query_source_paths.add(Path(record[key]))
                    if key != "data_path" and record["episode_index"] in prompt_ids.get(repo_id, set()):
                        prompt_source_paths.add(Path(record[key]))
    request = dict(
        schema=CACHE_SCHEMA, dataset_root=str(sim_root), dataset_revision=DATASET_REVISION,
        split=split, split_path=str(split_path.resolve()), split_sha256=file_sha256(split_path),
        prompt_source=prompt_source, prompt_selection=prompt_selection,
        prompt_root=str(human_root if prompt_source == "human" else sim_root),
        query_pools=query_pools, prompt_pools=prompt_pools,
        observation_horizon=OBSERVATION_HORIZON, action_horizon=ACTION_HORIZON,
        prompt_frames=prompt_frames, image_size=image_size, normalizer_mode=normalizer_mode,
        anchor_policy="one query anchor per released row; no task balancing",
        action_tail_policy="released final duplicate row retained; future padding repeats final action and is masked",
        prompt_policy=(
            "official paired human RGB only; all allowed human episodes, including same numeric episode id; "
            "deterministic seed/epoch/query draw;16 uniform frames including endpoints"
            if prompt_source == "human" else
            "same-base L0 RGB only; different episode index; deterministic seed/epoch/query draw;16 uniform frames including endpoints"
        ),
        source_signatures=[_signature(path) for path in sorted(query_source_paths | prompt_source_paths)],
    )
    request_identity = _identity(request)
    directory = cache_root / "manifests" / f"{split}-{prompt_source}-{request_identity[:24]}"
    manifest_path = directory / "manifest.json"
    with _file_lock(cache_root / ".locks" / f"{request_identity}.lock"):
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("request") != request:
                raise ValueError("Cached training request differs")
            unsigned = {key: value for key, value in manifest.items() if key not in ("manifest_identity", "normalizer")}
            if _identity(unsigned) != manifest.get("manifest_identity"):
                raise ValueError("Cached training manifest identity changed")
            _validate_artifacts(manifest, verify_hashes=verify_hashes)
            load_normalizer(manifest)
            LOGGER.info("Reusing %s (%d anchors)", manifest_path, manifest["length"])
            return manifest_path
        hashes: dict[str, str] = {}
        hash_lock = threading.Lock()

        def source_hash(path: str | Path) -> str:
            key = str(Path(path).resolve())
            with hash_lock:
                if key not in hashes:
                    hashes[key] = file_sha256(key)
                return hashes[key]

        def prepare_repository(item):
            repo_id, records = item
            try:
                numeric = _numeric_episodes(records) if any("data_path" in record for record in records) else {}
            except Exception as exc:
                raise RuntimeError(
                    f"Numeric preparation failed: repo={repo_id}, "
                    f"episodes={[row['episode_index'] for row in records]}: {exc}"
                ) from exc
            queries, prompts = [], []
            for record in records:
                episode = record["episode_index"]
                sampled_prompt = record["source_kind"] == "human"
                rgb_spec = dict(
                    video_sha256=source_hash(record["video_path"]),
                    from_timestamp=record["from_timestamp"], to_timestamp=record["to_timestamp"],
                    length=record["length"], fps=record["fps"], image_size=image_size,
                    rgb_reader="released_timestamp_and_exact_length_v1",
                )
                if sampled_prompt:
                    rgb_spec.update(rgb_reader="streamed_uniform_prompt_v1", frame_indices=uniform_prompt_indices(record["length"], prompt_frames).tolist())
                rgb_identity = _identity(rgb_spec)
                rgb_path = cache_root / ("prompt_rgb16/human" if sampled_prompt else "rgb") / repo_id / rgb_identity / f"episode_{episode:06d}.npy"

                def decode(record=record):
                    reference = ReferenceVideo(
                        "human_video_only" if sampled_prompt else "robot_video_only", record["task_base"], repo_id, episode,
                        Path(record["video_path"]), Path(record["info_path"]), Path(record["episode_metadata_path"]),
                        record["fps"], record["length"], record["from_timestamp"], record["to_timestamp"],
                    )
                    if sampled_prompt:
                        return read_uniform_prompt_rgb(reference, size=image_size, count=prompt_frames)
                    return read_reference_rgb(reference, size=image_size)

                try:
                    rgb = _array_artifact(rgb_path, (prompt_frames if sampled_prompt else record["length"], image_size, image_size, 3), "uint8", decode, verify_hashes=verify_hashes)
                except Exception as exc:
                    raise RuntimeError(
                        f"RGB preparation failed: repo={repo_id}, episode={episode}, "
                        f"length={record['length']}, fps={record['fps']}, "
                        f"range=[{record['from_timestamp']}, {record['to_timestamp']}], "
                        f"video={record['video_path']}, metadata={record['episode_metadata_path']}: {exc}"
                    ) from exc
                shared = {key: value for key, value in record.items() if key not in ("data_path", "dataset_from_index", "dataset_to_index")}
                shared["rgb"] = rgb
                if sampled_prompt:
                    shared["rgb_frame_indices"] = rgb_spec["frame_indices"]
                    shared["rgb_layout"] = "uniform_prompt_frames"
                if episode in query_ids.get(repo_id, set()):
                    numeric_key = source_hash(record["data_path"])
                    numerical_root = cache_root / "numeric" / repo_id / numeric_key
                    query = dict(shared, data_path=record["data_path"], dataset_from_index=record["dataset_from_index"], dataset_to_index=record["dataset_to_index"])
                    for key, values, width in (("states", numeric[episode][0], 18), ("actions", numeric[episode][1], 16)):
                        path = numerical_root / f"episode_{episode:06d}_{key}.npy"
                        query[key] = _array_artifact(path, (record["length"], width), "float32", lambda values=values: values, verify_hashes=verify_hashes)
                    queries.append(query)
                if episode in prompt_ids.get(repo_id, set()):
                    # RGB prompt records intentionally contain no numeric data/cache paths.
                    prompts.append(shared)
            LOGGER.info(
                "Prepared %s: %d query episodes, %d prompt episodes, %d query frames",
                repo_id, len(queries), len(prompts), sum(record["length"] for record in queries),
            )
            return queries, prompts

        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(prepare_repository, repositories.items()))
        query_records = sorted([record for queries, _ in prepared for record in queries], key=lambda row: (row["repo_id"], row["episode_index"]))
        prompt_records = sorted([record for _, prompts in prepared for record in prompts], key=lambda row: (row["repo_id"], row["episode_index"]))
        source_map = {signature["path"]: dict(signature, sha256=source_hash(signature["path"])) for signature in request["source_signatures"]}
        sources = [source_map[str(path.resolve())] for path in sorted(query_source_paths)]
        prompt_sources = [source_map[str(path.resolve())] for path in sorted(prompt_source_paths)]
        base_manifest = dict(
            schema=CACHE_SCHEMA, request=request, source_files=sources,
            prompt_source_files=prompt_sources, prompt_source=prompt_source,
            query_episodes=query_records, prompt_episodes=prompt_records,
            query_episode_count=len(query_records), prompt_episode_count=len(prompt_records),
            length=sum(row["length"] for row in query_records),
            source_video_fps_note="released video timestamps; not an action resampling instruction",
            execution_control_hz=20, states_feature=STATE_KEY, actions_feature=ACTION_KEY,
            prompt_modalities=["RGB"], numerical_prompt_inputs=False,
        )
        manifest_identity = _identity(base_manifest)
        normalizer = _fit_normalizer(query_records, dict(
            manifest_hash=manifest_identity, manifest_identity_scope="manifest excluding identity and normalizer descriptor",
            split=split, split_sha256=request["split_sha256"], query_pools=query_pools,
            query_source_identity=_identity(dict(query_pools=query_pools, source_files=sources)),
            fitting_scope="query training rows only; no prompt-only or held-out rows",
        ), normalizer_mode)
        normalizer_path = directory / "normalizer.json"
        _write_json(normalizer_path, normalizer)
        manifest = dict(base_manifest, manifest_identity=manifest_identity, normalizer=dict(path=str(normalizer_path.resolve()), sha256=file_sha256(normalizer_path)))
        _write_json(manifest_path, manifest)
    return manifest_path


def load_normalizer(manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text())
    descriptor = manifest["normalizer"]
    if file_sha256(descriptor["path"]) != descriptor["sha256"]:
        raise ValueError("Cached normalizer checksum changed")
    stats = json.loads(Path(descriptor["path"]).read_text())
    if stats.get("schema") != NORMALIZER_SCHEMA or stats["source"]["manifest_hash"] != manifest["manifest_identity"]:
        raise ValueError("Normalizer belongs to another training manifest")
    return stats


class RobotVideoTrainingDataset(Dataset):
    """Robot supervision with unbalanced anchors and RGB-only human prompts.

    Public sample keys: obs.agentview_rgb uint8[2,3,H,W],
    obs.qpos_gripper_states float32[2,18], prompt.agentview_rgb
    uint8[16,3,H,W], action float32[16,16], action_mask bool[16]
    (True=valid), and metadata. Values remain in original physical units.
    ``set_epoch`` uses shared state so persistent DataLoader workers see it.
    Use DataLoader multiprocessing_context="spawn" with num_workers>0 after
    video preparation or CUDA initialization, and collate_training_batch.
    """
    def __init__(self, manifest_path: str | Path, seed: int = 1, *, mmap_cache_size: int = 8):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != CACHE_SCHEMA:
            raise ValueError("Not a native TIG training manifest")
        unsigned = {key: value for key, value in self.manifest.items() if key not in ("manifest_identity", "normalizer")}
        if _identity(unsigned) != self.manifest["manifest_identity"]:
            raise ValueError("Training manifest identity changed")
        if mmap_cache_size < 1:
            raise ValueError("mmap_cache_size must be positive")
        self.manifest_identity = self.manifest["manifest_identity"]
        self.normalizer_stats = load_normalizer(self.manifest)
        # Legacy manifests are the explicitly supplementary robot-prompt run.
        self.prompt_source = self.manifest.get("prompt_source", self.manifest["request"].get("prompt_source", "robot"))
        if self.prompt_source not in PROMPT_SOURCES:
            raise ValueError("Unknown manifest prompt source")
        self.records = self.manifest["query_episodes"]
        self.seed = int(seed)
        self._epoch = multiprocessing.get_context("spawn").Value("q", 0)
        self._mmap_cache_size = mmap_cache_size
        self._rgb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._numeric = []
        self._ends = []
        self._prompt_pools: dict[str, list[dict[str, Any]]] = {}
        for record in self.manifest["prompt_episodes"]:
            if "states" in record or "actions" in record or "data_path" in record:
                raise ValueError("Numeric references are forbidden in prompt records")
            self._prompt_pools.setdefault(record["task_base"], []).append(record)
        total = 0
        for record in self.records:
            states = np.load(record["states"]["path"], allow_pickle=False)
            actions = np.load(record["actions"]["path"], allow_pickle=False)
            if states.shape != (record["length"], 18) or actions.shape != (record["length"], 16):
                raise ValueError("Query cache dimensions disagree with the manifest")
            states.setflags(write=False)
            actions.setflags(write=False)
            self._numeric.append((states, actions))
            total += record["length"]
            self._ends.append(total)
            candidates = self._prompt_candidates(record)
            if not candidates:
                raise ValueError("No allowed RGB prompt episode for a query")
        if total != self.manifest["length"] or not total:
            raise ValueError("Invalid training anchor count")

    def __len__(self) -> int:
        return self._ends[-1]

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _prompt_candidates(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        pool = self._prompt_pools.get(record["task_base"], [])
        if self.prompt_source == "robot":
            return [item for item in pool if item["episode_index"] != record["episode_index"]]
        # Human and robot episode numbers identify different demonstrations.
        return pool

    def _rgb(self, record: dict[str, Any]) -> np.ndarray:
        path = record["rgb"]["path"]
        if path not in self._rgb_cache:
            self._rgb_cache[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            if len(self._rgb_cache) > self._mmap_cache_size:
                self._rgb_cache.popitem(last=False)
        self._rgb_cache.move_to_end(path)
        return self._rgb_cache[path]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_rgb_cache"] = OrderedDict()
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect_right(self._ends, index)
        start = self._ends[record_index - 1] if record_index else 0
        anchor = index - start
        record = self.records[record_index]
        states, actions = self._numeric[record_index]
        history_indices = np.maximum(np.arange(anchor - 1, anchor + 1), 0)
        action_indices = anchor + np.arange(ACTION_HORIZON)
        valid = action_indices < record["length"]
        padded_indices = np.minimum(action_indices, record["length"] - 1)
        candidates = self._prompt_candidates(record)
        epoch = int(self._epoch.value)
        draw = _identity([self.seed, epoch, record["repo_id"], record["episode_index"], anchor])
        prompt = candidates[int(draw, 16) % len(candidates)]
        prompt_indices = uniform_prompt_indices(prompt["length"], self.manifest["request"]["prompt_frames"])
        if prompt.get("rgb_layout") == "uniform_prompt_frames":
            if prompt.get("rgb_frame_indices") != prompt_indices.tolist():
                raise ValueError("Cached prompt sampling differs from the requested endpoint-inclusive indices")
            prompt_rgb = self._rgb(prompt)
            if len(prompt_rgb) != len(prompt_indices):
                raise ValueError("Sampled prompt cache length differs from16")
        else:
            prompt_rgb = self._rgb(prompt)[prompt_indices]
        return dict(
            obs=dict(
                agentview_rgb=np.ascontiguousarray(self._rgb(record)[history_indices].transpose(0, 3, 1, 2)),
                qpos_gripper_states=states[history_indices].copy(),
            ),
            prompt=dict(agentview_rgb=np.ascontiguousarray(prompt_rgb.transpose(0, 3, 1, 2))),
            action=actions[padded_indices].copy(), action_mask=valid,
            metadata=dict(
                query_repo=record["repo_id"], query_episode=record["episode_index"], anchor=anchor,
                history_indices=history_indices.tolist(), prompt_repo=prompt["repo_id"],
                prompt_source=self.prompt_source,
                prompt_episode=prompt["episode_index"], prompt_frame_indices=prompt_indices.tolist(),
                epoch=epoch, sampling_seed=self.seed,
            ),
        )


def collate_training_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors without normalizing, scaling RGB or forwarding metadata."""
    import torch
    if not samples:
        raise ValueError("Cannot collate an empty training batch")
    return dict(
        obs={key: torch.from_numpy(np.stack([sample["obs"][key] for sample in samples])) for key in ("agentview_rgb", "qpos_gripper_states")},
        prompt={"agentview_rgb": torch.from_numpy(np.stack([sample["prompt"]["agentview_rgb"] for sample in samples]))},
        action=torch.from_numpy(np.stack([sample["action"] for sample in samples])),
        action_mask=torch.from_numpy(np.stack([sample["action_mask"] for sample in samples])),
        metadata=[sample["metadata"] for sample in samples],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare RGB/numeric caches and train-only statistics; does not train")
    prepare.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data/IG-10K-Dataset")
    prepare.add_argument("--tig-repo", type=Path, default=PROJECT_ROOT / "repos/The-Imitator-Game")
    prepare.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "cache/bpp_tig_training")
    prepare.add_argument("--split", choices=tuple(SPLIT_FILES), required=True)
    prepare.add_argument("--prompt-source", choices=PROMPT_SOURCES, default="human")
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--tasks", nargs="+")
    prepare.add_argument("--episodes", type=int, nargs="+")
    prepare.add_argument("--normalizer-mode", choices=("minmax", "quantile"), default="quantile")
    prepare.add_argument("--verify-hashes", action="store_true")
    values = vars(parser.parse_args(argv))
    values.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = prepare_training_data(**values)
    manifest = json.loads(path.read_text())
    print(json.dumps(dict(manifest=str(path), identity=manifest["manifest_identity"], query_episodes=manifest["query_episode_count"], anchors=manifest["length"], training_started=False), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""RGB-only TIG reference selection.

Only video features and episode/video metadata are read here. Frame-level
Parquet, demonstration actions, joint states, depth, masks and MANO are never
loaded. An L0 robot reference is deliberately shared by all levels of a task.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RGB_KEY = "observation.images.zed2i"
DATASET_REVISION = "9a57b27edcb54823d8da7d2b77efa61c092ffa95"
CONDITIONS = ("robot_video_only", "human_video_only")
REFERENCE_SELECTIONS = ("per_seed", "fixed")
REFERENCE_SEED_ORIGIN = 100000
_EPISODE_COLUMNS = (
    "episode_index", "length", "dataset_from_index", "dataset_to_index",
    f"videos/{RGB_KEY}/chunk_index", f"videos/{RGB_KEY}/file_index",
    f"videos/{RGB_KEY}/from_timestamp", f"videos/{RGB_KEY}/to_timestamp",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_task_level(task_id: str) -> tuple[str, str]:
    """Return level and the base Gym ID, rejecting ambiguous names."""
    level, separator, base = task_id.partition("_")
    if not separator or level not in ("L0", "L1", "L2", "L3"):
        raise ValueError(f"Expected L0_... through L3_... task alias: {task_id!r}")
    if not base.startswith("TwoRobot") or not base.endswith("-v1"):
        raise ValueError(f"Not a TIG dual-task alias: {task_id!r}")
    return level, base


def reference_episode_for_seed(
    seed: int, *, selection: str = "per_seed", reference_episode: int = 0,
) -> int:
    """Choose a reference independently of task level, condition or worker.

    Zero-WAM evaluates episode i with reset seed 100000+i and reference i.
    ``per_seed`` retains that relation, with ``reference_episode`` as an
    explicit offset. There is no modulo or reindexing by the worker's seed
    list. ``fixed`` always selects the specified reference episode.
    """
    if selection not in REFERENCE_SELECTIONS:
        raise ValueError(f"Unsupported reference selection: {selection}")
    if reference_episode < 0:
        raise ValueError("Reference episode offset must be non-negative")
    episode = reference_episode
    if selection == "per_seed":
        episode += seed - REFERENCE_SEED_ORIGIN
    if episode < 0:
        raise ValueError(
            f"Seed {seed} selects negative reference episode {episode}; "
            "increase --reference-episode or use --reference-selection fixed"
        )
    return int(episode)


@dataclass(frozen=True)
class ReferenceVideo:
    condition: str
    task_base: str
    dataset_id: str
    episode_index: int
    video_path: Path
    info_path: Path
    episode_metadata_path: Path
    fps: float
    length: int
    from_timestamp: float
    to_timestamp: float
    dataset_revision: str = DATASET_REVISION
    camera: str = "zed2i"

    def manifest(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = asdict(self)
        for key in ("video_path", "info_path", "episode_metadata_path"):
            result[key] = str(result[key])
        result["prompt_modalities"] = ["RGB"]
        result["feature_key"] = RGB_KEY
        result["reference_level"] = "L0" if self.condition == CONDITIONS[0] else None
        result["timebase"] = "released video/LeRobot timestamps; not inferred control Hz"
        if include_hash:
            result["video_sha256"] = file_sha256(self.video_path)
            result["info_sha256"] = file_sha256(self.info_path)
        return result


def _find_human_id(mapping_path: Path, base: str) -> str:
    mapping = json.loads(mapping_path.read_text())["task_mappings"]
    matches = []
    for item in mapping:
        sim_ids = item.get("sim_task_id", [])
        if any(value == f"L0_{base}" or value == base for value in sim_ids):
            matches.append(item["human_task_id"])
    if len(set(matches)) != 1:
        raise ValueError(f"Expected one human pairing for {base}, found {matches}")
    return matches[0]


def _episode_row(root: Path, episode: int) -> tuple[dict[str, Any], Path]:
    if episode < 0:
        raise ValueError("Reference episode must be non-negative")
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if paths:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow is required to select a LeRobot video episode") from exc
        for path in paths:
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            columns = [name for name in _EPISODE_COLUMNS if name in names]
            # Do not read stats/action/proprioception columns from metadata.
            for row in parquet.read(columns=columns).to_pylist():
                if int(row.get("episode_index", -1)) == episode:
                    return row, path
        raise IndexError(f"Episode {episode} absent from metadata under {root}")
    legacy = root / "meta" / "episodes.jsonl"
    if legacy.is_file():
        for line in legacy.read_text().splitlines():
            row = json.loads(line)
            if int(row.get("episode_index", -1)) == episode:
                return {key: row[key] for key in _EPISODE_COLUMNS if key in row}, legacy
        raise IndexError(f"Episode {episode} absent from {legacy}")
    raise FileNotFoundError(f"No episode video metadata found under {root / 'meta'}")


def select_reference(
    dataset_root: str | Path,
    task_mapping: str | Path,
    task_id: str,
    condition: str,
    episode_index: int = 0,
    *,
    dataset_revision: str = DATASET_REVISION,
) -> ReferenceVideo:
    """Choose a released ZED RGB episode without loading numerical demos."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported prompt condition: {condition}")
    _, base = split_task_level(task_id)
    dataset_root = Path(dataset_root)
    if condition == "robot_video_only":
        dataset_id = f"L0_{base}"
        root = dataset_root / "imitator_sim_v1_zed2i" / dataset_id
    else:
        dataset_id = _find_human_id(Path(task_mapping), base)
        root = dataset_root / "imitator_human_v1" / dataset_id
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feature = info.get("features", {}).get(RGB_KEY)
    if feature is None or feature.get("dtype") not in ("video", "image"):
        raise ValueError(f"Released reference has no ZED RGB video: {info_path}")
    if feature["dtype"] != "video":
        raise ValueError("This reference reader requires the released video layout")
    row, metadata_path = _episode_row(root, episode_index)
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS in {info_path}: {fps}")
    length = int(row.get("length", 0))
    if length <= 0:
        raise ValueError(f"Empty reference episode {dataset_id}:{episode_index}")
    prefix = f"videos/{RGB_KEY}/"
    start = float(row.get(prefix + "from_timestamp", 0.0))
    stop = float(row.get(prefix + "to_timestamp", start + length / fps))
    pattern = info["video_path"]
    replacements = dict(
        video_key=RGB_KEY, chunk_index=int(row.get(prefix + "chunk_index", 0)),
        file_index=int(row.get(prefix + "file_index", 0)),
        episode_index=episode_index,
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
    )
    video_path = root / pattern.format(**replacements)
    if not video_path.is_file():
        raise FileNotFoundError(f"Reference RGB video not downloaded: {video_path}")
    return ReferenceVideo(
        condition=condition, task_base=base, dataset_id=dataset_id,
        episode_index=episode_index, video_path=video_path, info_path=info_path,
        episode_metadata_path=metadata_path, fps=fps, length=length,
        from_timestamp=start, to_timestamp=stop, dataset_revision=dataset_revision,
    )


def resize_rgb(frame: Any, size: int = 224) -> np.ndarray:
    """RGB HWC uint8; never reorder or flip a TIG camera implicitly."""
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    value = np.asarray(frame)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected a single HWC RGB frame, got {value.shape}")
    if value.dtype != np.uint8:
        if not np.isfinite(value).all():
            raise ValueError("RGB frame contains non-finite values")
        if np.issubdtype(value.dtype, np.floating) and value.size and value.max() <= 1:
            value = value * 255
        value = np.clip(value, 0, 255).astype(np.uint8)
    if value.shape[:2] != (size, size):
        from PIL import Image
        value = np.asarray(Image.fromarray(value).resize((size, size), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(value)


def read_reference_rgb(reference: ReferenceVideo, size: int = 224) -> np.ndarray:
    """Decode exactly one episode from a potentially multi-episode video file."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required for RGB-only reference decoding") from exc
    expected_stop = reference.from_timestamp + reference.length / reference.fps
    # LeRobot versions have used either the last frame timestamp or an
    # exclusive end timestamp. The explicit episode length resolves that
    # one-frame ambiguity without borrowing a frame from the next episode.
    if abs(reference.to_timestamp - expected_stop) > 1.25 / reference.fps:
        raise ValueError("Reference video duration and episode length disagree")
    tolerance = 0.25 / reference.fps

    def decode(*, seek: bool) -> tuple[list[np.ndarray], list[float]]:
        frames, timestamps = [], []
        with av.open(str(reference.video_path)) as container:
            stream = container.streams.video[0]
            if seek and reference.from_timestamp > 0:
                # Some released HEVC files use open/reordered GOPs: seeking at
                # the exact episode boundary can omit that boundary's I-frame
                # even with backward=True. Decode a pre-roll, then apply the
                # SAME episode timestamp/length limits below.
                seek_time = max(0., reference.from_timestamp - 1.)
                timestamp = int(seek_time / float(stream.time_base))
                container.seek(timestamp, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.time is None:
                    raise ValueError("Reference video has no timestamps")
                if frame.time < reference.from_timestamp - tolerance:
                    continue
                if frame.time >= expected_stop - tolerance:
                    break
                frames.append(frame.reformat(width=size, height=size, format="rgb24").to_ndarray())
                timestamps.append(float(frame.time))
                if len(frames) >= reference.length:
                    break
        return frames, timestamps

    frames, timestamps = decode(seek=True)
    first_attempt_length = len(frames)
    if len(frames) != reference.length and reference.from_timestamp > 0:
        # A clean sequential decode can recover codec seek losses, but never
        # changes boundaries, repeats pixels, or fills truly missing frames.
        frames, timestamps = decode(seek=False)
    if not frames:
        raise ValueError(f"No RGB frames decoded for {reference.dataset_id}:{reference.episode_index}")
    # A missing tail must not quietly select frames from the next episode.
    if len(frames) != reference.length:
        raise ValueError(
            f"Short reference RGB episode: {len(frames)}/{reference.length} "
            f"for {reference.dataset_id}:{reference.episode_index}; "
            f"range=[{reference.from_timestamp}, {expected_stop}), "
            f"decoded_pts=[{timestamps[0]}, {timestamps[-1]}], "
            f"pre_roll_attempt_frames={first_attempt_length}"
        )
    return np.ascontiguousarray(np.stack(frames))


def history_batch(history: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack measured per-control-step observations as [batch=1,time,...]."""
    values = list(history)
    if len(values) != 2:
        raise ValueError("BPP-LIBERO requires the two most recent control observations")
    keys = ("agentview_rgb", "eye_in_hand_rgb", "ee_pos", "ee_ori", "gripper_states")
    return {key: np.stack([value[key] for value in values], axis=0)[None] for key in keys}

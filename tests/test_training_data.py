"""Native train-only data contracts, including a small real TIG fixture."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.baselines.bpp.bpp import training_data as data


PROJECT_ROOT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
TASK_BASE = "TwoRobotPickRemoteControl-v1"
L0 = f"L0_{TASK_BASE}"
L1 = f"L1_{TASK_BASE}"
HUMAN = "human_H4"
RUNTIME_DATA = PROJECT_ROOT / "data/IG-10K-Dataset"
RUNTIME_TIG = PROJECT_ROOT / "repos/The-Imitator-Game"
HAS_VIDEO_DATA = importlib.util.find_spec("av") is not None and importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(HAS_VIDEO_DATA, "PyAV and PyArrow required for native cache fixtures")
class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        (PROJECT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="bpp-training-tests-", dir=PROJECT_ROOT / "tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset_root = self.root / "datasets"
        self.tig_repo = self.root / "tig"
        self.cache_root = self.root / "cache"
        self.split_directory = self.tig_repo / "examples/baselines/lerobot_dataset/config/exp_configs"
        self.split_directory.mkdir(parents=True)
        self.rows = [dict(repo_id=f"L{level}_{TASK_BASE}", root=f"L{level}_{TASK_BASE}", train="0:3", test="") for level in range(4)]
        for filename in data.SPLIT_FILES.values():
            (self.split_directory / filename).write_text(json.dumps(self.rows))
        for filename in data.HUMAN_SPLIT_FILES.values():
            (self.split_directory / filename).write_text(json.dumps([dict(repo_id=HUMAN, root=HUMAN, train="0:3", test="")]))
        mapping = self.tig_repo / "examples/baselines/lerobot_dataset/task_mapping.json"
        mapping.write_text(json.dumps({"task_mappings": [{
            "human_task_id": HUMAN, "sim_task_id": [row["repo_id"] for row in self.rows],
        }]}))
        self.expected = {}
        self.make_repository(L0, numeric=False)
        self.make_repository(L1, numeric=True)
        self.make_repository(HUMAN, numeric=False, source="human", lengths=(21, 22, 23, 2))

    def make_repository(self, repo_id, *, numeric, source="robot", lengths=(3, 4, 2, 2)):
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = self.dataset_root / ("imitator_human_v1" if source == "human" else "imitator_sim_v1_zed2i") / repo_id
        (root / "meta/episodes/chunk-000").mkdir(parents=True)
        (root / "data/chunk-000").mkdir(parents=True)
        video = root / f"videos/{data.RGB_KEY}/chunk-000/file-000.mp4"
        video.parent.mkdir(parents=True)
        fps = 10
        info = dict(
            fps=fps, chunks_size=1000,
            features={data.RGB_KEY: {"dtype": "video", "shape": [3, 16, 16]},
                      data.STATE_KEY: {"shape": [18], "dtype": "float32"},
                      data.ACTION_KEY: {"shape": [16], "dtype": "float32"}},
            data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        )
        (root / "meta/info.json").write_text(json.dumps(info))
        metadata, frame_rows = [], []
        offset = 0
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for episode, length in enumerate(lengths):
                metadata.append({
                    "episode_index": episode, "length": length,
                    "dataset_from_index": offset, "dataset_to_index": offset + length,
                    "data/chunk_index": 0, "data/file_index": 0,
                    data._VIDEO_PREFIX + "chunk_index": 0, data._VIDEO_PREFIX + "file_index": 0,
                    data._VIDEO_PREFIX + "from_timestamp": offset / fps,
                    data._VIDEO_PREFIX + "to_timestamp": (offset + length) / fps,
                    # Whitelist reads must never treat metadata statistics as
                    # either prompt features or normalizer training rows.
                    "stats/action.qpos_gripper_actions/max": [99999.] * 16,
                })
                states, actions = [], []
                for frame in range(length):
                    value = float(10000 if episode == 3 else episode * 10 + frame)
                    state = np.full(18, value, dtype=np.float32) + np.arange(18, dtype=np.float32) / 100
                    action = np.full(16, value + .5, dtype=np.float32) + np.arange(16, dtype=np.float32) / 100
                    states.append(state)
                    actions.append(action)
                    frame_rows.append(dict(
                        episode_index=episode, frame_index=frame, index=offset + frame,
                        **{data.STATE_KEY: state.tolist(), data.ACTION_KEY: action.tolist()},
                    ))
                    image = np.full((16, 16, 3), 30 + episode * 40 + frame * 2, np.uint8)
                    video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                self.expected[(repo_id, episode)] = (np.stack(states), np.stack(actions))
                offset += length
            for packet in stream.encode():
                container.mux(packet)
        pq.write_table(pa.Table.from_pylist(metadata), root / "meta/episodes/chunk-000/file-000.parquet")
        path = root / "data/chunk-000/file-000.parquet"
        if numeric:
            pq.write_table(pa.Table.from_pylist(frame_rows), path)
        else:
            # L0 is prompt-only for these tests. Preparation must succeed even
            # when its numeric Parquet is unreadable.
            path.write_bytes(b"FORBIDDEN PROMPT NUMERICS; intentionally not Parquet")

    def prepare(self, **kwargs):
        options = dict(split="scratch", tasks=[L1], workers=2, image_size=8, prompt_source="robot")
        options.update(kwargs)
        return data.prepare_training_data(self.dataset_root, self.tig_repo, self.cache_root, **options)

    def test_human_is_default_allows_same_episode_id_and_persists_only_sixteen_rgb_frames(self):
        manifest_path = data.prepare_training_data(
            self.dataset_root, self.tig_repo, self.cache_root, split="scratch", workers=1,
            tasks=[L1], episodes=[0], image_size=8,
        )
        dataset = data.RobotVideoTrainingDataset(manifest_path)
        self.assertEqual(dataset.prompt_source, "human")
        self.assertEqual(len(dataset), 3)
        sample = dataset[0]
        self.assertEqual(sample["metadata"]["query_episode"], sample["metadata"]["prompt_episode"])
        self.assertEqual(sample["metadata"]["prompt_repo"], HUMAN)
        self.assertEqual(sample["metadata"]["prompt_frame_indices"][0], 0)
        self.assertEqual(sample["metadata"]["prompt_frame_indices"][-1], 20)
        self.assertEqual(set(sample["prompt"]), {"agentview_rgb"})
        self.assertEqual(sample["prompt"]["agentview_rgb"].shape, (16, 3, 8, 8))
        prompt = dataset.manifest["prompt_episodes"][0]
        self.assertEqual(prompt["rgb"]["shape"], [16, 8, 8, 3])
        self.assertEqual(prompt["length"], 21)
        self.assertTrue(set(prompt).isdisjoint({"states", "actions", "data_path"}))
        self.assertFalse((self.cache_root / "rgb" / HUMAN).exists())
        self.assertTrue(all("imitator_human_v1" not in row["path"] for row in dataset.manifest["source_files"]))
        self.assertTrue(any("imitator_human_v1" in row["path"] for row in dataset.manifest["prompt_source_files"]))
        self.assertFalse(any("/data/" in row["path"].replace(str(self.dataset_root), "") for row in dataset.manifest["prompt_source_files"]))
        self.assertEqual(dataset.normalizer_stats["state"]["count"], 3)

    def test_human_source_changes_identity_reuses_robot_query_cache_and_keeps_robot_experiment(self):
        robot_path = self.prepare()
        robot_text = robot_path.read_text()
        robot = data.RobotVideoTrainingDataset(robot_path)
        # Human preparation may read only its RGB stream; all robot query RGB
        # caches must already be reusable, and prompt numeric files are invalid.
        with patch.object(data, "read_reference_rgb", side_effect=AssertionError("Robot query RGB cache was decoded again")):
            human_path = self.prepare(prompt_source="human")
        human = data.RobotVideoTrainingDataset(human_path)
        self.assertNotEqual(robot_path, human_path)
        self.assertNotEqual(robot.manifest_identity, human.manifest_identity)
        self.assertEqual(robot_path.read_text(), robot_text)
        self.assertEqual([row["rgb"] for row in robot.records], [row["rgb"] for row in human.records])
        for field in ("state", "action", "selected_bounds"):
            self.assertEqual(robot.normalizer_stats[field], human.normalizer_stats[field])
        self.assertEqual(robot.normalizer_stats["source"]["query_source_identity"], human.normalizer_stats["source"]["query_source_identity"])
        self.assertEqual(sum(row["rgb"]["shape"][0] for row in human.manifest["prompt_episodes"]), 48)

    def test_human_task_pairing_is_shared_across_all_four_levels(self):
        _, queries, prompts, resources = data._selections(self.tig_repo, "scratch", None, [0], "human")
        self.assertEqual(len(queries), 4)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["repo_id"], HUMAN)
        self.assertEqual(resources["task_pairings"], {f"L{level}_{TASK_BASE}": HUMAN for level in range(4)})

    def test_all_scales_share_existing_query_and_human_rgb_artifacts(self):
        # The extra C45 human row has no query pairing and no local data at all.
        # It must be excluded instead of being opened or fitted accidentally.
        c45_human = self.split_directory / data.HUMAN_SPLIT_FILES["c45"]
        c45_human.write_text(json.dumps([
            dict(repo_id=HUMAN, root=HUMAN, train="0:3", test=""),
            dict(repo_id="human_H999", root="human_H999", train="0:3", test=""),
        ]))
        baseline = data.RobotVideoTrainingDataset(self.prepare(split="c15", prompt_source="human"))
        manifest_paths = {baseline.manifest_path}
        for split in ("c30", "c45"):
            with self.subTest(split=split), patch.object(data, "read_reference_rgb", side_effect=AssertionError("Query RGB duplicated across scales")), patch.object(data, "read_uniform_prompt_rgb", side_effect=AssertionError("Human prompt RGB duplicated across scales")):
                dataset = data.RobotVideoTrainingDataset(self.prepare(split=split, prompt_source="human"))
                manifest_paths.add(dataset.manifest_path)
                self.assertEqual([row["rgb"] for row in baseline.records], [row["rgb"] for row in dataset.records])
                self.assertEqual([row["states"] for row in baseline.records], [row["states"] for row in dataset.records])
                self.assertEqual([row["actions"] for row in baseline.records], [row["actions"] for row in dataset.records])
                self.assertEqual(baseline.manifest["prompt_episodes"], dataset.manifest["prompt_episodes"])
                self.assertEqual({row["repo_id"] for row in dataset.manifest["prompt_episodes"]}, {HUMAN})
                self.assertEqual(dataset.normalizer_stats["state"]["count"], 9)
                self.assertEqual(dataset.manifest["request"]["split"], split)
        self.assertEqual(len(manifest_paths), 3)

    def test_missing_paired_human_pool_is_rejected_instead_of_substituting_unpaired_data(self):
        for split in ("c15", "c30", "c45"):
            with self.subTest(split=split):
                path = self.split_directory / data.HUMAN_SPLIT_FILES[split]
                path.write_text(json.dumps([dict(repo_id="human_H999", root="human_H999", train="0:3", test="")]))
                with self.assertRaisesRegex(ValueError, "Paired human prompt task is absent"):
                    self.prepare(split=split, prompt_source="human")

    def test_windows_masks_and_l0_rgb_prompt_never_cross_episode_or_read_prompt_numeric(self):
        manifest_path = self.prepare()
        dataset = data.RobotVideoTrainingDataset(manifest_path, seed=7)
        self.assertEqual(len(dataset), 9)  # 3+4+2, no balancing or extra anchors.
        beginning = dataset[0]
        np.testing.assert_array_equal(beginning["obs"]["qpos_gripper_states"], np.repeat(self.expected[(L1, 0)][0][:1], 2, axis=0))
        end = dataset[2]
        np.testing.assert_array_equal(end["action"], np.repeat(self.expected[(L1, 0)][1][-1:], 16, axis=0))
        np.testing.assert_array_equal(end["action_mask"], [True] + [False] * 15)
        next_episode = dataset[3]
        self.assertEqual(next_episode["metadata"]["query_episode"], 1)
        np.testing.assert_array_equal(next_episode["obs"]["qpos_gripper_states"], np.repeat(self.expected[(L1, 1)][0][:1], 2, axis=0))
        for index in range(len(dataset)):
            sample = dataset[index]
            metadata = sample["metadata"]
            self.assertEqual(metadata["prompt_repo"], L0)
            self.assertNotEqual(metadata["query_episode"], metadata["prompt_episode"])
            self.assertEqual(set(sample["prompt"]), {"agentview_rgb"})
            self.assertEqual(sample["prompt"]["agentview_rgb"].shape, (16, 3, 8, 8))
            self.assertEqual(sample["prompt"]["agentview_rgb"].dtype, np.uint8)
            self.assertEqual(metadata["prompt_frame_indices"][0], 0)
            self.assertEqual(metadata["prompt_frame_indices"][-1], self.expected[(L0, metadata["prompt_episode"])][0].shape[0] - 1)
        manifest = dataset.manifest
        for prompt in manifest["prompt_episodes"]:
            self.assertTrue(set(prompt).isdisjoint({"states", "actions", "data_path"}))
        self.assertFalse(any("FORBIDDEN" in str(row) for row in dataset[0]["prompt"].values()))
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_normalizer_fits_only_query_train_rows_and_preserves_raw_units(self):
        for mode in ("quantile", "minmax"):
            with self.subTest(mode=mode):
                dataset = data.RobotVideoTrainingDataset(self.prepare(normalizer_mode=mode))
                states = np.concatenate([self.expected[(L1, episode)][0] for episode in range(3)])
                actions = np.concatenate([self.expected[(L1, episode)][1] for episode in range(3)])
                stats = dataset.normalizer_stats
                self.assertEqual(stats["source"]["manifest_hash"], dataset.manifest_identity)
                self.assertEqual(stats["source"]["query_pools"][0]["episode_ids"], [0, 1, 2])
                for field, values in (("state", states), ("action", actions)):
                    self.assertEqual(stats[field]["count"], 9)
                    np.testing.assert_allclose(stats[field]["min"], values.min(axis=0))
                    np.testing.assert_allclose(stats[field]["max"], values.max(axis=0))
                    bounds = stats["selected_bounds"][field]
                    expected = np.quantile(values, [.01, .99], axis=0) if mode == "quantile" else (values.min(axis=0), values.max(axis=0))
                    np.testing.assert_allclose(bounds["min"], expected[0])
                    np.testing.assert_allclose(bounds["max"], expected[1])
                np.testing.assert_array_equal(dataset[0]["action"][0], actions[0])
                self.assertLess(max(stats["state"]["max"]), 100.)

    def test_prepare_reuses_cache_but_changed_split_and_corruption_are_detected(self):
        first = self.prepare()
        with patch.object(data, "read_reference_rgb", side_effect=AssertionError("RGB decoded twice")), patch.object(data, "_numeric_episodes", side_effect=AssertionError("Numeric Parquet loaded twice")):
            self.assertEqual(self.prepare(), first)
        dataset = data.RobotVideoTrainingDataset(first)
        artifact = dataset.manifest["query_episodes"][0]["rgb"]
        array = np.load(artifact["path"], mmap_mode="r+")
        array[0, 0, 0, 0] ^= 255
        array.flush()
        del array
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            self.prepare(verify_hashes=True)
        split_path = self.split_directory / data.SPLIT_FILES["scratch"]
        split_path.write_text(json.dumps([dict(row, train="0:2") for row in self.rows]))
        second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertEqual(len(data.RobotVideoTrainingDataset(second)), 7)

    def test_split_restrictions_disallow_expansion_or_identical_prompt_episode(self):
        with self.assertRaisesRegex(ValueError, "expand the official train split"):
            self.prepare(episodes=[0, 3])
        with self.assertRaisesRegex(ValueError, "another allowed L0 episode"):
            self.prepare(episodes=[0])
        with self.assertRaisesRegex(ValueError, "Tasks absent"):
            self.prepare(tasks=["L2_TwoRobotStirSpoon-v1"])

    def test_decode_errors_identify_repository_episode_and_timestamp_bounds(self):
        with patch.object(data, "read_reference_rgb", side_effect=ValueError("Short reference RGB episode: 2/3")):
            with self.assertRaisesRegex(RuntimeError, rf"repo={L0}, episode=0, length=3, fps=10.0, range=\[0.0, 0.3\].*metadata=.*Short reference"):
                self.prepare()

    def test_prompt_sampling_is_worker_independent_and_epoch_updates_persistent_workers(self):
        dataset = data.RobotVideoTrainingDataset(self.prepare(), seed=91)
        direct0 = [dataset[index]["metadata"] for index in range(len(dataset))]
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=3, num_workers=2, persistent_workers=True,
            multiprocessing_context="spawn", collate_fn=data.collate_training_batch,
        )
        try:
            batches = list(loader)
            worker0 = [item for batch in batches for item in batch["metadata"]]
            self.assertEqual(direct0, worker0)
            self.assertEqual(batches[0]["obs"]["agentview_rgb"].dtype, torch.uint8)
            self.assertEqual(tuple(batches[0]["obs"]["qpos_gripper_states"].shape), (3, 2, 18))
            self.assertEqual(tuple(batches[0]["action"].shape), (3, 16, 16))
            self.assertEqual(batches[0]["action_mask"].dtype, torch.bool)
            dataset.set_epoch(3)
            direct3 = [dataset[index]["metadata"] for index in range(len(dataset))]
            worker3 = [item for batch in loader for item in batch["metadata"]]
            self.assertEqual(direct3, worker3)
            self.assertTrue(all(item["epoch"] == 3 for item in worker3))
            self.assertTrue(any(left["prompt_episode"] != right["prompt_episode"] for left, right in zip(direct0, direct3)))
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    @unittest.skipUnless((RUNTIME_DATA / "imitator_sim_v1_zed2i" / L0 / "meta/info.json").is_file(), "Real TIG robot fixture not downloaded")
    def test_real_remote_control_episodes_zero_and_one_have_exact_boundaries(self):
        manifest_path = data.prepare_training_data(
            RUNTIME_DATA, RUNTIME_TIG, self.root / "real_cache", split="scratch", workers=1,
            tasks=[L0], episodes=[0, 1], verify_hashes=True, prompt_source="robot",
        )
        dataset = data.RobotVideoTrainingDataset(manifest_path, seed=1)
        self.assertEqual([row["episode_index"] for row in dataset.records], [0, 1])
        first, second = dataset.records
        self.assertEqual((first["length"], second["length"]), (169, 180))
        self.assertEqual(len(dataset), 349)
        last_first = dataset[first["length"] - 1]
        first_second = dataset[first["length"]]
        expected_last_action = np.load(first["actions"]["path"])[-1]
        np.testing.assert_array_equal(last_first["action"], np.repeat(expected_last_action[None], 16, axis=0))
        np.testing.assert_array_equal(last_first["action_mask"], [True] + [False] * 15)
        expected_second_start = np.load(second["states"]["path"])[0]
        np.testing.assert_array_equal(first_second["obs"]["qpos_gripper_states"], np.repeat(expected_second_start[None], 2, axis=0))
        self.assertEqual(last_first["metadata"]["prompt_episode"], 1)
        self.assertEqual(first_second["metadata"]["prompt_episode"], 0)
        self.assertEqual(last_first["metadata"]["prompt_frame_indices"][-1], 179)
        self.assertEqual(first_second["metadata"]["prompt_frame_indices"][-1], 168)
        self.assertEqual(first_second["obs"]["agentview_rgb"].shape, (2, 3, 224, 224))
        self.assertEqual(dataset.normalizer_stats["state"]["count"], 349)
        self.assertEqual(dataset.normalizer_stats["action"]["count"], 349)
        with patch.object(data, "read_reference_rgb", side_effect=AssertionError("Real fixture decoded twice")):
            self.assertEqual(data.prepare_training_data(RUNTIME_DATA, RUNTIME_TIG, self.root / "real_cache", split="scratch", workers=1, tasks=[L0], episodes=[0, 1], prompt_source="robot"), manifest_path)


@unittest.skipUnless((RUNTIME_TIG / "examples/baselines/lerobot_dataset/task_mapping.json").is_file(), "Official TIG split resources unavailable")
class OfficialTrainingSplitTests(unittest.TestCase):
    def test_official_nested_scales_and_target_pools_have_exact_budgets_without_leakage(self):
        selections = {}
        for split, counts in (
            ("c15", (60, 3000, 15, 750)), ("c30", (120, 6000, 30, 1500)),
            ("c45", (180, 9000, 45, 2250)), ("scratch", (20, 200, 5, 50)),
        ):
            with self.subTest(split=split):
                _, queries, prompts, resources = data._selections(RUNTIME_TIG, split, None, None, "human")
                self.assertEqual((len(queries), sum(len(row["episode_ids"]) for row in queries), len(prompts), sum(len(row["episode_ids"]) for row in prompts)), counts)
                allowed = list(range(10 if split == "scratch" else 50))
                self.assertTrue(all(row["episode_ids"] == allowed for row in queries + prompts))
                self.assertEqual(set(resources["task_pairings"]), {row["repo_id"] for row in queries})
                self.assertEqual(set(resources["task_pairings"].values()), {row["repo_id"] for row in prompts})
                selections[split] = ({row["repo_id"] for row in queries}, {row["repo_id"] for row in prompts})
        for index in (0, 1):
            self.assertLess(selections["c15"][index], selections["c30"][index])
            self.assertLess(selections["c30"][index], selections["c45"][index])
            self.assertTrue(selections["c45"][index].isdisjoint(selections["scratch"][index]))
        path = RUNTIME_TIG / "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json"
        configured = {row["repo_id"] for row in json.loads(path.read_text())}
        self.assertEqual(len(configured), 48)
        self.assertEqual(configured - selections["c45"][1], {"human_H25", "human_H37", "human_H38"})

    def test_heldout_tasks_and_extra_episodes_cannot_expand_any_pretraining_scale(self):
        for split in ("c15", "c30", "c45"):
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "Tasks absent"):
                    data._selections(RUNTIME_TIG, split, [L0], None, "human")
                with self.assertRaisesRegex(ValueError, "expand the official train split"):
                    data._selections(RUNTIME_TIG, split, None, [0, 50], "human")
        with self.assertRaisesRegex(ValueError, "Tasks absent"):
            data._selections(RUNTIME_TIG, "scratch", ["L0_TwoRobotStirSpoon-v1"], None, "human")
        with self.assertRaisesRegex(ValueError, "expand the official train split"):
            data._selections(RUNTIME_TIG, "scratch", None, [0, 10], "human")


if __name__ == "__main__":
    unittest.main()

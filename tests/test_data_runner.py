"""Integration contracts for RGB references and the frozen evaluation loop.

Most cases use a deterministic environment/model double, so simulator assets
and checkpoint downloads are unnecessary. A real PyAV/Parquet fixture is also
checked when those optional runtime dependencies are installed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.baselines.bpp.bpp import runner
from examples.baselines.bpp.bpp.data import RGB_KEY, ReferenceVideo, read_reference_rgb, select_reference


TASK = "L0_TwoRobotPickRemoteControl-v1"
BASE = "TwoRobotPickRemoteControl-v1"
VIDEO_PREFIX = f"videos/{RGB_KEY}/"
PROJECT_TMP = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp")) / "tmp"


@dataclass
class _ArmState:
    tcp_pose_world: np.ndarray
    base_pose_world: np.ndarray
    finger_qpos: np.ndarray


class _Control:
    def __init__(self, active_arm="panda_wristcam-0"):
        self.active_arm = active_arm
        self.policy_from_world = np.eye(4)
        self.tcp_to_libero_eef = np.eye(4)
        self.osc_output_max = np.array([.05, .05, .05, .5, .5, .5])
        self.workspace_bounds = np.array([[-2.] * 3, [2.] * 3])

    def reset(self, states):
        self.initial_states = states

    def proprioception(self, state):
        return dict(
            ee_pos=state.tcp_pose_world[:3],
            ee_ori=np.array([1., 0., 0., 0., 1., 0.]),
            gripper_states=state.finger_qpos * [1., -1.],
        )

    def step(self, action, states):
        return types.SimpleNamespace(
            actions={uid: np.zeros(7, dtype=np.float32) for uid in states},
            diagnostics={"clip_events": {}},
        )


class _Model:
    model_info = {"variant": "frozen model test double"}

    def __init__(self, *args, failures=0, **kwargs):
        self.observations = []
        self.prompt_inputs = []
        self.reset_seeds = []
        self.remaining_failures = failures

    def reset(self, seed=None):
        self.reset_seeds.append(seed)

    def prepare_video(self, value, fps=None):
        if not isinstance(value, np.ndarray) or value.ndim != 4 or value.shape[-1] != 3:
            raise AssertionError("Prompt must be an RGB ndarray, not a sensorimotor dictionary")
        self.prompt_inputs.append(value.copy())
        return {"modalities": ["RGB"], "frames": len(value), "fps": fps}

    def get_action(self, obs):
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise RuntimeError("injected inference failure")
        self.observations.append({key: value.copy() for key, value in obs.items()})
        output = np.zeros((1, 16, 10), dtype=np.float32)
        output[:, :, 3] = 1.
        output[:, :, 7] = 1.
        return output


class _Environment:
    """Known success at step two, lost again before the episode ends."""
    control_freq = 20

    def __init__(self, config, task):
        self.unwrapped = self
        self.horizon = config.max_episode_steps
        self.t = 0
        self.actions = []
        self.closed = False
        pose = lambda x: types.SimpleNamespace(raw_pose=np.array([[x, 0., .5, 1., 0., 0., 0.]]))
        agents = {}
        for uid in ("panda_wristcam-0", "panda_wristcam-1"):
            agents[uid] = types.SimpleNamespace(
                tcp=types.SimpleNamespace(pose=pose(0.)),
                robot=types.SimpleNamespace(
                    pose=pose(0.),
                    get_qpos=lambda: np.array([[0.] * 7 + [.03, .03]]),
                ),
            )
        self.agent = types.SimpleNamespace(agents_dict=agents)

    def observation(self):
        image = np.full((1, 224, 224, 3), self.t, dtype=np.uint8)
        return {
            "sensor_data": {
                "zed2i": {"rgb": image},
                "panda_wristcam_0_hand_camera": {"rgb": image},
            },
            # These must never be forwarded to either policy interface.
            "extra": {"goal_pos": np.full(3, 9001.), "is_grasped": True},
            "privileged_simulator_state": np.full(36, 9001.),
        }

    def reset(self, seed):
        self.seed, self.t = seed, 0
        return self.observation(), self.evaluate()

    def step(self, actions):
        if set(actions) != {"panda_wristcam-0", "panda_wristcam-1"}:
            raise AssertionError("Unexpected actuator dictionary")
        if any(value.shape != (1, 7) for value in actions.values()):
            raise AssertionError("EEF targets must have a single explicit batch")
        self.actions.append(actions)
        self.t += 1
        self.agent.agents_dict["panda_wristcam-0"].tcp.pose.raw_pose[0, 0] = self.t
        return self.observation(), 0., self.t == 2, self.t == self.horizon, self.evaluate()

    def evaluate(self):
        return {
            "success": np.array([self.t == 2]),
            "peak_r_reach": np.array([min(self.t / 3., 1.)]),
        }

    def close(self):
        self.closed = True


class _Fixture(unittest.TestCase):
    def setUp(self):
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="bpp-runner-tests-", dir=PROJECT_TMP)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "frozen.ckpt"
        self.checkpoint.write_bytes(b"fixture weights; never loaded")
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps({
            "training": False,
            "conditions": [{"id": "robot_video_only"}, {"id": "human_video_only"}],
        }))
        self.video = self.root / "rgb.mp4"
        self.video.write_bytes(b"RGB video fixture; mock decoder")
        self.info = self.root / "info.json"
        self.info.write_text("{}")
        self.episodes = self.root / "episodes.parquet"
        self.episodes.write_bytes(b"metadata fixture")
        self.frames = np.zeros((5, 224, 224, 3), dtype=np.uint8)
        self.config = runner.RunConfig(
            checkpoint=self.checkpoint, output_dir=self.root / "results",
            dataset_root=self.root / "datasets", tig_repo=self.root / "tig",
            bpp_repo=self.root / "bpp", protocol_path=self.protocol,
            max_episode_steps=6, execution_horizon=3, capture_video=False,
        )
        self.fake_control_module = types.SimpleNamespace(ArmState=_ArmState, ControlAdapter=_Control)

    def reference(self, condition="robot_video_only", episode_index=0):
        return ReferenceVideo(
            condition, BASE,
            f"L0_{BASE}" if condition == "robot_video_only" else "human_H4",
            episode_index, self.video, self.info, self.episodes,
            30., len(self.frames), 0., len(self.frames) / 30.,
        )

    def episode(self, model, *, env_factory=_Environment, identity="experiment"):
        with patch.dict(sys.modules, {"examples.baselines.bpp.bpp.control": self.fake_control_module}):
            return runner.run_episode(
                self.config, model, self.reference(), self.frames, "robot_video_only",
                TASK, 100000, identity, env_factory=env_factory, control_factory=_Control,
            )


class EpisodeContractTests(_Fixture):
    def test_every_control_step_updates_history_and_no_privileged_inputs_escape(self):
        model = _Model()
        record = self.episode(model)
        self.assertEqual(record["status"], "completed", record)
        self.assertEqual(record["executed_control_steps"], 6)
        self.assertEqual(record["policy_queries"], 2)
        first, second = model.observations
        np.testing.assert_equal(first["ee_pos"][0, :, 0], [0., 0.])
        # Three actions were executed between queries, but history is adjacent
        # control observations (2,3), not adjacent policy queries (0,3).
        np.testing.assert_equal(second["ee_pos"][0, :, 0], [2., 3.])
        self.assertEqual(set(first), set(runner.CURRENT_KEYS))
        self.assertEqual(first["agentview_rgb"].shape, (1, 2, 3, 224, 224))
        self.assertEqual(first["gripper_states"][0, 0, 1], np.float32(-.03))
        self.assertFalse(any(np.any(value == 9001.) for value in first.values()))
        self.assertEqual(len(model.prompt_inputs), 1)
        self.assertEqual(model.prompt_inputs[0].dtype, np.uint8)
        np.testing.assert_array_equal(model.prompt_inputs[0], self.frames)
        self.assertTrue(record["success_once"])
        self.assertFalse(record["success_at_end"])
        self.assertEqual(record["phase_peaks"]["peak_r_reach"], 1.)

    def test_error_is_recorded_then_retry_and_completed_identity_are_enforced(self):
        model = _Model(failures=1)
        failed = self.episode(model)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertEqual(failed["executed_control_steps"], 0)
        completed = self.episode(model)
        self.assertEqual(completed["status"], "completed")
        forbidden_factory = lambda *args: self.fail("Completed case was executed twice")
        self.assertEqual(self.episode(model, env_factory=forbidden_factory), completed)
        with self.assertRaisesRegex(ValueError, "experiment identity"):
            self.episode(model, identity="other experiment")
        self.video.write_bytes(b"different RGB reference")
        with self.assertRaisesRegex(ValueError, "episode identity"):
            self.episode(model)


class ExperimentOrderTests(_Fixture):
    def orchestration_patches(self, execute):
        return (
            patch.dict(sys.modules, {
                "examples.baselines.bpp.bpp.model": types.SimpleNamespace(FrozenBPPModel=_Model),
                "examples.baselines.bpp.bpp.control": self.fake_control_module,
            }),
            patch.dict(os.environ, {"MS_ASSET_DIR": str(self.root / "assets")}),
            patch.object(runner, "select_reference", side_effect=lambda *a, **k: self.reference(a[3], a[4])),
            patch.object(runner, "read_reference_rgb", return_value=self.frames),
            patch.object(runner, "run_episode", side_effect=execute),
        )

    def test_human_cannot_start_without_completed_matching_robot_cases(self):
        cfg = replace(self.config, conditions=("human_video_only",))
        patches = self.orchestration_patches(lambda *a, **k: self.fail("Human started before robot"))
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaisesRegex(RuntimeError, "robot-video cases before human-video"):
                runner.run(cfg)

    def test_two_phases_are_ordered_and_changed_configuration_cannot_reuse_results(self):
        events = []

        def execute(config, model, reference, frames, condition, task, seed, identity):
            events.append((condition, identity))
            record = dict(
                status="completed", experiment_identity=identity, condition=condition,
                task_id=task, seed=seed, success_at_end=False, success_once=True,
            )
            runner._write_json(runner._case_path(config, condition, task, seed), record)
            return record

        patches = self.orchestration_patches(execute)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            summary = runner.run(replace(self.config, conditions=runner.CONDITIONS))
            self.assertEqual([event[0] for event in events], list(runner.CONDITIONS))
            self.assertEqual(events[0][1], events[1][1])
            self.assertEqual(summary["errors"], 0)
            with self.assertRaisesRegex(ValueError, "another experiment"):
                runner.run(replace(self.config, max_episode_steps=7))
            with self.assertRaisesRegex(ValueError, "another experiment"):
                runner.run(replace(self.config, reference_selection="fixed"))
            self.assertEqual(len(events), 2)
            source_hash = runner.file_sha256
            with patch.object(runner, "file_sha256", side_effect=lambda path: (
                "changed-control-source" if Path(path).name == "control.py" else source_hash(path)
            )):
                with self.assertRaisesRegex(ValueError, "another experiment"):
                    runner.run(self.config)
            self.assertEqual(len(events), 2)

    def test_zero_wam_reference_schedule_matches_all_ten_seeds_across_levels_and_conditions(self):
        events = []

        def execute(config, model, reference, frames, condition, task, seed, identity):
            events.append((condition, task, seed, reference.episode_index))
            record = dict(
                status="completed", experiment_identity=identity, condition=condition,
                task_id=task, seed=seed, success_at_end=False, success_once=False,
                reference=reference.manifest(),
            )
            runner._write_json(runner._case_path(config, condition, task, seed), record)
            return record

        config = replace(
            self.config, tasks=tuple(f"L{level}_{BASE}" for level in range(4)),
            seeds=tuple(range(100000, 100010)), conditions=runner.CONDITIONS,
        )
        patches = self.orchestration_patches(execute)
        with patches[0], patches[1], patches[2] as selector, patches[3], patches[4], patch("builtins.print"):
            result = runner.run(config)
        self.assertEqual(result["completed"], 80)
        self.assertEqual([event[0] for event in events[:40]], ["robot_video_only"] * 40)
        self.assertEqual([event[0] for event in events[40:]], ["human_video_only"] * 40)
        for condition, task, seed, episode in events:
            self.assertEqual(episode, seed - 100000, (condition, task, seed))
        # Each RGB episode is decoded once per condition, then shared across
        # all four task levels without treating the level as prompt input.
        self.assertEqual(selector.call_count, 20)
        manifest = json.loads((config.output_dir / "manifest.json").read_text())
        schedule = manifest["constants"]["reference_schedule"]
        self.assertEqual(schedule["selection"], "per_seed")
        self.assertEqual(schedule["seed_origin"], 100000)
        self.assertEqual(schedule["seed_to_reference_episode"], {
            str(seed): seed - 100000 for seed in config.seeds
        })
        self.assertTrue(schedule["shared_across_task_levels_and_conditions"])

    def test_sparse_seeds_are_stable_across_worker_subsets_and_fixed_is_explicit(self):
        def execute(config, model, reference, frames, condition, task, seed, identity):
            return dict(
                status="completed", experiment_identity=identity, condition=condition,
                task_id=task, seed=seed, success_at_end=False, success_once=False,
                reference=reference.manifest(),
            )

        for selection, offset, expected in (
            ("per_seed", 2, {100009: 11, 100003: 5, 100000: 2}),
            ("fixed", 4, {100009: 4, 100003: 4, 100000: 4}),
        ):
            with self.subTest(selection=selection):
                config = replace(
                    self.config, output_dir=self.root / selection,
                    seeds=(100009, 100003, 100000), reference_selection=selection,
                    reference_episode=offset,
                )
                patches = self.orchestration_patches(execute)
                with patches[0], patches[1], patches[2] as selector, patches[3], patches[4], patch("builtins.print"):
                    runner.run(config)
                    self.assertEqual(
                        [call.args[4] for call in selector.call_args_list],
                        list(dict.fromkeys(expected.values())),
                    )
                manifest = json.loads((config.output_dir / "manifest.json").read_text())
                schedule = manifest["constants"]["reference_schedule"]
                self.assertEqual(schedule["seed_to_reference_episode"], {
                    str(seed): episode for seed, episode in expected.items()
                })
                # A worker receiving only one sparse seed must choose the
                # same episode as the worker with the complete seed list.
                worker = replace(config, output_dir=self.root / f"{selection}-worker", seeds=(100003,))
                patches = self.orchestration_patches(execute)
                with patches[0], patches[1], patches[2] as selector, patches[3], patches[4], patch("builtins.print"):
                    runner.run(worker)
                self.assertEqual(selector.call_args.args[4], expected[100003])

    def test_missing_selected_reference_fails_only_its_case_without_substitution(self):
        executed = []

        def execute(config, model, reference, frames, condition, task, seed, identity):
            executed.append((seed, reference.episode_index))
            return dict(
                status="completed", experiment_identity=identity, condition=condition,
                task_id=task, seed=seed, success_at_end=False, success_once=False,
            )

        def select(*args, **kwargs):
            if args[4] == 1:
                raise IndexError("Episode 1 absent from fixture")
            return self.reference(args[3], args[4])

        patches = self.orchestration_patches(execute)
        with patches[0], patches[1], patches[2] as selector, patches[3], patches[4], patch("builtins.print"):
            selector.side_effect = select
            result = runner.run(replace(self.config, seeds=(100000, 100001, 100002)))
        self.assertEqual(executed, [(100000, 0), (100002, 2)])
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(result["conditions"]["robot_video_only"]["success_at_end"])
        failed = json.loads(runner._case_path(self.config, "robot_video_only", TASK, 100001).read_text())
        self.assertEqual(failed["reference_episode"], 1)
        self.assertEqual(failed["phase"], "reference_loading")

    def test_invalid_negative_schedule_is_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "negative reference episode"):
            replace(self.config, seeds=(99999,)).validate()
        with self.assertRaisesRegex(ValueError, "offset must be non-negative"):
            replace(self.config, reference_episode=-1).validate()
        # fixed reference is usable with other reset seed ranges.
        replace(self.config, seeds=(17,), reference_selection="fixed").validate()

    def test_errors_prevent_reporting_a_complete_experiment_success_rate(self):
        def execute(config, model, reference, frames, condition, task, seed, identity):
            record = dict(
                status="completed" if seed == 100000 else "error",
                experiment_identity=identity, condition=condition, task_id=task, seed=seed,
            )
            if record["status"] == "completed":
                record.update(success_at_end=True, success_once=True)
            else:
                record.update(error_type="RuntimeError", error="injected worker error")
            return record

        patches = self.orchestration_patches(execute)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            summary = runner.run(replace(self.config, seeds=(100000, 100001)))
        group = summary["conditions"]["robot_video_only"]
        self.assertEqual(group["planned"], 2)
        self.assertEqual(group["errors"], 1)
        self.assertFalse(group["complete"])
        self.assertIsNone(group["success_at_end"])
        self.assertEqual(group["success_at_end_completed_only"], 1.)


class ReferenceTests(_Fixture):
    def dataset(self, *, parquet=False):
        roots = []
        for group, name in (
            ("imitator_sim_v1_zed2i", f"L0_{BASE}"),
            ("imitator_human_v1", "human_H4"),
        ):
            root = self.root / "datasets" / group / name
            (root / "meta").mkdir(parents=True)
            (root / "meta/info.json").write_text(json.dumps({
                "fps": 10,
                "features": {RGB_KEY: {"dtype": "video", "shape": [3, 224, 224]}},
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }))
            row = {
                "episode_index": 1, "length": 4,
                VIDEO_PREFIX + "chunk_index": 0, VIDEO_PREFIX + "file_index": 0,
                VIDEO_PREFIX + "from_timestamp": .4, VIDEO_PREFIX + "to_timestamp": .8,
                "stats/action.qpos_gripper_actions": {"mean": [9001.] * 16},
            }
            (root / "meta/episodes.jsonl").write_text(json.dumps(row) + "\n")
            video = root / "videos" / RGB_KEY / "chunk-000/file-000.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"RGB-only placeholder")
            # Reading frame-level numerical demos would fail, so successful
            # selection demonstrates that they are unnecessary.
            numerical = root / "data/chunk-000/file-000.parquet"
            numerical.parent.mkdir(parents=True)
            numerical.write_bytes(b"NOT PARQUET: forbidden demo numeric fixture")
            if parquet:
                path = root / "meta/episodes/chunk-000/file-000.parquet"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"mocked metadata parquet")
            roots.append(root)
        mapping = self.root / "mapping.json"
        mapping.write_text(json.dumps({"task_mappings": [{
            "human_task_id": "human_H4",
            "sim_task_id": [f"L{level}_{BASE}" for level in range(4)],
        }]}))
        return roots, mapping, row

    def test_l0_robot_reference_is_shared_across_levels_and_human_uses_h4(self):
        _, mapping, _ = self.dataset()
        robot0 = select_reference(self.config.dataset_root, mapping, TASK, "robot_video_only", 1)
        robot2 = select_reference(self.config.dataset_root, mapping, f"L2_{BASE}", "robot_video_only", 1)
        human = select_reference(self.config.dataset_root, mapping, f"L3_{BASE}", "human_video_only", 1)
        self.assertEqual(robot0, robot2)
        self.assertEqual(human.dataset_id, "human_H4")
        self.assertEqual(robot0.manifest()["prompt_modalities"], ["RGB"])
        self.assertNotIn("action", robot0.__dict__)
        self.assertEqual(robot0.length, 4)

    def test_parquet_projection_excludes_demonstration_statistics(self):
        _, mapping, row = self.dataset(parquet=True)
        requested = []

        class Parquet:
            def __init__(self, path):
                if "/meta/episodes/" not in str(path):
                    raise AssertionError("Tried to read frame-level numerical data")
                self.schema_arrow = types.SimpleNamespace(names=list(row))

            def read(self, columns):
                requested.extend(columns)
                if any("qpos" in name or "action" in name for name in columns):
                    raise AssertionError("Requested demonstration numerical metadata")
                return types.SimpleNamespace(to_pylist=lambda: [{
                    key: row[key] for key in columns
                }])

        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = Parquet
        arrow = types.ModuleType("pyarrow")
        arrow.parquet = parquet
        with patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}):
            reference = select_reference(
                self.config.dataset_root, mapping, TASK, "robot_video_only", 1
            )
        self.assertEqual(reference.length, 4)
        self.assertTrue(requested)
        self.assertFalse(any("qpos" in name or "action" in name for name in requested))

    def test_episode_boundaries_keep_last_frame_and_never_take_the_next_episode(self):
        _, mapping, _ = self.dataset()
        reference = select_reference(
            self.config.dataset_root, mapping, TASK, "robot_video_only", 1
        )

        class Frame:
            def __init__(self, index):
                self.time, self.index = index / 10., index

            def reformat(self, **kwargs):
                return self

            def to_ndarray(self):
                return np.full((224, 224, 3), self.index, dtype=np.uint8)

        class Container:
            frame_count = 12

            def __init__(self):
                self.streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=1 / 90000)])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def seek(self, *args, **kwargs):
                pass

            def decode(self, stream):
                return (Frame(index) for index in range(self.frame_count))

        with patch.dict(sys.modules, {"av": types.SimpleNamespace(open=lambda *a: Container())}):
            for end in (.7, .8):  # last timestamp and exclusive-end metadata conventions
                frames = read_reference_rgb(replace(reference, to_timestamp=end))
                np.testing.assert_equal(frames[:, 0, 0, 0], [4, 5, 6, 7])
            Container.frame_count = 7
            with self.assertRaisesRegex(ValueError, "Short reference"):
                read_reference_rgb(reference)

    def test_seek_loss_is_recovered_without_changing_episode_boundaries(self):
        _, mapping, _ = self.dataset()
        reference = select_reference(self.config.dataset_root, mapping, TASK, "robot_video_only", 1)
        attempts = []

        class Frame:
            def __init__(self, index):
                self.time, self.index = index / 10., index

            def reformat(self, **kwargs):
                return self

            def to_ndarray(self):
                return np.full((224, 224, 3), self.index, dtype=np.uint8)

        class Container:
            def __init__(self):
                self.streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=1 / 90000)])
                self.sought = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def seek(self, timestamp, **kwargs):
                self.sought = True
                attempts.append(timestamp)

            def decode(self, stream):
                # Simulate a codec whose seeking still loses a leading frame
                # after pre-roll. The clean sequential retry has every frame.
                return (Frame(index) for index in range(12) if not (self.sought and index == 4))

        with patch.dict(sys.modules, {"av": types.SimpleNamespace(open=lambda *a: Container())}):
            frames = read_reference_rgb(reference)
        np.testing.assert_equal(frames[:, 0, 0, 0], [4, 5, 6, 7])
        self.assertEqual(len(attempts), 1)
        self.assertLess(attempts[0] / 90000., reference.from_timestamp)

    @unittest.skipUnless(
        importlib.util.find_spec("av") is not None and (
            PROJECT_TMP.parent / "data/IG-10K-Dataset/imitator_sim_v1_zed2i/L0_TwoRobotPlaceChipsRack-v1/meta/info.json"
        ).is_file(),
        "Real HEVC seek-loss regression fixture is unavailable",
    )
    def test_real_hevc_episode_matches_sequential_decode_with_no_missing_first_frame(self):
        import av
        task = "L0_TwoRobotPlaceChipsRack-v1"
        reference = select_reference(
            PROJECT_TMP.parent / "data/IG-10K-Dataset", self.root / "unused_mapping.json",
            task, "robot_video_only", 14,
        )
        actual = read_reference_rgb(reference, size=16)
        expected = []
        stop = reference.from_timestamp + reference.length / reference.fps
        tolerance = .25 / reference.fps
        with av.open(str(reference.video_path)) as container:
            for frame in container.decode(container.streams.video[0]):
                if frame.time < reference.from_timestamp - tolerance:
                    continue
                if frame.time >= stop - tolerance:
                    break
                expected.append(frame.reformat(width=16, height=16, format="rgb24").to_ndarray())
        self.assertEqual(len(expected), 256)
        np.testing.assert_array_equal(actual, np.stack(expected))

    @unittest.skipUnless(
        importlib.util.find_spec("av") is not None and importlib.util.find_spec("pyarrow") is not None,
        "Optional PyAV/pyarrow runtime dependencies are not installed yet",
    )
    def test_real_parquet_and_encoded_video_select_one_episode(self):
        import av
        import pyarrow as pa
        import pyarrow.parquet as pq

        roots, mapping, row = self.dataset(parquet=True)
        root = roots[0]
        parquet = root / "meta/episodes/chunk-000/file-000.parquet"
        pq.write_table(pa.Table.from_pylist([row]), parquet)
        video = root / "videos" / RGB_KEY / "chunk-000/file-000.mp4"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
            for index in range(12):
                frame = av.VideoFrame.from_ndarray(
                    np.full((16, 16, 3), index * 10, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        reference = select_reference(
            self.config.dataset_root, mapping, TASK, "robot_video_only", 1
        )
        frames = read_reference_rgb(reference)
        self.assertEqual(frames.shape, (4, 224, 224, 3))
        np.testing.assert_allclose(
            frames.mean(axis=(1, 2, 3)), [40., 50., 60., 70.], atol=4.,
        )


if __name__ == "__main__":
    unittest.main()

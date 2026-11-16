"""Local orchestration contracts; never launch GPUs, training or remote downloads."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.baselines.bpp.bpp import pipeline as p


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        (p.ROOT / "tmp").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="bpp-pipeline-tests-", dir=p.ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        base = self.root / "external.pt"
        base.write_bytes(b"common external initialization fixture")
        ready = {}
        for key, counts in (("c15", (3000, 750)), ("target", (200, 50))):
            data = self.root / f"{key}_data.json"
            feature = self.root / f"{key}_feature.json"
            p.atomic_json(data, dict(query_episode_count=counts[0], prompt_episode_count=counts[1], request=dict(split="scratch" if key == "target" else key, prompt_source="human")))
            p.atomic_json(feature, dict(complete=True, query_episode_count=counts[0], prompt_episode_count=counts[1], data_manifest_sha256=p.sha256(data), feature_fingerprint="common_frozen_vision"))
            ready[key] = dict(data_manifest=str(data), feature_manifest=str(feature))
        ready_file = self.root / "ready.json"
        p.atomic_json(ready_file, ready)
        self.config = p.PipelineConfig(run_root=self.root / "pipeline", c15_output=self.root / "pipeline/training/c15", reuse_prepared_inputs_from=None, c15_wrapper=self.root / "external_wrapper.py", ready_file=ready_file, external_checkpoint=base, poll_seconds=.001)
        self.artifacts = p.initial_artifacts(self.config)
        self.steps = p.build_steps(self.config)
        self.c15 = next(step for step in self.steps if step.name == "train.c15")

    @staticmethod
    def reader(path):
        return json.loads(Path(path).read_text())["metadata"]

    def make_training(self, step=None, *, interrupted=False, exit_code=0):
        step = step or self.c15
        key = "target" if step.regime in ("scratch", "finetune") else step.scale
        data, feature = Path(self.artifacts[key+".data"]), Path(self.artifacts[key+".features"])
        manifest = dict(action_normalization=p.ACTION_NORMALIZATION_V2, regime=step.regime, prompt_source="human", pilot=False, world_size=8, global_batch=64,
            max_updates=p.BUDGETS[step.regime], fixed_vision_preprocessing=True, init_seed=1,
            pretraining_scale=None if step.regime == "scratch" else step.scale,
            data_manifest=str(data), data_manifest_sha256=p.sha256(data), feature_manifest=str(feature), feature_manifest_sha256=p.sha256(feature),
            external_checkpoint=str(self.config.external_checkpoint), external_checkpoint_sha256=p.sha256(self.config.external_checkpoint),
            initialize_from=None, initialize_from_metadata=None, initialize_from_weights=None, **p.EXPECTED_TRAINING_RECIPE)
        for name, filename in (("trainer", "train.py"), ("native_model", "native_model.py"), ("training_data", "training_data.py"), ("feature_cache_source", "feature_cache.py")):
            manifest[name+"_sha256"] = p.sha256(Path(p.__file__).with_name(filename))
        if step.regime == "finetune":
            origin = Path(self.artifacts[step.scale+".checkpoint"])
            origin_meta = self.reader(origin)
            manifest.update(initialize_from=str(origin), initialize_from_sha256=p.sha256(origin), initialize_from_metadata=origin_meta, initialize_from_weights="ema")
        manifest["experiment_identity"] = p.identity(manifest, training=True)
        output = Path(step.output)
        p.atomic_json(output / "training_manifest.json", manifest)
        p.atomic_json(output / "process_status.json", dict(status="completed" if exit_code == 0 else "process_error", returncode=exit_code, command=["fixture", "--output-dir", str(output), "--regime", step.regime, "--action-normalization", p.ACTION_NORMALIZATION_V2]))
        count = 100 if interrupted else p.BUDGETS[step.regime]
        p.atomic_json(output / "completion.json", dict(status="interrupted" if interrupted else "completed", pilot=False, completed_updates=count, target_updates=p.BUDGETS[step.regime], experiment_identity=manifest["experiment_identity"]))
        metadata = dict(manifest, inference_sampling=dict(p.EXPECTED_SAMPLING), completed=not interrupted, train_step=count, init_head_sha256="same_new_heads", ema_state_complete=True, ema_optimization_step=count, normalizer_buffers_match_mode=True, model_input_perturbation=.1)
        p.atomic_json(output / ("interrupted.pt" if interrupted else "final.pt"), {"metadata": metadata})
        p.atomic_json(output / "model_initialization.json", dict(init_head_sha256="same_new_heads", action_normalization=p.ACTION_NORMALIZATION_V2))
        return manifest

    def make_evaluation(self, step=None):
        step = step or next(step for step in self.steps if step.name == "eval.c15.seen")
        tasks = p.official_tasks(self.config, step.regime)
        checkpoint = p.training_output(self.config, "scratch" if step.regime == "scratch" else step.scale+"_finetune" if step.regime == "pretrain_finetune" else step.scale) / "final.pt"
        expected_regime = step.scale if step.regime in ("seen", "zero_shot") else "finetune" if step.regime == "pretrain_finetune" else "scratch"
        training_metadata = (self.reader(checkpoint) if checkpoint.exists() else
            dict(p.EXPECTED_TRAINING_RECIPE, action_normalization=p.ACTION_NORMALIZATION_V2, train_step=p.BUDGETS[expected_regime]))
        manifest = dict(training_metadata=training_metadata, loaded_weight_selection="ema",
            loaded_ema_optimization_step=training_metadata["train_step"],
            action_normalization=p.ACTION_NORMALIZATION_V2, inference_sampling=dict(p.EXPECTED_SAMPLING), regime=step.regime, reference_source="human", training_prompt_source="human", is_pilot=False,
            tasks=tasks, seeds=list(range(100000, 100010)), max_episode_steps=500, execution_horizon=12,
            control_mode="pd_joint_pos", shader="rt-fast", pretraining_scale=None if step.regime == "scratch" else step.scale,
            checkpoint=str(checkpoint), checkpoint_sha256=p.sha256(checkpoint) if checkpoint.exists() else "checkpoint_hash")
        manifest["experiment_identity"] = p.identity(manifest)
        experiment = manifest["experiment_identity"]
        directory = Path(step.output)
        p.atomic_json(directory / "evaluation_manifest.json", manifest)
        cases = [(task, episode) for task in tasks for episode in range(10)]
        for index in range(16):
            name = f"shard_{index:03d}_of_016"
            p.atomic_json(directory / (name+".json"), dict(cases=cases[index::16], experiment_identity=experiment, attempt_id=str(index)))
            p.atomic_json(directory / ("process_status_"+name+".json"), dict(returncode=0, owns_evaluation_attempt=True, experiment_identity=experiment, attempt_id=str(index)))
        for task, episode in cases:
            case = dict(experiment_identity=experiment, task_id=task, seed=100000+episode, reference={"episode_index": episode}, reference_source="human")
            record = dict(case, case_id=p.identity(case), status="completed", process_exit_verified=True, process_returncode=0, is_pilot=False,
                          success_at_end=episode % 2 == 0, success_once=True, max_episode_steps=500, executed_control_steps=500, last_normalized_reward=.5)
            p.atomic_json(directory / "human" / task / f"seed_{100000+episode}" / "metrics.json", record)
        p.atomic_json(directory / "summary.json", dict(experiment_identity=experiment, planned_episode_count=200, incomplete_or_errors=False, process_exit_pending=False, success_at_end_rate=.5, success_once_rate=1.))
        return step, tasks

    def test_plan_has_strict_scale_gates_independent_initialization_and_shared_scratch(self):
        by_name = {step.name: step for step in self.steps}
        self.assertEqual(by_name["download.c30"].dependencies, ["complete.c15"])
        self.assertEqual(by_name["download.c45"].dependencies, ["complete.c30"])
        self.assertEqual(len([step for step in self.steps if step.name == "train.scratch"]), 1)
        for scale, budget in (("c15", 45000), ("c30", 90000), ("c45", 135000)):
            command = by_name[f"train.{scale}"].commands[0]
            self.assertNotIn("--initialize-from", command)
            self.assertIn("--nproc-per-node=8", command)
            self.assertEqual(command[command.index("--max-updates")+1], str(budget))
            ft = by_name[f"train.{scale}_finetune"].commands[0]
            self.assertEqual(ft[ft.index("--initialize-from")+1], "{"+scale+".checkpoint}")
            self.assertEqual(len(by_name[f"eval.{scale}.seen"].commands), 16)
            self.assertEqual(set(by_name[f"complete.{scale}"].dependencies), {f"eval.{scale}.seen", f"eval.{scale}.zero_shot", "eval.shared_scratch", f"eval.{scale}.pretrain_finetune"})

    def test_minmax_v2_plan_and_c15_directory_cannot_adopt_legacy(self):
        self.config.validate()
        for step in self.steps:
            if step.kind in ("train", "eval"):
                for command in step.commands:
                    self.assertEqual(command[command.index("--action-normalization")+1], p.ACTION_NORMALIZATION_V2)
        with self.assertRaisesRegex(p.ContractError, "C15 output"):
            replace(self.config, c15_output=self.root / "legacy_c15").validate()
        with self.assertRaisesRegex(p.ContractError, "normalization"):
            replace(self.config, action_normalization="legacy_quantile").validate()
        with self.assertRaisesRegex(p.ContractError, "EMA-v2"):
            replace(self.config, input_perturbation=0.0).validate()
        self.assertEqual(asdict(self.config)["ema_config"], p.EXPECTED_EMA_CONFIG)

    def reusable_source(self):
        source_root = self.root / "legacy_pipeline"
        source_artifacts = {}
        source_c15 = self.root / "legacy_c15"
        for key, counts in p.COUNTS.items():
            data = self.root / (key+"_data.json")
            feature = self.root / (key+"_feature.json")
            if not data.exists():
                p.atomic_json(data, dict(query_episode_count=counts[0], prompt_episode_count=counts[1],
                    request=dict(split=key, prompt_source="human")))
                p.atomic_json(feature, dict(complete=True, query_episode_count=counts[0], prompt_episode_count=counts[1],
                    data_manifest_sha256=p.sha256(data), feature_fingerprint="common_frozen_vision"))
            source_artifacts[key+".data"] = str(data)
            source_artifacts[key+".features"] = str(feature)
            source_artifacts[key+".checkpoint"] = str(source_root / "old_weights" / (key+".pt"))
            directory = source_c15 if key == "c15" else source_root / "training" / ("scratch" if key == "target" else key)
            manifest = dict(data_manifest=str(data), data_manifest_sha256=p.sha256(data),
                feature_manifest=str(feature), feature_manifest_sha256=p.sha256(feature))
            manifest["experiment_identity"] = p.identity(manifest, training=True)
            p.atomic_json(directory / "training_manifest.json", manifest)
        p.atomic_json(source_root / "pipeline_state.json", dict(status="measurements_complete_review_pending", active_step=None,
            config=dict(c15_output=str(source_c15)), artifacts=source_artifacts,
            steps={"complete."+scale:dict(status="complete") for scale in p.SCALES}))
        return source_root

    def test_reuse_keeps_scale_gates_skips_encoding_and_never_imports_old_weights(self):
        source = self.reusable_source()
        config = replace(self.config, reuse_prepared_inputs_from=source)
        config.validate()
        artifacts = p.initial_artifacts(config)
        steps = {step.name:step for step in p.build_steps(config)}
        self.assertEqual(len(steps), 29)
        for scale, previous in (("c30", "c15"), ("c45", "c30")):
            self.assertEqual(steps["download."+scale].dependencies, ["complete."+previous])
            for kind in ("download", "prepare", "features", "merge"):
                self.assertEqual(steps[kind+"."+scale].commands, [])
            self.assertEqual(artifacts[scale+".checkpoint"], str(config.run_root / "training" / scale / "final.pt"))
        self.assertEqual(artifacts["c15.checkpoint"], str(config.run_root / "training/c15/final.pt"))
        evidence = p.reused_input_contract(config, artifacts, "c45", "common_frozen_vision")
        self.assertTrue(evidence["reused_prepared_inputs"])
        self.assertFalse(evidence["launched_download_or_encoding"])
        feature = Path(artifacts["c45.features"])
        value = p.read_json(feature);value["complete"] = False;p.atomic_json(feature, value)
        with self.assertRaisesRegex(p.ContractError, "manifest changed"):
            p.reused_input_contract(config, artifacts, "c45", "common_frozen_vision")
        with self.assertRaisesRegex(p.ContractError, "source c45/features changed"):
            p.initial_artifacts(config)

    def test_normalization_and_saved_sampling_are_formal_training_gates(self):
        self.make_training()
        output = Path(self.c15.output)
        path = output / "training_manifest.json"
        manifest = p.read_json(path);manifest["action_normalization"] = "legacy_quantile"
        manifest["experiment_identity"] = p.identity({k:v for k,v in manifest.items() if k != "experiment_identity"}, training=True)
        p.atomic_json(path, manifest)
        with self.assertRaisesRegex(p.ContractError, "action_normalization"):
            p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)
        for field, invalid in (("clip_sample", False), ("num_inference_steps", 8), ("num_train_timesteps", 100)):
            with self.subTest(field=field):
                self.make_training()
                checkpoint = output / "final.pt";value = p.read_json(checkpoint)
                value["metadata"]["inference_sampling"][field] = invalid;p.atomic_json(checkpoint, value)
                with self.assertRaisesRegex(p.ContractError, "DDIM"):
                    p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)

    def test_completed_train_requires_checkpoint_identity_not_just_exit_zero(self):
        self.make_training()
        self.assertEqual(p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)["status"], "complete")
        checkpoint = Path(self.c15.output) / "final.pt"
        value = json.loads(checkpoint.read_text())
        value["metadata"]["completed"] = False
        p.atomic_json(checkpoint, value)
        with self.assertRaisesRegex(p.ContractError, "incomplete"):
            p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)
        self.make_training(interrupted=True)
        with self.assertRaisesRegex(p.ContractError, "Interrupted"):
            p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)

    def test_recipe_requires_original_perturbation_and_complete_ema_not_just_a_label(self):
        for field, value in (("input_perturbation", 0.0), ("use_ema", False)):
            with self.subTest(field=field):
                self.make_training()
                path = Path(self.c15.output) / "training_manifest.json"
                manifest = p.read_json(path);manifest[field] = value
                manifest["experiment_identity"] = p.identity({k:v for k,v in manifest.items() if k != "experiment_identity"}, training=True)
                p.atomic_json(path, manifest)
                with self.assertRaisesRegex(p.ContractError, "EMA-v2"):
                    p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)
        self.make_training()
        path = Path(self.c15.output) / "final.pt";value = p.read_json(path)
        value["metadata"]["ema_state_complete"] = False;p.atomic_json(path, value)
        with self.assertRaisesRegex(p.ContractError, "EMA state"):
            p.training_contract(self.config, self.c15, self.artifacts, checkpoint_reader=self.reader)

    def test_ema_summary_requires_every_shadow_shape_and_matching_update(self):
        class Shape:
            def __init__(self, value): self.shape = value
        saved = {k:v for k,v in p.EXPECTED_EMA_CONFIG.items() if k != "foreach"}
        saved.update(optimization_step=3, shadow_params=[Shape((2,))])
        payload = dict(state_dict_weight_type="raw", state_dict={"policy.weight":Shape((2,))},
            parameter_audit={"unique_parameters":2}, ema=dict(schema="bpp_tig_ema_v2",
                parameter_names=["policy.weight"], parameter_shapes=[[2]], state_dict=saved))
        self.assertTrue(p.checkpoint_ema_summary(payload, {"train_step":3})["ema_state_complete"])
        self.assertFalse(p.checkpoint_ema_summary(payload, {"train_step":4})["ema_state_complete"])
        saved["shadow_params"] = []
        self.assertFalse(p.checkpoint_ema_summary(payload, {"train_step":3})["ema_state_complete"])

    def test_checkpoint_inspection_checks_actual_saved_runtime_and_all_recipe_locations(self):
        class Tiny:
            def __init__(self, values): self.values = values; self.shape = (len(values),)
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.values
        recipe = dict(p.EXPECTED_TRAINING_RECIPE)
        metadata = dict(recipe, action_normalization=p.ACTION_NORMALIZATION_V2, train_step=3)
        raw = {"policy.weight":Tiny([0., 0.]), "state_scale":Tiny([1.]*18),
               "state_offset":Tiny([0.]*18), "action_scale":Tiny([1.]*16), "action_offset":Tiny([0.]*16)}
        saved = {k:v for k,v in p.EXPECTED_EMA_CONFIG.items() if k != "foreach"}
        saved.update(optimization_step=3, shadow_params=[Tiny([1., 1.])])
        bounds = lambda width: {"min":[-1.]*width, "max":[1.]*width}
        payload = dict(recipe, action_normalization=p.ACTION_NORMALIZATION_V2,
            metadata=metadata, config=dict(recipe, action_normalization=p.ACTION_NORMALIZATION_V2),
            state_dict_weight_type="raw", state_dict=raw, parameter_audit={"unique_parameters":2},
            normalizer_stats={"selected_bounds":{"state":bounds(18)}, "action":bounds(16)},
            model_config={"input_pertub":.1, "num_inference_steps":16,
                "noise_scheduler":{"_target_":"diffusers.DDIMScheduler", "clip_sample":True,
                                   "num_train_timesteps":50, "prediction_type":"epsilon"}},
            ema={"schema":"bpp_tig_ema_v2", "parameter_names":["policy.weight"],
                 "parameter_shapes":[[2]], "state_dict":saved})
        with patch.dict(sys.modules, {"torch":SimpleNamespace(load=lambda *args,**kwargs:payload)}):
            self.assertTrue(p.checkpoint_runtime_valid(p.checkpoint_summary(self.root / "not_loaded.pt")))
            payload["model_config"]["noise_scheduler"]["clip_sample"] = False
            self.assertFalse(p.checkpoint_runtime_valid(p.checkpoint_summary(self.root / "not_loaded.pt")))
            payload["config"]["use_ema"] = False
            with self.assertRaisesRegex(p.ContractError, "EMA-v2"):
                p.checkpoint_summary(self.root / "not_loaded.pt")

    def test_running_adoption_rejects_pid_reuse_and_dead_owner_needs_explicit_recovery(self):
        self.make_training(interrupted=True)
        path = Path(self.c15.output) / "process_status.json"
        status = p.read_json(path)
        status.update(status="running", pid=999999, started_utc=p.now())
        p.atomic_json(path, status)
        with patch.object(p, "process_observation", return_value={"alive": True, "matches": False}):
            with self.assertRaisesRegex(p.ContractError, "PID/command"):
                p.training_contract(self.config, self.c15, self.artifacts, allow_incomplete=True)
        with patch.object(p, "process_observation", return_value={"alive": False, "matches": False}), patch.object(p, "matching_training_producers", return_value=[]):
            self.assertEqual(p.training_contract(self.config, self.c15, self.artifacts, allow_incomplete=True)["status"], "pending")

    def test_finetune_requires_the_corresponding_complete_scale_lineage(self):
        self.make_training()
        ft = next(step for step in self.steps if step.name == "train.c15_finetune")
        self.make_training(ft)
        manifest_path = Path(ft.output) / "training_manifest.json"
        manifest = p.read_json(manifest_path)
        manifest["initialize_from_metadata"]["regime"] = "c30"
        manifest["experiment_identity"] = p.identity({key: value for key, value in manifest.items() if key != "experiment_identity"}, training=True)
        p.atomic_json(manifest_path, manifest)
        with self.assertRaisesRegex(p.ContractError, "lineage"):
            p.training_contract(self.config, ft, self.artifacts, checkpoint_reader=self.reader)

    def test_resume_never_relaunches_completed_training_and_failure_stops_later_scales(self):
        self.make_training()
        called = []
        def launch(config, step, commands, directory):
            called.append(step.name)
            return [{"returncode": 1, "log": str(directory / "failed.log")} for _ in commands]
        state = p.execute_pipeline(self.config, launcher=launch, checkpoint_reader=self.reader)
        self.assertEqual(called, ["eval.c15.seen"])
        self.assertEqual(state["steps"]["train.c15"]["status"], "complete")
        self.assertFalse(state["main_metrics_valid"])
        self.assertNotIn("download.c30", state["steps"])
        p.execute_pipeline(self.config, launcher=launch, checkpoint_reader=self.reader)
        self.assertEqual(called, ["eval.c15.seen"])

    def test_exit_zero_with_interrupted_completion_fails_before_any_evaluation(self):
        called = []
        def launch(config, step, commands, directory):
            called.append(step.name)
            self.make_training(step, interrupted=True)
            return [{"returncode": 0, "log": str(directory / "fixture.log")}]
        result = p.execute_pipeline(self.config, launcher=launch, checkpoint_reader=self.reader)
        self.assertEqual(called, ["train.c15"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["active_step"], "train.c15")
        self.assertFalse(result["main_metrics_valid"])

    def test_explicit_retry_resumes_only_a_matching_incomplete_checkpoint(self):
        self.make_training(interrupted=True, exit_code=-15)
        first = p.execute_pipeline(self.config, checkpoint_reader=self.reader, launcher=lambda *a: self.fail("Incomplete stage launched without retry"))
        self.assertEqual(first["status"], "failed")
        calls = []
        def launch(config, step, commands, directory):
            calls.append((step.name, commands))
            if step.kind == "train":
                self.assertIn("--resume-from", commands[0])
                self.make_training(step)
                return [{"returncode": 0, "log": str(directory / "resumed.log")}]
            return [{"returncode": 1, "log": str(directory / "stopped.log")} for _ in commands]
        result = p.execute_pipeline(self.config, retry_failed=True, checkpoint_reader=self.reader, launcher=launch)
        self.assertEqual([name for name, _ in calls], ["train.c15", "eval.c15.seen"])
        self.assertEqual(result["steps"]["train.c15"]["status"], "complete")

    def test_all_cases_and_owned_clean_shards_are_required_even_when_summary_claims_success(self):
        step, tasks = self.make_evaluation()
        self.assertEqual(p.evaluation_contract(self.config, step, checkpoint_hash="checkpoint_hash")["success_at_end_rate"], .5)
        status = Path(step.output) / "process_status_shard_015_of_016.json"
        value = json.loads(status.read_text())
        value["returncode"] = -11
        p.atomic_json(status, value)
        with self.assertRaisesRegex(p.ContractError, "exit cleanly"):
            p.evaluation_contract(self.config, step)
        value["returncode"] = 0
        p.atomic_json(status, value)
        case = Path(step.output) / "human" / tasks[0] / "seed_100000/metrics.json"
        record = json.loads(case.read_text())
        record["process_exit_verified"] = False
        p.atomic_json(case, record)
        with self.assertRaisesRegex(p.ContractError, "Invalid/missing/unclean"):
            p.evaluation_contract(self.config, step)

    def test_failed_evaluation_can_be_explicitly_retried_without_retraining_checkpoint(self):
        self.make_training()
        step, _ = self.make_evaluation()
        path = Path(step.output) / "process_status_shard_015_of_016.json"
        process = p.read_json(path)
        process["returncode"] = -11
        p.atomic_json(path, process)
        first = p.execute_pipeline(self.config, checkpoint_reader=self.reader, launcher=lambda *args: self.fail("Failed evaluation launched without retry"))
        self.assertEqual(first["active_step"], "eval.c15.seen")
        called = []
        def launch(config, current, commands, directory):
            called.append(current.name)
            if current.name == "eval.c15.seen":
                self.make_evaluation()
                return [{"returncode": 0, "log": str(directory / f"worker_{index}.log")} for index in range(len(commands))]
            return [{"returncode": 1, "log": str(directory / "fixture.log")} for _ in commands]
        result = p.execute_pipeline(self.config, retry_failed=True, checkpoint_reader=self.reader, launcher=launch)
        self.assertEqual(called, ["eval.c15.seen", "eval.c15.zero_shot"])
        self.assertEqual(result["steps"]["train.c15"]["status"], "complete")
        self.assertEqual(result["steps"]["eval.c15.seen"]["status"], "complete")
        self.assertTrue((Path(step.output) / "failed_attempts").is_dir())

    def test_complete_c15_fixture_reaches_review_pending_and_second_run_launches_nothing(self):
        config = replace(self.config, through="c15")
        calls = []
        def launch(config, step, commands, directory):
            calls.append(step.name)
            if step.kind == "train":
                self.make_training(step)
            elif step.kind == "eval":
                self.make_evaluation(step)
            else:
                self.fail(f"Unexpected fixture workload: {step.name}")
            return [{"returncode": 0, "log": str(directory / f"worker_{index}.log")} for index in range(len(commands))]
        state = p.execute_pipeline(config, checkpoint_reader=self.reader, launcher=launch)
        self.assertEqual(state["status"], "measurements_complete_review_pending", state)
        self.assertTrue(state["overall_review_pending"])
        self.assertEqual(len(state["steps"]["complete.c15"]["evidence"]["regimes"]), 4)
        self.assertTrue((config.run_root / "review_inputs.json").exists())
        before = list(calls)
        p.execute_pipeline(config, checkpoint_reader=self.reader, launcher=lambda *args: self.fail("Completed workload launched twice"))
        self.assertEqual(calls, before)

    def test_monitor_is_read_only_and_torch_free(self):
        config_path = self.root / "config.json"
        p.atomic_json(config_path, asdict(self.config))
        before = {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in self.root.rglob("*") if path.is_file()}
        self.assertTrue(p.monitor(self.config)["read_only"])
        entry = Path(p.__file__).parents[1] / "monitor_bpp_imitator.py"
        code = '''import importlib.abc,runpy,sys
class BlockTorch(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='torch':raise AssertionError('Monitor imported Torch')
sys.meta_path.insert(0,BlockTorch())
sys.argv=[sys.argv[1],'--config',sys.argv[2]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-c", code, str(entry), str(config_path)], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(self.config.tig_repo), PYTHONDONTWRITEBYTECODE="1"), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        after = {str(path): (path.stat().st_mtime_ns, path.stat().st_size) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_live_monitor_counts_clean_cases_without_publishing_formal_validity(self):
        step, tasks = self.make_evaluation()
        result = p.monitor(self.config)["evaluations"][step.name]
        self.assertEqual(result["completed_clean_cases"], 200)
        self.assertEqual(result["clean_worker_exits"], 16)
        self.assertFalse(result["formal_validity"])
        path = Path(step.output) / "human" / tasks[0] / "seed_100000/metrics.json"
        value = p.read_json(path)
        value["process_exit_verified"] = False
        p.atomic_json(path, value)
        result = p.monitor(self.config)["evaluations"][step.name]
        self.assertEqual(result["completed_clean_cases"], 199)
        self.assertEqual(result["pending_process_exit_cases"], 1)


if __name__ == "__main__":
    unittest.main()

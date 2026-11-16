import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

report = importlib.import_module("examples.baselines.bpp.bpp.report")


class ReportTests(unittest.TestCase):
    def fixture(self, root):
        tig, output, source = root / "tig", root / "evaluation", root / "source"
        config = tig / "examples/baselines/lerobot_dataset"
        (config / "eval/exp_list").mkdir(parents=True)
        source.mkdir()
        tasks = [f"L{level}_Task{base}-v1" for base in range(5) for level in range(4)]
        (config / "eval/exp_list/seen_5tasks_env_list.txt").write_text("\n".join(tasks))
        (config / "task_mapping.json").write_text(json.dumps({"task_mappings": [
            {"human_task_id": f"human_H{base}", "sim_task_id": tasks[base*4:base*4+4]} for base in range(5)]}))
        sources = {}
        for name in ("native_eval.py", "native_model.py", "data.py", "training_data.py"):
            (source / name).write_text(name)
            sources[name] = report.digest(source / name)
        manifest = {"regime": "seen", "tasks": tasks, "seeds": list(range(100000,100010)),
                    "reference_source": "human", "training_prompt_source": "human", "is_pilot": False,
                    "control_mode": "pd_joint_pos", "action_dimension": 16, "max_episode_steps": 500,
                    "execution_horizon": 12, "shader": "rt-fast", "dataset_revision": report.REVISION,
                    "prompt_modalities": ["RGB"], "prompt_frames": 16, "pretraining_scale": "c15",
                    "checkpoint": "/fixture/final.pt", "checkpoint_sha256": "1"*64, "source_sha256": sources}
        experiment = report.identity(manifest)
        manifest["experiment_identity"] = experiment
        report.atomic_json(output / "evaluation_manifest.json", manifest)
        cases = [(task, episode) for task in tasks for episode in range(10)]
        for index in range(16):
            name = f"shard_{index:03d}_of_016.json"
            report.atomic_json(output / name, {"cases": cases[index::16], "attempt_id": str(index), "experiment_identity": experiment})
            report.atomic_json(output / ("process_status_"+name), {"status": "completed", "returncode": 0,
                "owns_evaluation_attempt": True, "attempt_id": str(index), "experiment_identity": experiment})
        for task, episode in cases:
            base, level = int(task.split("Task")[1].split("-")[0]), int(task[1])
            row = {"experiment_identity": experiment, "task_id": task, "seed": 100000+episode,
                "reference": {"dataset_id": f"human_H{base}", "episode_index": episode,
                    "condition": "human_video_only", "prompt_modalities": ["RGB"], "prompt_frames": 16, "dataset_revision": report.REVISION},
                "actual_environment_flags": {"L1": level==1,"L2":level==2,"L3":level==3,"mirror_robot_pose":False},
                "executed_control_steps":500,"status":"completed","process_exit_verified":True,"process_returncode":0,
                "is_pilot":False,"reference_source":"human","max_episode_steps":500,"success_at_end":episode<3,"success_once":episode<7,"last_normalized_reward":0.25,"wall_seconds":30.0}
            row["case_id"]=report.identity({key:row[key] for key in ("experiment_identity","task_id","seed","reference","reference_source")})
            report.atomic_json(output / "human" / task / f"seed_{100000+episode}" / "metrics.json", row)
        report.atomic_json(output / "summary.json", {"experiment_identity":experiment,"planned_episode_count":200,"completed_episode_count":200,"error_episode_count":0,"is_pilot":False,"incomplete_or_errors":False,"process_exit_pending":False,"success_at_end_rate":.3,"success_once_rate":.7})
        return output, tig, source, tasks

    def test_end_once_and_continuous_proxy_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            output,tig,source,_ = self.fixture(Path(temporary))
            result,_ = report.inspect_evaluation(output,"c15","seen",tig,source_root=source)
            self.assertEqual(result["status"],"complete")
            self.assertEqual(result["success_at_end_rate"],.3)
            self.assertEqual(result["success_once_rate"],.7)
            self.assertEqual(result["SubSR_released_code_proxy"],.25)

    def test_native_crash_invalidates_formal_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            output,tig,source,_ = self.fixture(Path(temporary))
            path=output/"process_status_shard_000_of_016.json"
            value=report.read(path);value["returncode"]=-11;report.atomic_json(path,value)
            result,_=report.inspect_evaluation(output,"c15","seen",tig,source_root=source)
            self.assertEqual(result["status"],"incomplete_or_invalid")
            self.assertIsNone(result["success_at_end_rate"])
            self.assertIsNone(result["SubSR_released_code_proxy"])

    def test_wrong_human_reference_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            output,tig,source,tasks=self.fixture(Path(temporary))
            path=output/"human"/tasks[0]/"seed_100000/metrics.json"
            value=report.read(path);value["reference"]["dataset_id"]="human_H999";report.atomic_json(path,value)
            result,_=report.inspect_evaluation(output,"c15","seen",tig,source_root=source)
            self.assertIsNone(result["success_at_end_rate"])
            self.assertTrue(any("human reference differs" in item for item in result["issues"]))

    def test_missing_case_and_nan_cannot_be_averaged_away(self):
        clean={"status":"completed","process_exit_verified":True,"process_returncode":0,"is_pilot":False,
               "success_at_end":True,"success_once":True,"last_normalized_reward":.5}
        self.assertIsNone(report.metric_rows([clean],2)["success_at_end_rate"])
        invalid=dict(clean,last_normalized_reward=float("nan"))
        self.assertIsNone(report.metric_rows([invalid],1)["SubSR_released_code_proxy"])


    def test_case_hash_and_summary_identity_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            output,tig,source,tasks=self.fixture(Path(temporary))
            path=output/"human"/tasks[0]/"seed_100000/metrics.json"
            row=report.read(path);row["case_id"]="wrong";report.atomic_json(path,row)
            result,_=report.inspect_evaluation(output,"c15","seen",tig,source_root=source)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["success_at_end_rate"])
            row["case_id"]=report.identity({key:row[key] for key in ("experiment_identity","task_id","seed","reference","reference_source")})
            report.atomic_json(path,row)
            path=output/"summary.json";value=report.read(path);value["experiment_identity"]="wrong";report.atomic_json(path,value)
            result,_=report.inspect_evaluation(output,"c15","seen",tig,source_root=source)
            self.assertFalse(result["valid"])
            self.assertTrue(any("summary provenance" in item for item in result["issues"]))

    def test_pending_schema_and_empty_csv_do_not_keep_old_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            result,_=report.inspect_evaluation(root/"missing","c15","seen",root)
            self.assertEqual(result["planned_episodes"],200)
            self.assertIsNone(result["success_at_end_rate"])
            path=root/"scores.csv";path.write_text("old complete scores")
            report.write_csv(path,[])
            self.assertNotIn("old complete scores",path.read_text())
            self.assertIn("success_at_end_rate",path.read_text())


    def test_invalid_finetune_lineage_disables_derived_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/"run";c15=root/"c15"
            def training(output, regime, scale, **kwargs):
                output=Path(output)
                key="scratch" if regime=="scratch" else scale+"_finetune" if regime=="finetune" else scale
                origin=c15 if scale=="c15" else run/"training"/str(scale)
                return {"directory":str(output),"regime":regime,"scale":scale,"status":"complete","issues":[],
                        "checkpoint":str(output/"final.pt"),"checkpoint_sha256":key,"gpu_hours":1,
                        "initialize_from":str(origin/"final.pt") if regime=="finetune" else None,
                        "initialize_from_sha256":("wrong" if scale=="c15" else scale) if regime=="finetune" else None}
            def evaluation(directory, scale, regime, tig_repo):
                key="scratch" if regime=="scratch" else scale+"_finetune" if regime=="pretrain_finetune" else scale
                rows=[{"task_id":f"L{level}_Task{task}-v1","status":"completed","process_exit_verified":True,
                       "process_returncode":0,"is_pilot":False,"success_at_end":True,"success_once":True,
                       "last_normalized_reward":1.} for task in range(5) for level in range(4) for episode in range(10)]
                return {"scale":scale,"regime":regime,"directory":str(directory),"status":"complete","issues":[],
                        "checkpoint_sha256":key,**report.metric_rows(rows,200)},rows
            with patch.object(report,"inspect_training",side_effect=training),patch.object(report,"inspect_evaluation",side_effect=evaluation):
                result=report.build_report(run,root/"tig",c15,verify_checkpoints=True)
            ft=next(row for row in result["groups"] if row["scale"]=="c15" and row["regime"]=="pretrain_finetune")
            self.assertFalse(ft["valid"])
            self.assertIsNone(ft["success_at_end_rate"])
            self.assertEqual(result["verified_unique_rollouts"],1800)
            self.assertTrue(all(row["success_at_end_rate"] is None for row in result["per_task_level"] if row["scale"]=="c15" and row["regime"]=="pretrain_finetune"))


if __name__=="__main__":
    unittest.main()

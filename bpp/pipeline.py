"""Resumable C15→C30→C45 orchestration; planning/monitoring import no Torch."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable

ROOT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
TIG = Path(__file__).resolve().parents[4]
NAMESPACE = "examples.baselines.bpp.bpp"
BUDGETS = {"c15": 45000, "c30": 90000, "c45": 135000, "scratch": 3000, "finetune": 3000}
COUNTS = {"c15": (3000, 750), "c30": (6000, 1500), "c45": (9000, 2250), "target": (200, 50)}
SCALES = ("c15", "c30", "c45")
ACTION_NORMALIZATION_V2 = "train_minmax_joint14_gripper_identity_v2"
V1_RUN_ROOT = ROOT / "results/pipeline/bpp_human"
V2_RUN_ROOT = ROOT / "results/pipeline/bpp_human_minmax_v2"
EXPECTED_EMA_CONFIG = {"decay": 0.9999, "min_decay": 0.0, "update_after_step": 0,
                       "use_ema_warmup": False, "inv_gamma": 1.0, "power": 0.75, "foreach": False}
EXPECTED_TRAINING_RECIPE = {"training_recipe": "bpp_source_epsilon_perturbation_ema_v2",
    "input_perturbation": 0.1, "use_ema": True, "ema_config": EXPECTED_EMA_CONFIG,
    "evaluation_weights": "ema", "resume_weights": "raw"}
EXPECTED_SAMPLING = {"scheduler_class": "DDIMScheduler", "clip_sample": True,
                     "clip_sample_range": 1.0, "thresholding": False,
                     "num_inference_steps": 16, "num_train_timesteps": 50,
                     "prediction_type": "epsilon"}


class ContractError(RuntimeError):
    pass


class IncompleteEvaluation(ContractError):
    """Execution evidence may be retried without changing the formal identity."""


def read_json(path: Path, default=None):
    if not Path(path).is_file():
        return default
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.02)  # The external C15 wrapper uses a non-atomic status write.


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value, *, training=False):
    kwargs = {} if training else {"separators": (",", ":")}
    return hashlib.sha256(json.dumps(value, sort_keys=True, **kwargs).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def process_observation(status, *, wrapper: Path | None = None):
    """Check namespace PID AND command/start time, avoiding PID-reuse adoption."""
    pid = status.get("pid")
    result = {"pid": pid, "alive": False, "matches": False}
    if not isinstance(pid, int):
        return result
    try:
        os.kill(pid, 0)
        command = [item.decode(errors="replace") for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item]
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(tail[19])
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        started = boot + ticks / os.sysconf("SC_CLK_TCK")
        recorded = datetime.fromisoformat(status["started_utc"]).timestamp()
        same_command = command == status.get("command") or (wrapper is not None and str(wrapper) in command)
        result.update(alive=True, matches=same_command and abs(started-recorded) < 30,
                      cmdline=command, start_ticks=ticks)
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    return result


def matching_training_producers(output):
    """Inspect only command matches for this exact output; never signal them."""
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = [value.decode(errors="replace") for value in (path / "cmdline").read_bytes().split(b"\0") if value]
            index = command.index("--output-dir")
            if command[index+1] == str(output) and any("train_bpp_imitator" in value or "bpp.bpp.train" in value for value in command):
                found.append(int(path.name))
        except (OSError, ValueError, IndexError):
            pass
    return found


@dataclass(frozen=True)
class PipelineConfig:
    run_root: Path = V2_RUN_ROOT
    c15_output: Path = V2_RUN_ROOT / "training/c15"
    c15_wrapper: Path = ROOT / "tmp/run_formal_c15.py"
    ready_file: Path = ROOT / "results/protocols/c15_ready_manifests.json"
    python: Path = Path(os.environ.get("BPP_RUNTIME_PYTHON", str(ROOT / "envs/bpp_tig_torch271/bin/python")))
    downloader_python: Path = ROOT / "envs/downloader/bin/python"
    external_checkpoint: Path = Path(os.environ.get("BPP_CHECKPOINT", str(ROOT / "weights/bp_libero/libero_behavior_prompting.ckpt")))
    tig_repo: Path = TIG
    dataset_root: Path = ROOT / "data/IG-10K-Dataset"
    data_cache: Path = ROOT / "cache/bpp_tig_training"
    feature_cache: Path = ROOT / "cache/bpp_frozen_cls"
    downloader_manifests: Path = ROOT / "logs/bpp_downloads"
    through: str = "c45"
    eval_workers_per_gpu: int = 2
    poll_seconds: float = 10.
    seed: int = 1
    action_normalization: str = ACTION_NORMALIZATION_V2
    training_recipe: str = EXPECTED_TRAINING_RECIPE["training_recipe"]
    input_perturbation: float = 0.1
    use_ema: bool = True
    ema_config: dict[str, Any] = field(default_factory=lambda: dict(EXPECTED_EMA_CONFIG))
    evaluation_weights: str = "ema"
    resume_weights: str = "raw"
    reuse_prepared_inputs_from: Path | None = V1_RUN_ROOT

    @classmethod
    def load(cls, path=None, **overrides):
        values = read_json(Path(path), {}) if path else {}
        values.update({key: value for key, value in overrides.items() if value is not None})
        for key, default in asdict(cls()).items():
            if isinstance(default, Path) and key in values and values[key] is not None:
                values[key] = Path(values[key])
        return cls(**values)

    def validate(self):
        require_training_recipe(asdict(self))
        if self.seed != 1:
            raise ContractError("The formal minmax-v2 initialization/training seed is fixed at1")
        if self.action_normalization != ACTION_NORMALIZATION_V2:
            raise ContractError("This retraining pipeline requires the explicit minmax-v2 action normalization")
        if self.c15_output.resolve() != (self.run_root / "training/c15").resolve():
            raise ContractError("Minmax-v2 C15 output must be its own run-root/training/c15; do not adopt the legacy C15 model")
        if self.run_root.resolve() == V1_RUN_ROOT.resolve():
            raise ContractError("The archived v1 results root cannot hold minmax-v2 jobs")
        if self.reuse_prepared_inputs_from is not None:
            source = self.reuse_prepared_inputs_from.resolve()
            if not source.is_relative_to(ROOT.resolve()) or self.run_root.resolve().is_relative_to(source):
                raise ContractError("Prepared inputs must come from a separate project run root")
        if self.through not in SCALES or self.eval_workers_per_gpu not in (1, 2):
            raise ContractError("Expected C15/C30/C45 and one or two eval workers per GPU")
        if self.poll_seconds <= 0:
            raise ContractError("poll_seconds must be positive")
        for path in (self.run_root, self.c15_output, self.data_cache, self.feature_cache):
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise ContractError(f"Project output must remain below {ROOT}: {path}")


@dataclass
class Step:
    name: str
    kind: str
    scale: str
    dependencies: list[str]
    output: str
    commands: list[list[str]] = field(default_factory=list)
    regime: str | None = None


def training_output(config, key):
    return config.c15_output if key == "c15" else config.run_root / "training" / key


def evaluation_output(config, scale, regime):
    return config.run_root / "evaluation" / ("shared_scratch" if regime == "scratch" else f"{scale}_{regime}")


def initial_artifacts(config):
    ready = read_json(config.ready_file, {})
    values = {}
    for key in ("c15", "target"):
        if key in ready:
            values[f"{key}.data"] = ready[key]["data_manifest"]
            values[f"{key}.features"] = ready[key]["feature_manifest"]
    if config.reuse_prepared_inputs_from is not None:
        values.update(reusable_input_bindings(config))
    for scale in SCALES:
        values[f"{scale}.checkpoint"] = str(training_output(config, scale) / "final.pt")
        values[f"{scale}.finetune_checkpoint"] = str(training_output(config, scale+"_finetune") / "final.pt")
        if scale != "c15":
            values.setdefault(f"{scale}.features", str(config.run_root / "ready" / f"{scale}_features.complete.json"))
    values["scratch.checkpoint"] = str(training_output(config, "scratch") / "final.pt")
    return values


def reusable_input_bindings(config):
    """Reuse only verified raw inputs/CLS, never another run's learned weights."""
    source_root = Path(config.reuse_prepared_inputs_from)
    source_path = source_root / "pipeline_state.json"
    source = read_json(source_path, {})
    if (source.get("status") != "measurements_complete_review_pending" or source.get("active_step") is not None
            or any(source.get("steps", {}).get("complete."+scale, {}).get("status") != "complete" for scale in SCALES)):
        raise ContractError("Prepared-input source must have all three completed scale gates and no active stage")
    source_artifacts, values = source.get("artifacts", {}), {}
    for key in (*SCALES, "target"):
        training = (Path(source.get("config", {}).get("c15_output", source_root / "training/c15"))
                    if key == "c15" else source_root / "training" / ("scratch" if key == "target" else key))
        manifest_path = training / "training_manifest.json"
        manifest = read_json(manifest_path, {})
        unsigned = {name:value for name,value in manifest.items() if name != "experiment_identity"}
        if not manifest or identity(unsigned, training=True) != manifest.get("experiment_identity"):
            raise ContractError(f"Prepared-input source training identity differs: {key}")
        for role, field in (("data", "data_manifest"), ("features", "feature_manifest")):
            path = Path(source_artifacts.get(key+"."+role, "/nonexistent"))
            expected = manifest.get(field+"_sha256")
            if str(path.resolve()) != manifest.get(field) or not path.is_file() or sha256(path) != expected:
                raise ContractError(f"Prepared-input source {key}/{role} changed or differs from its training provenance")
            values[key+"."+role] = str(path.resolve())
            values[key+"."+role+"_sha256"] = expected
    values["reuse.source_run_root"] = str(source_root.resolve())
    values["reuse.source_pipeline_sha256"] = sha256(source_path)
    return values


def reused_input_contract(config, artifacts, key, fingerprint=None):
    for role in ("data", "features"):
        path = Path(artifacts[key+"."+role])
        if sha256(path) != artifacts.get(key+"."+role+"_sha256"):
            raise ContractError(f"Reused {key}/{role} manifest changed")
    actual_fingerprint = validate_ready(artifacts[key+".data"], artifacts[key+".features"], key, fingerprint)
    return {"status":"complete", "reused_prepared_inputs":True,
            "source_run_root":artifacts["reuse.source_run_root"],
            "source_pipeline_sha256":artifacts["reuse.source_pipeline_sha256"],
            "data_manifest":artifacts[key+".data"], "data_manifest_sha256":artifacts[key+".data_sha256"],
            "feature_manifest":artifacts[key+".features"], "feature_manifest_sha256":artifacts[key+".features_sha256"],
            "feature_fingerprint":actual_fingerprint, "launched_download_or_encoding":False}


def require_sampling(value):
    if not isinstance(value, dict) or any(value.get(key) != expected for key,expected in EXPECTED_SAMPLING.items()):
        raise ContractError("Expected the original DDIM clip1 / 16 inference / 50 training timestep configuration")
    if value.get("clip_sample") is not True or value.get("thresholding") is not False:
        raise ContractError("DDIM clipping/thresholding flags must be explicit booleans")


def require_training_recipe(value):
    if not isinstance(value, dict) or any(value.get(key) != expected for key,expected in EXPECTED_TRAINING_RECIPE.items()):
        raise ContractError("Expected the source epsilon input-perturbation0.1 and complete EMA-v2 training recipe")
    if value.get("use_ema") is not True:
        raise ContractError("EMA training must be explicitly enabled")


def checkpoint_runtime_valid(metadata):
    try:
        require_training_recipe(metadata)
        require_sampling(metadata.get("inference_sampling"))
    except ContractError:
        return False
    return (metadata.get("ema_state_complete") is True and metadata.get("normalizer_buffers_match_mode") is True
            and metadata.get("model_input_perturbation") == 0.1
            and metadata.get("ema_optimization_step") == metadata.get("train_step"))


def checkpoint_ema_summary(payload, metadata):
    ema = payload.get("ema", {})
    saved = ema.get("state_dict", {})
    names, shapes = ema.get("parameter_names", []), ema.get("parameter_shapes", [])
    shadows = saved.get("shadow_params", [])
    step = saved.get("optimization_step")
    complete = (ema.get("schema") == "bpp_tig_ema_v2" and payload.get("state_dict_weight_type") == "raw"
        and isinstance(names, list) and bool(names) and len(names) == len(set(names))
        and len(names) == len(shapes) == len(shadows) and type(step) is int and step == metadata.get("train_step")
        and all(saved.get(key) == value for key,value in EXPECTED_EMA_CONFIG.items() if key != "foreach"))
    numel = 0
    if complete:
        raw = payload.get("state_dict", {})
        try:
            for name, shape, shadow in zip(names, shapes, shadows):
                if tuple(raw[name].shape) != tuple(shape) or tuple(shadow.shape) != tuple(shape):
                    complete = False
                    break
                numel += math.prod(shape)
            complete = complete and numel == payload.get("parameter_audit", {}).get("unique_parameters")
        except (KeyError, TypeError, AttributeError):
            complete = False
    return {"ema_state_complete":bool(complete), "ema_optimization_step":step,
            "ema_parameter_count":len(names), "ema_parameter_numel":numel}


def checkpoint_normalizer_matches(payload):
    """Read only four tiny buffers; no large parameter tensor is materialized."""
    try:
        stats, raw = payload["normalizer_stats"], payload["state_dict"]
        bounds = stats.get("selected_bounds", {}).get("state")
        if bounds is None:
            bounds = {"min":stats["state"]["q01" if stats.get("mode")=="quantile" else "min"],
                      "max":stats["state"]["q99" if stats.get("mode")=="quantile" else "max"]}
        expected = {}
        for field, low, high in (("state", bounds["min"], bounds["max"]),
                                  ("action", stats["action"]["min"], stats["action"]["max"])):
            width = 18 if field == "state" else 16
            if len(low) != width or len(high) != width:
                return False
            scale, offset = [], []
            for index,(minimum,maximum) in enumerate(zip(low,high)):
                if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum:
                    return False
                if field == "action" and index in (7,15):
                    factor, shift = 1., 0.
                elif maximum-minimum < 1e-6:
                    factor, shift = 1., -minimum
                else:
                    factor = 2/(maximum-minimum)
                    shift = -1-minimum*factor
                scale.append(factor);offset.append(shift)
            expected[field+"_scale"], expected[field+"_offset"] = scale, offset
        for name,values in expected.items():
            actual = raw[name].detach().cpu().tolist()
            if len(actual) != len(values) or not all(math.isfinite(a) and math.isclose(a,b,rel_tol=2e-5,abs_tol=2e-5) for a,b in zip(actual,values)):
                return False
        return True
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def resolve(command, artifacts, *, strict=True):
    def substitute(match):
        key = match[1]
        if key not in artifacts and strict:
            raise ContractError(f"Artifact not ready: {key}")
        return str(artifacts.get(key, match[0]))
    return [re.sub(r"\{([^{}]+)\}", substitute, str(value)) for value in command]


def build_steps(config):
    py = str(config.python)
    steps = [Step("ready", "ready", "c15", [], str(config.ready_file))]

    def train(scale, regime, previous):
        key = "scratch" if regime == "scratch" else scale+"_finetune" if regime == "finetune" else scale
        data = "target" if regime in ("scratch", "finetune") else scale
        command = [py, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8", "-m", "examples.baselines.bpp.train_bpp_imitator",
                   "--regime", regime, "--prompt-source", "human", "--data-manifest", "{"+data+".data}", "--feature-manifest", "{"+data+".features}",
                   "--fixed-vision-preprocessing", "--action-normalization", config.action_normalization, "--checkpoint", str(config.external_checkpoint), "--output-dir", str(training_output(config, key)),
                   "--batch-size", "8", "--num-workers", "4", "--max-updates", str(BUDGETS[regime]), "--log-every", "200", "--checkpoint-every", "5000", "--seed", str(config.seed), "--init-seed", str(config.seed)]
        if regime == "finetune":
            command += ["--pretraining-scale", scale, "--initialize-from", "{"+scale+".checkpoint}"]
        name = f"train.{key}"
        steps.append(Step(name, "train", scale, [previous], str(training_output(config, key)), [command], regime))
        return name

    def evaluate(scale, regime, previous):
        key = "scratch" if regime == "scratch" else scale+".finetune_checkpoint" if regime == "pretrain_finetune" else scale+".checkpoint"
        checkpoint = "{scratch.checkpoint}" if key == "scratch" else "{"+key+"}"
        directory = evaluation_output(config, scale, regime)
        commands = []
        shards = 8 * config.eval_workers_per_gpu
        for index in range(shards):
            commands.append([py, "-u", "-m", "examples.baselines.bpp.eval_bpp_imitator", "--checkpoint", checkpoint,
                "--output-dir", str(directory), "--regime", regime, "--pretraining-scale", scale,
                "--dataset-root", str(config.dataset_root), "--tig-repo", str(config.tig_repo),
                "--reference-source", "human", "--training-prompt-source", "human", "--action-normalization", config.action_normalization, "--num-episodes", "10", "--max-episode-steps", "500",
                "--execution-horizon", "12", "--shader", "rt-fast", "--device", f"cuda:{index // config.eval_workers_per_gpu}",
                "--num-shards", str(shards), "--shard-index", str(index)])
        name = "eval.shared_scratch" if regime == "scratch" else f"eval.{scale}.{regime}"
        steps.append(Step(name, "eval", scale, [previous], str(directory), commands, regime))
        return name

    previous = "ready"
    scratch_step = None
    for scale in SCALES[:SCALES.index(config.through)+1]:
        if scale != "c15":
            name = f"download.{scale}"
            steps.append(Step(name, "download", scale, [previous], str(config.downloader_manifests / scale), [[str(config.downloader_python), "-u", "-m", "examples.baselines.bpp.download_bpp_imitator", "--scale", scale, "--manifest-root", str(config.downloader_manifests), "--dataset-root", str(config.dataset_root), "--tig-repo", str(config.tig_repo)]]))
            previous = name
            name = f"prepare.{scale}"
            steps.append(Step(name, "prepare", scale, [previous], str(config.data_cache), [[py, "-u", "-m", NAMESPACE+".training_data", "prepare", "--split", scale, "--prompt-source", "human", "--cache-root", str(config.data_cache), "--dataset-root", str(config.dataset_root), "--tig-repo", str(config.tig_repo), "--workers", "8"]]))
            previous = name
            name = f"features.{scale}"
            commands = [[py, "-u", "-m", NAMESPACE+".feature_cache", "prepare", "--data-manifest", "{"+scale+".data}", "--cache-root", str(config.feature_cache), "--checkpoint", str(config.external_checkpoint), "--device", f"cuda:{index}", "--shard-index", str(index), "--num-shards", "8", "--all-episodes", "--fixed-vision-preprocessing"] for index in range(8)]
            steps.append(Step(name, "features", scale, [previous], str(config.feature_cache), commands))
            previous = name
            name = f"merge.{scale}"
            command = [py, "-u", "-m", NAMESPACE+".feature_cache", "merge", "--data-manifest", "{"+scale+".data}", "--shards", *("{"+scale+f".feature_shard_{index}"+"}" for index in range(8)), "--output", "{"+scale+".features}"]
            steps.append(Step(name, "merge", scale, [previous], "{"+scale+".features}", [command]))
            previous = name
        previous = train(scale, scale, previous)
        seen = evaluate(scale, "seen", previous)
        zs = evaluate(scale, "zero_shot", seen)
        if scratch_step is None:
            scratch_train = train(scale, "scratch", zs)
            scratch_step = evaluate(scale, "scratch", scratch_train)
            previous = scratch_step
        else:
            previous = zs
        ft_train = train(scale, "finetune", previous)
        ft = evaluate(scale, "pretrain_finetune", ft_train)
        previous = f"complete.{scale}"
        steps.append(Step(previous, "gate", scale, [seen, zs, scratch_step, ft], str(config.run_root / "stages" / f"{scale}.json")))
    if config.reuse_prepared_inputs_from is not None:
        for step in steps:
            if step.kind in ("download", "prepare", "features", "merge"):
                step.commands = []
    return steps


def validate_ready(data_path, feature_path, key, fingerprint=None):
    data, feature = read_json(Path(data_path)), read_json(Path(feature_path))
    if not data or not feature:
        raise ContractError(f"Missing ready manifest for {key}")
    query, prompt = COUNTS[key]
    expected_split = "scratch" if key == "target" else key
    checks = [data.get("query_episode_count") == query, data.get("prompt_episode_count") == prompt,
              data.get("request", {}).get("split") == expected_split, data.get("request", {}).get("prompt_source") == "human",
              feature.get("complete") is True, feature.get("query_episode_count") == query,
              feature.get("prompt_episode_count") == prompt, feature.get("data_manifest_sha256") == sha256(Path(data_path))]
    if fingerprint is not None:
        checks.append(feature.get("feature_fingerprint") == fingerprint)
    if not all(checks):
        raise ContractError(f"Ready budget/prompt/source/fingerprint differs for {key}")
    return feature["feature_fingerprint"]


def training_health_contract(output, *, final=False):
    """A separate operational gate; it never changes the model training recipe."""
    from .training_health import HealthGateError, admission_check, require_no_health_stop
    try:
        require_no_health_stop(output)
        evidence = admission_check(output) if final else None
    except HealthGateError as exc:
        atomic_json(Path(output) / "training_health_rejection.json", {
            "schema": "training_health_rejection_v1", "observed_utc": now(),
            "error": str(exc), "decision": exc.decision, "formal_admission_blocked": True})
        raise ContractError(f"Training health admission rejected {output}: {exc}") from exc
    if evidence is not None:
        atomic_json(Path(output) / "training_health_admission.json", evidence)
    return evidence


def checkpoint_summary(path):
    """Only the internal inspection command imports Torch; tensors stay CPU/mmap."""
    path = Path(path)
    health = training_health_contract(path.parent, final=path.name == "final.pt")
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = value.get("metadata", {})
    keys = ("action_normalization", "experiment_identity", "regime", "prompt_source", "pretraining_scale", "pilot", "completed", "train_step", "max_updates", "init_head_sha256", "external_checkpoint_sha256", "initialize_from", "initialize_from_sha256", "data_manifest_sha256", "feature_manifest_sha256")
    result = {key: metadata.get(key) for key in (*keys, *EXPECTED_TRAINING_RECIPE, "initialize_from_weights")}
    modes = [value.get("action_normalization"), value.get("config", {}).get("action_normalization"), metadata.get("action_normalization")]
    if any(mode is not None for mode in modes) and (any(mode is None for mode in modes) or len(set(modes)) != 1):
        raise ContractError("Checkpoint action-normalization provenance conflicts or is incomplete")
    result["action_normalization"] = modes[0] or "legacy_quantile"
    if result["action_normalization"] == ACTION_NORMALIZATION_V2:
        for location in (value, value.get("config", {}), metadata):
            require_training_recipe(location)
    model = value.get("model_config", value.get("config", {}).get("model", {}))
    noise = model.get("noise_scheduler", {})
    result["inference_sampling"] = {"scheduler_class":noise.get("_target_", "").rsplit(".", 1)[-1],
        "clip_sample":noise.get("clip_sample"), "clip_sample_range":noise.get("clip_sample_range", 1.0),
        "thresholding":noise.get("thresholding", False), "num_inference_steps":model.get("num_inference_steps"),
        "num_train_timesteps":noise.get("num_train_timesteps"), "prediction_type":noise.get("prediction_type")}
    result["model_input_perturbation"] = model.get("input_pertub")
    result.update(checkpoint_ema_summary(value, metadata))
    result["normalizer_buffers_match_mode"] = checkpoint_normalizer_matches(value)
    if health is not None:
        result["training_health"] = health
    training_health_contract(path.parent)
    return result


def inspect_checkpoint(config, path):
    command = [str(config.python), "-m", NAMESPACE+".pipeline", "inspect-checkpoint", str(path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(config.tig_repo), PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(completed.stdout)


def pending_orphan_exit_observation(output, process):
    """Wait briefly for a live, identity-bound observer to publish a real reap code."""
    record = read_json(Path(output) / "exit_observer_status.json", {})
    if record.get("schema") != "orphan_exit_observer_status_v1" or record.get("status") != "observing":
        return None
    try:
        if Path(record["output_dir"]).resolve() != Path(output).resolve():
            return None
        expected = {key: process[key] for key in ("pid", "command", "started_utc")}
        if record.get("process_status_identity") != expected:
            return None
        producer = record["producer_process"]
        if producer["pid"] != process["pid"] or producer["command"] != process["command"]:
            return None
        observer = record["observer_process"]
        raw = Path(f"/proc/{observer['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        command = [x.decode() for x in Path(f"/proc/{observer['pid']}/cmdline").read_bytes().split(b"\0") if x]
        if raw[0] == "Z" or int(raw[19]) != observer["start_ticks"] or command != observer["command"]:
            return None
        source = Path(__file__).with_name("orphan_exit_observer.py")
        if Path(record["source_path"]).resolve() != source.resolve() or sha256(source) != record["source_sha256"]:
            return None
        current_path = Path(f"/proc/{producer['pid']}/stat")
        try:
            current = current_path.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            current = None
        if current is not None and (current[0] != "Z" or int(current[19]) != producer["start_ticks"]):
            return None  # A live mismatched or recycled PID is never adopted.
        current_time = datetime.now(timezone.utc)
        age = (current_time-datetime.fromisoformat(record["observed_utc"])).total_seconds()
        if not 0 <= age <= 15:
            return None
        wait_path = Path(output) / "exit_observer_wait.json"
        wait = read_json(wait_path, {})
        if wait.get("process_status_identity") != expected:
            wait = {"schema": "orphan_exit_wait_v1", "process_status_identity": expected,
                    "started_utc": current_time.isoformat(), "maximum_wait_seconds": 75}
            atomic_json(wait_path, wait)
        deadline_start = datetime.fromisoformat(wait["started_utc"])
        if record.get("producer_missing_since_utc"):
            deadline_start = min(deadline_start, datetime.fromisoformat(record["producer_missing_since_utc"]))
        elapsed = (current_time-deadline_start).total_seconds()
        if not 0 <= elapsed <= 75:
            return None
        return {"status": "running", "awaiting_verified_exit_observation": True,
                "observer_pid": observer["pid"], "waited_seconds": elapsed,
                "producer_pid": producer["pid"], "exit_status_not_yet_confirmed": True}
    except (OSError, KeyError, TypeError, ValueError):
        return None


def training_contract(config, step, artifacts, *, checkpoint_reader=None, allow_incomplete=False):
    output = Path(step.output)
    training_health_contract(output)
    manifest = read_json(output / "training_manifest.json")
    process = read_json(output / "process_status.json", {})
    if not manifest:
        if process.get("status") == "running":
            observed = process_observation(process, wrapper=None)
            if not observed["alive"] or not observed["matches"]:
                raise ContractError("Training startup PID/command differs")
            return {"status": "running", "process": observed}
        if any((output / name).exists() for name in ("final.pt", "latest.pt", "interrupted.pt", "completion.json", "progress.json")):
            raise ContractError("Training artifacts exist without their identity manifest")
        return {"status": "pending", "process": process}
    unsigned = {key: value for key, value in manifest.items() if key != "experiment_identity"}
    if identity(unsigned, training=True) != manifest.get("experiment_identity"):
        raise ContractError(f"Training manifest identity invalid: {output}")
    command = process.get("command", [])
    try:
        command_matches = command[command.index("--output-dir")+1] == str(output) and command[command.index("--regime")+1] == step.regime and command[command.index("--action-normalization")+1] == config.action_normalization
    except (ValueError, IndexError):
        command_matches = False
    if not command_matches:
        raise ContractError("Training process record belongs to another output/regime/action normalization")
    require_training_recipe(manifest)
    data_key = "target" if step.regime in ("scratch", "finetune") else step.scale
    data_path, feature_path = Path(artifacts[data_key+".data"]), Path(artifacts[data_key+".features"])
    expected = {"action_normalization": config.action_normalization, "regime": step.regime, "prompt_source": "human", "pilot": False, "world_size": 8, "global_batch": 64,
                "max_updates": BUDGETS[step.regime], "fixed_vision_preprocessing": True, "init_seed": config.seed,
                "pretraining_scale": None if step.regime == "scratch" else step.scale,
                "data_manifest": str(data_path.resolve()), "data_manifest_sha256": sha256(data_path),
                "feature_manifest": str(feature_path.resolve()), "feature_manifest_sha256": sha256(feature_path),
                "external_checkpoint": str(config.external_checkpoint.resolve())}
    for name in ("trainer", "native_model", "training_data", "feature_cache_source"):
        filename = {"trainer": "train.py", "feature_cache_source": "feature_cache.py"}.get(name, name+".py")
        expected[name+"_sha256"] = sha256(Path(__file__).with_name(filename))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ContractError(f"Training contract mismatch {key}: {output}")
    if step.regime == "finetune":
        if manifest.get("initialize_from_weights") != "ema":
            raise ContractError("Fresh fine-tuning must initialize from the corresponding Ck EMA weights")
        origin = Path(artifacts[step.scale+".checkpoint"])
        lineage = manifest.get("initialize_from_metadata") or {}
        if manifest.get("initialize_from") != str(origin.resolve()) or manifest.get("initialize_from_sha256") != sha256(origin) or lineage.get("regime") != step.scale or lineage.get("completed") is not True or lineage.get("pilot") is not False or lineage.get("train_step") != BUDGETS[step.scale] or lineage.get("action_normalization") != config.action_normalization:
            raise ContractError("Fine-tune lineage differs from its complete scale checkpoint")
        require_training_recipe(lineage)
    elif manifest.get("initialize_from") is not None or manifest.get("initialize_from_metadata") is not None:
        raise ContractError("Pretraining/Scratch must start independently from the common external model")
    common = read_json(config.c15_output / "training_manifest.json", manifest)
    if manifest.get("external_checkpoint_sha256") != common.get("external_checkpoint_sha256"):
        raise ContractError("External initialization differs between regimes")
    if process.get("status") == "running":
        observed = process_observation(process, wrapper=None)
        if not observed["alive"] or not observed["matches"]:
            latest = read_json(output / "process_status.json", {})
            if latest.get("status") != "running":
                return training_contract(config, step, artifacts, checkpoint_reader=checkpoint_reader, allow_incomplete=allow_incomplete)
            exit_wait = pending_orphan_exit_observation(output, process)
            if exit_wait is not None:
                return dict(exit_wait, experiment_identity=manifest["experiment_identity"])
            latest = read_json(output / "process_status.json", {})
            if latest.get("status") != "running":
                return training_contract(config, step, artifacts, checkpoint_reader=checkpoint_reader, allow_incomplete=allow_incomplete)
            if allow_incomplete and not observed["alive"] and not matching_training_producers(output):
                return {"status": "pending", "recovery_required": True}
            raise ContractError(f"Running training owner PID/command does not match: {output}")
        return {"status": "running", "process": observed, "experiment_identity": manifest["experiment_identity"]}
    completion = read_json(output / "completion.json")
    code = process.get("returncode", process.get("exit_code"))
    if completion is None or code != 0 or process.get("status") != "completed":
        if allow_incomplete:
            return {"status": "pending", "recovery_required": True}
        raise ContractError(f"Training lacks completed marker AND clean process exit: {output}")
    if completion.get("status") != "completed" or completion.get("pilot") is not False or completion.get("completed_updates") != BUDGETS[step.regime] or completion.get("target_updates") != BUDGETS[step.regime] or completion.get("experiment_identity") != manifest["experiment_identity"]:
        if allow_incomplete and completion.get("status") == "interrupted" and completion.get("experiment_identity") == manifest["experiment_identity"]:
            return {"status": "pending", "recovery_required": True}
        raise ContractError(f"Interrupted/pilot/incomplete training is not a completed stage: {output}")
    final = output / "final.pt"
    training_health_contract(output, final=True)
    if checkpoint_reader is None:
        return {"status": "candidate_complete", "experiment_identity": manifest["experiment_identity"]}
    metadata = checkpoint_reader(final)
    for key in ("action_normalization", "experiment_identity", "regime", "prompt_source", "pretraining_scale", "pilot", "max_updates", "external_checkpoint_sha256", "data_manifest_sha256", "feature_manifest_sha256"):
        if metadata.get(key) != manifest.get(key):
            raise ContractError(f"Final checkpoint metadata differs: {key}")
    require_training_recipe(metadata)
    require_sampling(metadata.get("inference_sampling"))
    if not checkpoint_runtime_valid(metadata):
        raise ContractError("Checkpoint lacks matching EMA state, original perturbation or minmax-v2 buffers")
    if metadata.get("initialize_from_weights") != manifest.get("initialize_from_weights"):
        raise ContractError("Checkpoint fine-tuning weight selection differs")
    if metadata.get("completed") is not True or metadata.get("train_step") != BUDGETS[step.regime]:
        raise ContractError("Final checkpoint is incomplete")
    initial = read_json(config.c15_output / "model_initialization.json", {})
    if initial.get("action_normalization") != config.action_normalization:
        raise ContractError("Common C15 initialization is not the minmax-v2 run")
    if step.regime != "finetune" and initial.get("init_head_sha256") and metadata.get("init_head_sha256") != initial["init_head_sha256"]:
        raise ContractError("Common new-head initialization differs")
    return {"status": "complete", "experiment_identity": manifest["experiment_identity"], "checkpoint": str(final), "checkpoint_sha256": sha256(final)}


def official_tasks(config, regime):
    filename = "seen_5tasks_env_list.txt" if regime == "seen" else "unseen_5tasks_env_list.txt"
    path = config.tig_repo / "examples/baselines/lerobot_dataset/eval/exp_list" / filename
    tasks = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if len(tasks) != 20 or len(set(tasks)) != 20:
        raise ContractError("Expected20 official task-level variants")
    return tasks


def evaluation_contract(config, step, *, checkpoint_hash=None):
    directory = Path(step.output)
    manifest = read_json(directory / "evaluation_manifest.json")
    if manifest is None:
        return {"status": "pending"}
    if manifest.get("checkpoint"):
        training_health_contract(Path(manifest["checkpoint"]).parent, final=True)
    experiment = manifest.get("experiment_identity")
    if identity({key: value for key, value in manifest.items() if key != "experiment_identity"}) != experiment:
        raise ContractError("Evaluation manifest identity changed")
    tasks = official_tasks(config, step.regime)
    required = {"action_normalization": config.action_normalization, "regime": step.regime, "reference_source": "human", "training_prompt_source": "human", "is_pilot": False,
                "tasks": tasks, "seeds": list(range(100000, 100010)), "max_episode_steps": 500, "execution_horizon": 12,
                "control_mode": "pd_joint_pos", "shader": "rt-fast"}
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ContractError(f"Evaluation contract mismatch: {key}")
    require_sampling(manifest.get("inference_sampling"))
    training_metadata = manifest.get("training_metadata", {})
    require_training_recipe(training_metadata)
    if (training_metadata.get("action_normalization") != config.action_normalization
            or manifest.get("loaded_weight_selection") != "ema"
            or manifest.get("loaded_ema_optimization_step") != training_metadata.get("train_step")):
        raise ContractError("Evaluation did not load the matching minmax-v2 EMA weights")
    if step.regime != "scratch" and manifest.get("pretraining_scale") != step.scale:
        raise ContractError("Evaluation scale lineage differs")
    if checkpoint_hash and manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ContractError("Evaluation checkpoint hash differs")
    cases = [(task, episode) for task in tasks for episode in range(10)]
    shards = 8 * config.eval_workers_per_gpu
    covered, statuses = set(), []
    for index in range(shards):
        name = f"shard_{index:03d}_of_{shards:03d}"
        claim = read_json(directory / (name+".json"))
        process = read_json(directory / ("process_status_"+name+".json"))
        expected = cases[index::shards]
        if not claim or not process:
            return {"status": "pending", "missing_worker": index}
        actual = [tuple(case) for case in claim.get("cases", [])]
        if actual != expected or covered.intersection(actual) or claim.get("experiment_identity") != experiment:
            raise ContractError("Evaluation shard coverage/identity differs")
        covered.update(actual)
        if process.get("experiment_identity") not in (None, experiment):
            raise ContractError("Evaluation process belongs to another experiment")
        if process.get("returncode") != 0 or process.get("owns_evaluation_attempt") is not True or process.get("attempt_id") != claim.get("attempt_id"):
            raise IncompleteEvaluation(f"Evaluation worker {index} did not exit cleanly with ownership")
        statuses.append(process)
    if covered != set(cases):
        raise ContractError("Evaluation cases are missing or duplicated")
    records = []
    for task, episode in cases:
        row = read_json(directory / "human" / task / f"seed_{100000+episode}" / "metrics.json")
        if not row or row.get("status") != "completed" or row.get("process_exit_verified") is not True or row.get("process_returncode") != 0:
            raise IncompleteEvaluation(f"Invalid/missing/unclean evaluation case: {task}/{episode}")
        if row.get("experiment_identity") != experiment or row.get("task_id") != task or row.get("seed") != 100000+episode or row.get("is_pilot") is not False or type(row.get("success_at_end")) is not bool or type(row.get("success_once")) is not bool:
            raise ContractError(f"Evaluation case identity/score contract differs: {task}/{episode}")
        records.append(row)
        reference = row.get("reference", {})
        expected_case = {"experiment_identity": experiment, "task_id": task, "seed": 100000+episode,
                         "reference": reference, "reference_source": "human"}
        if row.get("reference_source") != "human" or row.get("case_id") != identity(expected_case) or reference.get("episode_index") != episode or row.get("max_episode_steps") != 500 or row.get("executed_control_steps") != 500 or not isinstance(row.get("last_normalized_reward"), (int, float)) or not math.isfinite(row["last_normalized_reward"]):
            raise ContractError(f"Evaluation reference/horizon/reward/case identity differs: {task}/{episode}")
    sr = sum(row["success_at_end"] for row in records) / 200
    once = sum(row["success_once"] for row in records) / 200
    summary = read_json(directory / "summary.json", {})
    if summary.get("experiment_identity") != experiment or summary.get("planned_episode_count") != 200 or summary.get("incomplete_or_errors") is not False or summary.get("process_exit_pending") is not False or summary.get("success_at_end_rate") != sr or summary.get("success_once_rate") != once:
        raise ContractError("Global summary differs from the200 independently verified cases")
    return {"status": "complete", "output_dir": str(directory), "experiment_identity": experiment, "episodes": 200,
            "checkpoint": manifest.get("checkpoint"), "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "success_at_end_rate": sr, "success_once_rate": once, "worker_returncodes": [row["returncode"] for row in statuses]}


def download_contract(config, step):
    from .download_data import REPO_ID, REVISION, selected_folders, selected_path
    verification = read_json(Path(step.output) / "verification.json", {})
    source = read_json(Path(step.output) / "source.json", {})
    expected_folders = {row["repo_id"] for row in selected_folders(config.tig_repo, step.scale)[0]}
    if source.get("scale") != step.scale or {row["repo_id"] for row in source.get("folders", [])} != expected_folders or source.get("inventory_complete") is not True:
        raise ContractError("Download folder selection differs from the official scale+target union")
    if verification.get("status") != "complete" or verification.get("repo_id") != REPO_ID or verification.get("revision") != REVISION or source.get("revision") != REVISION or verification.get("errors"):
        raise ContractError("Download verification is incomplete or from another revision")
    actual = {row["filename"]: row for row in verification.get("files", [])}
    expected = {row["filename"]: row for row in source.get("files", [])}
    if not expected or set(actual) != set(expected) or verification.get("verified_files") != len(expected) or verification.get("verified_bytes") != sum(row["size"] for row in expected.values()):
        raise ContractError("Downloaded files/verified byte counts differ")
    for filename, row in expected.items():
        key = "actual_sha256" if row.get("sha256") else "actual_git_blob_sha1"
        digest = row.get("sha256") or row.get("git_blob_sha1")
        if not selected_path(filename, expected_folders) or actual[filename].get(key) != digest or actual[filename].get("size") != row["size"]:
            raise ContractError("Download content hash/size/modality differs")
    return {"status": "complete", "verification": str(Path(step.output) / "verification.json")}


def last_report(path, key):
    text = Path(path).read_text()[-262144:]
    for offset in reversed([match.start() for match in re.finditer(r"\{", text)]):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[offset:])
            if isinstance(value, dict) and key in value:
                return value[key]
        except json.JSONDecodeError:
            pass
    raise ContractError(f"No {key} result found in {path}")


def run_commands(config, step, commands, attempt_dir):
    """Own and supervise launched groups; never signal an adopted external job."""
    import fcntl
    jobs = []
    output_lock = None
    if step.kind == "train":
        output = Path(step.output)
        output.mkdir(parents=True, exist_ok=True)
        output_lock = (output / ".pipeline_launch.lock").open("a")
        try:
            fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            output_lock.close()
            raise ContractError("Another scheduler owns this training directory") from exc
        existing = read_json(output / "process_status.json", {})
        if (existing.get("status") == "running" and process_observation(existing)["alive"]) or matching_training_producers(output):
            output_lock.close()
            raise ContractError("A training producer is already alive; refusing a duplicate launch")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="1", PYTHONPATH=str(config.tig_repo)+os.pathsep+str(ROOT / "repos/behavior_prompting"))
    try:
        for index, command in enumerate(commands):
            log = attempt_dir / f"worker_{index:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("ab", buffering=0)
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=config.tig_repo, env=env, start_new_session=True)
            status_path = Path(step.output) / "process_status.json" if step.kind == "train" else attempt_dir / f"worker_{index:03d}.json"
            status = dict(status="running", pid=process.pid, command=command, started_utc=now(), stdout_log=str(log), formal=True)
            atomic_json(status_path, status)
            jobs.append((process, handle, status_path, status))
        while any(process.poll() is None for process, *_ in jobs):
            time.sleep(min(config.poll_seconds, 10.))
    except BaseException:
        for process, *_ in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, *_ in jobs:
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        for process, handle, status_path, status in jobs:
            handle.close()
            status.update(returncode=process.returncode, status="completed" if process.returncode == 0 else "process_error", finished_utc=now())
            atomic_json(status_path, status)
        if output_lock is not None:
            output_lock.close()
    return [{"returncode": process.returncode, "log": status["stdout_log"]} for process, _, _, status in jobs]


def archive_evaluation_retry(config, step):
    directory = Path(step.output)
    history = directory / "failed_attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    for path in directory.glob("*.json"):
        target = history / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for task in official_tasks(config, step.regime):
        for episode in range(10):
            case = directory / "human" / task / f"seed_{100000+episode}"
            record = read_json(case / "metrics.json", {})
            if case.exists() and (record.get("status") != "completed" or record.get("process_exit_verified") is not True):
                target = history / "human" / task / case.name
                target.parent.mkdir(parents=True, exist_ok=True)
                case.rename(target)


def active_attempt_workers(entry):
    if not entry.get("attempts"):
        return []
    directory = Path(entry["attempts"][-1]["directory"])
    active = []
    for path in directory.glob("worker_*.json"):
        record = read_json(path, {})
        if record.get("status") != "running":
            continue
        observed = process_observation(record)
        if observed["alive"] and not observed["matches"]:
            raise ContractError("Evaluation/feature worker PID was reused; refusing adoption")
        if observed["alive"]:
            active.append(observed)
    return active


def execute_pipeline(config, *, retry_failed=False, launcher=run_commands, checkpoint_reader=None):
    import fcntl
    config.validate()
    config.run_root.mkdir(parents=True, exist_ok=True)
    with (config.run_root / ".pipeline.lock").open("a") as master:
        try:
            fcntl.flock(master, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("Another orchestrator owns this pipeline") from exc
        config_values = json.loads(json.dumps(asdict(config), default=str))
        # Stage cap and polling do not alter training or evaluation identities.
        config_id = identity({key: value for key, value in config_values.items() if key not in ("through", "poll_seconds")})
        path = config.run_root / "pipeline_state.json"
        state = read_json(path, {"schema": "bpp_tig_pipeline_v1", "config": config_values, "config_identity": config_id, "artifacts": initial_artifacts(config), "steps": {}})
        if state.get("config_identity") != config_id:
            raise ContractError("Pipeline root belongs to another configuration")
        state["config"] = config_values
        artifacts = state["artifacts"]
        reader = checkpoint_reader or (lambda checkpoint: inspect_checkpoint(config, checkpoint))
        steps = build_steps(config)
        fingerprints = {}
        for step in steps:
            adopted_now = False
            entry = state["steps"].setdefault(step.name, {"status": "pending", "attempts": []})
            entry.update(kind=step.kind, scale=step.scale, regime=step.regime, output_dir=step.output)
            if any(state["steps"].get(dep, {}).get("status") != "complete" for dep in step.dependencies):
                raise ContractError(f"Dependencies not complete for {step.name}")
            if entry["status"] == "failed" and not retry_failed:
                state.update(status="failed", active_step=step.name, main_metrics_valid=False)
                atomic_json(path, state)
                return state
            try:
                if step.kind == "ready":
                    fingerprints["common"] = validate_ready(artifacts["c15.data"], artifacts["c15.features"], "c15")
                    validate_ready(artifacts["target.data"], artifacts["target.features"], "target", fingerprints["common"])
                    if config.reuse_prepared_inputs_from is not None:
                        for key in ("c15", "target"):
                            reused_input_contract(config, artifacts, key, fingerprints["common"])
                    common = read_json(config.c15_output / "training_manifest.json", {})
                    if common and common.get("action_normalization") != config.action_normalization:
                        raise ContractError("Legacy C15 training cannot be adopted into minmax-v2")
                    if common and common.get("external_checkpoint_sha256") != sha256(config.external_checkpoint):
                        raise ContractError("Common external checkpoint content changed")
                    evidence = {"status": "complete"}
                elif step.kind == "train":
                    data_key = "target" if step.regime in ("scratch", "finetune") else step.scale
                    validate_ready(artifacts[data_key+".data"], artifacts[data_key+".features"], data_key, fingerprints["common"])
                    evidence = training_contract(config, step, artifacts, checkpoint_reader=reader, allow_incomplete=retry_failed)
                    while evidence["status"] == "running":
                        adopted_now = True
                        entry.update(status="running", adopted=True, evidence=evidence)
                        state.update(status="running", active_step=step.name)
                        atomic_json(path, state)
                        time.sleep(config.poll_seconds)
                        evidence = training_contract(config, step, artifacts, checkpoint_reader=reader)
                elif step.kind == "eval":
                    checkpoint = resolve(step.commands[0], artifacts)[resolve(step.commands[0], artifacts).index("--checkpoint")+1]
                    checkpoint_hash = sha256(Path(checkpoint))
                    while True:
                        active = active_attempt_workers(entry)
                        try:
                            evidence = evaluation_contract(config, step, checkpoint_hash=checkpoint_hash)
                        except IncompleteEvaluation:
                            if not active and not retry_failed:
                                raise
                            evidence = {"status": "pending"}
                        if not active:
                            break
                        adopted_now = True
                        entry.update(status="running", adopted=True, evidence={"status": "running", "workers": active})
                        state.update(status="running", active_step=step.name)
                        atomic_json(path, state)
                        time.sleep(config.poll_seconds)
                elif step.kind == "gate":
                    evidence = {"status": "complete", "scale": step.scale, "regimes": {dep: state["steps"][dep]["evidence"] for dep in step.dependencies}, "overall_review_pending": True}
                    atomic_json(Path(step.output), evidence)
                elif config.reuse_prepared_inputs_from is not None and step.kind in ("download", "prepare", "features", "merge"):
                    if step.kind == "download":
                        evidence = dict(download_contract(config, step), reused_prepared_inputs=True,
                            source_run_root=str(config.reuse_prepared_inputs_from.resolve()), launched_download_or_encoding=False)
                    else:
                        evidence = reused_input_contract(config, artifacts, step.scale, fingerprints["common"])
                    evidence["operation"] = "local_reuse_" + step.kind
                elif step.kind == "merge" and entry.get("status") == "complete":
                    validate_ready(artifacts[step.scale+".data"], artifacts[step.scale+".features"], step.scale, fingerprints["common"])
                    evidence = entry["evidence"]
                elif step.kind == "download" and entry.get("status") == "complete":
                    evidence = download_contract(config, step)
                else:
                    evidence = entry.get("evidence", {"status": "pending"})
                if evidence["status"] != "complete":
                    if config.reuse_prepared_inputs_from is not None and step.kind in ("download", "prepare", "features", "merge"):
                        raise ContractError("Cached inputs failed validation; automatic download or re-encoding is disabled")
                    if adopted_now:
                        raise ContractError("An adopted stage became incomplete")
                    if step.kind == "eval" and (Path(step.output) / "evaluation_manifest.json").exists() and not retry_failed:
                        raise ContractError("Partial evaluation requires explicit --retry-failed; refusing duplicate workers")
                    commands = [resolve(command, artifacts) for command in step.commands]
                    if step.kind == "train" and (Path(step.output) / "training_manifest.json").exists():
                        if not retry_failed:
                            raise ContractError("Existing incomplete training requires explicit --retry-failed")
                        candidates = []
                        for name in ("interrupted.pt", "latest.pt"):
                            candidate = Path(step.output) / name
                            if candidate.is_file():
                                meta = reader(candidate)
                                manifest = read_json(Path(step.output) / "training_manifest.json")
                                if meta.get("experiment_identity") == manifest["experiment_identity"] and meta.get("regime") == step.regime and meta.get("pilot") is False and meta.get("completed") is False and meta.get("prompt_source") == "human" and meta.get("action_normalization") == config.action_normalization and checkpoint_runtime_valid(meta) and meta.get("max_updates") == BUDGETS[step.regime] and 0 < meta.get("train_step", 0) < BUDGETS[step.regime]:
                                    candidates.append((meta["train_step"], candidate))
                        if not candidates:
                            raise ContractError("No matching resumable checkpoint; refusing a fresh overwrite")
                        commands[0] += ["--resume-from", str(max(candidates)[1])]
                        if "--initialize-from" in commands[0]:
                            index = commands[0].index("--initialize-from")
                            del commands[0][index:index+2]
                    if step.kind == "eval" and retry_failed:
                        archive_evaluation_retry(config, step)
                    attempt_dir = config.run_root / "attempts" / step.name / str(len(entry["attempts"]))
                    attempt = {"started_utc": now(), "commands": commands, "directory": str(attempt_dir)}
                    if step.kind == "train":
                        for filename in ("process_status.json", "completion.json", "progress.json"):
                            old = Path(step.output) / filename
                            if old.exists():
                                archived = attempt_dir / "previous" / filename
                                archived.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(old, archived)
                    entry["attempts"].append(attempt)
                    entry["status"] = "running"
                    state.update(status="running", active_step=step.name)
                    atomic_json(path, state)
                    results = launcher(config, step, commands, attempt_dir)
                    attempt.update(finished_utc=now(), workers=results)
                    if any(result.get("returncode") != 0 for result in results) or len(results) != len(commands):
                        raise ContractError(f"Worker exited abnormally for {step.name}")
                    if step.kind == "train":
                        evidence = training_contract(config, step, artifacts, checkpoint_reader=reader)
                    elif step.kind == "eval":
                        checkpoint = Path(commands[0][commands[0].index("--checkpoint")+1])
                        evidence = evaluation_contract(config, step, checkpoint_hash=sha256(checkpoint))
                    elif step.kind == "prepare":
                        artifacts[step.scale+".data"] = last_report(Path(results[0]["log"]), "manifest")
                        data = read_json(Path(artifacts[step.scale+".data"]))
                        if data.get("query_episode_count") != COUNTS[step.scale][0] or data.get("prompt_episode_count") != COUNTS[step.scale][1] or data.get("request", {}).get("split") != step.scale or data.get("request", {}).get("prompt_source") != "human":
                            raise ContractError("Prepared data split/prompt budget differs")
                        evidence = {"status": "complete", "manifest": artifacts[step.scale+".data"]}
                    elif step.kind == "features":
                        for index, result in enumerate(results):
                            artifacts[step.scale+f".feature_shard_{index}"] = last_report(Path(result["log"]), "feature_manifest")
                        evidence = {"status": "complete"}
                    elif step.kind == "merge":
                        validate_ready(artifacts[step.scale+".data"], artifacts[step.scale+".features"], step.scale, fingerprints["common"])
                        evidence = {"status": "complete", "manifest": artifacts[step.scale+".features"]}
                    elif step.kind == "download":
                        evidence = download_contract(config, step)
                    if evidence.get("status") != "complete":
                        raise ContractError("Process exited zero but stage artifacts are not complete")
                entry.update(status="complete", evidence=evidence, checked_utc=now())
                atomic_json(path, state)
            except BaseException as exc:
                entry.update(status="failed", error_type=type(exc).__name__, error=str(exc), checked_utc=now())
                state.update(status="failed", active_step=step.name, main_metrics_valid=False)
                atomic_json(path, state)
                return state
        state.update(status="measurements_complete_review_pending", active_step=None, main_metrics_valid=True, overall_review_pending=True)
        atomic_json(path, state)
        atomic_json(config.run_root / "review_inputs.json", {"stages": {key: value["evidence"] for key, value in state["steps"].items() if key.startswith("complete.")}, "overall_review_pending": True})
        return state


def monitor(config):
    """Read JSON and /proc only; never write, import Torch or inspect tensor files."""
    state = read_json(config.run_root / "pipeline_state.json", {})
    training = {}
    evaluations = {}
    for key in ("c15", "scratch", "c15_finetune", "c30", "c30_finetune", "c45", "c45_finetune"):
        output = training_output(config, key)
        try:
            process = read_json(output / "process_status.json", {})
            progress = read_json(output / "progress.json", {})
            completion = read_json(output / "completion.json", {})
            if process or progress or completion:
                training[key] = {"output": str(output), "process_status": process.get("status"), "returncode": process.get("returncode", process.get("exit_code")),
                                 "process": process_observation(process, wrapper=config.c15_wrapper if key == "c15" else None),
                                 "step": progress.get("step"), "target": progress.get("max_updates"), "eta_seconds": progress.get("estimated_remaining_seconds"),
                                 "completion": completion.get("status")}
        except (OSError, ValueError) as exc:
            training[key] = {"read_error": str(exc)}
    for step in build_steps(config):
        if step.kind != "eval" or not Path(step.output).is_dir():
            continue
        try:
            directory = Path(step.output)
            rows = [read_json(path, {}) for path in directory.glob("human/*/seed_*/metrics.json")]
            exits = [read_json(path, {}) for path in directory.glob("process_status_shard_*.json")]
            clean = sum(row.get("status") == "completed" and row.get("process_exit_verified") is True and row.get("process_returncode") == 0 for row in rows)
            evaluations[step.name] = {
                "output_dir": step.output, "planned_cases": 200, "completed_clean_cases": clean,
                "error_cases": sum(row.get("status") == "error" for row in rows),
                "pending_process_exit_cases": sum(row.get("status") == "completed" and row.get("process_exit_verified") is not True for row in rows),
                "clean_worker_exits": sum(row.get("returncode") == 0 and row.get("owns_evaluation_attempt") is True for row in exits),
                "worker_errors": sum(row.get("returncode") != 0 or row.get("owns_evaluation_attempt") is not True for row in exits),
                "formal_validity": state.get("steps", {}).get(step.name, {}).get("status") == "complete",
            }
        except (OSError, ValueError) as exc:
            evaluations[step.name] = {"read_error": str(exc)}
    return {"read_only": True, "status": state.get("status", "not_started"), "active_step": state.get("active_step"),
            "main_metrics_valid": state.get("main_metrics_valid", False), "overall_review_pending": state.get("overall_review_pending", True),
            "steps": {name: {key: row.get(key) for key in ("status", "error", "checked_utc")} for name, row in state.get("steps", {}).items()}, "training": training, "evaluations": evaluations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "monitor"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--run-root", type=Path)
        command.add_argument("--c15-output", type=Path)
        command.add_argument("--action-normalization", choices=(ACTION_NORMALIZATION_V2,))
        command.add_argument("--reuse-prepared-inputs-from", type=Path)
        command.add_argument("--through", choices=SCALES)
        if name == "plan":
            command.add_argument("--output", type=Path)
        if name == "run":
            command.add_argument("--retry-failed", action="store_true")
    inspect = commands.add_parser("inspect-checkpoint")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect-checkpoint":
        print(json.dumps(checkpoint_summary(args.path)))
        return 0
    config = PipelineConfig.load(args.config, run_root=args.run_root, c15_output=args.c15_output, through=args.through, action_normalization=args.action_normalization, reuse_prepared_inputs_from=args.reuse_prepared_inputs_from)
    config.validate()
    if args.command == "plan":
        artifacts = initial_artifacts(config)
        result = {"schema": "bpp_tig_pipeline_plan_v1", "config": asdict(config), "launches_jobs": False,
                  "current": monitor(config), "steps": [dict(asdict(step), commands=[resolve(command, artifacts, strict=False) for command in step.commands]) for step in build_steps(config)]}
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, default=str, indent=2))
    elif args.command == "monitor":
        print(json.dumps(monitor(config), indent=2))
    else:
        result = execute_pipeline(config, retry_failed=args.retry_failed)
        print(json.dumps({key: result.get(key) for key in ("status", "active_step", "main_metrics_valid", "overall_review_pending")}, indent=2))
        return 0 if result["status"] == "measurements_complete_review_pending" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

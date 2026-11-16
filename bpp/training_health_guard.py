"""Attach a CPU-only health supervisor to existing, source-frozen training jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from .training_health import DEFAULT_POLICY, HealthGateError, admission_check, assess_training, load_policy, policy_identity

STAGES = ("c30", "c30_finetune", "c45", "c45_finetune")
TRAIN_MODULE = "examples.baselines.bpp.train_bpp_imitator"
STOP_NAME = "training_health_stop.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_snapshot(pid):
    """Read only identity fields; never return an unfiltered process environment."""
    path = Path("/proc") / str(int(pid))
    try:
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        command = (path / "cmdline").read_bytes().decode().strip("\0").split("\0")
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]),
                "pgrp": int(fields[2]), "session": int(fields[3]),
                "start_ticks": int(fields[19]), "command": command}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    current = process_snapshot(expected["pid"])
    return bool(current and current["state"] != "Z" and all(
        current[key] == expected[key] for key in ("start_ticks", "command")))


def signal_verified(expected, signum):
    """Bind the signal to the validated Linux task, even if a PID is recycled."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd signaling is required; refusing an unbound PID signal")
    if not same_process(expected):
        raise RuntimeError("Refusing to signal a process whose identity changed")
    fd = os.pidfd_open(expected["pid"], 0)
    try:
        if not same_process(expected):
            raise RuntimeError("Process identity changed while opening pidfd")
        signal.pidfd_send_signal(fd, signum)
    finally:
        os.close(fd)


def training_arguments(command):
    matches = [i for i in range(len(command)-1)
               if command[i] == "-m" and command[i+1] == TRAIN_MODULE]
    if len(matches) != 1:
        raise RuntimeError("Training module is not uniquely identified")
    return command[matches[0]+2:]


def flag_value(arguments, flag):
    if arguments.count(flag) != 1:
        raise RuntimeError(f"Missing or repeated training argument: {flag}")
    i = arguments.index(flag)
    if i+1 >= len(arguments):
        raise RuntimeError(f"Missing value: {flag}")
    return arguments[i+1]


def validated_producer(output, process):
    if process.get("status") != "running":
        return None
    current = process_snapshot(process["pid"])
    if not current or current["state"] == "Z":
        return None
    if current["command"] != process.get("command"):
        raise RuntimeError("Producer PID does not match its recorded complete command")
    if current["pgrp"] != current["pid"] or current["session"] != current["pid"]:
        raise RuntimeError("Producer is not the recorded isolated torchrun session")
    command = current["command"]
    if "torch.distributed.run" not in command:
        raise RuntimeError("Recorded producer is not torchrun")
    arguments = training_arguments(command)
    if Path(flag_value(arguments, "--output-dir")).resolve() != output.resolve():
        raise RuntimeError("Producer output directory differs")
    if flag_value(arguments, "--action-normalization") != "train_minmax_joint14_gripper_identity_v2":
        raise RuntimeError("Producer is outside the accepted v2 recipe")
    return current


def validated_workers(producer, output):
    expected_arguments = training_arguments(producer["command"])
    manifest = read_json(output / "training_manifest.json", {})
    world = manifest.get("world_size")
    if world != 8:
        raise RuntimeError("Health guardian expects the recorded eight-rank training stage")
    found = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            raw = (path / "stat").read_text()
            fields = raw[raw.rfind(")")+2:].split()
            if int(fields[1]) != producer["pid"]:
                continue
            current = process_snapshot(int(path.name))
            if not current or current["state"] == "Z":
                continue
            try:
                arguments = training_arguments(current["command"])
            except RuntimeError:
                continue
            if arguments != expected_arguments:
                continue
            env = {}
            for item in (path / "environ").read_bytes().split(b"\0"):
                key, separator, value = item.partition(b"=")
                if separator and key in (b"RANK", b"LOCAL_RANK", b"WORLD_SIZE", b"LOCAL_WORLD_SIZE", b"TORCHELASTIC_RUN_ID", b"TORCHELASTIC_MAX_RESTARTS"):
                    env[key.decode()] = value.decode()
            rank = int(env.get("RANK", "-1"))
            if (env.get("WORLD_SIZE") != "8" or env.get("LOCAL_WORLD_SIZE") != "8"
                    or env.get("LOCAL_RANK") != str(rank) or rank not in range(8) or rank in found):
                raise RuntimeError("DDP worker rank identity is inconsistent")
            if not env.get("TORCHELASTIC_RUN_ID") or env.get("TORCHELASTIC_MAX_RESTARTS") != "0":
                raise RuntimeError("DDP worker elastic identity differs from the fixed v2 run")
            found[rank] = dict(current, rank=rank, local_rank=rank, world_size=8,
                               elastic_run_id=env["TORCHELASTIC_RUN_ID"])
        except (FileNotFoundError, ProcessLookupError):
            continue
    if (set(found) != set(range(8)) or len({w["elastic_run_id"] for w in found.values()}) != 1
            or not same_process(producer)):
        raise RuntimeError("Could not uniquely validate all eight direct DDP workers")
    return [found[i] for i in range(8)]


def stop_stage(output, assessment, events, *, grace_seconds):
    """Sticky fence first, cooperative rank0 stop second; preserve producer records."""
    marker_path = output / STOP_NAME
    marker = read_json(marker_path)
    if marker is None:
        marker = {"schema": "training_health_stop_v1", "detected_utc": now(),
                  "detected_unix": time.time(), "stage": output.name,
                  "assessment": assessment, "formal_admission_blocked": True,
                  "action": "pending_identity_validation"}
        atomic_json(marker_path, marker)
        events("health_stop_detected", stage=output.name, rule=assessment.get("rule"),
               trigger_step=assessment.get("trigger_step"))
    process = read_json(output / "process_status.json", {})
    producer = validated_producer(output, process)
    if producer is None:
        marker.update(action="producer_exited", producer_status=process.get("status"),
                      updated_utc=now(), completion=read_json(output / "completion.json"),
                      preserved_checkpoints=[name for name in ("latest.pt", "interrupted.pt", "final.pt")
                                             if (output / name).is_file()])
        atomic_json(marker_path, marker)
        return marker
    if "producer" not in marker:
        # Persist a fixed identity and timeout before validating workers. Even a
        # partial worker shutdown must not restart the grace clock indefinitely.
        marker.update(producer=producer, cooperative_stop_started_unix=time.time(),
                      cooperative_stop_started_utc=now(), updated_utc=now())
        atomic_json(marker_path, marker)
    original = marker["producer"]
    if (producer["pid"] != original["pid"] or producer["start_ticks"] != original["start_ticks"]
            or not same_process(original)):
        raise RuntimeError("Refusing to stop a replacement producer behind a sticky health fence")
    if marker.get("action") == "launcher_sigterm_after_timeout":
        return marker
    if time.time()-marker["cooperative_stop_started_unix"] >= grace_seconds:
        signal_verified(original, signal.SIGTERM)
        marker.update(action="launcher_sigterm_after_timeout", updated_utc=now(),
                      warning="Graceful save exceeded its timeout; logs and prior atomic checkpoints are preserved, interrupted checkpoint is not guaranteed")
        atomic_json(marker_path, marker)
        events("launcher_sigterm_after_timeout", stage=output.name,
               producer_pid=original["pid"], grace_seconds=grace_seconds)
        return marker
    if marker.get("action") in ("rank0_sigterm_sent", "rank0_already_exiting"):
        return marker
    if "rank0" not in marker:
        workers = validated_workers(original, output)
        marker.update(workers=workers, rank0=workers[0],
                      action="rank0_signal_prepared", updated_utc=now())
        atomic_json(marker_path, marker)
    rank0 = marker["rank0"]
    if not same_process(rank0):
        marker.update(action="rank0_already_exiting", updated_utc=now())
        atomic_json(marker_path, marker)
        return marker
    signal_verified(rank0, signal.SIGTERM)
    marker.update(action="rank0_sigterm_sent", signal_sent_utc=now(),
                  signal_sent_unix=time.time(), updated_utc=now())
    atomic_json(marker_path, marker)
    events("rank0_sigterm_sent", stage=output.name, rank0_pid=rank0["pid"],
           producer_pid=original["pid"])
    return marker


def watch(run_root, *, poll_seconds=5.0, grace_seconds=300.0):
    run_root = run_root.resolve()
    if poll_seconds <= 0 or grace_seconds < 60:
        raise ValueError("poll_seconds must be positive and graceful timeout at least60s")
    policy_path = run_root / "training_health_policy.json"
    if not policy_path.is_file():
        raise RuntimeError("An explicit run health policy is required before attaching")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd signaling is unavailable")
    original_policy_digest = digest(policy_path)
    lock = (run_root / "training_health_guard.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Another guardian already owns this run") from exc
    identity = process_snapshot(os.getpid())
    guard_status = run_root / "training_health_guard_status.json"
    events_path = run_root / "training_health_guard_events.jsonl"
    exiting = {"requested": False}
    def request_exit(signum, frame):
        exiting["requested"] = True
    signal.signal(signal.SIGTERM, request_exit)
    signal.signal(signal.SIGINT, request_exit)
    def event(kind, **fields):
        row = {"observed_utc": now(), "event": kind, **fields}
        with events_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    source = {"guardian_sha256": digest(__file__),
              "policy_module_sha256": digest(Path(__file__).with_name("training_health.py")),
              "policy_file_sha256": original_policy_digest,
              "policy_identity": policy_identity(load_policy(run_root/"training"/"c30"))}
    event("guardian_attached", pid=os.getpid(), run_root=str(run_root), **source)
    last_steps = {}
    prepared_processes = {}
    status = {}
    try:
        while not exiting["requested"]:
            if digest(policy_path) != original_policy_digest:
                raise RuntimeError("The armed policy changed; refusing an unrecorded policy switch")
            if (digest(__file__) != source["guardian_sha256"] or
                    digest(Path(__file__).with_name("training_health.py")) != source["policy_module_sha256"]):
                raise RuntimeError("Armed monitoring source changed; refusing mixed monitoring identities")
            stages = {}
            for stage in STAGES:
                output = run_root / "training" / stage
                manifest = read_json(output / "training_manifest.json")
                if not manifest:
                    stages[stage] = {"status": "pending"}
                    continue
                try:
                    existing_stop = read_json(output / STOP_NAME)
                    log_path = output/"training.jsonl"
                    if existing_stop is None and (not log_path.exists() or log_path.stat().st_size == 0):
                        stages[stage] = {"status": "starting", "observed_utc": now()}
                        continue
                    assessment = (existing_stop.get("assessment") if existing_stop is not None
                                  else assess_training(output))
                    if assessment is None:
                        raise RuntimeError("Configured stage is outside the armed policy scope")
                    if assessment.get("status") == "stop" or (output / STOP_NAME).exists():
                        marker = stop_stage(output, assessment, event, grace_seconds=grace_seconds)
                        stage_status = {"status": "health_stopped", "assessment": assessment,
                                        "stop_action": marker.get("action"), "observed_utc": now()}
                    else:
                        process = read_json(output / "process_status.json", {})
                        completion = read_json(output / "completion.json", {})
                        stage_status = {**assessment, "observed_utc": now()}
                        if process.get("status") == "running":
                            producer = validated_producer(output, process)
                            if producer is not None:
                                cache_key = (producer["pid"], producer["start_ticks"])
                                if cache_key not in prepared_processes:
                                    prepared_processes[cache_key] = validated_workers(producer, output)
                                    event("stage_identity_verified", stage=stage,
                                          producer_pid=producer["pid"],
                                          rank0_pid=prepared_processes[cache_key][0]["pid"],
                                          worker_count=8)
                                stage_status["ready_to_stop"] = True
                                stage_status["producer_pid"] = producer["pid"]
                                stage_status["rank0_pid"] = prepared_processes[cache_key][0]["pid"]
                        if process.get("status") == "completed" and completion.get("status") == "completed":
                            admission = admission_check(output)
                            stage_status.update(status="passed", admission=admission)
                            atomic_json(output / "training_health_admission.json", admission)
                    atomic_json(output / "training_health_status.json", stage_status)
                    stages[stage] = stage_status
                    step = assessment.get("last_step", assessment.get("observed_step", assessment.get("trigger_step")))
                    if step != last_steps.get(stage):
                        event("stage_health", stage=stage, status=stage_status["status"], step=step)
                        last_steps[stage] = step
                except (HealthGateError, OSError, ValueError, RuntimeError) as exc:
                    stage_status = {"status": "monitor_error", "error": str(exc), "observed_utc": now()}
                    if (output / STOP_NAME).exists():
                        stage_status["formal_admission_blocked"] = True
                    atomic_json(output / "training_health_status.json", stage_status)
                    stages[stage] = stage_status
                    event("stage_monitor_error", stage=stage, error=str(exc))
            pipeline = read_json(run_root / "pipeline_state.json", {})
            status = {"schema": "training_health_guard_status_v1", "status": "running",
                      "observed_utc": now(), "guardian_process": identity, "source": source,
                      "run_root": str(run_root), "poll_seconds": poll_seconds,
                      "graceful_stop_timeout_seconds": grace_seconds,
                      "pipeline_status": pipeline.get("status"), "active_step": pipeline.get("active_step"),
                      "stages": stages}
            atomic_json(guard_status, status)
            active = any(read_json(run_root/"training"/stage/"process_status.json", {}).get("status") == "running"
                         and (snap := process_snapshot(read_json(run_root/"training"/stage/"process_status.json", {}).get("pid", -1)))
                         and snap["state"] != "Z" for stage in STAGES)
            if pipeline.get("status") in ("failed", "measurements_complete_review_pending", "complete") and not active:
                status.update(status="health_stopped" if any((run_root/"training"/stage/STOP_NAME).exists() for stage in STAGES)
                              else "pipeline_failed" if pipeline.get("status") == "failed" else "pipeline_finished", observed_utc=now())
                atomic_json(guard_status, status)
                event("guardian_finished", status=status["status"], pipeline_status=pipeline.get("status"))
                return 2 if status["status"] == "health_stopped" else 1 if status["status"] == "pipeline_failed" else 0
            time.sleep(poll_seconds)
        status.update(status="guardian_detached", observed_utc=now())
        atomic_json(guard_status, status)
        event("guardian_detached", reason="signal", training_signalled=False)
        return 0
    except Exception as exc:
        status.update(schema="training_health_guard_status_v1", status="guardian_error",
                      observed_utc=now(), error=str(exc), guardian_process=identity, source=source)
        atomic_json(guard_status, status)
        event("guardian_error", error=str(exc))
        raise
    finally:
        lock.close()


def arm(run_root):
    """Record one immutable operational policy; never modify a training run."""
    run_root = run_root.resolve()
    expected = json.loads(json.dumps(DEFAULT_POLICY))
    identity = policy_identity(expected)
    source = Path(__file__).parent
    frozen = {name: digest(source / name) for name in
              ("train.py", "native_model.py", "training_data.py", "feature_cache.py")}
    manifest = read_json(run_root / "training/c30/training_manifest.json")
    if manifest:
        fields = {"train.py": "trainer_sha256", "native_model.py": "native_model_sha256",
                  "training_data.py": "training_data_sha256", "feature_cache.py": "feature_cache_source_sha256"}
        if any(frozen[name] != manifest.get(key) for name, key in fields.items()):
            raise RuntimeError("In-flight training source changed; refusing to arm over a different recipe")
    run_root.mkdir(parents=True, exist_ok=True)
    policy_path = run_root / "training_health_policy.json"
    required_path = run_root / "training_health_required.json"
    required = read_json(required_path)
    if required is not None:
        if required.get("schema") != "training_health_required_v1" or required.get("policy_sha256") != identity:
            raise RuntimeError("A different required-policy identity is already armed")
        if not policy_path.exists() or policy_identity(read_json(policy_path)) != identity:
            raise RuntimeError("Required policy is missing or changed; refusing an implicit repair")
        return required
    if policy_path.exists():
        if policy_identity(read_json(policy_path)) != identity:
            raise RuntimeError("A different health policy is already present")
    else:
        atomic_json(policy_path, expected)
    required = {"schema": "training_health_required_v1", "armed_utc": now(),
                "policy_sha256": identity, "training_recipe_unchanged": True,
                "frozen_training_source_sha256": frozen,
                "monitoring_source_sha256": {"training_health.py": digest(source/"training_health.py"),
                                             "training_health_guard.py": digest(__file__)}}
    atomic_json(required_path, required)
    return required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("arm")
    setup.add_argument("--run-root", type=Path, required=True)
    start = sub.add_parser("watch")
    start.add_argument("--run-root", type=Path, required=True)
    start.add_argument("--poll-seconds", type=float, default=5.0)
    start.add_argument("--grace-seconds", type=float, default=300.0)
    status = sub.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "arm":
        print(json.dumps(arm(args.run_root), ensure_ascii=False, indent=2))
        return
    if args.command == "status":
        print(json.dumps(read_json(args.run_root/"training_health_guard_status.json", {}), ensure_ascii=False, indent=2))
        return
    raise SystemExit(watch(args.run_root, poll_seconds=args.poll_seconds, grace_seconds=args.grace_seconds))


if __name__ == "__main__":
    main()

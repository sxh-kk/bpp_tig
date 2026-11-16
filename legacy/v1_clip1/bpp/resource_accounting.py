"""CPU-only accounting of recorded formal BPP jobs; never imports Torch/NVML.

GPU hours are unions of assigned-GPU process wall intervals, not utilization,
energy, exclusive reservations or the total cost of this project.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid

ROOT = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp"))
GPUS = tuple(range(8))


def read(path, default=None, diagnostics=None):
    reason = None
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            reason = "missing_file"
            break
        except json.JSONDecodeError:
            reason = "invalid_json"
            if attempt < 2:
                time.sleep(.02)
        except (OSError, UnicodeError):
            reason = "unreadable_file"
            break
    if diagnostics is not None:
        diagnostics.append({"source":str(path), "reason":reason})
    return default


def read_object(path, diagnostics, **context):
    errors = []
    value = read(path, diagnostics=errors)
    if not isinstance(value, dict):
        diagnostics.extend(dict(context, **issue) for issue in
                           (errors or [{"source":str(path), "reason":"non_object_json"}]))
        return None
    return value


def stage_started(entry):
    return bool(entry.get("attempts") or entry.get("adopted") or entry.get("status") in
                ("running", "complete", "adopted_status_only"))


def artifact_expected(state, name):
    # Pipeline initial_artifacts pre-registers these two merge outputs, unlike
    # data/shard manifests, which are registered only after successful creation.
    if name not in ("c30.features", "c45.features"):
        return True
    scale = name.split(".")[0]
    for entry in state.get("steps", {}).values():
        if entry.get("scale") != scale:
            continue
        if entry.get("kind") == "train" and stage_started(entry):
            return True
        if entry.get("kind") == "merge" and (entry.get("status") == "complete" or
                any(attempt.get("finished_utc") for attempt in entry.get("attempts", []))):
            return True
    return False


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc).timestamp() if result.tzinfo is None else result.timestamp()
    except ValueError:
        return None


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def option(command, flag):
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return command[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def assigned_gpus(kind, command):
    """Original pipeline commands use physical IDs in a fixed 0..7 namespace."""
    if kind == "train":
        return list(GPUS) if option(command, "--nproc-per-node") == "8" else []
    if kind not in ("eval", "features"):
        return []
    device = option(command, "--device")
    match = re.fullmatch(r"cuda:([0-7])", device or "")
    return [int(match[1])] if match else []


def matching_active(status):
    """Read /proc only: no signal, ptrace, subprocess, GPU query or model import."""
    pid = status.get("pid")
    if type(pid) is not int:
        return False
    try:
        folder = Path(f"/proc/{pid}")
        actual = [x.decode(errors="replace") for x in (folder / "cmdline").read_bytes().split(b"\0") if x]
        expected = status.get("command", [])
        tail = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        if not actual or tail[0] in ("Z", "X"):
            return False
        boot = next(int(x.split()[1]) for x in Path("/proc/stat").read_text().splitlines() if x.startswith("btime "))
        born = boot + int(tail[19]) / os.sysconf("SC_CLK_TCK")
        recorded = timestamp(status.get("started_utc"))
        if recorded is None or abs(born - recorded) > 30:
            return False
        if actual == expected:
            return True
        # The published eval entry narrows visibility and re-execs with cuda:0.
        device = option(expected, "--device")
        if not re.fullmatch(r"cuda:[0-7]", device or ""):
            return False
        narrowed = list(expected)
        if "--device" in narrowed:
            narrowed[narrowed.index("--device") + 1] = "cuda:0"
        else:
            narrowed = ["--device=cuda:0" if item.startswith("--device=") else item for item in narrowed]
        selected = device.split(":", 1)[1].encode()
        environment = dict(x.split(b"=", 1) for x in (folder / "environ").read_bytes().split(b"\0") if b"=" in x)
        return actual == narrowed and environment.get(b"CUDA_VISIBLE_DEVICES") == selected
    except (OSError, ValueError, IndexError, StopIteration):
        return False


def interval(status, snapshot, active_match=matching_active):
    start, end = timestamp(status.get("started_utc")), timestamp(status.get("finished_utc"))
    if start is None or start > snapshot:
        return None, "missing_or_future_start"
    active = end is None
    if active:
        if not active_match(status):
            return None, "unfinished_status_without_matching_live_pid_command_start"
        end = snapshot
    if end < start or end > snapshot + 1:
        return None, "invalid_end_timestamp"
    return (start, min(end, snapshot), active), None


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def seconds(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def stage_groups(state):
    """Only identical shared-Scratch outputs alias; feature stages stay distinct."""
    groups = {}
    for name, entry in state.get("steps", {}).items():
        output = str(Path(entry.get("output_dir", ".")).absolute())
        shared = entry.get("regime") == "scratch" and entry.get("kind") in ("train", "eval")
        key = (entry.get("kind"), "shared_scratch", output) if shared else (name,)
        group = groups.setdefault(key, {"name": name, "aliases": [], "entries": []})
        group["aliases"].append(name)
        group["entries"].append(entry)
    return list(groups.values())


def collect_statuses(group, diagnostics=None):
    diagnostics = [] if diagnostics is None else diagnostics
    paths, fallbacks = {}, []
    kind = group["entries"][0].get("kind")
    for entry in group["entries"]:
        output = Path(entry.get("output_dir", "."))
        if kind == "train":
            paths.update((str(p), None) for p in output.glob("**/*process_status*.json"))
            if stage_started(entry):
                paths.setdefault(str(output / "process_status.json"), None)
        for attempt in entry.get("attempts", []):
            folder = Path(attempt.get("directory", "/nonexistent"))
            commands = attempt.get("commands", [])
            workers = sorted(folder.glob("worker_*.json"))
            for p in workers:
                match = re.search(r"worker_(\d+)", p.name)
                index = int(match[1]) if match else -1
                command = commands[index] if 0 <= index < len(commands) else None
                paths[str(p)] = command
            if kind == "train":
                paths.update((str(p), None) for p in folder.glob("**/*process_status*" ) if p.is_file())
            else:
                for index, command in enumerate(commands):
                    paths.setdefault(str(folder / f"worker_{index:03d}.json"), command)
            if kind == "train" or not workers:
                for command in commands:
                    fallbacks.append(dict(started_utc=attempt.get("started_utc"),
                        finished_utc=attempt.get("finished_utc"), command=command,
                        status="attempt_envelope", source=str(folder), envelope=True))
    rows = []
    for path, fallback in paths.items():
        value = read_object(path, diagnostics, stage=group["name"], category="status")
        if value is None:
            continue
        value = dict(value, source=path)
        if not value.get("command") and fallback:
            value["command"] = fallback
        rows.append(value)
    # Prefer process records; preserve envelopes for older overwritten attempts.
    for fallback in fallbacks:
        start = timestamp(fallback.get("started_utc"))
        if not any(row.get("command") == fallback["command"] and start is not None
                   and timestamp(row.get("started_utc")) is not None
                   and abs(timestamp(row["started_utc"]) - start) < 30 for row in rows):
            if fallback.get("finished_utc"):
                rows.append(fallback)
    return rows


def account(state, snapshot=None, active_match=matching_active):
    snapshot = datetime.now(timezone.utc).timestamp() if snapshot is None else snapshot
    stages, events, excluded = [], [], []
    global_gpu = {gpu: [] for gpu in GPUS}
    for group in stage_groups(state):
        entry = group["entries"][0]
        kind = entry.get("kind")
        local_gpu = {gpu: [] for gpu in GPUS}
        walls = []
        rows = collect_statuses(group, excluded)
        for row in rows:
            value, reason = interval(row, snapshot, active_match)
            if value is None:
                excluded.append({"stage": group["name"], "source": row.get("source"), "reason": reason})
                continue
            start, end, active = value
            devices = assigned_gpus(kind, row.get("command", []))
            walls.append((start, end))
            for gpu in devices:
                local_gpu[gpu].append((start, end))
                global_gpu[gpu].append((start, end))
            events.append({"stage": group["name"], "kind": kind, "pid": row.get("pid"),
                "source": row.get("source"), "status": row.get("status"),
                "returncode": row.get("returncode", row.get("exit_code")),
                "started_utc": utc(start), "finished_or_snapshot_utc": utc(end),
                "active_pid_command_start_verified": active, "assigned_physical_gpus": devices,
                "wall_seconds": end-start, "attempt_envelope_fallback": bool(row.get("envelope"))})
        completion = read(Path(entry.get("output_dir", ".")) / "completion.json", {}) if kind == "train" else {}
        manifest = read(Path(entry.get("output_dir", ".")) / "training_manifest.json", {}) if kind == "train" else {}
        updates = completion.get("completed_updates")
        batch = manifest.get("global_batch")
        stages.append({"stage": group["name"], "aliases": group["aliases"], "kind": kind,
            "scale": "shared_target" if entry.get("regime") == "scratch" else entry.get("scale"),
            "status": entry.get("status"), "output_dir": entry.get("output_dir"),
            "recorded_process_wall_union_seconds": seconds(walls) if walls else None,
            "recorded_wall_span_seconds": max(x[1] for x in walls)-min(x[0] for x in walls) if walls else None,
            "assigned_gpu_wall_hours": sum(seconds(v) for v in local_gpu.values())/3600,
            "gpu_wall_seconds_by_physical_id": {str(k):seconds(v) for k,v in local_gpu.items()},
            "included_records": len(walls),
            "failed_closed_process_records": sum(row.get("returncode", row.get("exit_code")) not in (None, 0)
                for row in rows if timestamp(row.get("finished_utc")) is not None),
            "final_completed_updates": updates,
            "final_query_windows": updates*batch if updates is not None and batch is not None else None,
            "global_batch": batch, "optimizer_loop_wall_seconds": completion.get("wall_seconds"),
            "peak_cuda_allocated_bytes_rank0": completion.get("peak_memory_allocated_rank0")})
    gpu_rows = [{"physical_gpu": gpu, "assigned_wall_seconds": seconds(spans),
                 "assigned_gpu_wall_hours": seconds(spans)/3600,
                 "merged_intervals_utc": [[utc(a),utc(b)] for a,b in merge_intervals(spans)]}
                for gpu,spans in global_gpu.items()]
    earliest = min((timestamp(row["started_utc"]) for row in events), default=None)
    latest = max((timestamp(row["finished_or_snapshot_utc"]) for row in events), default=None)
    elapsed = latest - earliest if earliest is not None else None
    window = {"started_utc":utc(earliest) if earliest is not None else None,
              "latest_completed_or_verified_active_utc":utc(latest) if latest is not None else None,
              "seconds":elapsed, "hours":elapsed/3600 if elapsed is not None else None,
              "scope":"Earliest included formal process (including adopted C15 training) to latest completed or PID-verified active observation; includes CPU stages and gaps."}
    return {"snapshot_utc":utc(snapshot),"stages":stages,"interval_records":events,
            "formal_pipeline_elapsed_wall":window,
            "hypothetical_continuous_eight_gpu_reservation_hours":8*elapsed/3600 if elapsed is not None else None,
            "hypothetical_reservation_note":"Planning upper bound under the assumption that all eight GPUs are reserved exclusively throughout the elapsed window; separate from assigned-GPU interval totals. CPU preparation with zero assigned GPU time does not imply a free node allocation. No monetary cost is inferred.",
            "excluded_records":excluded,"physical_gpus":gpu_rows,
            "recorded_assigned_gpu_wall_hours":sum(x["assigned_gpu_wall_hours"] for x in gpu_rows)}


def storage(state, project_root=ROOT):
    """Stat only referenced artifact paths and shallow known training checkpoints."""
    project_root = Path(os.path.abspath(project_root))
    candidates, manifests, visited = set(), [], set()
    required, contexts, manifest_errors = {}, {}, []
    def walk(value):
        if isinstance(value, dict):
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
        elif isinstance(value, str) and value.startswith(str(project_root)+os.sep):
            candidates.add(os.path.abspath(value))
    queue = [(v, artifact_expected(state, k), "artifact:"+k)
             for k,v in state.get("artifacts",{}).items()
             if ".data" in k or ".features" in k or ".feature_shard_" in k]
    config = state.get("config",{})
    if config.get("external_checkpoint"): candidates.add(config["external_checkpoint"])
    for group in stage_groups(state):
        entry=group["entries"][0];output=Path(entry.get("output_dir","."))
        if entry.get("kind")=="train":
            for p in output.glob("*.pt"): candidates.add(str(p.absolute()))
            errors=[]
            metadata=read_object(output/"training_manifest.json",errors) or {}
            manifest_errors.extend(dict(issue, expectation="expected" if stage_started(entry) else "not_yet_expected") for issue in errors)
            for key in ("data_manifest","feature_manifest"):
                if metadata.get(key):queue.append((metadata[key], True, str(output/"training_manifest.json")))
        if entry.get("kind")=="download": queue.append((str(output/"source.json"), entry.get("status")=="complete", group["name"]))
        if entry.get("kind")=="eval":
            errors=[]
            meta=read_object(output/"evaluation_manifest.json",errors) or {}
            manifest_errors.extend(dict(issue, expectation="expected" if stage_started(entry) else "not_yet_expected") for issue in errors)
            for task in meta.get("tasks",[]):
                for seed in meta.get("seeds",[]):
                    lane=output/"human"/task/f"seed_{seed}"
                    candidates.update(str((lane/name).absolute()) for name in ("metrics.json","rollout.mp4"))
    while queue:
        raw, expected, source=queue.pop();path=Path(raw)
        key=os.path.abspath(path)
        required[key]=required.get(key,False) or expected
        contexts.setdefault(key,set()).add(source)
        if key in visited:continue
        visited.add(key)
        value=read_object(path,manifest_errors)
        if value is None:continue
        manifests.append(key);candidates.add(key);walk(value)
        for name in ("data_manifest","feature_manifest","manifest"):
            if isinstance(value.get(name),str) and value[name].endswith(".json"): queue.append((value[name], True, key))
        queue.extend((x, True, key) for x in value.get("shard_manifests",[]) if isinstance(x,str))
        if value.get("schema")=="bpp_tig_download_source_v1" or ("files" in value and "folders" in value):
            dataset=Path(config.get("dataset_root",project_root/"data/IG-10K-Dataset"))
            candidates.update(str(dataset/x["filename"]) for x in value.get("files",[]) if x.get("filename"))
    for issue in manifest_errors:
        key=os.path.abspath(issue["source"])
        issue.setdefault("expectation", "expected" if required.get(key,True) else "not_yet_expected")
        issue["referenced_by"]=sorted(contexts.get(key,[]))
        if issue["expectation"]=="expected":candidates.add(key)
    files,missing,seen_inode=[],[],set()
    for raw in sorted(candidates):
        path=Path(raw)
        if not path.is_absolute() or not path.is_relative_to(project_root):continue
        try:info=path.stat()
        except OSError:
            missing.append(raw);continue
        if not stat.S_ISREG(info.st_mode):continue
        inode=(info.st_dev,info.st_ino)
        if inode in seen_inode:continue
        seen_inode.add(inode)
        relative=path.relative_to(project_root)
        if relative.parts[0]=="data":category="referenced_raw_dataset"
        elif "bpp_frozen_cls" in relative.parts and path.suffix==".npy":category="frozen_cls_cache"
        elif "bpp_tig_training" in relative.parts and path.suffix==".npy":category="rgb_numeric_cache"
        elif path.suffix in (".pt",".ckpt"):category="known_checkpoints"
        elif "evaluation" in relative.parts:category="current_formal_evaluation_outputs"
        else:category="referenced_manifests_and_other_files"
        files.append({"path":raw,"category":category,"logical_bytes":info.st_size})
    categories=[]
    for key in sorted({x["category"] for x in files}):
        members=[x for x in files if x["category"]==key]
        categories.append({"category":key,"file_count":len(members),"logical_bytes":sum(x["logical_bytes"] for x in members)})
    usage=shutil.disk_usage(project_root)
    return {"categories":categories,"known_existing_logical_bytes":sum(x["logical_bytes"] for x in files),
            "known_file_count":len(files),"manifest_paths_read":manifests,
            "missing_references_count":len(missing),"missing_references_sample":missing[:30],
            "expected_manifest_issues":[x for x in manifest_errors if x["expectation"]=="expected"],
            "not_yet_expected":[x for x in manifest_errors if x["expectation"]=="not_yet_expected"],
            "deduplication":"absolute path then filesystem device/inode; symlinks followed",
            "shared_filesystem":{"path":str(project_root),"total_bytes":usage.total,"free_bytes":usage.free,
                "note":"Current shared filesystem capacity; not this project's quota or exclusive free allocation."},
            "files":files}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");temp.replace(path)
    finally:temp.unlink(missing_ok=True)


def write_csv(path,rows):
    path=Path(path)
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ["no_records"]
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def build_report(run_root,project_root=ROOT):
    run_root=Path(run_root);state=read(run_root/"pipeline_state.json")
    if not isinstance(state,dict):raise ValueError("A recorded pipeline_state.json is required")
    c15=state.get("config",{}).get("c15_output")
    if c15 and not any(x.get("output_dir")==c15 for x in state.get("steps",{}).values()):
        state["steps"]["train.c15.adopted"]={"kind":"train","regime":"c15","scale":"c15",
            "output_dir":c15,"attempts":[],"status":"adopted_status_only"}
    result=account(state)
    result.update(schema="bpp_tig_formal_resource_accounting_v1",pipeline_state=str(run_root/"pipeline_state.json"),
        pipeline_status=state.get("status"),active_step=state.get("active_step"),
        pipeline_snapshot_canonical_json_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),
        accounting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        measurement="Recorded assigned-GPU process wall-time proxy; not kernel utilization, energy, NVML-owned memory, exclusive/billable cost or a project total.",
        physical_gpu_mapping="This project's original CUDA_VISIBLE_DEVICES is fixed to 0,1,2,3,4,5,6,7. Torchrun8 maps to all eight; eval/features original --device cuda:N maps to physical N, even after the entry narrows visibility.",
        coverage_limits=["Initial C15/target CLS preparation, downloads/preparation before pipeline records, development pilots and unrecorded attempts are NOT included in GPU-time totals.",
          "Recorded worker finish timestamps may be stamped when the whole attempt exits; this is an allocation-wall proxy, not precise per-worker compute lifetime.",
          "Stages without usable timestamps and unfinished statuses without a matching live PID/command/start time are uncharged and explicitly listed.",
          "Storage covers referenced data/cache files, shallow known formal checkpoints and current planned evaluation outputs only; environments, simulator assets, pilots, unreferenced caches/checkpoints, archived outputs and temporary files are not fully inventoried.",
          "Per-stage GPU hours may overlap between stages; the headline total unions all intervals separately for each physical GPU.",
          "Peak CUDA allocated memory is completion.json rank0's tensor allocator peak, not total VRAM usage across processes.",
          "The hypothetical eight-GPU reservation window also excludes pre-formal development/downloads/initial CLS work; CPU-only stages do not imply a free node allocation.",
          "Referenced files are stat-ed during report generation; live checkpoints or outputs may grow, so storage is not an atomic filesystem snapshot."],
        storage=storage(state,project_root))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=ROOT/"results/pipeline/bpp_human")
    parser.add_argument("--output",type=Path,default=ROOT/"results/review/formal_resource_accounting.json")
    args=parser.parse_args(argv);result=build_report(args.run_root)
    files=result["storage"].pop("files")
    write_json(args.output,result)
    totals = [{"snapshot_utc":result["snapshot_utc"],
        "recorded_assigned_gpu_wall_hours":result["recorded_assigned_gpu_wall_hours"],
        "formal_pipeline_elapsed_wall_seconds":result["formal_pipeline_elapsed_wall"]["seconds"],
        "hypothetical_continuous_eight_gpu_reservation_hours":result["hypothetical_continuous_eight_gpu_reservation_hours"],
        "known_existing_logical_bytes":result["storage"]["known_existing_logical_bytes"],
        "current_shared_filesystem_free_bytes":result["storage"]["shared_filesystem"]["free_bytes"]}]
    for suffix,rows in (("totals",totals),("stages",result["stages"]),("gpus",result["physical_gpus"]),
                        ("storage",result["storage"]["categories"]),("known_files",files)):
        write_csv(args.output.with_name(args.output.stem+"_"+suffix+".csv"),rows)
    print(json.dumps({"report":str(args.output),"recorded_assigned_gpu_wall_hours":result["recorded_assigned_gpu_wall_hours"],
                      "known_existing_logical_bytes":result["storage"]["known_existing_logical_bytes"],
                      "coverage":"partial recorded formal artifacts; not a project total"}))
    return 0

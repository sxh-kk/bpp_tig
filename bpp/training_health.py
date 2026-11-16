"""Read-only, standard-library training-health gates for the primary v2 run.

The policy concerns pre-clipping rank-zero gradient diagnostics and training loss,
never evaluation success rates. Every historical window is examined: recovery in
later rows cannot erase an earlier stop decision. This module never writes files,
loads a checkpoint, imports Torch, signals a process, or changes the recipe.

``assess_training`` accepts a snapshot of an actively appended JSONL file and
ignores only its final record without a terminating newline. ``admission_check``
is for a completed final checkpoint's stage, requires the complete log and clean
original completion/process records, and rechecks the sticky stop marker before
returning. Its caller remains responsible for checkpoint metadata validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


POLICY_SCHEMA = "training_health_policy_v1"
SCOPED_STAGES = ("c30", "c30_finetune", "c45", "c45_finetune")
_FIXED_WINDOW_FIELDS = {
    "window_points": 10, "min_window_span_updates": 1800,
    "required_high_gradient_points": 8,
}
_FIXED_RULE_FIELDS = {
    "sustained_extreme_gradient": {"grad_norm_threshold": 100.0},
    "sustained_high_gradient_loss_stagnation": {
        "grad_norm_threshold": 10.0, "min_loss_median_ratio": 0.98,
    },
}
DEFAULT_POLICY = {
    "schema": POLICY_SCHEMA,
    "scope": list(SCOPED_STAGES),
    **_FIXED_WINDOW_FIELDS,
    "rules": {name: dict(fields) for name, fields in _FIXED_RULE_FIELDS.items()},
}


class HealthGateError(RuntimeError):
    """The available artifacts cannot safely admit the requested stage.

    ``decision`` contains the complete assessment when a historical health rule
    caused the rejection; malformed/missing evidence has no fabricated decision.
    """

    def __init__(self, message: str, *, decision: dict[str, Any] | None = None):
        super().__init__(message)
        self.decision = decision


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise HealthGateError(f"Cannot encode canonical health evidence: {error}") from error


def policy_identity(policy: Mapping[str, Any]) -> str:
    """SHA256 of sorted compact UTF-8 JSON (including any policy metadata)."""
    _validate_policy(policy)
    return hashlib.sha256(_canonical_bytes(dict(policy))).hexdigest()


def _validate_policy(policy: Mapping[str, Any]) -> None:
    if not isinstance(policy, dict) or policy.get("schema") != POLICY_SCHEMA:
        raise HealthGateError(f"Expected policy schema {POLICY_SCHEMA}")
    scope = policy.get("scope")
    if (not isinstance(scope, list) or len(scope) != len(SCOPED_STAGES)
            or any(not isinstance(stage, str) for stage in scope)
            or set(scope) != set(SCOPED_STAGES)):
        raise HealthGateError("Health policy scope must be exactly C30/C45 and their fine-tuning stages")
    for field, expected in _FIXED_WINDOW_FIELDS.items():
        if type(policy.get(field)) is not int or policy[field] != expected:
            raise HealthGateError(f"{POLICY_SCHEMA} requires {field}={expected}")
    rules = policy.get("rules")
    if not isinstance(rules, dict) or set(rules) != set(_FIXED_RULE_FIELDS):
        raise HealthGateError("Health policy requires both specified gradient rules")
    for name, expected in _FIXED_RULE_FIELDS.items():
        rule = rules[name]
        if not isinstance(rule, dict):
            raise HealthGateError(f"Health policy rule is not an object: {name}")
        for field, value in expected.items():
            number = rule.get(field)
            # Comparing to finite fixed constants also rejects infinity, NaN,
            # booleans and arbitrarily large JSON integers without float overflow.
            if type(number) not in (int, float) or number != value:
                raise HealthGateError(f"{POLICY_SCHEMA} requires {name}.{field}={value}")
    _canonical_bytes(policy)


def _context(output: str | Path) -> tuple[Path, str, Path] | None:
    try:
        directory = Path(output).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HealthGateError(f"Cannot resolve training output: {error}") from error
    if directory.parent.name != "training" or directory.name not in SCOPED_STAGES:
        return None
    return directory, directory.name, directory.parent.parent / "training_health_policy.json"


def _read_bytes(path: Path, *, require_stable: bool = False) -> bytes:
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            # Bound the snapshot to the size observed when the descriptor opened.
            # Appends after that point belong to the guardian's next assessment.
            data = stream.read(before.st_size)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except (OSError, ValueError) as error:
        raise HealthGateError(f"Cannot reliably read {path}: {error}") from error
    if (len(data) != before.st_size or after.st_size < before.st_size
            or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
            or (after.st_size == before.st_size and after.st_mtime_ns != before.st_mtime_ns)
            or (require_stable and ((after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns)
                                    or (current.st_size, current.st_mtime_ns) != (before.st_size, before.st_mtime_ns)))):
        raise HealthGateError(f"Artifact was replaced/truncated/rewritten while reading: {path}")
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _decode_object(data: bytes, source: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise HealthGateError(f"Invalid JSON at {source}: {error}") from error
    if not isinstance(value, dict):
        raise HealthGateError(f"JSON record is not an object: {source}")
    return value


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    data = _read_bytes(path, require_stable=True)
    return _decode_object(data, str(path)), data


def load_policy(output: str | Path) -> dict[str, Any] | None:
    """Load runroot/training_health_policy.json for a scoped training output.

    Outputs outside ``training/{c30,c30_finetune,c45,c45_finetune}``, or a run
    without a policy file or required-policy sentinel, return None. A present
    invalid policy raises instead of silently disabling protection. Once
    runroot/training_health_required.json exists, the policy must remain present
    and match its pinned canonical SHA256. The v1 schema fixes the thresholds.
    """
    context = _context(output)
    if context is None:
        return None
    _, _, path = context
    required_path = path.with_name("training_health_required.json")
    required_exists = required_path.exists() or required_path.is_symlink()
    if not path.exists() and not path.is_symlink():
        if required_exists:
            raise HealthGateError(f"Required training-health policy is missing: {path}")
        return None
    policy, _ = _read_object(path)
    _validate_policy(policy)
    if required_exists:
        required, _ = _read_object(required_path)
        if required.get("schema") != "training_health_required_v1":
            raise HealthGateError(f"Invalid required training-health sentinel: {required_path}")
        if required.get("policy_sha256") != policy_identity(policy):
            raise HealthGateError(f"Training-health policy differs from required sentinel: {required_path}")
    return policy


def require_no_health_stop(output: str | Path) -> None:
    """Reject any sticky scoped training_health_stop.json, including a malformed marker.

    This light check is also suitable for latest/interrupted checkpoint recovery;
    it imposes no final-completion requirement and does not read model weights.
    """
    context = _context(output)
    if context is None:
        return
    path = context[0] / "training_health_stop.json"
    if path.exists() or path.is_symlink():
        raise HealthGateError(f"Sticky training-health stop exists; stage cannot be admitted or resumed: {path}")


def _finite_nonnegative(value: Any, field: str, source: str) -> None:
    if type(value) not in (int, float):
        raise HealthGateError(f"Expected numeric {field} at {source}")
    try:
        valid = math.isfinite(value) and value >= 0
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise HealthGateError(f"Expected finite nonnegative {field} at {source}")


def _validate_rows(rows: list[dict[str, Any]], source: str) -> None:
    previous_step = 0
    budget = None
    for index, row in enumerate(rows, 1):
        location = f"{source}, record {index}"
        if not isinstance(row, dict):
            raise HealthGateError(f"Training row is not an object: {location}")
        step = row.get("step")
        if type(step) is not int or step <= previous_step:
            raise HealthGateError(f"Training steps must be positive and strictly increasing: {location}, step={step!r}")
        previous_step = step
        for field in ("loss", "grad_norm_rank0"):
            _finite_nonnegative(row.get(field), field, location)
        modules = row.get("preclip_grad_norm_by_module_rank0")
        if not isinstance(modules, dict) or not modules:
            raise HealthGateError(f"Missing pre-clipping module gradient diagnostics: {location}")
        for name, norm in modules.items():
            if not isinstance(name, str) or not name:
                raise HealthGateError(f"Invalid gradient module name: {location}")
            _finite_nonnegative(norm, f"module gradient {name}", location)
        if "max_updates" in row:
            maximum = row["max_updates"]
            if type(maximum) is not int or maximum < step or (budget is not None and maximum != budget):
                raise HealthGateError(f"Invalid/inconsistent max_updates: {location}")
            budget = maximum
    if not rows:
        raise HealthGateError(f"No complete training diagnostic rows: {source}")


def _training_snapshot(output: str | Path) -> tuple[list[dict[str, Any]], bytes, int]:
    path = Path(output) / "training.jsonl"
    data = _read_bytes(path)
    lines = data.split(b"\n")
    # Only a record terminated by newline is committed by the training logger.
    # Never skip an invalid interior line or a malformed newline-terminated tail.
    tail_bytes = len(lines[-1])
    rows = [_decode_object(line, f"{path}:{index}") for index, line in enumerate(lines[:-1], 1)]
    _validate_rows(rows, str(path))
    return rows, data, tail_bytes


def read_training_rows(output: str | Path) -> list[dict[str, Any]]:
    """Read training.jsonl, ignoring only a final line without its newline.

    Missing logs, zero complete rows, invalid committed JSON, nonfinite/missing
    diagnostics, or non-increasing steps raise HealthGateError. Files stay intact.
    """
    return _training_snapshot(output)[0]


def _median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    # All values are nonnegative: halving before adding avoids finite-input
    # overflow in the ordinary (lower + upper) / 2 median formula.
    return ordered[middle] if len(ordered) % 2 else ordered[middle-1] / 2 + ordered[middle] / 2


def _window_summary(rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    norms = [row["grad_norm_rank0"] for row in rows]
    span = rows[-1]["step"] - rows[0]["step"]
    rules = policy["rules"]
    return {
        "points": len(rows), "start_step": rows[0]["step"], "end_step": rows[-1]["step"],
        "span_updates": span,
        "eligible": len(rows) == policy["window_points"] and span >= policy["min_window_span_updates"],
        "loss_median": _median(row["loss"] for row in rows),
        "grad_norm_min": min(norms), "grad_norm_median": _median(norms),
        "grad_norm_max": max(norms),
        "extreme_gradient_points": sum(value >= rules["sustained_extreme_gradient"]["grad_norm_threshold"] for value in norms),
        "high_gradient_points": sum(value >= rules["sustained_high_gradient_loss_stagnation"]["grad_norm_threshold"] for value in norms),
    }


def assess_rows(rows: Iterable[dict[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Assess all ordered logged windows, retaining the earliest stop forever.

    A stop needs 8/10 preclip norms >=100 across >=1800 updates, OR 8/10 norms
    >=10 and a loss median >=0.98 times the preceding 10-point median. The
    latter needs two adjacent windows each spanning >=1800 updates. One spike
    can produce a warning but cannot produce a stop. Evidence retains raw rows,
    including the per-module norms, without modifying the caller's data.
    """
    _validate_policy(policy)
    try:
        ordered = list(rows)
    except (TypeError, ValueError) as error:
        raise HealthGateError(f"Cannot read diagnostic rows: {error}") from error
    _validate_rows(ordered, "training rows")
    count = policy["window_points"]
    required = policy["required_high_gradient_points"]
    ratio = policy["rules"]["sustained_high_gradient_loss_stagnation"]["min_loss_median_ratio"]
    first_stop = None
    examined = 0
    eligible_extreme = eligible_comparison = 0
    for end in range(count, len(ordered) + 1):
        current_rows = ordered[end-count:end]
        current = _window_summary(current_rows, policy)
        previous_rows = ordered[end-2*count:end-count] if end >= 2*count else []
        previous = _window_summary(previous_rows, policy)
        examined += 1
        eligible_extreme += int(current["eligible"])
        comparison = bool(current["eligible"] and previous is not None and previous["eligible"])
        eligible_comparison += int(comparison)
        rule = None
        if current["eligible"] and current["extreme_gradient_points"] >= required:
            rule = "sustained_extreme_gradient"
        elif (comparison and current["high_gradient_points"] >= required
              and current["loss_median"] >= ratio * previous["loss_median"]):
            rule = "sustained_high_gradient_loss_stagnation"
        if rule is not None and first_stop is None:
            first_stop = {
                "status": "stop", "rule": rule, "trigger_step": current_rows[-1]["step"],
                "reason": "historical_persistent_gradient_anomaly",
                "window_summaries": {"current": current, "previous": previous if rule.endswith("stagnation") else None},
                "evidence_rows": [dict(row) for row in (previous_rows + current_rows if rule.endswith("stagnation") else current_rows)],
            }
    latest_rows = ordered[-count:]
    preceding_rows = ordered[-2*count:-count] if len(ordered) >= 2*count else []
    latest = _window_summary(latest_rows, policy)
    preceding = _window_summary(preceding_rows, policy)
    elevated = latest["high_gradient_points"] > 0
    result = first_stop or {
        "status": "warning" if elevated else "healthy", "rule": None, "trigger_step": None,
        "reason": "elevated_gradient_without_persistent_stop" if elevated else "no_persistent_gradient_anomaly",
        "window_summaries": {"current": latest, "previous": preceding},
        "evidence_rows": [dict(row) for row in latest_rows],
    }
    result.update({
        "schema": "training_health_decision_v1", "row_count": len(ordered),
        "first_step": ordered[0]["step"], "last_step": ordered[-1]["step"],
        "historical_windows_examined": examined,
        "eligible_extreme_gradient_windows": eligible_extreme,
        "eligible_loss_comparison_windows": eligible_comparison,
        "latest_window_summary": latest,
    })
    return result


def _manifest(output: Path, stage: str) -> tuple[dict[str, Any], bytes]:
    path = output / "training_manifest.json"
    manifest, data = _read_object(path)
    experiment = manifest.get("experiment_identity")
    unsigned = {key: value for key, value in manifest.items() if key != "experiment_identity"}
    # Match train.py exactly: its training identity uses the default JSON spaces.
    try:
        expected = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError, OverflowError) as error:
        raise HealthGateError(f"Invalid training identity manifest: {path}: {error}") from error
    if not isinstance(experiment, str) or experiment != expected:
        raise HealthGateError(f"Training manifest identity is invalid: {path}")
    regime = "finetune" if stage.endswith("_finetune") else stage
    scale = stage.split("_", 1)[0]
    if manifest.get("regime") != regime or manifest.get("pretraining_scale") != scale:
        raise HealthGateError(f"Training manifest stage lineage differs from output path: {path}")
    maximum = manifest.get("max_updates")
    if type(maximum) is not int or maximum <= 0:
        raise HealthGateError(f"Training manifest max_updates is not positive: {path}")
    return manifest, data


def _assess_training(output: Path, stage: str, policy_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    rows, data, tail_bytes = _training_snapshot(output)
    manifest, manifest_bytes = _manifest(output, stage)
    maximum = manifest["max_updates"]
    if any(row.get("max_updates") != maximum for row in rows):
        raise HealthGateError(f"Training log update budget differs from manifest: {output}")
    if any(row.get("pilot") is not False for row in rows):
        raise HealthGateError(f"Scoped training health requires original non-pilot logs: {output}")
    result = assess_rows(rows, policy)
    policy_sha = policy_identity(policy)
    log_sha = hashlib.sha256(data).hexdigest()
    result.update({
        "stage": stage, "output_dir": str(output), "source": str(output / "training.jsonl"),
        "policy_schema": policy["schema"], "policy_identity": policy_sha,
        "policy_sha256": policy_sha, "policy_path": str(policy_path),
        "log_sha256": log_sha, "source_sha256": log_sha, "log_bytes": len(data),
        "ignored_incomplete_tail": bool(tail_bytes), "ignored_incomplete_tail_bytes": tail_bytes,
        "experiment_identity": manifest["experiment_identity"],
        "training_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    })
    return result


def assess_training(output: str | Path) -> dict[str, Any] | None:
    """Return full-log health evidence for a scoped policy, or None if disabled.

    Active training need not have completed; original logs and manifest are only
    read. A stop decision remains a stop even if later training appears healthy.
    A training_health_stop.json marker is enforced separately by admission/recovery checks.
    """
    policy = load_policy(output)
    if policy is None:
        return None
    directory, stage, policy_path = _context(output)
    return _assess_training(directory, stage, policy_path, policy)


def _option(command: Any, option: str) -> str | None:
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise HealthGateError("Original training process command must be a string list")
    values = []
    for index, item in enumerate(command):
        if item == option:
            if index + 1 >= len(command) or command[index + 1].startswith("--"):
                raise HealthGateError(f"Missing value in original process command: {option}")
            values.append(command[index + 1])
        elif item.startswith(option + "="):
            values.append(item.split("=", 1)[1])
    if len(values) > 1:
        raise HealthGateError(f"Repeated option in original process command: {option}")
    return values[0] if values else None


def _logging_interval(process: dict[str, Any], manifest: dict[str, Any]) -> int:
    value = _option(process.get("command"), "--log-every")
    if value is None:
        value = manifest.get("log_every", 200)
    try:
        if type(value) is bool or (isinstance(value, float) and not value.is_integer()):
            raise ValueError("not an integer")
        interval = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HealthGateError(f"Invalid original logging interval: {value!r}") from error
    if interval <= 0:
        raise HealthGateError("Original logging interval must be positive")
    if "log_every" in manifest and manifest["log_every"] != interval:
        raise HealthGateError("Process and manifest logging intervals conflict")
    return interval


def admission_check(output: str | Path) -> dict[str, Any] | None:
    """Admit a completed scoped stage using fresh, bound health evidence.

    Raises HealthGateError on a sticky marker, any historical stop, incomplete or
    unreliable logs, missing scheduled points, or abnormal/incomplete training.
    Outside this policy's scope it returns None. Use require_no_health_stop for
    interrupted/latest checkpoints rather than imposing this final-stage gate.
    """
    require_no_health_stop(output)
    policy = load_policy(output)
    if policy is None:
        return None
    directory, stage, policy_path = _context(output)
    result = _assess_training(directory, stage, policy_path, policy)
    if result["status"] == "stop":
        raise HealthGateError(f"Historical training-health stop: {stage}, {result['rule']} at step {result['trigger_step']}", decision=result)
    if result["ignored_incomplete_tail"]:
        raise HealthGateError(f"Final-stage log has an uncommitted trailing line: {directory}")
    manifest, manifest_bytes = _manifest(directory, stage)
    if hashlib.sha256(manifest_bytes).hexdigest() != result["training_manifest_sha256"]:
        raise HealthGateError("Training manifest changed during health admission")
    completion, completion_bytes = _read_object(directory / "completion.json")
    process, process_bytes = _read_object(directory / "process_status.json")
    maximum = manifest["max_updates"]
    experiment = manifest["experiment_identity"]
    if (completion.get("status") != "completed" or completion.get("pilot") is not False
            or completion.get("completed_updates") != maximum or completion.get("target_updates") != maximum
            or completion.get("experiment_identity") != experiment or manifest.get("pilot") is not False):
        raise HealthGateError(f"Original completion is not a complete formal stage: {directory}")
    code = process.get("returncode", process.get("exit_code"))
    if (process.get("status") != "completed" or type(code) is not int or code != 0
            or ("exit_code" in process and (type(process["exit_code"]) is not int or process["exit_code"] != 0))
            or process.get("experiment_identity") not in (None, experiment)):
        raise HealthGateError(f"Original process did not complete with return code zero: {directory}")
    command = process.get("command")
    command_output = _option(command, "--output-dir")
    command_regime = _option(command, "--regime")
    if command_output is None or Path(command_output).resolve() != directory or command_regime != manifest["regime"]:
        raise HealthGateError("Original process command belongs to another output or regime")
    interval = _logging_interval(process, manifest)
    rows = read_training_rows(directory)
    if rows[-1]["step"] != maximum:
        raise HealthGateError(f"Final training log step does not equal max_updates={maximum}: {directory}")
    expected_steps = {1, maximum, *range(interval, maximum + 1, interval)}
    actual_steps = {row["step"] for row in rows}
    missing = sorted(expected_steps - actual_steps)
    if missing:
        raise HealthGateError(f"Training log is missing required scheduled steps: {missing}")
    # All bound artifacts must still be the assessed originals. No guardian
    # heartbeat or previously generated health-status file is trusted here.
    originals = (
        (directory / "training.jsonl", result["log_sha256"]),
        (directory / "training_manifest.json", result["training_manifest_sha256"]),
        (directory / "completion.json", hashlib.sha256(completion_bytes).hexdigest()),
        (directory / "process_status.json", hashlib.sha256(process_bytes).hexdigest()),
    )
    for path, expected_sha in originals:
        if hashlib.sha256(_read_bytes(path, require_stable=True)).hexdigest() != expected_sha:
            raise HealthGateError(f"Original artifact changed during health admission: {path}")
    current_policy = load_policy(directory)
    if current_policy is None or policy_identity(current_policy) != result["policy_sha256"]:
        raise HealthGateError("Training-health policy changed/disappeared during admission")
    result.update({
        "admitted": True, "admission_schema": "training_health_admission_v1",
        "completed_updates": maximum, "log_every": interval,
        "required_log_points": len(expected_steps), "observed_log_points": len(rows),
        "completion_sha256": hashlib.sha256(completion_bytes).hexdigest(),
        "process_status_sha256": hashlib.sha256(process_bytes).hexdigest(),
        "completion_status": completion["status"], "process_status": process["status"], "returncode": code,
    })
    result["evidence_identity"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    require_no_health_stop(directory)
    return result


def main(argv: list[str] | None = None) -> int:
    """Read-only CLI: check a final stage, or assess a running stage's logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("check", "Admit a final stage using original full-log evidence"),
                            ("assess", "Assess all available training rows without requiring completion")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output", required=True, type=Path, help="Stage output under runroot/training")
    args = parser.parse_args(argv)
    try:
        result = admission_check(args.output) if args.command == "check" else assess_training(args.output)
    except HealthGateError as error:
        failure = {"status": "error", "error": str(error), "output_dir": str(args.output)}
        if error.decision is not None:
            failure["decision"] = error.decision
        print(json.dumps(failure, ensure_ascii=False, allow_nan=False))
        return 2
    if result is None:
        result = {"status": "not_scoped", "output_dir": str(args.output)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "stop" else 0


if __name__ == "__main__":
    raise SystemExit(main())

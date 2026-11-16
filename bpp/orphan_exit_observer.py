"""Recover the actual exit status of an existing orphan from supervisor's new log.

This CPU-only observer never launches, signals, resumes, or imports training.
A completion artifact alone is not evidence of a successful process exit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

DEFAULT_SUPERVISOR_LOG = Path("/mlplatform/supervisord/supervisord.log")
TRAIN_MODULE = "examples.baselines.bpp.train_bpp_imitator"
STATUS_IDENTITY = ("pid", "command", "started_utc")
MAX_MISSING_SECONDS = 60.0
REAP = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3,6})"
    r"\s+INFO\s+reaped unknown pid (?P<pid>\d+) \((?P<outcome>[^()\r\n]+)\)\r?\n$"
)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def source_digest():
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def proc_stat(pid):
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {
        "pid": pid, "state": fields[0], "ppid": int(fields[1]),
        "pgrp": int(fields[2]), "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def process_snapshot(pid):
    """Bind the complete command to a stable /proc start time, never to PID alone."""
    try:
        first = proc_stat(pid)
        command = [
            item.decode("utf-8", errors="surrogateescape")
            for item in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        second = proc_stat(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None
    if first["start_ticks"] != second["start_ticks"]:
        raise RuntimeError(f"PID {pid} was reused during identity inspection")
    return dict(second, command=command)


def process_state(expected):
    current = process_snapshot(expected["pid"])
    if current is None:
        return "missing"
    if current["start_ticks"] != expected["start_ticks"]:
        raise RuntimeError("Producer PID was reused; refusing exit attribution")
    if current["state"] in ("Z", "X", "x"):
        return "missing"
    if current["command"] != expected["command"]:
        raise RuntimeError("Producer's complete command changed; refusing exit attribution")
    return "alive"


def flag_value(arguments, flag):
    if arguments.count(flag) != 1:
        raise RuntimeError(f"Missing or repeated producer argument: {flag}")
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise RuntimeError(f"Missing producer argument value: {flag}")
    return arguments[index + 1]


def validate_producer(output, recorded):
    if recorded.get("status") != "running":
        raise RuntimeError("process_status.json must still describe running training")
    pid = recorded.get("pid")
    command = recorded.get("command")
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("Invalid producer PID")
    if not isinstance(command, list) or not command or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise RuntimeError("A complete producer command is required")
    producer = process_snapshot(pid)
    if producer is None or producer["state"] in ("Z", "X", "x"):
        raise RuntimeError("Cannot arm after the producer has exited")
    if producer["command"] != command:
        raise RuntimeError("Producer command differs from process_status.json")
    if producer["ppid"] != 1:
        raise RuntimeError("Producer is not an existing orphan owned by PID 1")
    if producer["pgrp"] != pid or producer["session"] != pid:
        raise RuntimeError("Producer is not the recorded isolated torchrun session")
    if not any(
        command[index:index + 2] == ["-m", "torch.distributed.run"]
        for index in range(len(command) - 1)
    ):
        raise RuntimeError("Producer is not the recorded torchrun launcher")
    modules = [
        index for index in range(len(command) - 1)
        if command[index:index + 2] == ["-m", TRAIN_MODULE]
    ]
    if len(modules) != 1:
        raise RuntimeError("Producer training module is not uniquely identified")
    arguments = command[modules[0] + 2:]
    if Path(flag_value(arguments, "--output-dir")).resolve() != output:
        raise RuntimeError("Producer --output-dir is outside the selected stage")
    if flag_value(arguments, "--prompt-source") != "human":
        raise RuntimeError("Producer does not use the accepted human prompt")
    if flag_value(arguments, "--action-normalization") != "train_minmax_joint14_gripper_identity_v2":
        raise RuntimeError("Producer does not use the accepted v2 recipe")
    started = datetime.fromisoformat(recorded["started_utc"])
    if started.tzinfo is None:
        raise RuntimeError("Producer's recorded start time must be timezone aware")
    boot = None
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("btime "):
            boot = int(line.split()[1])
            break
    if boot is None:
        raise RuntimeError("Cannot bind the producer start ticks to its recorded start time")
    actual_start = boot + producer["start_ticks"] / os.sysconf("SC_CLK_TCK")
    if abs(actual_start - started.timestamp()) >= 30:
        raise RuntimeError("Producer start ticks do not match its recorded start time")
    return producer


def current_status(path, original):
    latest = read_json(path)
    if latest.get("status") != "running" or any(
        latest.get(key) != original.get(key) for key in STATUS_IDENTITY
    ):
        raise RuntimeError("Original process_status identity or running state changed")
    return latest


class LogCursor:
    """Read complete new lines, draining the old descriptor before inode rotation."""

    def __init__(self, path, *, from_eof):
        self.path = path
        self.handle = path.open("rb", buffering=0)
        self.identity = self.stat_identity(os.fstat(self.handle.fileno()))
        self.pending = b""
        self.pending_offset = 0
        self.skip_partial = False
        if from_eof:
            end = self.handle.seek(0, os.SEEK_END)
            if end:
                self.handle.seek(end - 1)
                self.skip_partial = self.handle.read(1) != b"\n"
                self.handle.seek(end)
        self.offset = self.handle.tell()

    @staticmethod
    def stat_identity(stat):
        return {"device": stat.st_dev, "inode": stat.st_ino}

    def close(self):
        self.handle.close()

    def lines(self):
        # Bound each drain so heartbeat publication remains responsive during logging.
        for _ in range(10000):
            start = self.handle.tell()
            fragment = self.handle.readline()
            self.offset = self.handle.tell()
            if not fragment:
                return
            if self.skip_partial:
                if fragment.endswith(b"\n"):
                    self.skip_partial = False
                continue
            if not self.pending:
                self.pending_offset = start
            self.pending += fragment
            if len(self.pending) > 1024 * 1024:
                raise RuntimeError("Supervisor log line exceeds the safe observation limit")
            if not self.pending.endswith(b"\n"):
                return
            complete, offset = self.pending, self.pending_offset
            self.pending = b""
            yield complete, offset, self.offset, dict(self.identity)

    def rotate_if_drained(self):
        old = os.fstat(self.handle.fileno())
        if old.st_size < self.offset:
            raise RuntimeError("Supervisor log was truncated; an exit record may have been lost")
        if old.st_size > self.offset:
            return
        try:
            destination = self.path.stat()
        except FileNotFoundError:
            return
        if self.stat_identity(destination) == self.identity:
            return
        if self.pending or self.skip_partial:
            raise RuntimeError("Supervisor log rotated with an incomplete observed line")
        replacement = self.path.open("rb", buffering=0)
        identity = self.stat_identity(os.fstat(replacement.fileno()))
        if identity == self.identity:
            replacement.close()
            return
        self.handle.close()
        self.handle = replacement
        self.identity = identity
        self.offset = 0


def parse_reap(raw, pid, armed):
    marker = f"reaped unknown pid {pid} ".encode()
    if marker not in raw:
        return None
    line = raw.decode("utf-8", errors="strict")
    match = REAP.fullmatch(line)
    if match is None or int(match["pid"]) != pid:
        raise RuntimeError("Target producer has an unparseable supervisor reap line")
    finished = datetime.strptime(match["timestamp"], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
    # Supervisord timestamps have millisecond precision; never import older history.
    boundary = armed.replace(microsecond=(armed.microsecond // 1000) * 1000)
    if finished < boundary:
        raise RuntimeError("Target reap timestamp predates observer arming")
    outcome = match["outcome"]
    if re.fullmatch(r"exit status \d+", outcome):
        returncode = int(outcome.removeprefix("exit status "))
        if not 0 <= returncode <= 255:
            raise RuntimeError("Supervisor exit status is outside the POSIX exit range")
    elif re.fullmatch(r"terminated by SIG[A-Z0-9]+", outcome):
        name = outcome.removeprefix("terminated by ")
        try:
            returncode = -int(signal.Signals[name])
        except KeyError as exc:
            raise RuntimeError(f"Unknown supervisor signal name: {name}") from exc
    else:
        raise RuntimeError("Target supervisor reap outcome is not a recorded exit or signal")
    return returncode, finished.isoformat(), line


def observe(output, supervisor_log):
    output = output.resolve(strict=True)
    if not output.is_dir():
        raise RuntimeError("--output must be an existing training stage directory")
    supervisor_log = supervisor_log.resolve()
    status_path = output / "exit_observer_status.json"
    process_path = output / "process_status.json"
    observation_path = output / "exit_observation.json"
    with (output / "exit_observer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            # Never overwrite the heartbeat belonging to the lock holder.
            raise RuntimeError("Another exit observer already holds this stage lock") from exc
        status = {
            "schema": "orphan_exit_observer_status_v1",
            "status": "error",
            "output_dir": str(output),
            "observer_process": process_snapshot(os.getpid()),
            "producer_process": None,
            "source_path": str(Path(__file__).resolve()),
            "source_sha256": source_digest(),
            "armed_utc": None,
            "observed_utc": now(),
            "log_path": str(supervisor_log),
            "log_identity": None,
            "offset": None,
        }
        cursor = None

        def heartbeat():
            status["observed_utc"] = now()
            if cursor is not None:
                status.update(log_identity=dict(cursor.identity), offset=cursor.offset)
            atomic_json(status_path, status)

        try:
            if observation_path.exists():
                raise RuntimeError("An exit observation already exists; refusing to overwrite evidence")
            original = read_json(process_path)
            producer = validate_producer(output, original)
            status["producer_process"] = producer
            status["process_status_identity"] = {key: original[key] for key in STATUS_IDENTITY}
            cursor = LogCursor(supervisor_log, from_eof=True)
            armed = datetime.now(timezone.utc)
            status["armed_utc"] = armed.isoformat()
            # Arming is allowed only while that exact process is still alive.
            if process_state(producer) != "alive":
                raise RuntimeError("Producer exited while establishing the observation boundary")
            current_status(process_path, original)
            status["status"] = "observing"
            missing_since = None
            heartbeat()
            while True:
                if source_digest() != status["source_sha256"]:
                    raise RuntimeError("Observer source changed after arming")
                current_status(process_path, original)
                producer_state = process_state(producer)
                for raw, offset, end_offset, identity in cursor.lines():
                    event = parse_reap(raw, producer["pid"], armed)
                    if event is None:
                        continue
                    returncode, finished_utc, raw_line = event
                    # A log line cannot authorize replacing a current live/reused PID.
                    if process_state(producer) != "missing":
                        raise RuntimeError("Supervisor reap conflicts with the producer's live identity")
                    latest = current_status(process_path, original)
                    observed_utc = now()
                    observation = {
                        "schema": "orphan_exit_observation_v1",
                        "returncode": returncode,
                        "finished_utc": finished_utc,
                        "armed_utc": status["armed_utc"],
                        "observed_utc": observed_utc,
                        "raw_line": raw_line,
                        "producer_process": producer,
                        "observer_process": status["observer_process"],
                        "process_status_identity": status["process_status_identity"],
                        "source_path": status["source_path"],
                        "source_sha256": status["source_sha256"],
                        "supervisor_log": {
                            "path": str(supervisor_log), **identity,
                            "byte_offset": offset, "line_end_offset": end_offset,
                        },
                    }
                    atomic_json(observation_path, observation)
                    # Recheck after persisting evidence and preserve every original field.
                    latest = current_status(process_path, original)
                    if process_state(producer) != "missing":
                        raise RuntimeError("Producer identity changed before exit status publication")
                    latest.update(
                        status="completed" if returncode == 0 else "process_error",
                        returncode=returncode,
                        finished_utc=finished_utc,
                        exit_status_source="supervisor_reap_log",
                        exit_observation_path=str(observation_path),
                    )
                    atomic_json(process_path, latest)
                    status.update(
                        status="exit_recorded", returncode=returncode,
                        finished_utc=finished_utc,
                        exit_observation_path=str(observation_path),
                    )
                    heartbeat()
                    return 0
                cursor.rotate_if_drained()
                if producer_state == "missing":
                    if missing_since is None:
                        missing_since = time.monotonic()
                        status["producer_missing_since_utc"] = now()
                    if time.monotonic() - missing_since >= MAX_MISSING_SECONDS:
                        raise RuntimeError("Producer disappeared without an observed supervisor reap within 60 seconds")
                heartbeat()
                time.sleep(1.0)
        except Exception as exc:
            status.update(status="error", error_type=type(exc).__name__, error=str(exc))
            heartbeat()
            raise
        finally:
            if cursor is not None:
                cursor.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervisor-log", type=Path, default=DEFAULT_SUPERVISOR_LOG)
    args = parser.parse_args()
    try:
        return observe(args.output, args.supervisor_log)
    except Exception as exc:
        print(f"orphan exit observer failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

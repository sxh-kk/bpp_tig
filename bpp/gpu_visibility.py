"""Select an eval worker's existing logical GPU before CUDA libraries load.

This module uses only the standard library. CUDA_VISIBLE_DEVICES tokens are
opaque: logical index N selects token N, without resolving or renumbering a
physical GPU. A missing/ambiguous mask or a loaded CUDA runtime is left alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import sys
from typing import Mapping, MutableMapping, Sequence, TextIO


@dataclass(frozen=True)
class GPUVisibilityPlan:
    arguments: tuple[str, ...]
    original_device: str | None
    effective_device: str | None
    original_visible_devices: str | None
    effective_visible_devices: str | None
    selected_physical_token: str | None
    changed: bool
    reason: str

    def provenance(self) -> dict:
        value = asdict(self)
        value.pop("arguments")
        return value


def _explicit_device(arguments: Sequence[str]) -> tuple[str | None, int | None, bool]:
    """Return the last explicit option, matching argparse's repeated-option rule."""
    device, position, joined = None, None, False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument == "--device" and index + 1 < len(arguments):
            device, position, joined = arguments[index + 1], index + 1, False
            index += 1
        elif argument.startswith("--device="):
            device, position, joined = argument.partition("=")[2], index, True
        index += 1
    return device, position, joined


def select_gpu_visibility(arguments: Sequence[str], visible_devices: str | None,
                          *, blocked_reason: str | None = None) -> GPUVisibilityPlan:
    """Plan a same-physical-GPU mapping, without importing CUDA or mutating state."""
    original = tuple(arguments)
    device, position, joined = _explicit_device(original)

    def unchanged(reason):
        return GPUVisibilityPlan(original, device, device, visible_devices,
                                 visible_devices, None, False, reason)

    if "--help" in original or "-h" in original:
        return unchanged("help")
    if device is None:
        return unchanged("no_explicit_device")
    match = re.fullmatch(r"cuda:([0-9]+)", device)
    if match is None:
        return unchanged("non_indexed_cuda_device")
    if visible_devices is None:
        return unchanged("visible_devices_unset")
    tokens = tuple(token.strip() for token in visible_devices.split(","))
    # Invalid/negative/duplicate masks can truncate CUDA enumeration. Do not
    # turn an originally invalid ordinal into an accidentally valid selection.
    if (not tokens or len(set(tokens)) != len(tokens)
            or any(not re.fullmatch(r"[0-9]+|GPU-[^\s,]+|MIG-[^\s,]+", token)
                   for token in tokens)):
        return unchanged("visible_devices_ambiguous")
    logical_index = int(match.group(1))
    if logical_index >= len(tokens):
        return unchanged("logical_device_out_of_range")
    if blocked_reason:
        return unchanged(blocked_reason)
    selected = tokens[logical_index]
    rewritten = list(original)
    rewritten[position] = "--device=cuda:0" if joined else "cuda:0"
    return GPUVisibilityPlan(tuple(rewritten), device, "cuda:0", visible_devices,
                             selected, selected,
                             tuple(rewritten) != original or selected != visible_devices,
                             "same_physical_gpu")


def _loaded_cuda_runtime_reason(modules: Mapping[str, object]) -> str | None:
    """Inspect already-loaded modules only; never import or initialize Torch."""
    torch = modules.get("torch")
    cuda = modules.get("torch.cuda")
    if cuda is None and torch is not None:
        cuda = getattr(torch, "__dict__", {}).get("cuda")
    if torch is not None or cuda is not None:
        initialized = getattr(cuda, "is_initialized", None)
        if not callable(initialized):
            return "cuda_initialization_state_unknown"
        try:
            if initialized():
                return "cuda_already_initialized"
        except Exception:
            return "cuda_initialization_state_unknown"
    # Other CUDA consumers do not expose a common noninitializing query. A
    # conservative skip preserves their existing device enumeration/context.
    consumers = ("sapien", "cupy", "pycuda", "tensorflow", "jaxlib",
                 "cuda.cudart", "cuda.bindings.runtime", "cuda.core")
    if any(name == prefix or name.startswith(prefix + ".")
           for name in modules for prefix in consumers):
        return "cuda_consumer_already_loaded"
    return None


def configure_eval_gpu_visibility(arguments: Sequence[str], *,
                                  environ: MutableMapping[str, str] | None = None,
                                  modules: Mapping[str, object] | None = None,
                                  stderr: TextIO | None = None) -> list[str]:
    """Apply a safe plan before eval dispatch; record the resource mapping only."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    stderr = sys.stderr if stderr is None else stderr
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    plan = select_gpu_visibility(arguments, visible)
    if plan.reason == "same_physical_gpu":
        plan = select_gpu_visibility(arguments, visible,
                                     blocked_reason=_loaded_cuda_runtime_reason(modules))
        if plan.reason == "same_physical_gpu":
            environ["CUDA_VISIBLE_DEVICES"] = plan.effective_visible_devices
    if plan.original_device and re.fullmatch(r"cuda:[0-9]+", plan.original_device) and plan.reason != "help":
        print(json.dumps({"bpp_eval_gpu_visibility": plan.provenance()}, sort_keys=True),
              file=stderr, flush=True)
    return list(plan.arguments)

"""Dispatch the TIG baseline entrypoints to their canonical local core."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
import sys

from .config import BPPPaths, CORE_MODULES


def dispatch(command: str, argv: list[str] | None = None) -> int:
    """Keep core CLI/defaults/checkpoint behavior, including multiprocessing."""
    if command not in CORE_MODULES:
        raise ValueError(f"Unknown BPP entrypoint: {command}")
    arguments = list(sys.argv[1:] if argv is None else argv)
    paths = BPPPaths.from_environment()
    paths.validate()
    importlib.invalidate_caches()
    name = CORE_MODULES[command]
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(paths.canonical_root):
        raise ImportError(f"Cannot resolve {name} inside the canonical BPP package: {paths.canonical_root}")
    module = importlib.import_module(name)
    if command in ("prepare", "monitor") and (not arguments or arguments[0] != command):
        arguments.insert(0, command)

    # Import normal modules, rather than executing source with runpy: spawned
    # evaluator workers must retain importable canonical function identities.
    if inspect.signature(module.main).parameters:
        result = module.main(arguments)
    else:
        previous = sys.argv
        sys.argv = [previous[0], *arguments]
        try:
            result = module.main()
        finally:
            sys.argv = previous
    return result if isinstance(result, int) else 0

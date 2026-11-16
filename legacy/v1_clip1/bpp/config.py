"""Paths for the authoritative BPP implementation inside the TIG baseline."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


CORE_MODULES = {
    "pipeline": "examples.baselines.bpp.bpp.pipeline",
    "monitor": "examples.baselines.bpp.bpp.pipeline",
    "download": "examples.baselines.bpp.bpp.download_data",
    "train": "examples.baselines.bpp.bpp.train",
    "eval": "examples.baselines.bpp.bpp.native_eval",
    "prepare": "examples.baselines.bpp.bpp.training_data",
}


@dataclass(frozen=True)
class BPPPaths:
    project_root: Path
    canonical_root: Path

    @classmethod
    def from_environment(cls) -> "BPPPaths":
        project = Path(os.environ.get("BPP_ROOT", "/gaozt-test1/sxh/bpp")).expanduser().resolve()
        return cls(project, Path(__file__).resolve().parent)

    def validate(self) -> None:
        if not (self.canonical_root / "__init__.py").is_file():
            raise FileNotFoundError(
                f"The canonical BPP package is missing under {self.canonical_root}."
            )

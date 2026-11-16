"""Prepare train-only data for the canonical TIG BPP baseline."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.baselines.bpp.bpp.runtime import dispatch


def main(argv=None):
    return dispatch("prepare", argv)


if __name__ == "__main__":
    raise SystemExit(main())

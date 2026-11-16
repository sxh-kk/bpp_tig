"""Plan or run the resumable, strictly gated TIG BPP scale pipeline."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.baselines.bpp.bpp.runtime import dispatch


def main(argv=None):
    return dispatch("pipeline", argv)


if __name__ == "__main__":
    raise SystemExit(main())

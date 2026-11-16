"""Run the fixed, independently supervised C15 prompt diagnostic."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.baselines.bpp.bpp.prompt_diagnostic import main


if __name__ == "__main__":
    raise SystemExit(main())

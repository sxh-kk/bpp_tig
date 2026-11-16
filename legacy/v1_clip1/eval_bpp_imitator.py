"""Evaluate the canonical dual-arm TIG BPP baseline."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.baselines.bpp.bpp.gpu_visibility import configure_eval_gpu_visibility


def main(argv=None):
    arguments = configure_eval_gpu_visibility(sys.argv[1:] if argv is None else argv)
    # Select visibility before dispatch can import Torch/SAPIEN in native_eval.
    from examples.baselines.bpp.bpp.runtime import dispatch
    return dispatch("eval", arguments)


if __name__ == "__main__":
    raise SystemExit(main())

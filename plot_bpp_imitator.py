"""Export static scientific figures from the checkpoint-verified BPP review."""
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from examples.baselines.bpp.bpp.scientific_plot import main

if __name__=='__main__':raise SystemExit(main())

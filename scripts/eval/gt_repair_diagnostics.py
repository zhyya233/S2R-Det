#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from s2r_det.cli import main

if __name__ == "__main__":
    main(["diagnostics"] + sys.argv[1:])

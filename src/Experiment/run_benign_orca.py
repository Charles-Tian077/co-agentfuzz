"""Run only the isolated arithmetic benchmark; never import attack modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benign_orca.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())

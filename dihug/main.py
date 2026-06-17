"""Module entrypoint for DiHuG.

This wrapper keeps the inherited SparseDiff training script intact while
allowing `python -m dihug.main ...` from the DiHuG project root.
"""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPARSE_DIFFUSION = ROOT / "sparse_diffusion"
for path in (ROOT, SPARSE_DIFFUSION):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from sparse_diffusion.main import main  # noqa: E402


if __name__ == "__main__":
    main()

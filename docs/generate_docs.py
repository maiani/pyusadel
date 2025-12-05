"""
Utility script to build HTML API documentation with pdoc.

The script installs no dependencies on its own; install the optional
``pdoc`` package first (see README instructions) and then run this file
from the repository root::

    python docs/generate_docs.py

The generated site ends up in ``docs/site``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "docs" / "site"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        "pyusadel",
        "-o",
        str(output_dir),
    ]
    subprocess.check_call(cmd, cwd=repo_root)
    print(f"Documentation generated in {output_dir}")


if __name__ == "__main__":
    main()

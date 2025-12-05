"""
Utility script to build HTML API documentation with Sphinx.

Install the optional documentation dependencies first (see README instructions)
and then run this file from the repository root::

    python docs/generate_docs.py

The generated site ends up in ``docs/_build/html``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sphinx.cmd.build import main as sphinx_main


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs"
    html_dir = docs_dir / "_build" / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    status = sphinx_main(["-b", "html", str(docs_dir), str(html_dir)])
    if status != 0:
        sys.exit(status)

    print(f"Documentation generated in {html_dir}")


if __name__ == "__main__":
    main()

"""Configuration for Sphinx documentation."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so autodoc can import the package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

project = "pyUsadel"
author = "Andrea Maiani"

current_year = datetime.now().year
copyright = f"{current_year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

html_theme = "alabaster"

html_static_path: list[str] = []

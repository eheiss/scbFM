"""Small path bootstrap shared by data-preparation and plotting notebooks."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(os.environ.get("SCBFM_ROOT_DIR", REPO_ROOT.parent)).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("SCBFM_OUTPUT_DIR", ROOT_DIR / "output")).expanduser()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = ROOT_DIR / OUTPUT_DIR
FIGURE_DIR = Path(os.environ.get("SCBFM_FIGURE_DIR", OUTPUT_DIR / "figures")).expanduser().resolve()

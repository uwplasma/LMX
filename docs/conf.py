from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "LMX"
author = "LMX contributors"
copyright = f"{datetime.now(timezone.utc).year}, LMX contributors"
release = "1.3.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx_design",
]

source_suffix = {
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# These DOI resolvers return 403 to GitHub's link checker while remaining valid
# in browsers. The bibliography retains the persistent identifiers; executable
# links for the associated software and preprints are checked separately.
linkcheck_ignore = [
    r"https://doi\.org/10\.1115/1\.2960953",
    r"https://doi\.org/10\.1063/5\.0230242",
    r"https://doi\.org/10\.1145/347837\.347846",
    r"https://doi\.org/10\.1137/10078356X",
    r"https://doi\.org/10\.1073/pnas\.2101784118",
    r"https://doi\.org/10\.1080/01495728408961817",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

html_theme = "furo"
html_title = "LMX"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/uwplasma/LMX/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

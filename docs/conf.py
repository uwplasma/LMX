from __future__ import annotations

from datetime import datetime, timezone

project = "LMX"
author = "LMX contributors"
copyright = f"{datetime.now(timezone.utc).year}, LMX contributors"
release = "1.2.0"

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
    r"https://doi\.org/10\.1063/5\.0230242",
    r"https://doi\.org/10\.1145/347837\.347846",
    r"https://doi\.org/10\.1073/pnas\.2101784118",
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

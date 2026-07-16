from __future__ import annotations

from datetime import datetime

project = "LMX"
author = "LMX contributors"
copyright = f"{datetime.now().year}, LMX contributors"
release = "1.1.3"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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

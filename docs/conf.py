from __future__ import annotations

from datetime import datetime

project = "LMX"
author = "LMX contributors"
copyright = f"{datetime.now().year}, LMX contributors"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

html_theme = "furo"
html_title = "LMX"
html_static_path = ["_static"]

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate import generate_from_readmes

generate_from_readmes()

project = "mvsr"
author = "Ansgar Lößer, Max Schlecht"
copyright = f"%Y, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
html_theme = "furo"
html_static_path = ["static"]
html_css_files = ["theme-overrides.css"]
myst_fence_as_directive = ["automodule", "mermaid"]
add_module_names = False
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autodoc_typehints_format = "fully-qualified"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

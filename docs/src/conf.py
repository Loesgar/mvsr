import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fixes import setup as setup
from generate import generate_from_readmes

generate_from_readmes()

project = "mvsr"
author = "Ansgar Lößer, Max Schlecht"
copyright = f"%Y, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
html_theme = "furo"
html_static_path = ["static"]
html_css_files = ["theme-overrides.css"]
myst_fence_as_directive = ["automodule", "mermaid"]
myst_heading_anchors = 6
add_module_names = False
always_use_bars_union = True
autodoc_default_options = {"members": True}
autodoc_member_order = "bysource"
autodoc_mock_imports = ["_typeshed"]
autodoc_preserve_defaults = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
typehints_document_rtype = False

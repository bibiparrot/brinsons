"""Sphinx configuration for the brinsons reference and method guide."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "brinsons"
author = "brinsons contributors"
release = "0.1.0"
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
html_theme = "furo"
html_title = "brinsons | Performance attribution"
html_theme_options = {"navigation_with_keys": True}
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath", "deflist"]
myst_heading_anchors = 3
nitpicky = True
nitpick_ignore = [("py:class", "polars.dataframe.frame.DataFrame"), ("py:class", "datetime.date")]
autodoc_typehints = "description"
exclude_patterns = ["_build"]

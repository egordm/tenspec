"""Sphinx configuration for the Tenspec documentation site.

One sidebar carries every route, and each route is a task a reader arrives with. Build the
site with `uv run --only-group docs poe docs`.
"""

# Import it under another name. Sphinx reads every module-level name in this file as a config
# value, and a bare `version` would shadow the `version` setting with this function.
from importlib.metadata import version as installed_version

project = "tenspec"
author = "Egor Dmitriev"
copyright = "2026, Egor Dmitriev"  # noqa: A001
release = installed_version("tenspec")
version = release

# myst_nb supersedes myst_parser: it parses the same Markdown and adds the notebook formats.
# Loading both would register two parsers for the same suffix.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# This file sits in the source directory, and the notebook format below claims `.py`, so name it
# here or Sphinx reads its own configuration as a document.
exclude_patterns = ["_build", "conf.py"]

# The tutorials are committed as Jupytext percent-format Python, and nothing else. MyST-NB reads
# that source directly, so no notebook or saved output is committed beside it.
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}

# Execute on every build, and fail the build on an unexpected cell error. A page therefore shows
# the output of this build, never a saved transcript.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 180

# `myst_heading_anchors` gives every heading down to level three a slug, so a link to one
# section keeps working on the site and on GitHub.
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence", "deflist"]

# Napoleon reads the Google-style docstrings the package already writes.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# No default members. Expanding every member pulled in the Pydantic schema-integration hook and
# its whole CoreSchema union. Each entry in api.md asks for the members a caller actually uses.
# The signature goes in the description, because a tensor declaration makes a heading unreadable.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

html_theme = "pydata_sphinx_theme"
html_title = "tenspec"
html_static_path = ["_static"]
html_css_files = ["brand.css"]
html_favicon = "_static/favicon.svg"
html_show_sourcelink = False

# The sidebar carries every route, so the header carries no second copy of them: `navbar_center`
# is empty, and the header keeps the logo, the search field, the theme switch and the repository
# icon. Every page sits directly under the root document, so every page's sidebar renders the
# whole reader map rather than one section of it.
# The mark is named for the mode it serves: `mark-light.svg` carries dark ink for the light page.
# `alt_text` is empty so the mark is decorative, and the visible wordmark alone names the home
# link. Both together read as "tenspec tenspec".
html_theme_options = {
    "github_url": "https://github.com/egordm/tenspec",
    "navbar_align": "left",
    "navbar_center": [],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": [],
    "logo": {
        "text": "tenspec",
        "image_light": "_static/mark-light.svg",
        "image_dark": "_static/mark-dark.svg",
        "alt_text": "",
    },
}

# `sidebar-reader-map` calls the theme's renderer over the root toctrees, so every page shows
# every route. `_templates/sidebar-reader-map.html` holds the reason.
templates_path = ["_templates"]
html_sidebars = {"**": ["sidebar-collapse", "sidebar-reader-map"]}

# `default_mode` is a theme context value, not a theme option. The theme reads it for a visitor
# who has stored no choice yet, and its own switch still writes and reads that choice.
html_context = {"default_mode": "dark"}

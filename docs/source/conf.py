"""
Sphinx configuration file for Trittention-Transformer documentation.
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('../..'))

# Project information
project = 'Trittention-Transformer'
copyright = f'{datetime.now().year}, Mudit Bhargava'
author = 'Mudit Bhargava'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# Add extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
    'myst_parser'
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Configure napoleon for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns to exclude from source
exclude_patterns = []

# The theme to use for HTML documentation
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'lightning': ('https://pytorch-lightning.readthedocs.io/en/stable/', None),
}

# Enable todo items
todo_include_todos = True

# Enable markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

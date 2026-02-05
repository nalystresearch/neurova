# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

# Neurova Documentation Configuration
# Sphinx build configuration file

import os
import sys
from datetime import datetime

# Add neurova to path for autodoc
sys.path.insert(0, os.path.abspath('../../neurova'))

# -- Project information -----------------------------------------------------

project = 'Neurova'
copyright = f'{datetime.now().year}, Neurova Authors'
author = 'Neurova Authors'

# The full version, including alpha/beta/rc tags
version = '1.0.0'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
]

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The master document
master_doc = 'index'

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
}

# MyST parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'

html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#2980B9',
        'color-brand-content': '#2980B9',
        'color-api-name': '#1a1a1a',
        'color-api-pre-name': '#1a1a1a',
    },
    'dark_css_variables': {
        'color-brand-primary': '#56a0d3',
        'color-brand-content': '#56a0d3',
    },
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    'top_of_page_button': 'edit',
    'source_repository': 'https://github.com/neurova/neurova/',
    'source_branch': 'main',
    'source_directory': 'doc/source/',
}

html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

html_title = 'Neurova Documentation'
html_short_title = 'Neurova'
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# HTML context for templates
html_context = {
    'display_github': True,
    'github_user': 'neurova',
    'github_repo': 'neurova',
    'github_version': 'main',
    'conf_py_path': '/doc/source/',
}

# -- Options for LaTeX output ------------------------------------------------

latex_engine = 'xelatex'

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
}

latex_documents = [
    (master_doc, 'neurova.tex', 'Neurova Documentation',
     'Neurova Authors', 'manual'),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

# -- Options for PDF output --------------------------------------------------

pdf_documents = [
    (master_doc, 'neurova', 'Neurova Documentation', 'Neurova Authors'),
]

# -- Extension configuration -------------------------------------------------

# Todo extension
todo_include_todos = True

# Coverage extension
coverage_show_missing_items = True

# GraphViz configuration
graphviz_output_format = 'svg'

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Custom setup ------------------------------------------------------------

def setup(app):
    """Custom Sphinx setup."""
    # Add custom CSS
    app.add_css_file('custom.css')
    
    # Register custom directives if needed
    pass

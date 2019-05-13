# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../src/explorepy/'))


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ['pybluez', 'bluetooth', 'pylsl', 'bokeh','bokeh.layouts']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]
if os.getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_US'

source_suffix = '.rst'
master_doc = 'index'
project = 'explorepy'
year = '2018-2019'
author = 'Mohamad Atayi'
copyright = '{0}, {1}'.format(year, author)
version = release = '0.3.0'

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/Mentalab-hub/explorepy/issues/%s', '#'),
    'pr': ('https://github.com/Mentalab-hub/explorepy/pull/%s', 'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

# If false, no module index is generated.
html_domain_indices = True

napoleon_google_docstring = True

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

autoclass_content = 'both'
autodoc_default_flags = [
    'members',
    'inherited-members',
    'private-members',
    'show-inheritance',
]
autodoc_member_order = 'bysource'
autosummary_generate = True

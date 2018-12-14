
Python API for Mentalab biosignal aquisition devices

Under development

.. image:: https://static1.squarespace.com/static/5428308ae4b0701411ea8aaf/t/59be746a90bade209a205adb/1542922455492/?format=1500w
   :width: 200 px
   :scale: 30 %
   :align: center

========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/explorepy/badge/?style=flat
    :target: https://readthedocs.org/projects/explorepy
    :alt: Documentation Status


.. |travis| image:: https://travis-ci.org/bmeatayi/explorepy.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/bmeatayi/explorepy

.. |version| image:: https://img.shields.io/pypi/v/explorepy.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/explorepy

.. |commits-since| image:: https://img.shields.io/github/commits-since/bmeatayi/explorepy/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/bmeatayi/explorepy/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/explorepy.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/explorepy

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/explorepy.svg
    :alt: Supported versions
    :target: https://pypi.org/project/explorepy

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/explorepy.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/explorepy


.. end-badges



Installation
============

::

    pip install explorepy

Documentation
=============


https://explorepy.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

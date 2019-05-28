==================
explorepy overview
==================

.. image:: logo.png
   :scale: 100 %
   :align: center


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

.. |travis| image:: https://travis-ci.org/Mentalab-hub/explorepy.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/Mentalab-hub/explorepy

.. |version| image:: https://img.shields.io/pypi/v/explorepy.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/explorepy

.. |commits-since| image:: https://img.shields.io/github/commits-since/Mentalab-hub/explorepy/v0.3.1.svg
    :alt: Commits since latest release
    :target: https://github.com/Mentalab-hub/explorepy/compare/v0.3.1...master

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

A Python API for Mentalab biosignal aquisition devices

Requirements
============
* Python 3.5 or newer version
* `numpy <https://github.com/pybluez/pybluez>`_
* `pybluez <https://github.com/pybluez/pybluez>`_ (check their repo for the requirements of pybluez)
* `pylsl <https://github.com/labstreaminglayer/liblsl-Python>`_
* `scipy <https://github.com/scipy/scipy>`_
* `bokeh <https://github.com/bokeh/bokeh>`_


Installation
============
To install ``explorepy`` from PyPI run:
::

    pip install explorepy


To install the latest development version run:
::

    pip install git+https://github.com/Mentalab-hub/explorepy


Example use
===========
CLI command:
``explorepy acquire -n Explore_XXXX``

Enter ``explorepy -h`` for help.


The following code connects to the Explore device and prints the data.

::

    import explorepy
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX")  # Put your device Bluetooth name
    explorer.acquire()

You can also visualize signal in real-time.

::

    import explorepy
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX")  # Put your device Bluetooth name
    explorer.visualize(n_chan=4, bp_freq=(1, 30), notch_freq=50)  # Give the number of channels, frequencies of bandpass and notch filter

EEG:

.. image:: /images/Dashboard_EEG.jpg
  :width: 800
  :alt: EEG Dashboard

ECG with heart beat detection:

.. image:: /images/Dashboard_ECG.jpg
  :width: 800
  :alt: ECG Dashboard

Documentation
=============

To see full documentation of the API, visit: https://explorepy.readthedocs.io/


Authors
=======
- `Sebastian Herberger`_
- `Mohamad Atayi`_
- `Philipp Jakovleski`_
- `Andreas Gutsche`_

.. _Sebastian Herberger: https://github.com/SHerberger
.. _Mohamad Atayi: https://github.com/bmeatayi
.. _Philipp Jakovleski: https://github.com/philippjak
.. _Andreas Gutsche: https://github.com/andyman410







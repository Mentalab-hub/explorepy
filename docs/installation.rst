============
Installation
============


Requirements
------------
* Python 3.5 or newer version
* `numpy <https://github.com/pybluez/pybluez>`_
* `scipy <https://github.com/scipy/scipy>`_
* `pybluez <https://github.com/pybluez/pybluez>`_ (check their repo for the requirements of pybluez)
* `pylsl <https://github.com/labstreaminglayer/liblsl-Python>`_
* `bokeh <https://github.com/bokeh/bokeh>`_

``explorepy`` is using ``pybluez`` as the bluetooth backend. ``pybluez`` has different dependencies in different operating systems.

Windows
^^^^^^^
* Visual C++ 2010 Express for build
* Visual Studio 2017 community edition (In order to build 64-bit debug and release executables)

Linux
^^^^^
* Python distutils
* Bluez and header files
* ``sudo apt-get install libbluetooth-dev``

Mac OS
^^^^^^
* Xcode
* PyObjc 3.1b or later


Installation commands
^^^^^^^^^^^^^^^^^^^^^
To install ``explorepy`` from PyPI run:
::

    pip install explorepy


To install the latest development version run:
::

    pip install git+https://github.com/Mentalab-hub/explorepy

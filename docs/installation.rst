============
Installation
============


Requirements
------------
* Python 3.5 or newer version
* `numpy <https://github.com/pybluez/pybluez>`_
* `pybluez <https://github.com/pybluez/pybluez>`_ (check their repo for the requirements of pybluez)
* `pylsl <https://github.com/labstreaminglayer/liblsl-Python>`_

``explorepy`` is using ``pybluez`` as the bluetooth backend. ``pybluez`` has different dependencies in different operating systems.

Windows
^^^^^^^
* Visual C++ 2010 Express for build
* Visual Studio 2015 community edition (Probably for 64-bit systems)

Linux
^^^^^
* Python distutils
* Bluez and header files
* ``sudo apt-get install libbluetooth-dev``

Mac OS
^^^^^^
* ?


Installation commands
^^^^^^^^^^^^^^^^^^^^^
To install ``explorepy`` from PyPI run:
::

    pip install explorepy


To install the latest development version run:
::

    pip install git+https://github.com/Mentalab-hub/explorepy

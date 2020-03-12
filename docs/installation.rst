============
Installation
============


Requirements
------------
* Python 3.5 or newer version
* Bluetooth adapter on your device

``explorepy`` is using ``pybluez`` as the bluetooth backend. ``pybluez`` has different dependencies in different operating systems.

Windows
^^^^^^^
* Visual C++ build tools
* Visual Studio 2015 community edition (In order to build 64-bit debug and release executables)

Ubuntu 16.04 or 18.04
^^^^^^^^^^^^^^^^^^^^^
* ``sudo apt-get install libbluetooth-dev``

Mac OS
^^^^^^
* Xcode
* PyObjc 3.1b or later


For more details on pybluez dependencies please see `pybluez docs <https://github.com/pybluez/pybluez>`_.


How to install (Windows)
--------------

This instructions guides you to install Explorepy API with all its dependencies on Windows.

1. Install Python 3 on your computer. It is recommended to install Anaconda Python package. Download and install Anaconda Python 3.7 Windows installer from `here <https://www.anaconda.com/distribution/#download-section>`_.
2. Download and install MS Visual Studio Community Edition 2015 from this `link <https://visualstudio.microsoft.com/vs/older-downloads/>`_. Make sure you install Build Tools for Visual Studio 2017 (version 15.9) from this `link <https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads>`_.
3. We recommend using a virtual environment.

  * In Conda command prompt: ``conda create -n myenv python=3.6``
  * Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``

4. To install ``explorepy`` from PyPI run: ``pip install explorepy``

Quick test
----------

* Open Conda command prompt

* Activate the virtual environment: ``conda activate myenv``

* ``explorepy visualize -n <YOUR-DEVICE-NAME> -c 4`` (Change the number of channels if needed)

* To stop visualization press Ctrl+c

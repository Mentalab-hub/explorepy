.. image:: logo.png
   :scale: 100 %
   :align: center



.. start-badges

|docs| |version| |wheel| |supported-versions| |commits-since| |travis|

.. |docs| image:: https://readthedocs.org/projects/explorepy/badge/?style=flat
    :target: https://readthedocs.org/projects/explorepy
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/Mentalab-hub/explorepy.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/Mentalab-hub/explorepy

.. |version| image:: https://img.shields.io/pypi/v/explorepy.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/explorepy

.. |commits-since| image:: https://img.shields.io/github/commits-since/Mentalab-hub/explorepy/v1.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/Mentalab-hub/explorepy/compare/v1.0.0...master

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

==================
Explorepy overview
==================

Explorepy is an open-source Python-based biosignal acquisition API for Mentalab's Explore device. It provides users the following features:

* Real-time streaming of ExG, orientation and environmental data
* Real-time visualization
* Data recording in CSV and BDF+ formats
* LSL integration
* Impedance measurement
* Explore device configuration


Quick installation
==================

Requirements
------------

* Python 3.5 or newer version
* Visual Studio 2015 community edition (only Windows)
* Bluetooth header files (only Linux -> use this command: ``sudo apt-get install libbluetooth-dev``)


Please check `installation page <https://explorepy.readthedocs.io/en/latest/installation.html>`_ for more detailed instructions.

To install ``explorepy`` from PyPI run:
::

    pip install explorepy


To install the latest development version (git must be installed before running this command):
::

    pip install git+https://github.com/Mentalab-hub/explorepy


Get started
===========

CLI command
-----------

``explorepy acquire -n Explore_XXXX``

Enter ``explorepy -h`` for help.


Python code
-----------

The following code connects to the Explore device and prints the data.

::

    import explorepy
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX")  # Put your device Bluetooth name
    explorer.acquire()

You can also visualize signals in real-time.

::

    import explorepy
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX")  # Put your device Bluetooth name
    explorer.visualize(bp_freq=(.5, 30), notch_freq=50)

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
- `Masooma Fazelian`_
- `Salman Rahman`_
- `Philipp Jakovleski`_
- `Andreas Gutsche`_


.. _Sebastian Herberger: https://github.com/SHerberger
.. _Mohamad Atayi: https://github.com/bmeatayi
.. _Masooma Fazelian: https://github.com/fazelian
.. _Salman Rahman: https://github.com/salman2135
.. _Philipp Jakovleski: https://github.com/philippjak
.. _Andreas Gutsche: https://github.com/andyman410


License
=======
This project is licensed under the `MIT <https://github.com/Mentalab-hub/explorepy/blob/master/LICENSE>`_ license.





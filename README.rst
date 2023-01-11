.. image:: logo.png
   :scale: 100 %
   :align: center


.. start-badges

|docs| |version| |wheel| |supported-versions| |commits-since|

.. |docs| image:: https://readthedocs.org/projects/explorepy/badge/?style=flat
    :target: https://readthedocs.org/projects/explorepy
    :alt: Documentation Status


.. |version| image:: https://img.shields.io/pypi/v/explorepy.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/explorepy


.. |commits-since| image:: https://img.shields.io/github/commits-since/Mentalab-hub/explorepy/v1.7.0.svg
    :alt: Commits since latest release
    :target: https://github.com/Mentalab-hub/explorepy/compare/v1.7.0...master


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

=========================
``explorepy`` overview
=========================

``explorepy`` is an open-source Python API designed to collect and process ExG data using Mentalab's Explore device. Amongst other things, ``explorepy`` provides the following features:

* Real-time streaming of ExG, orientation and environmental data.
* Real-time visualization of ExG, orientation and environmental data.
* Data recording in CSV and BDF+ formats.
* Integration with LabStreaming Layer.
* Electrode impedance measurements.
* Explore device configuration.

Quick installation
==================
For Windows users, the best way to install ``explorepy`` is to download the latest ``explorepy`` version from the `release page <https://github.com/Mentalab-hub/explorepy/releases>`_. Please note that dependencies will install automatically from the release page.

For other operating systems, or to build the package manually on Windows, please refer to the information below. MAC OSX is currently not supported. 


Requirements
------------

* Python 3.7 to Python 3.10.
* Visual Studio 2015 community edition (Windows only. For package building).
* Bluetooth header files (Linux only. Use: ``sudo apt-get install libbluetooth-dev``).


Detailed installation instructions can be found on the `installation page <https://explorepy.readthedocs.io/en/latest/installation.html>`_.

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
To check ``explorepy`` is running use:
::
    explorepy acquire -n Explore_XXXX

For help, use:
::
    explorepy -h


Python code
-----------

In Python you can connect to the Explore device and print data using:

::

    import explorepy
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX")  # Put your device Bluetooth name
    explorer.acquire()

You can also visualize the data in real-time.

::

    import explorepy
    explorer = explorepy.Explore()
    explorer.connect(device_name="Explore_XXXX")  # Put your device Bluetooth name

Documentation
=============

For full API documentation, visit: https://explorepy.readthedocs.io/.

Troubleshooting
===============
If you are having problems, please check the `troubleshooting <https://explorepy.readthedocs.io/en/latest/installation.html#troubleshooting>`_
section of the documentation.

If you are still having problems, send us your error log via Sentry (note: Explorepy will send the log
automatically once you have provided permission), or send us the log file via email to contact@mentalab.com. The log file is usually found under:

* Windows: ``<Windows Drive>:\Users\<USER_NAME>\AppData\Local\mentalab\explorepy\Logs\explorepy.log``
* Ubuntu: ``/home/<USER_NAME>/.cache/explorepy/log/explorepy.log``
* Mac OS: ``/Users/<USER_NAME>/Library/Logs/explorepy/explorepy.log``

You can also create a new issue in the GitHub repository.

Authors
=======
- `Mohamad Atayi`_
- `Salman Rahman`_
- `Andrea Escartin`_
- `Alex Platt`_
- `Andreas Gutsche`_
- `Masooma Fazelian`_
- `Philipp Jakovleski`_
- `Florian Sesser`_
- `Sebastian Herberger`_


.. _Mohamad Atayi: https://github.com/bmeatayi
.. _Salman Rahman: https://github.com/salman2135
.. _Andrea Escartin: https://github.com/andrea-escartin
.. _Alex Platt: https://github.com/Nujanauss
.. _Andreas Gutsche: https://github.com/andyman410
.. _Masooma Fazelian: https://github.com/fazelian
.. _Philipp Jakovleski: https://github.com/philippjak
.. _Florian Sesser : https://github.com/hacklschorsch
.. _Sebastian Herberger: https://github.com/SHerberger

License
=======
This project is licensed under the `MIT <https://github.com/Mentalab-hub/explorepy/blob/master/LICENSE>`_ license. You can reach us at contact@mentalab.com.

============
Installation
============

Minimal Requirements
--------------------
* Python 3.12 and upwards. We recommend using Python 3.12.
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 6GB RAM (minimum 1GB *free* RAM during the session)
* Intel i5 or higher (2x2.5GHz) CPU

Recommended Requirements
------------------------
* Python 3.12
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 8GB RAM
* Intel i7 or higher CPU

How to install
--------------

Option 1: Installing via installer file (basic)
"""""""""""""""""""""""""""""""""""""""""""""""

Windows and Mac
^^^^^^^^^^^^^^^

*This option is best for users who only intend to use functionalities offered by* ``explorepy`` *via a graphical user interface*

For example, if you want to quickly visualize and record data and don't need the command line interface or to use it in your own Python script, use this option.

If you intend to call ``explorepy`` from the command line or a Python script (e.g. from an experiment script), install ``explorepy`` via Anaconda/pip instead.

For Windows, Mac and Linux the standalone desktop software ExploreDesktop can be installed using the installer files uploaded to the
`release page <https://github.com/Mentalab-hub/explore-desktop-release/releases/latest/>`_. Please note that the dependencies will be installed automatically and bundled locally with the installed software.


Option 2: Installing from Python Package Index (PyPI) and pip (advanced)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


.. note::

   Explore legacy device support is **deprecated** in ExplorePy. See the documentation on :doc:`explore_legacy_devices`.


*To install explorepy for any Python version below 3.10, please contact support@mentalab.com*

*This option is best for users who intend to include* ``explorepy`` *functionalities in their own Python scripts or use it from the command line.*

For an overview of ``explorepy`` commands, click `here <https://explorepy.readthedocs.io/en/latest/usage.html#command-line-interface>`_.

Windows
^^^^^^^

To install the ``explorepy`` API and all its dependencies using pip on Windows:

1. Install Anaconda (or any other Python distribution; these instructions pertain to Anaconda only). Download and run the `Anaconda installer for windows <https://www.anaconda.com/download/success>`_.
2. Install `Microsoft Build Tools for Visual Studio 2019 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_. Select *Desktop development with C++* in the workloads tab. Make sure that *MSVCv142 - VS 2019 C++ x64/x86 build tools* and the latest version of *Windows 10 SDK* are checked.
3. Open the Anaconda command prompt.
4. We recommend using a conda environment. To do this:

   a. In the Anaconda command prompt: ``conda create -n myenv python=3.12``
   b. Activate the conda environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Install liblsl: ``conda install -c conda-forge liblsl``
7. Run ``pip install explorepy`` to install ``explorepy`` from PyPI.

Ubuntu
^^^^^^
1. From the terminal, run these commands one by one: ``sudo apt-get install libbluetooth-dev`` and ``sudo apt-get install build-essential``.
2. We recommend installing Anaconda. Download and installer `Anaconda<https://www.anaconda.com/download>`/Miniconda/.
3. We recommend using a virtual environment in Conda. To do this:

   a. In the Anaconda command prompt: ``conda create -n myenv python=3.12``
   b. Activate the conda environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``
5. Install liblsl: ``conda install -c conda-forge liblsl``
6. Run ``pip install explorepy`` to install ``explorepy`` from PyPI.
7. From the terminal, run: ``sudo apt install libxcb-cursor0``

Set up USB streaming in Linux
+++++++++++++++++++++++++++++

a. Set up ``udev`` rules for appropiate permission to ``/dev/ttyACM*`` in Linux

    *Steps to Execute the Udev Script Manually*

    1. Create a file named ``setup_udev_rule.sh`` and include the following script

        ::

            #!/bin/bash

            RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="stm_virtual_com", MODE="0666"'

            echo "Creating udev rule..."
            echo "$RULE" | sudo tee /etc/udev/rules.d/99-stm-virtual-com.rules > /dev/null

            sudo udevadm control --reload-rules && sudo udevadm trigger
            echo "udev rule has been created successfully!"
            echo "You can access your device at /dev/$SYMLINK_NAME when it is connected."

    2. Make the ``setup_udev_rule.sh`` executable ::

         chmod +x setup_udev_rule.sh

    3. Execute the script ::

        ./setup_udev_rule.sh

    *To remove the udev rule when no longer required*  ::

        sudo rm /etc/udev/rules.d/99-stm-virtual-com.rules


b. Alternate method: Temporarily granting appropriate permissions to ``/dev/ttyACM*``


    1. Identify the device (ttyACM0, ttyACM1, ttyACM2, etc) in ``/dev`` directory


    2. Execute this command in the terminal (replace * with appropiate id) ::

            chmod 666 /dev/ttyACM*

Mac
^^^
1. Install ``XCode`` from the Mac App store. For this, you may need to upgrade to the latest version of MacOS. For older versions of MacOS, find compatible versions of ``XCode`` `here <https://en.wikipedia.org/wiki/Xcode>`_. All old ``XCode`` versions are available `here <https://developer.apple.com/download/more/>`_.
2. Accept the license agreement: ``sudo xcodebuild -license``.
3. It is best to install Anaconda. Download  and run the `Anaconda installer for Mac <https://www.anaconda.com/download/success>`_. For older versions of MacOS, compatible version of Anaconda can be found in `this table <https://docs.continuum.io/anaconda/install/#old-os>`_ and downloaded `here <https://repo.anaconda.com/archive/index.html>`_.
4. We recommend using a conda environment.

   a. In the Anaconda command prompt: ``conda create -n myenv python=3.10``
   b. Activate the conda environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Install liblsl: ``conda install -c conda-forge liblsl``
7. Run ``pip install explorepy`` to install ``explorepy`` from PyPI.
8. Connect your Explore device from the Bluetooth menu and run your Python script.


Quick test
----------

*Note: If you installed the graphical user interface ExploreDesktop as outlined above, explorepy won't be available from the command line.*

* Open the Anaconda command prompt.
* Activate the virtual environment that you made before installing explorepy: ``conda activate myenv``
* Run ``explorepy acquire -n DEVICE-NAME``
* To stop the command execution, press ``Ctrl+C``

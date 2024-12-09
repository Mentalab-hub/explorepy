============
Installation
============

Minimal Requirements
------------
* Python 3.10 to Python 3.12. We recommend Python 3.12.
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 6GB RAM (minimum 1GB *free* RAM during the session)
* Intel i5 or higher (2x2.5GHz) CPU

Recommended Requirements
------------
* Python 3.7 to Python 3.12
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 8GB RAM
* Intel i7 or higher CPU

How to install
--------------

Windows
^^^^^^^

Option 1: Installing via installer file (basic)
""""""""

*This option is best for users who only intend to use* ``explorepy`` *via graphical user interface*

For example, if you want to quickly visualize and record data, use Option 1.
If you intend to add ``explorepy`` commands to a Python script
(e.g. an experiment script), install ``explorepy`` via Anaconda instead.

For an overview of ``explorepy`` commands, click `here <https://explorepy.readthedocs.io/en/latest/usage.html#command-line-interface>`_.

On a Windows and Mac operating systems, standalone desktop software ``explorepy-desktop`` can be installed using the .exe installable file uploaded to
`release page <https://github.com/Mentalab-hub/explore-desktop-release/releases/latest/>`_. Please note that the dependencies will be installed automatically.


A standalone Mac installer is also available. Please contact Mentalab support to get Mac installer.


Option 2: Installing from Python Package Index (PyPI) and pip (advanced)
""""""""
*To install explorepy for any python version except 3.10, please contact support@mentalab.com.*

*This option is best for users who intend to include* ``explorepy`` *functionalities in their own Python scripts.*
To install the ``explorepy`` API and all its dependencies using pip on Windows:

1. Install Anaconda (or any other Python distribution; these instructions pertain to Anaconda only). Download and install the `Anaconda Windows installer <https://www.anaconda.com/distribution/#download-section>`_.
2. Install `Microsoft Build Tools for Visual Studio 2019 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_. Select *Desktop development with C++* in the workloads tab. Make sure that *MSVCv142 - VS 2019 C++ x64/x86 build tools* and the latest version of *Windows 10 SDK* are checked.
3. Open the Conda command prompt.
4. We recommend using a virtual environment. To do this:

   a. In Conda command prompt: ``conda create -n myenv python=3.10``
   b. Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.

Ubuntu
^^^^^^
1. From Linux Terminal, run these commands one by one: ``sudo apt-get install libbluetooth-dev`` and ``sudo apt-get install build-essential`` and ``conda install -c conda-forge liblsl``.
2. We recommend installing Anaconda. Download and installer `Anaconda<https://www.anaconda.com/download>`/Miniconda/.
3. We recommend using a virtual environment in Conda. To do this:

   a. In Conda command prompt: ``conda create -n myenv python=3.10``
   b. Activate the virtual environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``
5. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.
6. From Linux Terminal, run: ``sudo apt install libxcb-cursor0``


Set up USB streaming in Linux
"""""""""""""""""""""""""""""
a. Set up ``udev`` rules for appropiate permission to ``/dev/ttyACM*`` in Linux

    *Steps to Execute the Udev Script Manually*

    1. Create a file named ``setup_udev_rule.sh`` and include the following script

Mac
^^^
1. Install ``XCode`` from the Mac App store. For this, you may need to upgrade to the latest version of MacOS. For older versions of MacOS, find compatible versions of ``XCode`` `here <https://en.wikipedia.org/wiki/Xcode>`_. All old ``XCode`` versions are available `here <https://developer.apple.com/download/more/>`_.
2. Accept the license agreement: ``sudo xcodebuild -license``
3. It is best to install Anaconda. Download  and install the `Anaconda Python 3.7 Mac installer <https://www.anaconda.com/distribution/#download-section>`_. For older versions of MacOS, compatible version of Anaconda can be found in `this table <https://docs.continuum.io/anaconda/install/#old-os>`_ and downloaded `here <https://repo.anaconda.com/archive/index.html>`_.
4. We recommend using a virtual environment in Conda.

   a. In Conda command prompt: ``conda create -n myenv python=3.10``
   b. Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.
7. Run: ``brew install blueutil``, to install blueutil for bluetooth communication
7. Connect your Explore device from Mac Bluetooth menu and run your Python script.


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


    2. Execute this command in the terminal (replcae * with appropiate id) ::

            chmod 666 /dev/ttyACM*

Mac
^^^
1. Install ``XCode`` from the Mac App store. For this, you may need to upgrade to the latest version of MacOS. For older versions of MacOS, find compatible versions of ``XCode`` `here <https://en.wikipedia.org/wiki/Xcode>`_. All old ``XCode`` versions are available `here <https://developer.apple.com/download/more/>`_.
2. Accept the license agreement: ``sudo xcodebuild -license``
3. Download and installer `Anaconda<https://www.anaconda.com/download>`/Miniconda/.
4. We recommend using a virtual environment in Conda.

   a. In Conda command prompt: ``conda create -n myenv python=3.10``
   b. Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.
7. Connect your Explore device from Mac Bluetooth menu and run your Python script.

Quick test
----------

* Open the Conda command prompt (if you used pip) or Windows command prompt (if you used the installable file).
* Activate the virtual environment (this step is only necessary in the Conda command prompt): ``conda activate myenv``
* Run ``explorepy acquire -n DEVICE-NAME``
* To stop the command execution, press ``Ctrl+C``

Troubleshooting
---------------

**1. OSError: A socket operation was attempted to an unreachable network.**

Solution: Ensure that your Explore device is paired with your computer and try again.

---------------------

**2. ValueError: Error opening socket.**

Solution: Ensure the Bluetooth module of your computer's operating system is on and working.

---------------------

**3. OSError: The handle is invalid.**

Solution: Ensure the Bluetooth module of your computer's operating system is on and working.

---------------------

If your issue persists, please send a screenshot and brief error description to support@mentalab.com, and we will quickly help you solve it.

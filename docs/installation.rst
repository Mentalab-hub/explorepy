============
Installation
============


Requirements
------------
* Python 3.5 or newer version
* Visual Studio 2015 community edition (only Windows)



How to install
--------------

Windows
^^^^^^^
This instructions guides you to install Explorepy API with all its dependencies on Windows.

1. Install Python 3 on your computer. It is recommended to install Anaconda Python package. Download and install Anaconda Python 3.7 Windows installer from `here <https://www.anaconda.com/distribution/#download-section>`_.
2. Download and install MS Visual Studio Community Edition 2015 from this `link <https://visualstudio.microsoft.com/vs/older-downloads/>`_. You may need to sign in to your Microsoft account to be able to download it. During the installation, select custom installation and in the features list make sure you check **Common tools for Visual C++ 2015** under Visual C++ section and  **Python tools for Visual Studio (January 2017)**.
3. We recommend using a virtual environment.

  * In Conda command prompt: ``conda create -n myenv python=3.6``
  * Activate the virtual environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``
5. To install ``explorepy`` from PyPI run: ``pip install explorepy``


Ubuntu
^^^^^^
1. Install Bluetooth header files using ``sudo apt-get install libbluetooth-dev``
2. It is recommended to install Anaconda Python package. Download and install Anaconda Python 3.7 Windows installer from `here <https://www.anaconda.com/distribution/#download-section>`_.
3. We recommend using a virtual environment in Conda.

  * In Conda command prompt: ``conda create -n myenv python=3.6``
  * Activate the virtual environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``

5. To install ``explorepy`` from PyPI run: ``pip install explorepy``

Mac
^^^^
1. Install XCode from Mac App store. An upgrade to the latest version of MacOS might be required for installation of XCode. For older versions of MacOS, you can find compatible versions of XCode in this `page <https://en.wikipedia.org/wiki/Xcode>`_. All old Xcode versions are available `here <https://developer.apple.com/download/more/>`_.
2. Accept the license agreement: ``sudo xcodebuild -license``
3. It is recommended to install Anaconda Python package. Download and install Anaconda Python 3.7 Mac installer from `here <https://www.anaconda.com/distribution/#download-section>`_. For older versions of MacOS, compatible version of Anaconda can be found in `this table <https://docs.continuum.io/anaconda/install/#old-os>`_ and can be downloaded from `here <https://repo.anaconda.com/archive/index.html>`_.
4. We recommend using a virtual environment in Conda.

  * In Conda command prompt: ``conda create -n myenv python=3.6``
  * Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``

6. To install ``explorepy`` from PyPI run: ``pip install explorepy``


Quick test
----------

* Open Conda command prompt

* Activate the virtual environment: ``conda activate myenv``

* ``explorepy visualize -n <YOUR-DEVICE-NAME> -c 4`` (Change the number of channels if needed)

* To stop visualization press Ctrl+c

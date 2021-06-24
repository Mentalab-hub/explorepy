============
Installation
============


Minimal Requirements
------------
* Python 3.6 or newer version
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 6GB RAM (minimum 1GB *free* RAM during the session)
* Intel i5 or higher (2x2.5GHz) CPU


Recommended Requirements
------------
* Python 3.6 or newer version
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 8GB RAM
* Intel i7 or higher CPU


How to install
--------------

Windows
^^^^^^^
On a Windows machine, Explorepy can be installed using the installable file uploaded in
the `release page <https://github.com/Mentalab-hub/explorepy/releases>`_ or using pip. Please note using the installable
files, the dependencies will be installed automatically.

The following instructions guides you to install Explorepy API using pip with all its dependencies on Windows.

1. Install Anaconda Python package (or any other python distribution, but these instructions assume that anaconda is
    installed). Download and install Anaconda Python 3.8 Windows installer from
    `here <https://www.anaconda.com/distribution/#download-section>`_.
2. Install Microsoft Build Tools for Visual Studio 2019 from `here <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_.
    Select **Desktop development with C++** in workloads tab, make sure that  **MSVCv142 - VS 2019 C++ x64/x86 build tools** and latest version of **Windows 10 SDK** are checked.

3. We recommend using a virtual environment.

  * In Conda command prompt: ``conda create -n myenv python=3.7``
  * Activate the virtual environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``
5. To install ``explorepy`` from PyPI run: ``pip install explorepy``


Ubuntu
^^^^^^
1. Install Bluetooth header files using ``sudo apt-get install libbluetooth-dev``
2. It is recommended to install Anaconda Python package. Download and install Anaconda Python 3.7 Ubuntu installer from `here <https://www.anaconda.com/distribution/#download-section>`_.
3. We recommend using a virtual environment in Conda.

  * In Conda command prompt: ``conda create -n myenv python=3.6``
  * Activate the virtual environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``


5. To install ``explorepy`` from PyPI run: ``pip install explorepy``


Mac
^^^
1. Install XCode from Mac App store. An upgrade to the latest version of MacOS might be required for installation of XCode.
For older versions of MacOS, you can find compatible versions of XCode in this `page <https://en.wikipedia.org/wiki/Xcode>`_.
All old Xcode versions are available in `this link <https://developer.apple.com/download/more/>`_.

2. Accept the license agreement: ``sudo xcodebuild -license``

3. It is recommended to install Anaconda Python package. Download and install Anaconda Python 3.7 Mac installer from `here <https://www.anaconda.com/distribution/#download-section>`_. For older versions of MacOS, compatible version of Anaconda can be found in `this table <https://docs.continuum.io/anaconda/install/#old-os>`_ and can be downloaded from `here <https://repo.anaconda.com/archive/index.html>`_.

4. We recommend using a virtual environment in Conda.

  * In Conda command prompt: ``conda create -n myenv python=3.6``
  * Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``

6. To install ``explorepy`` from PyPI run: ``pip install explorepy``


Quick test
----------

* Open Conda command prompt (if you have installed with pip) or Windows command prompt (if you used the installable file)

* Activate the virtual environment (this step is only for Conda command prompt): ``conda activate myenv``

* Run ``explorepy visualize -n <YOUR-DEVICE-NAME> -lf 1 -hf 40``

* To stop visualization press Ctrl+c


Troubleshooting
---------------

**1. Pylsl import issue**

::

        self._handle = _dlopen(self._name, mode)
    OSError: [WinError 126] The specified module could not be found


To fix this problem, an older version of Pylsl can be installed using this command: ::

    pip install pylsl==1.13.1

Alternatively, MS Visual C++ redistributable (vc_redist) can be installed via this `download page <https://support.microsoft.com/en-ca/help/2977003/the-latest-supported-visual-c-downloads>`_.

--------------------


**2. Anaconda asyncio events library raises "NotImplementedError" error in Windows**

::

    File "c:\users\jose\anaconda3\lib\site-packages\bokeh\server\server.py", line 407, in __init__
        http_server.add_sockets(sockets)
    File "c:\users\jose\anaconda3\lib\site-packages\tornado\tcpserver.py", line 165, in add_sockets
        self._handlers[sock.fileno()] = add_accept_handler(
    File "c:\users\jose\anaconda3\lib\site-packages\tornado\netutil.py", line 279, in add_accept_handler
        io_loop.add_handler(sock, accept_handler, IOLoop.READ)
    File "c:\users\jose\anaconda3\lib\site-packages\tornado\platform\asyncio.py", line 100, in add_handler
        self.asyncio_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
    File "c:\users\jose\anaconda3\lib\asyncio\events.py", line 501, in add_reader
        raise NotImplementedError

Solution: Downgrade Anaconda distribution version to one of 3.6 or 3.7 versions.

---------------------

**3. No module named 'numpy'**

::

    ERROR: Command errored out with exit status 4294967295:
     command: 'C:\Users\mh\Anaconda3\envs\test130_38\python.exe' -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\mh_at\\AppData\\Local\\Temp\\pip-install-6fpsl7b1\\pyedflib_e1c2dcc3a4dc46af9468c24083cbda2b\\setup.py'"'"'; __file__='"'"'C:\\Users\\mh_at\\AppData\\Local\\Temp\\pip-install-6fpsl7b1\\pyedflib_e1c2dcc3a4dc46af9468c24083cbda2b\\setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base 'C:\Users\mh_at\AppData\Local\Temp\pip-pip-egg-info-48yn2fu3'
         cwd: C:\Users\mh\AppData\Local\Temp\pip-install-6fpsl7b1\pyedflib_e1c2dcc3a4dc46af9468c24083cbda2b\
    Complete output (4 lines):
    No module named 'numpy'
    *** package "numpy" not found ***
    pyEDFlib requires a version of NumPy, even for setup.
    Please get it from http://numpy.scipy.org/ or install it through your package manager.
    ----------------------------------------
    ERROR: Command errored out with exit status 4294967295: python setup.py egg_info Check the logs for full command output.


Solution: To fix this error, install numpy separately by ``pip install numpy`` and then install explorepy.


**4. OSError: A socket operation was attempted to an unreachable network.**

Solution: Make sure that the device is paired with your computer and try again.

**5. ValueError: Error opening socket.**

Solution: Make sure the Bluetooth module of operating system is on and working.

**6. OSError: The handle is invalid.**

Solution: Make sure the Bluetooth module of operating system is on and working.

**7. DeviceNotFoundError: No device found with the name: Explore_####**

Solution: Make sure the device is on and in advertising mode (blinking in blue at 1Hz). If the Bluetooth module of your
computer is off, you may also get this error.



============
Installation
============

Minimal Requirements
------------
* Python 3.6 to Python 3.10
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 6GB RAM (minimum 1GB *free* RAM during the session)
* Intel i5 or higher (2x2.5GHz) CPU

Recommended Requirements
------------
* Python 3.6 to Python 3.10
* Microsoft Build Tools for Visual Studio 2019 (only Windows)
* 8GB RAM
* Intel i7 or higher CPU

How to install
--------------

Windows
^^^^^^^

Option 1: Installing .exe file (basic)
""""""""

*This option is best for users who only intend to use* ``explorepy`` *from a command-line prompt.*

For example, if you want to quickly visualize and record data, use Option 1.
If you intend to add ``explorepy`` commands to a Python script
(e.g. an experiment script), install ``explorepy`` via Anaconda instead.

For an overview of ``explorepy`` commands, click `here <https://explorepy.readthedocs.io/en/latest/usage.html#command-line-interface>`_.

On a Windows machine, ``explorepy`` can be installed using the .exe installable file uploaded to
``explorepy``'s `release page <https://github.com/Mentalab-hub/explorepy/releases/download/v1.5.0/MentaLab_ExplorePy_1.5.0.exe>`_. Please note that the dependencies will be installed automatically.

Option 2: Installing from Python Package Index (PyPI) and pip (advanced)
""""""""

*This option is best for users who intend to include* ``explorepy`` *functionalities in their own Python scripts.*

To install the ``explorepy`` API and all its dependencies using pip on Windows:

1. Install Anaconda (or any other Python distribution; these instructions pertain to Anaconda only). Download and install the `Anaconda Windows installer <https://www.anaconda.com/distribution/#download-section>`_.
2. Install `Microsoft Build Tools for Visual Studio 2019 <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_. Select *Desktop development with C++* in the workloads tab. Make sure that *MSVCv142 - VS 2019 C++ x64/x86 build tools* and the latest version of *Windows 10 SDK* are checked.
3. Open the Conda command prompt.
4. We recommend using a virtual environment. To do this:

   a. In Conda command prompt: ``conda create -n myenv python=3.8``
   b. Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.

Ubuntu
^^^^^^
1. Install Bluetooth header files using ``sudo apt-get install libbluetooth-dev``
2. We recommend installing Anaconda. Download and install the `Anaconda Python 3.7 Ubuntu installer <https://www.anaconda.com/distribution/#download-section>`_.
3. We recommend using a virtual environment in Conda. To do this:

   a. In Conda command prompt: ``conda create -n myenv python=3.8``
   b. Activate the virtual environment: ``conda activate myenv``

4. Upgrade your pip: ``python -m pip install --upgrade pip``
5. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.

Mac
^^^
Please note that Mac OSX is not supported at the moment due to some bluetooth bug from Apple OS updates.
1. Install ``XCode`` from the Mac App store. For this, you may need to upgrade to the latest version of MacOS. For older versions of MacOS, find compatible versions of ``XCode`` `here <https://en.wikipedia.org/wiki/Xcode>`_. All old ``XCode`` versions are available `here <https://developer.apple.com/download/more/>`_.
2. Accept the license agreement: ``sudo xcodebuild -license``
3. It is best to install Anaconda. Download  and install the `Anaconda Python 3.7 Mac installer <https://www.anaconda.com/distribution/#download-section>`_. For older versions of MacOS, compatible version of Anaconda can be found in `this table <https://docs.continuum.io/anaconda/install/#old-os>`_ and downloaded `here <https://repo.anaconda.com/archive/index.html>`_.
4. We recommend using a virtual environment in Conda.

   a. In Conda command prompt: ``conda create -n myenv python=3.8``
   b. Activate the virtual environment: ``conda activate myenv``

5. Upgrade your pip: ``python -m pip install --upgrade pip``
6. Run: ``pip install explorepy``, to install ``explorepy`` from PyPI.

Quick test
----------

* Open the Conda command prompt (if you used pip) or Windows command prompt (if you used the installable file).
* Activate the virtual environment (this step is only necessary in the Conda command prompt): ``conda activate myenv``
* Run ``explorepy acquire -n DEVICE-NAME``
* To stop the command execution, press ``Ctrl+C``

Troubleshooting
---------------

**1. Pylsl import issue**

::

        self._handle = _dlopen(self._name, mode)
    OSError: [WinError 126] The specified module could not be found


Solution: Install an older version of Pylsl. To do this, run: ::

    pip install pylsl==1.13.1

Alternatively, install `MS Visual C++ redistributable (vc_redist) <https://support.microsoft.com/en-ca/help/2977003/the-latest-supported-visual-c-downloads>`_.

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

Solution: Downgrade your Anaconda distribution to version 3.6 or 3.7.

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


Solution: Install numpy separately using ``pip install numpy`` and then install ``explorepy``.

---------------------

**4. OSError: A socket operation was attempted to an unreachable network.**

Solution: Ensure that your Explore device is paired with your computer and try again.

---------------------

**5. ValueError: Error opening socket.**

Solution: Ensure the Bluetooth module of your computer's operating system is on and working.

---------------------

**6. OSError: The handle is invalid.**

Solution: Ensure the Bluetooth module of your computer's operating system is on and working.

---------------------

If your issue persists, please send a screenshot and brief error description to support@mentalab.com, and we will quickly help you solve it.

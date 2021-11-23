# Explorepy Installer for Windows

This creates an installer for Windows 10, 64 Bit, using the NSIS installer creator.

![Screenshot of PyNSISt Explorepy Installer for Windows](screenshot.png)


## About

The installer is created using PyNSISt, a Python front end to Nullsoft's great & free NSIS installer creator.
It gets almost all software as Python wheels from PyPI, and hopefully soon all of it.
PyNSISt creates installers for Windows, but it can also be run under Linux, which is very handy when creating installers from CI pipelines.


## Building

Tested on Windows 10; should be working on Linux as well.


### Dependencies

#### Software

Install this software on the machine you want to create the installer on.
Tested versions in (brackets).

  - [NSIS - the Nullsoft Scriptable Install System](https://nsis.sourceforge.io/) (3.06.1)
  - [Microsoft Build Tools for Visual Studio 2019](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16)
  - A local python installation with virtual env support (3.7.7)


#### Python wheels

NSIS pulls most software from the net.
At the moment, we still need to build bokeh module locally:


`local-wheels/bokeh-2.2.3-py3-none-any.whl`

  1. Get the bokeh version sdist: https://pypi.org/project/bokeh/2.2.3/#files
  2. Extract it
  3. Run `python setup.py bdist_wheel`.
  4. Copy the resulting file in `dist/` to the local-wheels folder.

This step hopefully will be unneeded when bokeh has wheels online in the Python cheese shop, as most other packages already do.
@hacklschorsch tries to help with that: [See GitHub issue](https://github.com/bokeh/bokeh/issues/10572).



## Do the thing

I successfully used [PyNSISt](https://pynsist.readthedocs.io/) version 2.5.1 for this, but the latest version should be preferable.

In the directory `explorepy\installer\windows`, issue the following commands:


    python3 -m venv venv
    .\venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install pynsist

    pynsist installer.cfg


The last command should have created an executable installer at `build\nsis\MentaLab_ExplorePy_1.1.exe`.

Next, sign the installer so it shows fewer warnings on our customers' machines.
(not yet - FIXME)

Congratulations! You have yourself a fine installer for Windows!  └|∵|┐♪


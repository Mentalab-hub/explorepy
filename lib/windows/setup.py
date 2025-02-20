# Copy this file to the top level folder(where pyproject.toml file resides)
# Then, run pip install command to install explorepy for legacy devices
import os
import sys

from setuptools import (
    Extension,
    setup
)
import numpy

ext_modules_list = []
current_platform = sys.platform
libPath = "lib"
srcPath = "src"

if current_platform == 'win32' or current_platform == 'win64':
    windows_lib_path = os.path.join(libPath, 'windows')
    explorepy_path = os.path.join(srcPath, "explorepy")
    ext_modules_list.append(Extension(
        name='explorepy._exploresdk',
        sources=[os.path.join(windows_lib_path, 'swig_interface_wrap.cxx'),
                 os.path.join(windows_lib_path, 'BluetoothHelpers.cpp'),
                 os.path.join(windows_lib_path, 'DeviceINQ.cpp'),
                 os.path.join(windows_lib_path, 'BTSerialPortBinding.cpp')],
        swig_opts=['-c++']
    ))
    ext_modules_list.append(Extension(
        name='explorepy.int24to32',
        sources=[os.path.join(explorepy_path, 'int24to32.c')],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['/O2']
    ))

elif current_platform.startswith('linux'):
    linux_lib_path = os.path.join(libPath, 'linux')
    explorepy_path = os.path.join(srcPath, "explorepy")
    ext_modules_list.append(Extension(
        name='explorepy._exploresdk',
        sources=[os.path.join(linux_lib_path, 'swig_interface_wrap.cxx'),
                 os.path.join(linux_lib_path, 'DeviceINQ.cpp'),
                 os.path.join(linux_lib_path, 'BTSerialPortBinding.cpp')],
        extra_link_args=["-lbluetooth"],
        swig_opts=['-c++']
    ))
    ext_modules_list.append(Extension(
        name='explorepy.int24to32',
        sources=[os.path.join(explorepy_path, 'int24to32.c')],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3']
    ))
setup_args = dict(
    ext_modules=ext_modules_list,
)

setup(**setup_args)

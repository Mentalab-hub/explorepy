#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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

ext_modules_list.append(Extension(
    name='explorepy.int24to32',
    sources=['src/int24to32.c'],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3']
))

if current_platform == 'win32' or current_platform == 'win64':
    windows_lib_path = os.path.join(libPath, 'windows')
    ext_modules_list.append(Extension(
        name='explorepy._exploresdk',
        sources=[os.path.join(windows_lib_path, 'swig_interface_wrap.cxx'),
                 os.path.join(windows_lib_path, 'BluetoothHelpers.cpp'),
                 os.path.join(windows_lib_path, 'DeviceINQ.cpp'),
                 os.path.join(windows_lib_path, 'BTSerialPortBinding.cpp')],
        swig_opts=['-c++']
    ))

elif current_platform.startswith('linux'):
    linux_lib_path = os.path.join(libPath, 'linux')
    ext_modules_list.append(Extension(
        name='explorepy._exploresdk',
        sources=[os.path.join(linux_lib_path, 'swig_interface_wrap.cxx'),
                 os.path.join(linux_lib_path, 'DeviceINQ.cpp'),
                 os.path.join(linux_lib_path, 'BTSerialPortBinding.cpp')],
        extra_link_args=["-lbluetooth"],
        swig_opts=['-c++']
    ))
setup_args = dict(
    ext_modules=ext_modules_list,
)

setup(**setup_args)

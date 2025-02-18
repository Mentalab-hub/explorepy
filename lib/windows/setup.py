# Copy this file to the top level folder(where pyproject.toml file resides)
# Then, run pip install command to install explorepy for legacy devices
import os
import sys

from setuptools import (
    Extension,
    setup
)

ext_modules_list = []
current_platform = sys.platform
libPath = "lib"
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

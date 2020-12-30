#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
import os
import sys
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from setuptools import Extension
from setuptools.command.develop import develop
from setuptools.command.install import install


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


my_req = ['numpy', 'scipy', 'pyedflib==0.1.15', 'click==7.0', 'appdirs==1.4.3']
ext_modules_list = []
current_platform = sys.platform

if not os.environ.get('READTHEDOCS'):
    my_req.append('pylsl')
    my_req.append('bokeh==2.2.3')

    if current_platform != 'darwin':
        my_req.append('pybluez==0.22')

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
    else:
        if sys.version_info >= (3, 6):
            my_req.append('pyobjc-core>=6')
            my_req.append('pyobjc-framework-Cocoa>=6')
        else:
            my_req.append('pyobjc-core>=3.1,<6')
            my_req.append('pyobjc-framework-Cocoa>=3.1,<6')
        os.system('cp  lib/mac/_exploresdk.so  src/explorepy')
        os.system('cp  lib/mac/btScan  src/explorepy')
        os.system('cp  lib/mac/exploresdk.py  src/explorepy')
setup(
    name='explorepy',
    version='1.3.0',

    license='MIT license',
    description='Python API for Mentalab biosignal aquisition devices',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Mentalab GmbH.',
    author_email='mohamad.atayi@mentalab.com',
    url='https://github.com/Mentalab-hub/explorepy',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    ext_modules=ext_modules_list,
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    keywords=[
        'Mentalab', 'Explorepy', 'EEG signal',
    ],
    install_requires=my_req,
    extras_require={},
    entry_points='''
        [console_scripts]
        explorepy=explorepy.cli:cli
    ''',
)

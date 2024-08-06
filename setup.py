#!/usr/bin/env python
"""
BPMFwFFT
Calculate the binding potential of mean force (BPMF) using the FFT.
"""
import sys
import os
import platform
from setuptools import setup, Extension, find_packages
import versioneer
import numpy as np

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

try:
    from setuptools import setup, Extension, find_packages
    from Cython.Build import cythonize
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Distutils import build_ext

# Detect the system and set appropriate flags
extra_compile_args=["-std=c++17", "-fopenmp"]
extra_link_args = ["-fopenmp"]

# Get the conda prefix (where Anaconda is installed)
conda_prefix = sys.prefix

# Construct the path to Eigen
eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')

if platform.machine() in ['x86_64', 'AMD64']:
    extra_compile_args.extend(["-msse2", "-msse3"])
elif platform.machine().startswith('arm'):
    extra_compile_args.append("-mfpu=neon")

if os.name == 'posix':  # For Linux and macOS
    extra_compile_args.extend(["-fopenmp", "-O3"])
    extra_link_args.append("-fopenmp")
elif os.name == 'nt':   # For Windows
    extra_compile_args.extend(["/openmp", "/O2"])
    extra_link_args.append("/openmp")

metadata = dict(
    name='bpmfwfft',
    author='Trung Hai Nguyen, Jim Tufts',
    author_email='jtufts@hawk.iit.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    zip_safe=False,
)

def extension():
    util_extension = Extension('bpmfwfft.util',
        sources=['bpmfwfft/util.pyx'],
        include_dirs=[np.get_include()],
        language='c',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)
    
    sasa_extension = Extension(
        "bpmfwfft.sasa_wrapper",
        sources=["bpmfwfft/sasa_wrapper.pyx", "bpmfwfft/sasa.cpp"],
        include_dirs=[np.get_include(), "bpmfwfft"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)
    
    charge_extension = Extension(
        "bpmfwfft.charge_grid_wrapper",
        sources=["bpmfwfft/charge_grid_wrapper.pyx", "bpmfwfft/charge_grid.cpp"],
        include_dirs=[np.get_include(), "bpmfwfft", eigen_include],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)
    
    extensions = [util_extension, sasa_extension]
    
    # Apply different language levels
    util_extension = cythonize(util_extension, compiler_directives={'language_level': "2"})[0]
    sasa_extension = cythonize(sasa_extension, compiler_directives={'language_level': "3"})[0]
    charge_extension = cythonize(charge_extension, compiler_directives={'language_level': "3"})[0]
    
    return [util_extension, sasa_extension, charge_extension]

if __name__ == '__main__':
    run_build = True
    if run_build:
        extensions = extension()
        try:
            import Cython as _c
            if _c.__version__ < '0.29':
                raise ImportError("Too old")
        except ImportError as e:
            print('setup depends on Cython (>=0.29). Install it prior invoking setup.py')
            print(e)
            sys.exit(1)
        try:
            import numpy as np
        except ImportError:
            print('setup depends on NumPy. Install it prior invoking setup.py')
            sys.exit(1)
        for e in extensions:
            e.include_dirs.append(np.get_include())
        # Remove the cythonize call here since we've already done it in the extension() function
        metadata['ext_modules'] = extensions
    setup(**metadata)

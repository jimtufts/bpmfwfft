#!/usr/bin/env python

"""
BPMFwFFT
Calculate the binding potential of mean force (BPMF) using the FFT.
"""

#!/usr/bin/env python

import sys
from setuptools import setup, Extension, find_packages
import versioneer

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
    from setuptools import setup
    from setuptools import Extension
    from setuptools import find_packages
    from Cython.Build import cythonize
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy

metadata = \
    dict(name='bpmfwfft',
    author='Trung Hai Nguyen, Jim Tufts',
    author_email='jtufts@hawk.iit.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    ext_modules = cythonize("bpmfwfft/util.pyx", compiler_directives={'language_level' : "2"}),
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    zip_safe=False,
    )

def extension():
    return [
            Extension('bpmfwfft.util',
                sources=['bpmfwfft/util.pyx',],
                include_dirs=[numpy.get_include()],
                language='c'),
        ]

if __name__ == '__main__':
    run_build = True
    if run_build:
        extensions = extension()
        try:
            import Cython as _c
            from Cython.Build import cythonize

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
        metadata['ext_modules'] = cythonize(extensions, language_level=sys.version_info[0])

    setup(**metadata)
# from setuptools import find_packages, setup
# from Cython.Build import cythonize
# import numpy as np

# with open("README.md", 'r') as f:
#     long_description = f.read()

# if __name__ == "__main__":
#     setuptools.setup()

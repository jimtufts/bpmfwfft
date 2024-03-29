try:
    from setuptools import setup
    from setuptools import Extension
    from Cython.Build import cythonize
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy as np

ext_modules = cythonize([Extension("util",["util.pyx"])], compiler_directives={'language_level' : "2"})

setup(
    name= 'util',
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = ext_modules,
    )

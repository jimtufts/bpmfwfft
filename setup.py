import os
import sys
import platform
from setuptools import setup, Extension, find_packages
import versioneer
import numpy as np
import subprocess

from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# Get the conda prefix (where Anaconda is installed)
conda_prefix = sys.prefix
eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')

# Set compiler flags
extra_compile_args = ["-std=c++17", "-O3", "-DNDEBUG", "-DEIGEN_NO_DEBUG"]
extra_link_args = []

# Add OpenMP support if available
if platform.system() != 'Darwin':  # Not macOS
    extra_compile_args.append("-fopenmp")
    extra_link_args.append("-fopenmp")

# Add architecture-specific flags
if platform.system() != 'Darwin':
    extra_compile_args.append("-march=native")

if platform.machine() in ['x86_64', 'AMD64']:
    if platform.system() != 'Darwin':
        extra_compile_args.extend(["-msse2", "-msse3", "-mavx", "-mavx2"])
elif platform.machine().startswith('arm'):
    if platform.system() != 'Darwin':
        extra_compile_args.append("-mfpu=neon")

# CUDA setup
cuda_toolkit_path = os.getenv('CUDA_HOME', '/usr/local/cuda')
cuda_include_dir = os.path.join(cuda_toolkit_path, 'include')
cuda_lib_dir = os.path.join(cuda_toolkit_path, 'lib64')

class CUDA_build_ext(build_ext):
    def run(self):
        nvcc = os.path.join(cuda_toolkit_path, 'bin', 'nvcc')

        if os.path.exists(nvcc):
            # Compile charge grid CUDA library
            charge_grid_lib = os.path.join("bpmfwfft", "libcharge_grid_cuda.so")
            charge_grid_sources = [
                os.path.join("bpmfwfft", "charge_grid_cuda.cu"),
                os.path.join("bpmfwfft", "nnls_cusolver.cu")
            ]

            self._compile_cuda_library(
                "charge grid CUDA library with NNLS",
                charge_grid_lib,
                charge_grid_sources,
                ['-lcublas', '-lcusolver']
            )

            # Compile potential grid CUDA library
            potential_grid_lib = os.path.join("bpmfwfft", "libpotential_grid_cuda.so")
            potential_grid_sources = [
                os.path.join("bpmfwfft", "potential_grid_cuda.cu")
            ]

            self._compile_cuda_library(
                "potential grid CUDA library",
                potential_grid_lib,
                potential_grid_sources,
                []
            )

            # Compile SASA CUDA library
            sasa_lib = os.path.join("bpmfwfft", "libsasa_cuda.so")
            sasa_sources = [
                os.path.join("bpmfwfft", "sasa_cuda.cu")
            ]

            self._compile_cuda_library(
                "SASA CUDA library",
                sasa_lib,
                sasa_sources,
                []
            )
        else:
            print("CUDA toolkit not found. Building without GPU support...")

        super().run()

    def _compile_cuda_library(self, name, cuda_lib, cuda_sources, extra_libs):
        """Helper method to compile a CUDA library"""
        nvcc = os.path.join(cuda_toolkit_path, 'bin', 'nvcc')

        compile_command = [
            nvcc,
            '-shared',
            '-Xcompiler', '-fPIC',
            '-O3',
            '-arch=sm_60',
            '-o', cuda_lib,
        ] + extra_libs + cuda_sources

        # Check if rebuild is needed
        needs_rebuild = not os.path.exists(cuda_lib)
        if not needs_rebuild:
            lib_mtime = os.path.getmtime(cuda_lib)
            for src in cuda_sources:
                if os.path.exists(src) and os.path.getmtime(src) > lib_mtime:
                    needs_rebuild = True
                    break

        if needs_rebuild:
            print(f"Compiling {name}...")
            try:
                subprocess.check_call(compile_command)
                build_lib = os.path.abspath(self.build_lib)
                target_dir = os.path.join(build_lib, 'bpmfwfft')
                os.makedirs(target_dir, exist_ok=True)
                self.copy_file(cuda_lib, target_dir)
            except subprocess.CalledProcessError as e:
                print(f"Warning: {name} compilation failed: {e}")
                print("Continuing without this CUDA component...")

    def build_extension(self, ext):
        # Apply C++ flags only to C++ extensions
        if ext.language == 'c++':
            ext.extra_compile_args = extra_compile_args
            ext.extra_link_args = extra_link_args
        else:
            # For C extensions, use a subset of flags (no -std=c++17)
            c_compile_args = [arg for arg in extra_compile_args if not arg.startswith('-std=c++')]
            ext.extra_compile_args = c_compile_args
            ext.extra_link_args = extra_link_args
        super().build_extension(ext)

def get_extensions():
    extensions = [
        Extension('bpmfwfft.util',
                  sources=['bpmfwfft/util.pyx'],
                  include_dirs=[np.get_include()],
                  language='c'),
        Extension("bpmfwfft.sasa_wrapper",
                  sources=["bpmfwfft/sasa_wrapper.pyx", "bpmfwfft/sasa.cpp"],
                  include_dirs=[np.get_include(), "bpmfwfft", eigen_include],
                  language="c++"),
        Extension("bpmfwfft.charge_grid_wrapper",
                  sources=["bpmfwfft/charge_grid_wrapper.pyx", "bpmfwfft/charge_grid.cpp"],
                  include_dirs=[np.get_include(), "bpmfwfft", eigen_include],
                  language="c++"),
        Extension("bpmfwfft.potential_grid_wrapper",
                  sources=["bpmfwfft/potential_grid_wrapper.pyx",
                           "bpmfwfft/potential_grid.cpp",
                           "bpmfwfft/charge_grid.cpp"],  # Need charge_grid.cpp for helper functions
                  include_dirs=[np.get_include(), "bpmfwfft", eigen_include],
                  language="c++"),
    ]

    # Only add CUDA extension if CUDA is available
    nvcc = os.path.join(cuda_toolkit_path, 'bin', 'nvcc')
    if os.path.exists(nvcc):
        print("CUDA detected, adding GPU extensions...")
        extensions.append(
            Extension("bpmfwfft.charge_grid_cuda_wrapper",
                      sources=["bpmfwfft/charge_grid_cuda_wrapper.pyx",
                               "bpmfwfft/charge_grid_cuda_handler.cpp"],
                      include_dirs=[np.get_include(), cuda_include_dir, "bpmfwfft"],
                      library_dirs=[cuda_lib_dir, "bpmfwfft"],
                      libraries=['cudart', 'cublas', 'charge_grid_cuda'],
                      runtime_library_dirs=["$ORIGIN"],
                      language="c++")
        )
        extensions.append(
            Extension("bpmfwfft.potential_grid_cuda_wrapper",
                      sources=["bpmfwfft/potential_grid_cuda_wrapper.pyx",
                               "bpmfwfft/potential_grid_cuda_handler.cpp"],
                      include_dirs=[np.get_include(), cuda_include_dir, "bpmfwfft"],
                      library_dirs=[cuda_lib_dir, "bpmfwfft"],
                      libraries=['cudart', 'potential_grid_cuda'],
                      runtime_library_dirs=["$ORIGIN"],
                      language="c++")
        )
        extensions.append(
            Extension("bpmfwfft.sasa_cuda_wrapper",
                      sources=["bpmfwfft/sasa_cuda_wrapper.pyx",
                               "bpmfwfft/sasa_cuda_handler.cpp"],
                      include_dirs=[np.get_include(), cuda_include_dir, "bpmfwfft"],
                      library_dirs=[cuda_lib_dir, "bpmfwfft"],
                      libraries=['cudart', 'sasa_cuda'],
                      runtime_library_dirs=["$ORIGIN"],
                      language="c++")
        )
    else:
        print("CUDA not detected, skipping GPU extensions...")

    return cythonize(extensions, compiler_directives={'language_level': "3"})

cmdclass = versioneer.get_cmdclass()
cmdclass.update({'build_ext': CUDA_build_ext})

setup(
    name='bpmfwfft',
    author='Trung Hai Nguyen, Jim Tufts',
    author_email='jtufts@hawk.iit.edu',
    description='Calculate the binding potential of mean force (BPMF) using the FFT.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    setup_requires=['cython>=0.29', 'numpy'],
    ext_modules=get_extensions(),
    zip_safe=False,
)

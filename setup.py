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
extra_compile_args = ["-std=c++17", "-fopenmp", "-O3", "-march=native", "-DNDEBUG", "-DEIGEN_NO_DEBUG"]
extra_link_args = ["-fopenmp"]

if platform.machine() in ['x86_64', 'AMD64']:
    extra_compile_args.extend(["-msse2", "-msse3", "-mavx", "-mavx2"])
elif platform.machine().startswith('arm'):
    extra_compile_args.append("-mfpu=neon")

# CUDA setup
cuda_toolkit_path = os.getenv('CUDA_HOME', '/usr/local/cuda')
cuda_include_dir = os.path.join(cuda_toolkit_path, 'include')
cuda_lib_dir = os.path.join(cuda_toolkit_path, 'lib64')

class CUDA_build_ext(build_ext):
    def run(self):
        cuda_lib = os.path.join("bpmfwfft", "libcharge_grid_cuda.so")
        cuda_source = os.path.join("bpmfwfft", "charge_grid_cuda.cu")

        # Compile CUDA source only if nvcc is available
        nvcc = os.path.join(cuda_toolkit_path, 'bin', 'nvcc')

        if os.path.exists(nvcc):
            compile_command = [
                nvcc,
                '-shared',
                '-Xcompiler', '-fPIC',
                '-O3',
                '-arch=sm_60',  # Adjust this based on your GPU architecture
                '-o', cuda_lib,
                cuda_source
            ]

            if not os.path.exists(cuda_lib) or os.path.getmtime(cuda_source) > os.path.getmtime(cuda_lib):
                print("Compiling CUDA library...")
                try:
                    subprocess.check_call(compile_command)
                    # Ensure the CUDA library is copied to the correct location
                    build_lib = os.path.abspath(self.build_lib)
                    target_dir = os.path.join(build_lib, 'bpmfwfft')
                    os.makedirs(target_dir, exist_ok=True)
                    self.copy_file(cuda_lib, target_dir)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: CUDA compilation failed: {e}")
                    print("Continuing without CUDA support...")
        else:
            print("CUDA toolkit not found. Building without GPU support...")

        super().run()

    def build_extension(self, ext):
        ext.extra_compile_args = extra_compile_args
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
    ]

    # Only add CUDA extension if CUDA is available
    nvcc = os.path.join(cuda_toolkit_path, 'bin', 'nvcc')
    if os.path.exists(nvcc):
        print("CUDA detected, adding GPU extension...")
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
    else:
        print("CUDA not detected, skipping GPU extension...")

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

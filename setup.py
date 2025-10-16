# This setup.py script uses setuptools to build the pybind11 extension.
# It uses the pybind11.setup_helpers.Pybind11Extension which simplifies CMake configuration.

import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# The C++ files to compile
cpp_files = [
    "src/py_engine.cpp",
    "src/engine.cc",
]

# Define the pybind11 extension module
ext_modules = [
    Pybind11Extension(
        # This is the name of the module Python will import: `import py_engine`
        "smolgrad", 
        # Source files required for the compilation
        sources=cpp_files,
        # Standard C++17 is required for std::shared_ptr and the modern STL features
        # used in the C++ code.
        cxx_std=17,
    ),
]

setup(
    name="py-engine",
    version="0.1.0",
    description="Python wrapper for the C++ Value autograd engine.",
    author="Philip Park",
    ext_modules=ext_modules,
    # This ensures that the build process uses the pybind11-aware build extension
    cmdclass={"build_ext": build_ext},
    # The zip_safe option disables the installation of the extension module 
    # as a zipped file, which is necessary for C/C++ extensions.
    zip_safe=False,
)

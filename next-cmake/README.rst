.. highlight:: rst
.. |project_name| replace:: rocBLAS

==============
|project_name|
==============

-----------------
Quick Start Guide
-----------------

This section describes how to configure and build the |project_name| project. We assume the user has a
ROCm installation, Python 3.8 or newer and CMake 3.25.0 or newer.

The |project_name| project consists of three components:

1. host library
2. device libraries
3. client applications

Each component has a corresponding subdirectory. The host and device libraries are independently
configurable and buildable but the client applications require the host library build time and the
device libraries at runtime.

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake fnuctionality with exception of
some project specific options. As such, users are advised to refer to the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options see the options section.

Full build of |project_name|
-----------------------

   .. code-block:: cmake
      :linenos:

      cd rocBLAS/next-cmake
      # configure
      cmake -B build                                       \
            -S .                                           \
            -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
            -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
            -D CMAKE_BUILD_TYPE=Release                    \
            -D CMAKE_PREFIX_PATH=/opt/rocm                 \
            -D GPU_TARGETS=gfx90a
      # build
      cmake --build build --parallel 32

Building device libraries
-------------------------
   .. code-block:: cmake
      :linenos:

      cd rocBLAS/next-cmake
      # configure
      cmake -B build                                       \
            -S .                                           \
            -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
            -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
            -D CMAKE_BUILD_TYPE=Release                    \
            -D CMAKE_PREFIX_PATH=/opt/rocm                 \
            -D GPU_TARGETS=gfx90a                          \
            -D ROCBLAS_ENABLE_DEVICE=ON                    \
            -D ROCBLAS_ENABLE_HOST=OFF                     \
            -D ROCBLAS_ENABLE_CLIENT=OFF
      # build
      cmake --build build --parallel 32

.. tip::
      **For Developers**

      View debugging info by adding ``--log-level=VERBOSE`` to the configure command.


Options
-------

*CMake options*:

* `CMAKE_BUILD_TYPE`: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* `CMAKE_INSTALL_PREFIX`: Base installation directory (defaults to /opt/rocm on Linux, C:/hipSDK on Windows)
* `CMAKE_PREFIX_PATH`: Find package search path (consider setting to ``$ROCM_PATH``)
* `CMAKE_EXPORT_COMPILE_COMMANDS`: Export compile_commands.json for clang tooling support (default: `ON`)

*Project wide options*:

* `ROCBLAS_ENABLE_HOST`: Enables generation of host library (default: `ON`)
* `ROCBLAS_ENABLE_DEVICE`: Enables generation of device libraries (default: `ON`)
* `ROCBLAS_ENABLE_CLIENT`: Enables generation of client applications (default: `ON` if `ROCBLAS_ENABLE_HOST` is `ON`, `OFF` otherwise)
* `ROCBLAS_ENABLE_ASAN`: Build with address sanitizer enabled (default: `OFF`)
* `ROCBLAS_ENABLE_COVERAGE`: Build with gcov support (default: `OFF`)

*Host library options*:

* `ROCBLAS_BUILD_SHARED_LIBS`: Build the |project_name| shared or static library (default: `ON`)
* `ROCBLAS_ENABLE_BLIS`: Enable BLIS support (default: `ON`)
* `ROCBLAS_ENABLE_OPENMP`: Enable OpenMP support (default: `ON`)
* `ROCBLAS_ENABLE_TENSILE`: Build |project_name| host library with Tensile backend (default: `ON`)
* `ROCBLAS_ENABLE_HIPBLASLT`: Build |project_name| host library with hipBLASLt backend (default: `OFF`)
* `ROCBLAS_CONFIG_DIR`: Path placed into ldconfig file (default: `${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}`)

*Device libraries options*:

* `ROCBLAS_TENSILE_LIBRARY_DIR`: Path to tensile library (default: `${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/rocblas` on Linux, `${CPACK_PACKAGING_INSTALL_PREFIX}rocblas/bin` on Windows)

*Client options*:

* `ROCBLAS_ENABLE_BENCHMARKS`: Build benchmark client (default: `ON`)
* `ROCBLAS_ENABLE_TESTS`: Build test client (default: `ON`)
* `ROCBLAS_ENABLE_SAMPLES`: Build client samples (default: `ON`)
* `ROCBLAS_ENABLE_FORTRAN`: Build Fortran clients (default: `OFF`)
* `ROCBLAS_REQUIRE_ROCM_SMI`: Require rocm_smi (default: `ON` on Linux, `OFF` on Windows)

CMake Targets
-------------

* `roc::rocblas`

---------------
Physical Design
---------------

|project_name| consists of three components:

1. host library
2. device libraries built by Tensile
3. client applications

The host library and clients are built by |project_name|, while the device libraries are built by Tensile.
Note that the client applications require the host library to build and the device libraries to run.

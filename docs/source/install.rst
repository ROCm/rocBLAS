
.. toctree::
   :maxdepth: 4
   :caption: Contents:

***********************
Building and Installing
***********************

Prerequisites
=============

-  A ROCm enabled platform, more information `here <https://rocm.github.io/>`_.


Installing pre-built packages
=============================

rocBLAS can be installed on Ubuntu using

::

   sudo apt-get update
   sudo apt-get install rocblas

Once installed, rocBLAS can be used just like any other library with a C API.
The header file will need to be included in the user code in order to make calls
into rocBLAS, and the rocBLAS shared library will become link-time and run-time
dependent for the user applciation.

Building from source
====================

Building from source is not necessary, as rocBLAS can be used after installing the pre-built
packages as described above. If desired, the following instructions can be used to build rocBLAS from source.

Download rocBLAS
----------------

The rocBLAS source code is available at the `rocBLAS github page <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_. Download the master branch of rocBLAS from github using:

::

   git clone -b master https://github.com/ROCmSoftwarePlatform/rocBLAS.git
   cd rocBLAS

Below are steps to build either (dependencies + library) or
(dependencies + library + client). You only need (dependencies +
library) if you call rocBLAS from your code, or if you need to install
rocBLAS for other users. The client contains the test code and examples.

It is recommended that the script install.sh be used to build rocBLAS.
If you need individual commands, they are also given.

Use install.sh to build (library dependencies + library)
--------------------------------------------------------

Common uses of install.sh to build (library dependencies + library) are
in the table below.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
| install.sh command                        | Description              |
+===========================================+==========================+
| ``./install.sh -h``                       | Help information.        |
+-------------------------------------------+--------------------------+
| ``./install.sh -d``                       | Build library            |
|                                           | dependencies and library |
|                                           | in your local directory. |
|                                           | The -d flag only needs   |
|                                           | to be used once. For     |
|                                           | subsequent invocations   |
|                                           | of install.sh it is not  |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``./install.sh``                          | Build library in your    |
|                                           | local directory. It is   |
|                                           | assumed dependencies     |
|                                           | have been built.         |
+-------------------------------------------+--------------------------+
| ``./install.sh -i``                       | Build library, then      |
|                                           | build and install        |
|                                           | rocBLAS package in       |
|                                           | /opt/rocm/rocblas. You   |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | rocBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+

Use install.sh to build (library dependencies + client dependencies + library + client)
---------------------------------------------------------------------------------------

The client contains executables in the table below.

=============== ====================================================
executable name description
=============== ====================================================
rocblas-test    runs Google Tests to test the library
rocblas-bench   executable to benchmark or test individual functions
example-sscal   example C code calling rocblas_sscal function
=============== ====================================================

Common uses of install.sh to build (dependencies + library + client) are
in the table below.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
| install.sh command                        | Description              |
+===========================================+==========================+
| ``./install.sh -h``                       | Help information.        |
+-------------------------------------------+--------------------------+
| ``./install.sh -dc``                      | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | and client in your local |
|                                           | directory. The -d flag   |
|                                           | only needs to be used    |
|                                           | once. For subsequent     |
|                                           | invocations of           |
|                                           | install.sh it is not     |
|                                           | necessary to rebuild the |
|                                           | dependencies.            |
+-------------------------------------------+--------------------------+
| ``./install.sh -c``                       | Build library and client |
|                                           | in your local directory. |
|                                           | It is assumed the        |
|                                           | dependencies have been   |
|                                           | built.                   |
+-------------------------------------------+--------------------------+
| ``./install.sh -idc``                     | Build library            |
|                                           | dependencies, client     |
|                                           | dependencies, library,   |
|                                           | client, then build and   |
|                                           | install the rocBLAS      |
|                                           | package. You will be     |
|                                           | prompted for sudo        |
|                                           | access. It is expected   |
|                                           | that if you want to      |
|                                           | install for all users    |
|                                           | you use the -i flag. If  |
|                                           | you want to keep rocBLAS |
|                                           | in your local directory, |
|                                           | you do not need the -i   |
|                                           | flag.                    |
+-------------------------------------------+--------------------------+
| ``./install.sh -ic``                      | Build and install        |
|                                           | rocBLAS package, and     |
|                                           | build the client. You    |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | rocBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+

Build (library dependencies + library) using individual commands
----------------------------------------------------------------

Before building the library please install the library dependencies
CMake, Python 2.7, Python 3, and Python-yaml.

CMake 3.5 or later
******************

The build infrastructure for rocBLAS is based on
`Cmake <https://cmake.org/>`__ v3.5. This is the version of cmake
available on ROCm supported platforms. If you are on a headless machine
without the x-windows system, we recommend using **ccmake**; if you have
access to X-windows, we recommend using **cmake-gui**.

Install one-liners cmake: \* Ubuntu: ``sudo apt install cmake-qt-gui``
\* Fedora: ``sudo dnf install cmake-gui``

Python
******

By default both python2 and python3 are on Ubuntu.
Python is used in Tensile, and Tensile is part of rocBLAS.
To build rocBLAS both Python 2.7 and Python 3 are needed.

Python-yaml
***********

PyYAML files contain training information from Tensile that is used to
build gemm kernels in rocBLAS.

Install one-liners PyYAML:

* Ubuntu: ``sudo apt install python2.7 python-yaml``

* Fedora: ``sudo dnf install python PyYAML``

Build library
*************

The rocBLAS library contains both host and device code, so the HCC
compiler must be specified during cmake configuration to properly
initialize build tools. Example steps to build rocBLAS:

.. code:: bash

   # after downloading and changing to rocblas directory:
   mkdir -p build/release
   cd build/release
   # Default install path is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other install path
   # Default build config is 'Release', define -DCMAKE_BUILD_TYPE=Debug to specify Debug configuration
   CXX=/opt/rocm/bin/hcc cmake ../..
   make -j$(nproc)
   #if you want to install in /opt/rocm or the directory set in cmake with -DCMAKE_INSTALL_PREFIX
   sudo make install # sudo required if installing into system directory such as /opt/rocm

Build (library dependencies + client dependencies + library + client) using individual commands
-----------------------------------------------------------------------------------------------

Additional dependencies for the rocBLAS clients
***********************************************

The unit tests and benchmarking applications in the client introduce the
following dependencies:

#. `boost <http://www.boost.org/>`__

2. `fortran <http://gcc.gnu.org/wiki/GFortran>`__

3. `lapack <https://github.com/Reference-LAPACK/lapack-release>`__ - lapack itself brings a dependency on a fortran compiler

4. `googletest <https://github.com/google/googletest>`__

boost
`````

Linux distros typically have an easy installation mechanism for boost
through the native package manager.

-  Ubuntu: ``sudo apt install libboost-program-options-dev``
-  Fedora: ``sudo dnf install boost-program-options``

Unfortunately, googletest and lapack are not as easy to install. Many
distros do not provide a googletest package with pre-compiled libraries,
and the lapack packages do not have the necessary cmake config files for
cmake to configure linking the cblas library. rocBLAS provide a cmake
script that builds the above dependencies from source. This is an
optional step; users can provide their own builds of these dependencies
and help cmake find them by setting the CMAKE_PREFIX_PATH definition.
The following is a sequence of steps to build dependencies and install
them to the cmake default /usr/local.

gfortran and lapack
```````````````````

LAPACK is used in the client to test rocBLAS. LAPACK is a Fortran
Library, so gfortran is required for building the client.

\*Ubuntu ``apt-get update``

``apt-get install gfortran``

\*Fedora ``yum install gcc-gfortran``

.. code:: bash

   mkdir -p build/release/deps
   cd build/release/deps
   cmake -DBUILD_BOOST=OFF ../../deps   # assuming boost is installed through package manager as above
   make -j$(nproc) install

Build library and client using individual commands
--------------------------------------------------

Once dependencies are available on the system, it is possible to
configure the clients to build. This requires a few extra cmake flags to
the library cmake configure script. If the dependencies are not
installed into system defaults (like /usr/local ), you should pass the
CMAKE_PREFIX_PATH to cmake to help find them. \*
``-DCMAKE_PREFIX_PATH="<semicolon separated paths>"``

.. code:: bash

   # after downloading and changing to rocblas directory:
   mkdir -p build/release
   cd build/release
   # Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
   CXX=/opt/rocm/bin/hcc cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON ../..
   make -j$(nproc)
   sudo make install   # sudo required if installing into system directory such as /opt/rocm

Use of Tensile
--------------

The rocBLAS library uses
`Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`__, which
supplies the high-performance implementation of xGEMM. Tensile is
downloaded by cmake during library configuration and automatically
configured as part of the build, so no further action is required by the
user to set it up.

Common build problems
---------------------

-  **Issue:** Could not find a configuration file for package “LLVM”
   that is compatible with requested version “7.0”.

   **Solution:** You may have outdated rocBLAS dependencies in
   /usr/local. If you do not have anything other than rocBLAS
   dependencies in /usr/local, then rename /usr/local and re-build
   rocBLAS dependencies by running install.sh with the -d flag. If you
   have other software in /usr/local, then uninstall the rocBLAS
   dependencies, and re-install by running install.sh with the -d flag.

-  **Issue:** “Tensile could not be found because dependency Python
   Interp could not be found”.

   **Solution:** Due to a bug in Tensile, you may need cmake-gui 3.5 and
   above, though in the cmakefiles it requires 2.8.

-  **Issue:** HIP (/opt/rocm/hip) was built using hcc
   1.0.xxx-xxx-xxx-xxx, but you are using /opt/rocm/hcc/hcc with version
   1.0.yyy-yyy-yyy-yyy from hipcc. (version does not match) . Please
   rebuild HIP including cmake or update HCC_HOME variable.

   **Solution:** Download HIP from github and use hcc to `build from
   source <https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md>`__
   and then use the build HIP instead of /opt/rocm/hip one or singly
   overwrite the new build HIP to this location.

-  **Issue:** For MI25 (Vega10 Server) - HCC RUNTIME ERROR: Fail to find
   compatible kernel

   **Solution:** export HCC_AMDGPU_TARGET=gfx900

-  **Issue:** Could not find a package configuration file provided by
   “ROCM” with any of the following names:

   ROCMConfig.cmake

   rocm-config.cmake

   **Solution:** Install `ROCm cmake
   modules <https://github.com/RadeonOpenCompute/rocm-cmake>`__

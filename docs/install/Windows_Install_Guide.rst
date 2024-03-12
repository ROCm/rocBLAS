.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _windows-install:

********************************************************************
Installation and Building for Windows
********************************************************************

=====================================
Prerequisites
=====================================

- An AMD HIP SDK enabled platform. Find more information on the :doc:`System requirements (Windows) <rocm-install-on-windows:reference/system-requirements>` page.
- rocBLAS is supported on the same Windows versions and toolchains that are supported by the HIP SDK.

.. note::
   The AMD HIP SDK is quickly evolving and will have more up-to-date information regarding installing and building for Windows.

============================
Installing Prebuilt Packages
============================

rocBLAS can be installed on Windows 11 or Windows 10 using the AMD HIP SDK installer.

The simplest way to use rocBLAS in your code would be using CMake for which you would add the SDK installation location to your
``CMAKE_PREFIX_PATH`` in your CMake configure step.

.. note::
   You must use quotes as the path contains a space.

::

    -DCMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\5.5"


Then in your ``CMakeLists.txt`` use:

::

    find_package(rocblas)
    target_link_libraries( your_exe PRIVATE roc::rocblas )


Examples of consuming rocBLAS on Windows with CMake can be found at `rocBLAS-Examples github page <https://github.com/ROCm/rocBLAS-Examples>`_.

Once installed, rocBLAS can be used just like any other library with a C API.
The ``rocblas.h`` header file must be included in your code to make calls
into rocBLAS, and the rocBLAS import library and dynamic link library will become respective link-time and run-time
dependencies for your application.

.. note::
   An additional runtime dependency beyond the dynamic link library (``.dll``) file is the entire ``rocblas/``
   subdirectory found in the HIP SDK bin folder. This must be kept in the same directory as the ``rocblas.dll``
   or can be located elsewhere if setting the environment variable ``ROCBLAS_TENSILE_LIBPATH`` to the
   non-standard location. The contents are read at execution time much like additional DLL files.

Once installed, find ``rocblas.h`` in the HIP SDK ``\\include\\rocblas``
directory. Only use these two installed files when needed in user code.
Find other rocBLAS included files in HIP SDK ``\\include\\rocblas\\internal``, however,
do not include these files directly into source code.

===============================
Building and Installing rocBLAS
===============================

For most users, building from source is not necessary, as rocBLAS can be used after installing the prebuilt
packages as described above. If desired, users can use the following instructions to build rocBLAS from source.
The codebase used for rocBLAS for the HIP SDK is the same as used for linux ROCm distribution.
However as these two distributions have different stacks the code and build process have subtle variations.


Requirements
------------

As a rough estimate, 64GB of system memory is required for a full rocBLAS build. This value can be lower if
rocBLAS is built with a different Tensile logic target (see the ``--logic`` command from ``rmake.py --help``). This value
may also increase in the future as more functions are added to rocBLAS and dependencies such as Tensile grow.


Download rocBLAS
----------------

The rocBLAS source code, which is the same as for the ROCm linux distributions, is available at the `rocBLAS github page <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_.
The version of the ROCm HIP SDK may be shown in the path of default installation, but
you can run the HIP SDK compiler to report the verison from the bin/ folder with:

::

    hipcc --version

The HIP version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, HIP version could be 5.4.22880-135e1ab4;
this corresponds to major = 5, minor = 4, patch = 22880, build identifier 135e1ab4.
There are GitHub branches at the rocBLAS site with names release/rocm-rel-major.minor where major and minor are the same as in the HIP version.
For example for you can use the following to download rocBLAS:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCmSoftwarePlatform/rocBLAS.git
   cd rocBLAS

Replace x.y in the above command with the version of HIP SDK installed on your machine. For example, if you have HIP 5.5 installed, then use ``-b release/rocm-rel-5.5``
You can can add the SDK tools to your path with an entry like:

::

   %HIP_PATH%\bin

Building
--------

Below are steps to build using the ``rmake.py`` script. The user can install dependencies and build either:

* dependencies + library

* dependencies + library + client

You only need (dependencies + library) if you call rocBLAS from your code and only want the library built.
The client contains testing and benchmark tools.  ``rmake.py`` will print to the screen the full cmake command being used to configure rocBLAS based on your rmake command line options.
This full ``cmake`` command can be used in your own build scripts if you want to bypass the python helper script for a fixed set of build options.

Library Dependencies
--------------------

Dependencies installed by the python script rdeps.py are listed in the rdeps.xml configuration file. The -d flag passed to rmake.py installs dependencies the same as if
running ``rdeps.py`` directly.
Currently ``rdeps.py`` uses ``vcpkg`` and ``pip`` to install the build dependencies, with ``vcpkg`` being cloned into environment variable ``VCPKG_PATH`` or defaults into ``C:\\github\\vckpg``.
``pip`` will install into your current python3 environment.

The minimum version requirement for CMake is listed in the top level ``CMakeLists.txt`` file. CMake installed with Visual Studio 2022 meets this requirement.
The ``vcpkg`` version tag is specified at the top of the ``rdeps.py`` file.

However, for the test and benchmark clients' host reference BLAS, it is recommended that you manually download and install AMD's ILP64 version of AOCL-BLAS 4.2 from https://www.amd.com/en/developer/aocl.html.
If you download and run the full Windows AOCL installer into the default locations ( `C:\Program Files\AMD\AOCL-Windows\` ) then the AOCL reference BLAS (amd-blis) should be found
by the clients CMakeLists.txt.  

.. note::
   If instead of the AOCL reference library you use OpenBLAS with vcpkg from rdeps.py you may experience `rocblas-test` stress test failures due to 32-bit integer overflow
   on the host reference code unless you exclude the ILP64 stress tests via command line argument `--gtest_filter=-*I64*`.


Build Library dependencies + Library
------------------------------------

Common uses of rmake.py to build (library dependencies + library) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+--------------------+--------------------------+
| Command            | Description              |
+====================+==========================+
| ``./rmake.py -h``  | Help information.        |
+--------------------+--------------------------+
| ``./rmake.py -d``  | Build library            |
|                    | dependencies and library |
|                    | in your local directory. |
|                    | The -d flag only needs   |
|                    | to be used once.         |
+--------------------+--------------------------+
| ``./rmake.py``     | Build library. It is     |
|                    | assumed dependencies     |
|                    | have been built.         |
+--------------------+--------------------------+
| ``./rmake.py -i``  | Build library, then      |
|                    | build and install        |
|                    | rocBLAS package.         |
|                    | If you want to keep      |
|                    | rocBLAS in your local    |
|                    | tree, you do not         |
|                    | need the -i flag.        |
+--------------------+--------------------------+


Build Library Dependencies + Client Dependencies + Library + Client
-------------------------------------------------------------------

Some client executables (.exe) are listed in the table below:

====================== =================================================
executable name        description
====================== =================================================
rocblas-test           runs Google Tests to test the library
rocblas-bench          executable to benchmark or test functions
rocblas-example-sscal  example C code calling rocblas_sscal function
====================== =================================================

Common uses of rmake.py to build (dependencies + library + client) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+------------------------+--------------------------+
| Command                | Description              |
+========================+==========================+
| ``./rmake.py -h``      | Help information.        |
+------------------------+--------------------------+
| ``./rmake.py -dc``     | Build library            |
|                        | dependencies, client     |
|                        | dependencies, library,   |
|                        | and client in your local |
|                        | directory. The d flag    |
|                        | only needs to be used    |
|                        | once. For subsequent     |
|                        | invocations of           |
|                        | rmake.py it is not       |
|                        | necessary to rebuild the |
|                        | dependencies.            |
+------------------------+--------------------------+
| ``./rmake.py -c``      | Build library and client |
|                        | in your local directory. |
|                        | It is assumed the        |
|                        | dependencies have been   |
|                        | installed.               |
+------------------------+--------------------------+
| ``./rmake.py -idc``    | Build library            |
|                        | dependencies, client     |
|                        | dependencies, library,   |
|                        | client, then build and   |
|                        | install the rocBLAS      |
|                        | package. If              |
|                        | you want to keep rocBLAS |
|                        | in your local directory, |
|                        | you do not need the -i   |
|                        | flag.                    |
+------------------------+--------------------------+
| ``./rmake.py -ic``     | Build and install        |
|                        | rocBLAS package, and     |
|                        | build the client.        |
|                        | If you want to keep      |
|                        | rocBLAS in your local    |
|                        | directory, you do not    |
|                        | need the -i flag.        |
+------------------------+--------------------------+

Build Clients without Library
-----------------------------

The rocBLAS clients can be built on their own using ``rmake.py`` with a pre-existing rocBLAS library.

Note that the version of the rocBLAS clients being built should match the version of the installed rocBLAS.
You can determine the version of the installed rocBLAS in the HIP SDK directory from the file ``include\\rocblas\\internal\\rocblas-version.h``.
Find the version of rocBLAS being built if you have grep (e.g. in a git bash) with command ``grep "VERSION_STRING" CMakeLists.txt`` in the rocBLAS directory where you are building the clients.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------+--------------------------+
| Command                             | Description              |
+=====================================+==========================+
| ``./rmake.py --clients-only``       | Build rocBLAS clients    |
|                                     | and use an installed     |
|                                     | rocBLAS library at       |
|                                     | HIP_PATH if no           |
|                                     | --library-path specified |
+-------------------------------------+--------------------------+
| ``./rmake.py --clients-only``       | Build rocBLAS clients    |
| ``--library-path /path/to/rocBLAS`` | and use a rocBLAS        |
|                                     | library at the specified |
|                                     | location.                |
+-------------------------------------+--------------------------+

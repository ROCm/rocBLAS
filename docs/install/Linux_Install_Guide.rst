.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _linux-install:

********************************************************************
Installation and Building for Linux
********************************************************************

Prerequisites
===================================

- A ROCm enabled platform. Find more information on the :doc:`System requirements (Linux) <rocm-install-on-linux:reference/system-requirements>` page.

Installing Prebuilt Packages
===================================

rocBLAS can be installed on Ubuntu(R) or Debian using:

::

   sudo apt-get update
   sudo apt-get install rocblas

rocBLAS can be installed on CentOS using:

::

    sudo yum update
    sudo yum install rocblas

rocBLAS can be installed on SLES using:

::

    sudo dnf upgrade
    sudo dnf install rocblas

rocBLAS can be installed on Fedora using:

::

    sudo dnf install rocblas
    sudo dnf install rocblas-devel

Once installed, rocBLAS can be used just like any other library with a C API.
The ``rocblas.h`` header file must be included in the user code to make calls
into rocBLAS, and the rocBLAS shared library will become link-time and run-time
dependent for the user application.

The header files ``rocblas.h`` and ``rocblas_module.f90`` are installed in ``/opt/rocm/include/rocblas``.
The library file ``librocblas.so`` is installed in ``/opt/rocm/lib``.


Static Library
----------------

Note for non-standard static library builds there is an additional runtime dependency which is the entire subdirectory ``rocblas/`` located in the ``/opt/rocm/lib`` folder.
This runtime folder can be moved elsewhere if setting the environment variable ``ROCBLAS_TENSILE_LIBPATH`` to the new location, or if running an executable
linked against the static library ``librocblas.a`` the same directory as the executable will be searched for the rocblas subdirectory.
The contents of the files in this ``rocblas/`` subdirectory are read at execution time much like shared library files would be.
They contain GPU code objects and their meta-data.


Building and Installing rocBLAS
===================================

For most users, building from source is not necessary, as rocBLAS can be used after installing
the prebuilt packages as described above. However, you can use following instructions to build
rocBLAS from source if necessary.


Requirements
------------

As a rule, 64GB of system memory is required for a full rocBLAS fat binary build. This value can be lower if
rocBLAS is built for specific architectures using the ``-a`` option to ``install.sh``. More information is available
from ``./install.sh --help``.



Download rocBLAS
----------------

The rocBLAS source code is available at the `rocBLAS github page <https://github.com/ROCm/rocBLAS>`_. Check the ROCm version on your system. For Ubuntu(R), use:

::

    apt show rocm-libs -a

For Centos, use:

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example, ROCm version could be 4.0.0.40000-23; this corresponds to major = 4, minor = 0, patch = 0, build identifier 40000-23.
There are GitHub branches at the rocBLAS site with names rocm-major.minor.x where major and minor are the same as in the ROCm version. To download rocBLAS, you can use the following command:

::

   git clone -b release/rocm-rel-x.y https://github.com/ROCm/rocBLAS.git
   cd rocBLAS

Replace x.y in the above command with the version of ROCm installed on your machine. For example, if you have ROCm 6.0 installed, then replace release/rocm-rel-x.y with release/rocm-rel-6.0.


Below are steps to build using ``install.sh`` script. The user can build either:

* dependencies + library

* dependencies + library + client

You only need (dependencies + library) if you call rocBLAS from your code.
The client contains the test and benchmark code.

Library Dependencies
--------------------

CMake has a minimum version requirement listed in the file ``install.sh``. See ``--cmake_install`` flag in ``install.sh`` to upgrade automatically.

Dependencies are listed in the script ``install.sh``. Passing the ``-d`` flag to ``install.sh`` installs the dependencies.

However, for the test and benchmark clients' host reference BLAS, it is recommended that you manually download and install AMD's ILP64 version of AOCL-BLAS 4.2 from https://www.amd.com/en/developer/aocl.html.
If you download and install the full AOCL packages into their default locations, or only download the BLIS archive files and extract into the build directory deps subfolder, then this reference BLAS should be found
by the clients ``CMakeLists.txt``.  Note, if you only use the ``install.sh -d`` dependency script based BLIS download and install, you may experience ``rocblas-test`` stress test failures due to 32-bit integer overflow on the host unless you exclude the stress tests via command line argument ``--gtest_filter=-*stress*``.

Build Library dependencies + Library
------------------------------------

Common uses of ``install.sh`` to build (library dependencies + library) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+----------------------+-----------------------------+
|  Command             | Description                 |
+======================+=============================+
| ``./install.sh -h``  | Help information.           |
+----------------------+-----------------------------+
| ``./install.sh -d``  | Build library               |
|                      | dependencies and library    |
|                      | in your local directory.    |
|                      | The ``-d``` flag only needs |
|                      | to be used once. For        |
|                      | subsequent invocations      |
|                      | of ``install.sh``` it is not|
|                      | necessary to rebuild the    |
|                      | dependencies.               |
+----------------------+-----------------------------+
| ``./install.sh``     | Build library in your       |
|                      | local directory. It is      |
|                      | assumed dependencies        |
|                      | have been built.            |
+----------------------+-----------------------------+
| ``./install.sh -i``  | Build library, then         |
|                      | build and install           |
|                      | rocBLAS package in          |
|                      | ``/opt/rocm/rocblas``. You  |
|                      | will be prompted for        |
|                      | sudo access. This will      |
|                      | install for all users.      |
|                      | If you want to keep         |
|                      | rocBLAS in your local       |
|                      | directory, you do not       |
|                      | need the ``-i`` flag.       |
+----------------------+-----------------------------+


Build Library Dependencies + Client Dependencies + Library + Client
-------------------------------------------------------------------

Some client executables are listed in the table below:

====================== =================================================
executable name        description
====================== =================================================
rocblas-test           runs Google Tests to test the library
rocblas-bench          executable to benchmark or test functions
rocblas-example-sscal  example C code calling rocblas_sscal function
====================== =================================================

Common uses of ``install.sh`` to build (dependencies + library + client) are
in the table below:

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+------------------------+----------------------------+
| Command                | Description                |
+========================+============================+
| ``./install.sh -h``    | Help information.          |
+------------------------+----------------------------+
| ``./install.sh -dc``   | Build library              |
|                        | dependencies, client       |
|                        | dependencies, library,     |
|                        | and client in your local   |
|                        | directory. The ``-d`` flag |
|                        | only needs to be used      |
|                        | once. For subsequent       |
|                        | invocations of             |
|                        | ``install.sh`` it is not   |
|                        | necessary to rebuild the   |
|                        | dependencies.              |
+------------------------+----------------------------+
| ``./install.sh -c``    | Build library and client   |
|                        | in your local directory.   |
|                        | It is assumed the          |
|                        | dependencies have been     |
|                        | built.                     |
+------------------------+----------------------------+
| ``./install.sh -idc``  | Build library              |
|                        | dependencies, client       |
|                        | dependencies, library,     |
|                        | client, then build and     |
|                        | install the rocBLAS        |
|                        | package. You will be       |
|                        | prompted for sudo          |
|                        | access. It is expected     |
|                        | that if you want to        |
|                        | install for all users      |
|                        | you use the ``-i`` flag. If|
|                        | you want to keep rocBLAS   |
|                        | in your local directory,   |
|                        | you do not need the ``-i`` |
|                        | flag.                      |
+------------------------+----------------------------+
| ``./install.sh -ic``   | Build and install          |
|                        | rocBLAS package, and       |
|                        | build the client. You      |
|                        | will be prompted for       |
|                        | sudo access. This will     |
|                        | install for all users.     |
|                        | If you want to keep        |
|                        | rocBLAS in your local      |
|                        | directory, you do not      |
|                        | need the ``-i`` flag.      |
+------------------------+----------------------------+

Build Clients without Library
-----------------------------

The rocBLAS clients can be built on their own using ``install.sh`` with a preexisting rocBLAS library.

Note that the version of the rocBLAS clients being built should match the version of the installed rocBLAS. Find the version of the installed rocBLAS in the installed rocBLAS directory in the file ``include/internal/rocblas-version.h``. Find the version of rocBLAS being built by running ``grep"VERSION_STRING" CMakeLists.txt`` in the rocBLAS directory being built.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------+----------------------------+
| Command                             | Description                |
+=====================================+============================+
| ``./install.sh --clients-only``     | Build rocBLAS clients      |
|                                     | and use an installed       |
|                                     | rocBLAS library at         |
|                                     | ROCM_PATH (``/opt/rocm`` if|
|                                     | not specified).            |
+-------------------------------------+----------------------------+
| ``./install.sh --clients-only``     | Build rocBLAS clients      |
| ``--library-path /path/to/rocBLAS`` | and use a rocBLAS          |
|                                     | library at the specified   |
|                                     | location.                  |
+-------------------------------------+----------------------------+

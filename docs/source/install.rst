***********************
Building and Installing
***********************

Prerequisites
=============

-  A ROCm enabled platform, more information `here <https://rocm.github.io/>`_.

Installing pre-built packages
=============================

rocBLAS can be installed on Ubuntu or Debian using

::

   sudo apt-get update
   sudo apt-get install rocblas

rocBLAS can be installed on CentOS using

::

    sudo yum update
    sudo yum install rocblas

rocBLAS can be installed on SLES using

::

    sudo dnf upgrade
    sudo dnf install rocblas

Once installed, rocBLAS can be used just like any other library with a C API.
The rocblas.h header file will need to be included in the user code in order to make calls
into rocBLAS, and the rocBLAS shared library will become link-time and run-time
dependent for the user applciation.

Once installed, rocblas.h and rocblas_module.f90 can be found in the /opt/rocm/include/rocblas
directory. Only these two installed files should be used when needed in user code.
Other rocBLAS files can be found in /opt/rocm/include/internal, however these files
should not be directly included.


Building from source using install.sh
=====================================

For most users building from source is not necessary, as rocBLAS can be used after installing the pre-built
packages as described above. If desired, the following instructions can be used to build rocBLAS from source.


Requirements
------------

As a general rule, 64GB of system memory is required for a full rocBLAS build. This value can be lower if
rocBLAS is built with a different Tensile logic target (see the --logic command for ./install.sh). This value
may also increase in the future as more functions are added to rocBLAS and dependencies such as Tensile grow.


Download rocBLAS
----------------

The rocBLAS source code is available at the `rocBLAS github page <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_. Check the ROCm Version on your system. For Ubuntu use

::

    apt show rocm-libs -a

For Centos use

::

    yum info rocm-libs

The ROCm version has major, minor, and patch fields, possibly followed by a build specific identifier. For example the ROCm version could be 4.0.0.40000-23, this corresponds to major = 4, minor = 0, patch = 0, build identifier 40000-23. There are GitHub branches at the rocBLAS site with names rocm-major.minor.x where major and minor are the same as in the ROCm version. For ROCm version 4.0.0.40000-23 you need to use the following to download rocBLAS:

::

   git clone -b rocm-4.0.x https://github.com/ROCmSoftwarePlatform/rocBLAS.git
   cd rocBLAS

For ROCm versions with other major and minor fields, clone the branch rocm-major.minor.x in place of rocm-4.0.x.

Below are steps to build either

* dependencies + library

* dependencies + library + client

You only need (dependencies + library) if you call rocBLAS from your code.
The client contains the test and benchmark code.


Build library dependencies + library
------------------------------------

Common uses of install.sh to build (library dependencies + library) are
in the table below.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
|  Command                                  | Description              |
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
|                                           | /opt/rocm. You   |
|                                           | will be prompted for     |
|                                           | sudo access. This will   |
|                                           | install for all users.   |
|                                           | If you want to keep      |
|                                           | rocBLAS in your local    |
|                                           | directory, you do not    |
|                                           | need the -i flag.        |
+-------------------------------------------+--------------------------+


Build library dependencies + client dependencies + library + client
-------------------------------------------------------------------

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
| Command                                   | Description              |
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

Build clients without library
-----------------------------

The rocBLAS clients can be built on their own using install.sh with a preexisting rocBLAS library.

Note that the version of the rocBLAS clients being built should match the version of the installed rocBLAS. The version of the installed rocBLAS can be found in the installed rocBLAS directory, in the file include/rocblas/internal/rocblas-version.h. The version of rocBLAS being built can be found by running ``grep"VERSION_STRING" CMakeLists.txt`` in the rocBLAS directory being built.

.. tabularcolumns::
   |\X{1}{4}|\X{3}{4}|

+-------------------------------------------+--------------------------+
| Command                                   | Description              |
+===========================================+==========================+
| ``./install.sh --clients-only``           | Build rocBLAS clients    |
|                                           | and use an installed     |
|                                           | rocBLAS library at       |
|                                           | ROCM_PATH (/opt/rocm if  |
|                                           | not specified).          |
+-------------------------------------------+--------------------------+
| ``./install.sh --clients-only``           | Build rocBLAS clients    |
|     ``--library-path /path/to/rocBLAS``   | and use a rocBLAS        |
|                                           | library at the specified |
|                                           | location.                |
+-------------------------------------------+--------------------------+

Dependencies
============

Dependencies are listed in the script install.sh. The -d flag to install.sh installs dependencies.

CMake has a minimum version requirement listed in the file install.sh. See --cmake_install flag in install.sh to upgrade automatically.

Use of Tensile
==============

The rocBLAS library uses
`Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`__, which
supplies the high-performance implementation of xGEMM. Tensile is
downloaded by cmake during library configuration and automatically
configured as part of the build, so no further action is required by the
user to set it up.

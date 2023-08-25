=====================
rocBLAS Documentation
=====================

------------
Introduction
------------

This document contains instructions for installing, using, and contributing to rocBLAS.
The quickest way to install is from prebuilt packages. Alternatively, there are instructions to build from source. The document also contains an API Reference Guide, Programmer's Guide, and Contributor's Guide.

Documentation Roadmap
^^^^^^^^^^^^^^^^^^^^^
The following is a list of rocBLAS documents in the suggested reading order:

 - Installation Guide (either `Linux <Linux_Install_Guide.html>`__ or `Windows <Windows_Install_Guide.html>`__): Describes how to install and configure the rocBLAS library; designed to get users up and running quickly with the library
 - `API Reference Guide <API_Reference_Guide.html>`__: Provides detailed information about rocBLAS functions, data types and other programming constructs
 - `Programmer's Guide <Programmers_Guide.html>`__: Describes the code organization, Design implementation detail, Optimizations used in the library, and those that should be considered for new development and Testing & Benchmarking detail
 - `Contributor's Guide <Contributors_Guide.html>`__: Describes coding guidelines for contributors


Use of Tensile
^^^^^^^^^^^^^^

The rocBLAS library internally uses
`Tensile <https://github.com/ROCmSoftwarePlatform/Tensile>`__, which
supplies the high-performance implementation of GEMM. It requires no separate installation as it is installed as part of the rocBLAS package.
rocBLAS uses CMake for build automation, and CMake downloads Tensile during library configuration and automatically
configures it as part of the build, so no further action is required by the
user to set it up.  No external facing API for Tensile is provided.

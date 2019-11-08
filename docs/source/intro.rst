
.. toctree::
   :maxdepth: 4 
   :caption: Contents:

************
Introduction
************

Overview
========

A `BLAS <http://www.netlib.org/blas/>`__ implementation on top of AMD’s Radeon Open Compute `ROCm <https://rocm.github.io/install.html>`__ runtime and toolchains.
rocBLAS is implemented in the `HIP <https://github.com/ROCm-Developer-Tools/HIP>`__ programming language and optimized for AMD’s latest 
discrete GPUs.

======== =========
Acronym  Expansion
======== =========
**BLAS**    **B**\ asic **L**\ inear **A**\ lgebra **S**\ ubprograms
**ROCm**    **R**\ adeon **O**\ pen **C**\ ompute Platfor\ **m**
**HIP**     **H**\ eterogeneous-Compute **I**\ nterface for **P**\ ortability
======== =========

hipBLAS
=======

hipBLAS is a BLAS marshalling library, with multiple supported backends. It sits between the application and a 'worker' BLAS library, marshalling inputs
into the backend library and marshalling results back to the application. hipBLAS exports an interface that does not require the client to change,
regardless of the chosen backend. Currently hipBLAS supports rocBLAS and cuBLAS as backends.

hipBLAS focuses on convenience and portability. If performance outweighs these factors then using rocBLAS itself is recommended.

hipBLAS can be found on github `here <https://github.com/ROCmSoftwarePlatform/hipBLAS/>`__.
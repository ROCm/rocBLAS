.. toctree::
   :maxdepth: 4 
   :caption: Contents:

****
API
****

This section provides details of the library API

Types
=====
Definitions
------



Enums
------
.. doxygenenum:: rocblas_status

.. doxygenenum:: rocblas_datatype

.. doxygenenum:: rocblas_pointer_mode


Functions
=========

Level 1 BLAS
-------------

.. doxygenfunction:: rocblas_sscal

.. doxygenfunction:: rocblas_scopy

.. doxygenfunction:: rocblas_dcopy

rocblas_<type>dot()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdot

.. doxygenfunction:: rocblas_ddot

rocblas_<type>swap()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sswap

.. doxygenfunction:: rocblas_dswap

rocblas_<type>axpy()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_daxpy
    :brief

.. doxygenfunction:: rocblas_saxpy

.. doxygenfunction:: rocblas_haxpy
    :brief


.. doxygengroup:: Group2

Build Information
=================
rocblas_get_version_string()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_version_string
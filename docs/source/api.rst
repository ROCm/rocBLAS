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
rocblas_<type>copy
^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dcopy

.. doxygenfunction:: rocblas_scopy

rocblas_<type>dot()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ddot

.. doxygenfunction:: rocblas_sdot

rocblas_<type>swap()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sswap

.. doxygenfunction:: rocblas_dswap

rocblas_<type>axpy()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_daxpy

.. doxygenfunction:: rocblas_saxpy

.. doxygenfunction:: rocblas_haxpy


.. doxygengroup:: Group2

Build Information
=================
rocblas_get_version_string()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_version_string
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

rocblas_<type>scal()
^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dscal

.. doxygenfunction:: rocblas_sscal
rocblas_<type>copy()
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

rocblas_<type>asum()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dasum

.. doxygenfunction:: rocblas_sasum


rocblas_<type>nrm2()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dnrm2

.. doxygenfunction:: rocblas_snrm2


Build Information
=================
rocblas_get_version_string()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_version_string
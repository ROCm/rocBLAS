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

rocblas_int
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_int

rocblas_long
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_long

rocblas_float_complex
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_float_complex

rocblas_double_complex
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_double_complex

rocblas_half
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_half

rocblas_half_complex
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_half_complex

rocblas_handle
^^^^^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_handle

Enums
------
Enumeration constants have numbering that is consistent with CBLAS, ACML and most standard C BLAS libraries.

rocblas_operation
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_operation

rocblas_fill
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_fill

rocblas_diagonal
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_diagonal

rocblas_side
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_side

rocblas_status
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_status

rocblas_datatype
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_datatype

rocblas_pointer_mode
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_pointer_mode

rocblas_layer_mode
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_layer_mode

rocblas_gemm_algo
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_gemm_algo

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


rocblas_i<type>amax()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_idamax

.. doxygenfunction:: rocblas_isamax

rocblas_i<type>amin()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_idamin

.. doxygenfunction:: rocblas_isamin

Level 2 BLAS
-------------
rocblas_<type>gemv()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dgemv

.. doxygenfunction:: rocblas_sgemv

rocblas_<type>trsv()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dtrsv

.. doxygenfunction:: rocblas_strsv

rocblas_<type>ger()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dger

.. doxygenfunction:: rocblas_sger

rocblas_<type>syr()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dsyr

.. doxygenfunction:: rocblas_ssyr

Level 3 BLAS
-------------
rocblas_<type>trtri_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dtrtri_batched

.. doxygenfunction:: rocblas_strtri_batched

rocblas_<type>trsm()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dtrsm

.. doxygenfunction:: rocblas_strsm

rocblas_<type>gemm()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dgemm

.. doxygenfunction:: rocblas_sgemm

.. doxygenfunction:: rocblas_hgemm

rocblas_<type>gemm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dgemm_strided_batched

.. doxygenfunction:: rocblas_sgemm_strided_batched

.. doxygenfunction:: rocblas_hgemm_strided_batched

rocblas_<type>gemm_kernel_name()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dgemm_kernel_name

.. doxygenfunction:: rocblas_sgemm_kernel_name

.. doxygenfunction:: rocblas_hgemm_kernel_name

rocblas_<type>geam()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_dgeam

.. doxygenfunction:: rocblas_sgeam

BLAS Extensions
---------------
rocblas_gemm_ex()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_gemm_ex

rocblas_gemm_strided_batched_ex()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_gemm_strided_batched_ex

Build Information
-----------------

rocblas_get_version_string()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_version_string

Auxiliary
---------

rocblas_pointer_to_mode()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_pointer_to_mode

rocblas_create_handle()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_create_handle

rocblas_destroy_handle()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_destroy_handle

rocblas_add_stream()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_add_stream

rocblas_set_stream()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_stream

rocblas_get_stream()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_stream

rocblas_set_pointer_mode()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_pointer_mode

rocblas_get_pointer_mode()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_pointer_mode

rocblas_set_vector()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_vector

rocblas_get_vector()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_vector

rocblas_set_matrix()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_matrix

rocblas_get_matrix()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_matrix

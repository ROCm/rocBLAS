.. toctree::
   :maxdepth: 4
   :caption: Contents:
.. _api_label:

****
API
****

This section provides details of the library API

Types
=====
Definitions
-----------

rocblas_int
^^^^^^^^^^^
.. doxygentypedef:: rocblas_int

rocblas_stride
^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_stride

rocblas_half
^^^^^^^^^^^^
.. doxygenstruct:: rocblas_half

rocblas_bfloat16
^^^^^^^^^^^^^^^^
.. doxygenstruct:: rocblas_bfloat16

rocblas_float_complex
^^^^^^^^^^^^^^^^^^^^^
.. doxygenstruct:: rocblas_float_complex

rocblas_double_complex
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenstruct:: rocblas_double_complex

rocblas_handle
^^^^^^^^^^^^^^
.. doxygentypedef:: rocblas_handle

Enums
-----
Enumeration constants have numbering that is consistent with CBLAS, ACML and most standard C BLAS libraries.

rocblas_operation
^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_operation

rocblas_fill
^^^^^^^^^^^^
.. doxygenenum:: rocblas_fill

rocblas_diagonal
^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_diagonal

rocblas_side
^^^^^^^^^^^^
.. doxygenenum:: rocblas_side

rocblas_status
^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_status

rocblas_datatype
^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_datatype

rocblas_pointer_mode
^^^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_pointer_mode

rocblas_layer_mode
^^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_layer_mode

rocblas_gemm_algo
^^^^^^^^^^^^^^^^^
.. doxygenenum:: rocblas_gemm_algo

Functions
=========

Level 1 BLAS
-------------

rocblas_<type>scal()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sscal

.. doxygenfunction:: rocblas_dscal

.. doxygenfunction:: rocblas_cscal

.. doxygenfunction:: rocblas_zscal

.. doxygenfunction:: rocblas_csscal

.. doxygenfunction:: rocblas_zdscal

rocblas_<type>scal_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sscal_batched

.. doxygenfunction:: rocblas_dscal_batched

.. doxygenfunction:: rocblas_cscal_batched

.. doxygenfunction:: rocblas_zscal_batched

.. doxygenfunction:: rocblas_csscal_batched

.. doxygenfunction:: rocblas_zdscal_batched

rocblas_<type>scal_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sscal_strided_batched

.. doxygenfunction:: rocblas_dscal_strided_batched

.. doxygenfunction:: rocblas_cscal_strided_batched

.. doxygenfunction:: rocblas_zscal_strided_batched

.. doxygenfunction:: rocblas_csscal_strided_batched

.. doxygenfunction:: rocblas_zdscal_strided_batched

rocblas_<type>copy()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_scopy

.. doxygenfunction:: rocblas_dcopy

.. doxygenfunction:: rocblas_ccopy

.. doxygenfunction:: rocblas_zcopy

rocblas_<type>copy_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_scopy_batched

.. doxygenfunction:: rocblas_dcopy_batched

.. doxygenfunction:: rocblas_ccopy_batched

.. doxygenfunction:: rocblas_zcopy_batched

rocblas_<type>copy_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_scopy_strided_batched

.. doxygenfunction:: rocblas_dcopy_strided_batched

.. doxygenfunction:: rocblas_ccopy_strided_batched

.. doxygenfunction:: rocblas_zcopy_strided_batched

rocblas_<type>dot()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdot

.. doxygenfunction:: rocblas_ddot

.. doxygenfunction:: rocblas_hdot

.. doxygenfunction:: rocblas_bfdot

.. doxygenfunction:: rocblas_cdotu

.. doxygenfunction:: rocblas_cdotc

.. doxygenfunction:: rocblas_zdotu

.. doxygenfunction:: rocblas_zdotc

rocblas_<type>dot_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdot_batched

.. doxygenfunction:: rocblas_ddot_batched

.. doxygenfunction:: rocblas_hdot_batched

.. doxygenfunction:: rocblas_bfdot_batched

.. doxygenfunction:: rocblas_cdotu_batched

.. doxygenfunction:: rocblas_cdotc_batched

.. doxygenfunction:: rocblas_zdotu_batched

.. doxygenfunction:: rocblas_zdotc_batched

rocblas_<type>dot_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdot_strided_batched

.. doxygenfunction:: rocblas_ddot_strided_batched

.. doxygenfunction:: rocblas_hdot_strided_batched

.. doxygenfunction:: rocblas_bfdot_strided_batched

.. doxygenfunction:: rocblas_cdotu_strided_batched

.. doxygenfunction:: rocblas_cdotc_strided_batched

.. doxygenfunction:: rocblas_zdotu_strided_batched

.. doxygenfunction:: rocblas_zdotc_strided_batched

rocblas_<type>swap()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sswap

.. doxygenfunction:: rocblas_dswap

.. doxygenfunction:: rocblas_cswap

.. doxygenfunction:: rocblas_zswap

rocblas_<type>swap_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sswap_batched

.. doxygenfunction:: rocblas_dswap_batched

.. doxygenfunction:: rocblas_cswap_batched

.. doxygenfunction:: rocblas_zswap_batched

rocblas_<type>swap_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sswap_strided_batched

.. doxygenfunction:: rocblas_dswap_strided_batched

.. doxygenfunction:: rocblas_cswap_strided_batched

.. doxygenfunction:: rocblas_zswap_strided_batched

rocblas_<type>axpy()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_saxpy

.. doxygenfunction:: rocblas_daxpy

.. doxygenfunction:: rocblas_haxpy

.. doxygenfunction:: rocblas_caxpy

.. doxygenfunction:: rocblas_zaxpy

rocblas_<type>asum()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sasum

.. doxygenfunction:: rocblas_dasum

.. doxygenfunction:: rocblas_scasum

.. doxygenfunction:: rocblas_dzasum

rocblas_<type>asum_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sasum_batched

.. doxygenfunction:: rocblas_dasum_batched

.. doxygenfunction:: rocblas_scasum_batched

.. doxygenfunction:: rocblas_dzasum_batched

rocblas_<type>asum_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sasum_strided_batched

.. doxygenfunction:: rocblas_dasum_strided_batched

.. doxygenfunction:: rocblas_scasum_strided_batched

.. doxygenfunction:: rocblas_dzasum_strided_batched


rocblas_<type>nrm2()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_snrm2

.. doxygenfunction:: rocblas_dnrm2

.. doxygenfunction:: rocblas_scnrm2

.. doxygenfunction:: rocblas_dznrm2

rocblas_<type>nrm2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_snrm2_batched

.. doxygenfunction:: rocblas_dnrm2_batched

.. doxygenfunction:: rocblas_scnrm2_batched

.. doxygenfunction:: rocblas_dznrm2_batched

rocblas_<type>nrm2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_snrm2_strided_batched

.. doxygenfunction:: rocblas_dnrm2_strided_batched

.. doxygenfunction:: rocblas_scnrm2_strided_batched

.. doxygenfunction:: rocblas_dznrm2_strided_batched


rocblas_i<type>amax()
^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_isamax

.. doxygenfunction:: rocblas_idamax

.. doxygenfunction:: rocblas_icamax

.. doxygenfunction:: rocblas_izamax

rocblas_i<type>amax_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_isamax_batched

.. doxygenfunction:: rocblas_idamax_batched

.. doxygenfunction:: rocblas_icamax_batched

.. doxygenfunction:: rocblas_izamax_batched

rocblas_i<type>amax_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_isamax_strided_batched

.. doxygenfunction:: rocblas_idamax_strided_batched

.. doxygenfunction:: rocblas_icamax_strided_batched

.. doxygenfunction:: rocblas_izamax_strided_batched


rocblas_i<type>amin()
^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_isamin

.. doxygenfunction:: rocblas_idamin

.. doxygenfunction:: rocblas_icamin

.. doxygenfunction:: rocblas_izamin

rocblas_i<type>amin_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_isamin_batched

.. doxygenfunction:: rocblas_idamin_batched

.. doxygenfunction:: rocblas_icamin_batched

.. doxygenfunction:: rocblas_izamin_batched

rocblas_i<type>amin_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_isamin_strided_batched

.. doxygenfunction:: rocblas_idamin_strided_batched

.. doxygenfunction:: rocblas_icamin_strided_batched

.. doxygenfunction:: rocblas_izamin_strided_batched

rocblas_<type>rot()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srot

.. doxygenfunction:: rocblas_drot

.. doxygenfunction:: rocblas_crot

.. doxygenfunction:: rocblas_csrot

.. doxygenfunction:: rocblas_zrot

.. doxygenfunction:: rocblas_zdrot

rocblas_<type>rot_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srot_batched

.. doxygenfunction:: rocblas_drot_batched

.. doxygenfunction:: rocblas_crot_batched

.. doxygenfunction:: rocblas_csrot_batched

.. doxygenfunction:: rocblas_zrot_batched

.. doxygenfunction:: rocblas_zdrot_batched

rocblas_<type>rot_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srot_strided_batched

.. doxygenfunction:: rocblas_drot_strided_batched

.. doxygenfunction:: rocblas_crot_strided_batched

.. doxygenfunction:: rocblas_csrot_strided_batched

.. doxygenfunction:: rocblas_zrot_strided_batched

.. doxygenfunction:: rocblas_zdrot_strided_batched

rocblas_<type>rotg()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotg

.. doxygenfunction:: rocblas_drotg

.. doxygenfunction:: rocblas_crotg

.. doxygenfunction:: rocblas_zrotg

rocblas_<type>rotg_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotg_batched

.. doxygenfunction:: rocblas_drotg_batched

.. doxygenfunction:: rocblas_crotg_batched

.. doxygenfunction:: rocblas_zrotg_batched

rocblas_<type>rotg_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotg_strided_batched

.. doxygenfunction:: rocblas_drotg_strided_batched

.. doxygenfunction:: rocblas_crotg_strided_batched

.. doxygenfunction:: rocblas_zrotg_strided_batched

rocblas_<type>rotm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotm

.. doxygenfunction:: rocblas_drotm

rocblas_<type>rotm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotm_batched

.. doxygenfunction:: rocblas_drotm_batched

rocblas_<type>rotm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotm_strided_batched

.. doxygenfunction:: rocblas_drotm_strided_batched

rocblas_<type>rotmg()
^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotmg

.. doxygenfunction:: rocblas_drotmg

rocblas_<type>rotmg_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotmg_batched

.. doxygenfunction:: rocblas_drotmg_batched

rocblas_<type>rotmg_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_srotmg_strided_batched

.. doxygenfunction:: rocblas_drotmg_strided_batched

Level 2 BLAS
-------------
rocblas_<type>gemv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemv

.. doxygenfunction:: rocblas_dgemv

.. doxygenfunction:: rocblas_cgemv

.. doxygenfunction:: rocblas_zgemv

rocblas_<type>gemv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemv_batched

.. doxygenfunction:: rocblas_dgemv_batched

.. doxygenfunction:: rocblas_cgemv_batched

.. doxygenfunction:: rocblas_zgemv_batched

rocblas_<type>gemv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemv_strided_batched

.. doxygenfunction:: rocblas_dgemv_strided_batched

.. doxygenfunction:: rocblas_cgemv_strided_batched

.. doxygenfunction:: rocblas_zgemv_strided_batched

rocblas_<type>trsv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsv

.. doxygenfunction:: rocblas_dtrsv

rocblas_<type>trsv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsv_batched

.. doxygenfunction:: rocblas_dtrsv_batched

rocblas_<type>trsv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsv_strided_batched

.. doxygenfunction:: rocblas_dtrsv_strided_batched

rocblas_<type>ger()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sger

.. doxygenfunction:: rocblas_dger

rocblas_<type>ger_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sger_batched

.. doxygenfunction:: rocblas_dger_batched

rocblas_<type>ger_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sger_strided_batched

.. doxygenfunction:: rocblas_dger_strided_batched

rocblas_<type>syr()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr

.. doxygenfunction:: rocblas_dsyr

rocblas_<type>syr_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr_batched

.. doxygenfunction:: rocblas_dsyr_batched

rocblas_<type>syr_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr_strided_batched

.. doxygenfunction:: rocblas_dsyr_strided_batched

Level 3 BLAS
-------------
rocblas_<type>trtri()
^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strtri

.. doxygenfunction:: rocblas_dtrtri

rocblas_<type>trtri_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strtri_batched

.. doxygenfunction:: rocblas_dtrtri_batched

rocblas_<type>trtri_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strtri_strided_batched

.. doxygenfunction:: rocblas_dtrtri_strided_batched

rocblas_<type>trsm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsm

.. doxygenfunction:: rocblas_dtrsm

rocblas_<type>trsm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsm_batched

.. doxygenfunction:: rocblas_dtrsm_batched

rocblas_<type>trsm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsm_strided_batched

.. doxygenfunction:: rocblas_dtrsm_strided_batched

rocblas_<type>trmm()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmm

.. doxygenfunction:: rocblas_dtrmm

rocblas_<type>gemm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemm

.. doxygenfunction:: rocblas_dgemm

.. doxygenfunction:: rocblas_hgemm

.. doxygenfunction:: rocblas_cgemm

.. doxygenfunction:: rocblas_zgemm

rocblas_<type>gemm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemm_batched

.. doxygenfunction:: rocblas_dgemm_batched

.. doxygenfunction:: rocblas_hgemm_batched

.. doxygenfunction:: rocblas_cgemm_batched

.. doxygenfunction:: rocblas_zgemm_batched

rocblas_<type>gemm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemm_strided_batched

.. doxygenfunction:: rocblas_dgemm_strided_batched

.. doxygenfunction:: rocblas_hgemm_strided_batched

.. doxygenfunction:: rocblas_cgemm_strided_batched

.. doxygenfunction:: rocblas_zgemm_strided_batched

rocblas_<type>gemm_kernel_name()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgemm_kernel_name

.. doxygenfunction:: rocblas_dgemm_kernel_name

.. doxygenfunction:: rocblas_hgemm_kernel_name

rocblas_<type>geam()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgeam

.. doxygenfunction:: rocblas_dgeam

BLAS Extensions
---------------
rocblas_gemm_ex()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_gemm_ex

rocblas_gemm_batched_ex()
^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_gemm_batched_ex

rocblas_gemm_strided_batched_ex()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_gemm_strided_batched_ex

rocblas_trsm_ex()
^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_trsm_ex

rocblas_trsm_batched_ex()
^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_trsm_batched_ex

rocblas_trsm_strided_batched_ex()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_trsm_strided_batched_ex

Build Information
-----------------

rocblas_get_version_string()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_version_string

Auxiliary
---------

rocblas_pointer_to_mode()
^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_pointer_to_mode

rocblas_create_handle()
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_create_handle

rocblas_destroy_handle()
^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_destroy_handle

rocblas_add_stream()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_add_stream

rocblas_set_stream()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_stream

rocblas_get_stream()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_stream

rocblas_set_pointer_mode()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_pointer_mode

rocblas_get_pointer_mode()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_pointer_mode

rocblas_set_vector()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_vector

rocblas_set_vector_async()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_vector_async

rocblas_get_vector()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_vector

rocblas_get_vector_async()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_vector_async

rocblas_set_matrix()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_matrix

rocblas_set_matrix_async()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_matrix_async

rocblas_get_matrix()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_matrix

rocblas_get_matrix_async()
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_matrix_async

rocblas_start_device_memory_size_query()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_start_device_memory_size_query

rocblas_stop_device_memory_size_query()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stop_device_memory_size_query

rocblas_get_device_memory_size()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_device_memory_size

rocblas_set_device_memory_size()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_set_device_memory_size


rocblas_is_managing_device_memory()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_is_managing_device_memory

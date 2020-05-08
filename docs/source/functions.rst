.. toctree::
   :maxdepth: 5
   :caption: Contents:
.. _api_label:

*****************
rocBLAS functions
*****************

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


rocblas_<type>axpy()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_saxpy

.. doxygenfunction:: rocblas_daxpy

.. doxygenfunction:: rocblas_haxpy

.. doxygenfunction:: rocblas_caxpy

.. doxygenfunction:: rocblas_zaxpy

rocblas_<type>axpy_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_saxpy_batched

.. doxygenfunction:: rocblas_daxpy_batched

.. doxygenfunction:: rocblas_haxpy_batched

.. doxygenfunction:: rocblas_caxpy_batched

.. doxygenfunction:: rocblas_zaxpy_batched

rocblas_<type>axpy_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_saxpy_strided_batched

.. doxygenfunction:: rocblas_daxpy_strided_batched

.. doxygenfunction:: rocblas_haxpy_strided_batched

.. doxygenfunction:: rocblas_caxpy_strided_batched

.. doxygenfunction:: rocblas_zaxpy_strided_batched


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


Level 2 BLAS
-------------
rocblas_<type>gbmv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgbmv

.. doxygenfunction:: rocblas_dgbmv

.. doxygenfunction:: rocblas_cgbmv

.. doxygenfunction:: rocblas_zgbmv

rocblas_<type>gbmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgbmv_batched

.. doxygenfunction:: rocblas_dgbmv_batched

.. doxygenfunction:: rocblas_cgbmv_batched

.. doxygenfunction:: rocblas_zgbmv_batched

rocblas_<type>gbmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgbmv_strided_batched

.. doxygenfunction:: rocblas_dgbmv_strided_batched

.. doxygenfunction:: rocblas_cgbmv_strided_batched

.. doxygenfunction:: rocblas_zgbmv_strided_batched


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


rocblas_<type>ger()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sger

.. doxygenfunction:: rocblas_dger

.. doxygenfunction:: rocblas_cgeru

.. doxygenfunction:: rocblas_zgeru

.. doxygenfunction:: rocblas_cgerc

.. doxygenfunction:: rocblas_zgerc

rocblas_<type>ger_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sger_batched

.. doxygenfunction:: rocblas_dger_batched

.. doxygenfunction:: rocblas_cgeru_batched

.. doxygenfunction:: rocblas_zgeru_batched

.. doxygenfunction:: rocblas_cgerc_batched

.. doxygenfunction:: rocblas_zgerc_batched

rocblas_<type>ger_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sger_strided_batched

.. doxygenfunction:: rocblas_dger_strided_batched

.. doxygenfunction:: rocblas_cgeru_strided_batched

.. doxygenfunction:: rocblas_zgeru_strided_batched

.. doxygenfunction:: rocblas_cgerc_strided_batched

.. doxygenfunction:: rocblas_zgerc_strided_batched


rocblas_<type>sbmv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssbmv

.. doxygenfunction:: rocblas_dsbmv

rocblas_<type>sbmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssbmv_batched

.. doxygenfunction:: rocblas_dsbmv_batched

rocblas_<type>sbmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssbmv_strided_batched

.. doxygenfunction:: rocblas_dsbmv_strided_batched


rocblas_<type>spmv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspmv

.. doxygenfunction:: rocblas_dspmv

rocblas_<type>spmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspmv_batched

.. doxygenfunction:: rocblas_dspmv_batched

rocblas_<type>spmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspmv_strided_batched

.. doxygenfunction:: rocblas_dspmv_strided_batched


rocblas_<type>spr()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspr

.. doxygenfunction:: rocblas_dspr

.. doxygenfunction:: rocblas_cspr

.. doxygenfunction:: rocblas_zspr

rocblas_<type>spr_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspr_batched

.. doxygenfunction:: rocblas_dspr_batched

.. doxygenfunction:: rocblas_cspr_batched

.. doxygenfunction:: rocblas_zspr_batched

rocblas_<type>spr_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspr_strided_batched

.. doxygenfunction:: rocblas_dspr_strided_batched

.. doxygenfunction:: rocblas_cspr_strided_batched

.. doxygenfunction:: rocblas_zspr_strided_batched


rocblas_<type>spr2()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspr2

.. doxygenfunction:: rocblas_dspr2

rocblas_<type>spr2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspr2_batched

.. doxygenfunction:: rocblas_dspr2_batched

rocblas_<type>spr2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sspr2_strided_batched

.. doxygenfunction:: rocblas_dspr2_strided_batched


rocblas_<type>symv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssymv

.. doxygenfunction:: rocblas_dsymv

.. doxygenfunction:: rocblas_csymv

.. doxygenfunction:: rocblas_zsymv

rocblas_<type>symv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssymv_batched

.. doxygenfunction:: rocblas_dsymv_batched

.. doxygenfunction:: rocblas_csymv_batched

.. doxygenfunction:: rocblas_zsymv_batched

rocblas_<type>symv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssymv_strided_batched

.. doxygenfunction:: rocblas_dsymv_strided_batched

.. doxygenfunction:: rocblas_csymv_strided_batched

.. doxygenfunction:: rocblas_zsymv_strided_batched


rocblas_<type>syr()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr

.. doxygenfunction:: rocblas_dsyr

.. doxygenfunction:: rocblas_csyr

.. doxygenfunction:: rocblas_zsyr

rocblas_<type>syr_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr_batched

.. doxygenfunction:: rocblas_dsyr_batched

.. doxygenfunction:: rocblas_csyr_batched

.. doxygenfunction:: rocblas_zsyr_batched

rocblas_<type>syr_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr_strided_batched

.. doxygenfunction:: rocblas_dsyr_strided_batched

.. doxygenfunction:: rocblas_csyr_strided_batched

.. doxygenfunction:: rocblas_zsyr_strided_batched


rocblas_<type>syr2()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr2

.. doxygenfunction:: rocblas_dsyr2

.. doxygenfunction:: rocblas_csyr2

.. doxygenfunction:: rocblas_zsyr2

rocblas_<type>syr2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr2_batched

.. doxygenfunction:: rocblas_dsyr2_batched

.. doxygenfunction:: rocblas_csyr2_batched

.. doxygenfunction:: rocblas_zsyr2_batched

rocblas_<type>syr2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr2_strided_batched

.. doxygenfunction:: rocblas_dsyr2_strided_batched

.. doxygenfunction:: rocblas_csyr2_strided_batched

.. doxygenfunction:: rocblas_zsyr2_strided_batched


rocblas_<type>tbmv()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stbmv

.. doxygenfunction:: rocblas_dtbmv

.. doxygenfunction:: rocblas_ctbmv

.. doxygenfunction:: rocblas_ztbmv

rocblas_<type>tbmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stbmv_batched

.. doxygenfunction:: rocblas_dtbmv_batched

.. doxygenfunction:: rocblas_ctbmv_batched

.. doxygenfunction:: rocblas_ztbmv_batched

rocblas_<type>tbmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stbmv_strided_batched

.. doxygenfunction:: rocblas_dtbmv_strided_batched

.. doxygenfunction:: rocblas_ctbmv_strided_batched

.. doxygenfunction:: rocblas_ztbmv_strided_batched


rocblas_<type>tbsv()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stbsv

.. doxygenfunction:: rocblas_dtbsv

.. doxygenfunction:: rocblas_ctbsv

.. doxygenfunction:: rocblas_ztbsv

rocblas_<type>tbsv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stbsv_batched

.. doxygenfunction:: rocblas_dtbsv_batched

.. doxygenfunction:: rocblas_ctbsv_batched

.. doxygenfunction:: rocblas_ztbsv_batched

rocblas_<type>tbsv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stbsv_strided_batched

.. doxygenfunction:: rocblas_dtbsv_strided_batched

.. doxygenfunction:: rocblas_ctbsv_strided_batched

.. doxygenfunction:: rocblas_ztbsv_strided_batched


rocblas_<type>tpmv()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stpmv

.. doxygenfunction:: rocblas_dtpmv

.. doxygenfunction:: rocblas_ctpmv

.. doxygenfunction:: rocblas_ztpmv

rocblas_<type>tpmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stpmv_batched

.. doxygenfunction:: rocblas_dtpmv_batched

.. doxygenfunction:: rocblas_ctpmv_batched

.. doxygenfunction:: rocblas_ztpmv_batched

rocblas_<type>tpmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stpmv_strided_batched

.. doxygenfunction:: rocblas_dtpmv_strided_batched

.. doxygenfunction:: rocblas_ctpmv_strided_batched

.. doxygenfunction:: rocblas_ztpmv_strided_batched


rocblas_<type>tpsv()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stpsv

.. doxygenfunction:: rocblas_dtpsv

.. doxygenfunction:: rocblas_ctpsv

.. doxygenfunction:: rocblas_ztpsv

rocblas_<type>tpsv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stpsv_batched

.. doxygenfunction:: rocblas_dtpsv_batched

.. doxygenfunction:: rocblas_ctpsv_batched

.. doxygenfunction:: rocblas_ztpsv_batched

rocblas_<type>tpsv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_stpsv_strided_batched

.. doxygenfunction:: rocblas_dtpsv_strided_batched

.. doxygenfunction:: rocblas_ctpsv_strided_batched

.. doxygenfunction:: rocblas_ztpsv_strided_batched


rocblas_<type>trmv()
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmv

.. doxygenfunction:: rocblas_dtrmv

.. doxygenfunction:: rocblas_ctrmv

.. doxygenfunction:: rocblas_ztrmv

rocblas_<type>trmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmv_batched

.. doxygenfunction:: rocblas_dtrmv_batched

.. doxygenfunction:: rocblas_ctrmv_batched

.. doxygenfunction:: rocblas_ztrmv_batched

rocblas_<type>trmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmv_strided_batched

.. doxygenfunction:: rocblas_dtrmv_strided_batched

.. doxygenfunction:: rocblas_ctrmv_strided_batched

.. doxygenfunction:: rocblas_ztrmv_strided_batched


rocblas_<type>trsv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsv

.. doxygenfunction:: rocblas_dtrsv

.. doxygenfunction:: rocblas_ctrsv

.. doxygenfunction:: rocblas_ztrsv

rocblas_<type>trsv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsv_batched

.. doxygenfunction:: rocblas_dtrsv_batched

.. doxygenfunction:: rocblas_ctrsv_batched

.. doxygenfunction:: rocblas_ztrsv_batched

rocblas_<type>trsv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsv_strided_batched

.. doxygenfunction:: rocblas_dtrsv_strided_batched

.. doxygenfunction:: rocblas_ctrsv_strided_batched

.. doxygenfunction:: rocblas_ztrsv_strided_batched


rocblas_<type>hemv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chemv

.. doxygenfunction:: rocblas_zhemv

rocblas_<type>hemv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chemv_batched

.. doxygenfunction:: rocblas_zhemv_batched

rocblas_<type>hemv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chemv_strided_batched

.. doxygenfunction:: rocblas_zhemv_strided_batched


rocblas_<type>hbmv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chbmv

.. doxygenfunction:: rocblas_zhbmv

rocblas_<type>hbmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chbmv_batched

.. doxygenfunction:: rocblas_zhbmv_batched

rocblas_<type>hbmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chbmv_strided_batched

.. doxygenfunction:: rocblas_zhbmv_strided_batched


rocblas_<type>hpmv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpmv

.. doxygenfunction:: rocblas_zhpmv

rocblas_<type>hpmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpmv_batched

.. doxygenfunction:: rocblas_zhpmv_batched

rocblas_<type>hpmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpmv_strided_batched

.. doxygenfunction:: rocblas_zhpmv_strided_batched


rocblas_<type>hpmv()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpmv

.. doxygenfunction:: rocblas_zhpmv

rocblas_<type>hpmv_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpmv_batched

.. doxygenfunction:: rocblas_zhpmv_batched

rocblas_<type>hpmv_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpmv_strided_batched

.. doxygenfunction:: rocblas_zhpmv_strided_batched


rocblas_<type>her()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher

.. doxygenfunction:: rocblas_zher

rocblas_<type>her_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher_batched

.. doxygenfunction:: rocblas_zher_batched

rocblas_<type>her_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher_strided_batched

.. doxygenfunction:: rocblas_zher_strided_batched


rocblas_<type>her2()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher2

.. doxygenfunction:: rocblas_zher2

rocblas_<type>her2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher2_batched

.. doxygenfunction:: rocblas_zher2_batched

rocblas_<type>her2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher2_strided_batched

.. doxygenfunction:: rocblas_zher2_strided_batched


rocblas_<type>hpr()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpr

.. doxygenfunction:: rocblas_zhpr

rocblas_<type>hpr_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpr_batched

.. doxygenfunction:: rocblas_zhpr_batched

rocblas_<type>hpr_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpr_strided_batched

.. doxygenfunction:: rocblas_zhpr_strided_batched


rocblas_<type>hpr2()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpr2

.. doxygenfunction:: rocblas_zhpr2

rocblas_<type>hpr2_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpr2_batched

.. doxygenfunction:: rocblas_zhpr2_batched

rocblas_<type>hpr2_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chpr2_strided_batched

.. doxygenfunction:: rocblas_zhpr2_strided_batched



Level 3 BLAS
-------------

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


rocblas_<type>symm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssymm

.. doxygenfunction:: rocblas_dsymm

.. doxygenfunction:: rocblas_csymm

.. doxygenfunction:: rocblas_zsymm

rocblas_<type>symm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssymm_batched

.. doxygenfunction:: rocblas_dsymm_batched

.. doxygenfunction:: rocblas_csymm_batched

.. doxygenfunction:: rocblas_zsymm_batched

rocblas_<type>symm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssymm_strided_batched

.. doxygenfunction:: rocblas_dsymm_strided_batched

.. doxygenfunction:: rocblas_csymm_strided_batched

.. doxygenfunction:: rocblas_zsymm_strided_batched


rocblas_<type>syrk()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyrk

.. doxygenfunction:: rocblas_dsyrk

.. doxygenfunction:: rocblas_csyrk

.. doxygenfunction:: rocblas_zsyrk

rocblas_<type>syrk_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyrk_batched

.. doxygenfunction:: rocblas_dsyrk_batched

.. doxygenfunction:: rocblas_csyrk_batched

.. doxygenfunction:: rocblas_zsyrk_batched

rocblas_<type>syrk_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyrk_strided_batched

.. doxygenfunction:: rocblas_dsyrk_strided_batched

.. doxygenfunction:: rocblas_csyrk_strided_batched

.. doxygenfunction:: rocblas_zsyrk_strided_batched


rocblas_<type>syr2k()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr2k

.. doxygenfunction:: rocblas_dsyr2k

.. doxygenfunction:: rocblas_csyr2k

.. doxygenfunction:: rocblas_zsyr2k

rocblas_<type>syr2k_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr2k_batched

.. doxygenfunction:: rocblas_dsyr2k_batched

.. doxygenfunction:: rocblas_csyr2k_batched

.. doxygenfunction:: rocblas_zsyr2k_batched

rocblas_<type>syr2k_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyr2k_strided_batched

.. doxygenfunction:: rocblas_dsyr2k_strided_batched

.. doxygenfunction:: rocblas_csyr2k_strided_batched

.. doxygenfunction:: rocblas_zsyr2k_strided_batched


rocblas_<type>syrkx()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyrkx

.. doxygenfunction:: rocblas_dsyrkx

.. doxygenfunction:: rocblas_csyrkx

.. doxygenfunction:: rocblas_zsyrkx

rocblas_<type>syrkx_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyrkx_batched

.. doxygenfunction:: rocblas_dsyrkx_batched

.. doxygenfunction:: rocblas_csyrkx_batched

.. doxygenfunction:: rocblas_zsyrkx_batched

rocblas_<type>syrkx_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_ssyrkx_strided_batched

.. doxygenfunction:: rocblas_dsyrkx_strided_batched

.. doxygenfunction:: rocblas_csyrkx_strided_batched

.. doxygenfunction:: rocblas_zsyrkx_strided_batched


rocblas_<type>trmm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmm

.. doxygenfunction:: rocblas_dtrmm

.. doxygenfunction:: rocblas_ctrmm

.. doxygenfunction:: rocblas_ztrmm

rocblas_<type>trmm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmm_batched

.. doxygenfunction:: rocblas_dtrmm_batched

.. doxygenfunction:: rocblas_ctrmm_batched

.. doxygenfunction:: rocblas_ztrmm_batched

rocblas_<type>trmm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strmm_strided_batched

.. doxygenfunction:: rocblas_dtrmm_strided_batched

.. doxygenfunction:: rocblas_ctrmm_strided_batched

.. doxygenfunction:: rocblas_ztrmm_strided_batched


rocblas_<type>trsm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsm

.. doxygenfunction:: rocblas_dtrsm

.. doxygenfunction:: rocblas_ctrsm

.. doxygenfunction:: rocblas_ztrsm

rocblas_<type>trsm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsm_batched

.. doxygenfunction:: rocblas_dtrsm_batched

.. doxygenfunction:: rocblas_ctrsm_batched

.. doxygenfunction:: rocblas_ztrsm_batched

rocblas_<type>trsm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_strsm_strided_batched

.. doxygenfunction:: rocblas_dtrsm_strided_batched

.. doxygenfunction:: rocblas_ctrsm_strided_batched

.. doxygenfunction:: rocblas_ztrsm_strided_batched


rocblas_<type>hemm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chemm

.. doxygenfunction:: rocblas_zhemm

rocblas_<type>hemm_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chemm_batched

.. doxygenfunction:: rocblas_zhemm_batched

rocblas_<type>hemm_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_chemm_strided_batched

.. doxygenfunction:: rocblas_zhemm_strided_batched


rocblas_<type>herk()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cherk

.. doxygenfunction:: rocblas_zherk

rocblas_<type>herk_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cherk_batched

.. doxygenfunction:: rocblas_zherk_batched

rocblas_<type>herk_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cherk_strided_batched

.. doxygenfunction:: rocblas_zherk_strided_batched


rocblas_<type>her2k()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher2k

.. doxygenfunction:: rocblas_zher2k

rocblas_<type>her2k_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher2k_batched

.. doxygenfunction:: rocblas_zher2k_batched

rocblas_<type>her2k_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cher2k_strided_batched

.. doxygenfunction:: rocblas_zher2k_strided_batched


rocblas_<type>herkx()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cherkx

.. doxygenfunction:: rocblas_zherkx

rocblas_<type>herkx_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cherkx_batched

.. doxygenfunction:: rocblas_zherkx_batched

rocblas_<type>herkx_strided_batched()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_cherkx_strided_batched

.. doxygenfunction:: rocblas_zherkx_strided_batched


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

rocblas_<type>geam()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgeam

.. doxygenfunction:: rocblas_dgeam

.. doxygenfunction:: rocblas_cgeam

.. doxygenfunction:: rocblas_zgeam

rocblas_<type>geam()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgeam_batched

.. doxygenfunction:: rocblas_dgeam_batched

.. doxygenfunction:: rocblas_cgeam_batched

.. doxygenfunction:: rocblas_zgeam_batched

rocblas_<type>geam()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sgeam_strided_batched

.. doxygenfunction:: rocblas_dgeam_strided_batched

.. doxygenfunction:: rocblas_cgeam_strided_batched

.. doxygenfunction:: rocblas_zgeam_strided_batched


rocblas_<type>dgmm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdgmm

.. doxygenfunction:: rocblas_ddgmm

.. doxygenfunction:: rocblas_cdgmm

.. doxygenfunction:: rocblas_zdgmm

rocblas_<type>dgmm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdgmm_batched

.. doxygenfunction:: rocblas_ddgmm_batched

.. doxygenfunction:: rocblas_cdgmm_batched

.. doxygenfunction:: rocblas_zdgmm_batched

rocblas_<type>dgmm()
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_sdgmm_strided_batched

.. doxygenfunction:: rocblas_ddgmm_strided_batched

.. doxygenfunction:: rocblas_cdgmm_strided_batched

.. doxygenfunction:: rocblas_zdgmm_strided_batched


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


Device Memory functions
-----------------------

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


Build Information
-----------------

rocblas_get_version_string()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: rocblas_get_version_string


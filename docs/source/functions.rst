.. _api_label:

*************
rocBLAS Types
*************

Definitions
===========

rocblas_int
-----------
.. doxygentypedef:: rocblas_int

rocblas_stride
--------------
.. doxygentypedef:: rocblas_stride

rocblas_half
------------
.. doxygenstruct:: rocblas_half

rocblas_bfloat16
----------------
.. doxygenstruct:: rocblas_bfloat16

rocblas_float_complex
---------------------
.. doxygenstruct:: rocblas_float_complex

rocblas_double_complex
-----------------------
.. doxygenstruct:: rocblas_double_complex

rocblas_handle
--------------
.. doxygentypedef:: rocblas_handle

Enums
=====
Enumeration constants have numbering that is consistent with CBLAS, ACML and most standard C BLAS libraries.

rocblas_operation
-----------------
.. doxygenenum:: rocblas_operation

rocblas_fill
------------
.. doxygenenum:: rocblas_fill

rocblas_diagonal
----------------
.. doxygenenum:: rocblas_diagonal

rocblas_side
------------
.. doxygenenum:: rocblas_side

rocblas_status
--------------
.. doxygenenum:: rocblas_status

rocblas_datatype
----------------
.. doxygenenum:: rocblas_datatype

rocblas_pointer_mode
--------------------
.. doxygenenum:: rocblas_pointer_mode

rocblas_atomics_mode
--------------------
.. doxygenenum:: rocblas_atomics_mode

rocblas_layer_mode
------------------
.. doxygenenum:: rocblas_layer_mode

rocblas_gemm_algo
-----------------
.. doxygenenum:: rocblas_gemm_algo

rocblas_gemm_flags
-----------------
.. doxygenenum:: rocblas_gemm_flags

*****************
rocBLAS Functions
*****************

Level 1 BLAS
============

rocblas_iXamax + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_isamax
   :outline:
.. doxygenfunction:: rocblas_idamax
   :outline:
.. doxygenfunction:: rocblas_icamax
   :outline:
.. doxygenfunction:: rocblas_izamax

.. doxygenfunction:: rocblas_isamax_batched
   :outline:
.. doxygenfunction:: rocblas_idamax_batched
   :outline:
.. doxygenfunction:: rocblas_icamax_batched
   :outline:
.. doxygenfunction:: rocblas_izamax_batched

.. doxygenfunction:: rocblas_isamax_strided_batched
   :outline:
.. doxygenfunction:: rocblas_idamax_strided_batched
   :outline:
.. doxygenfunction:: rocblas_icamax_strided_batched
   :outline:
.. doxygenfunction:: rocblas_izamax_strided_batched


rocblas_iXamin + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_isamin
   :outline:
.. doxygenfunction:: rocblas_idamin
   :outline:
.. doxygenfunction:: rocblas_icamin
   :outline:
.. doxygenfunction:: rocblas_izamin

.. doxygenfunction:: rocblas_isamin_batched
   :outline:
.. doxygenfunction:: rocblas_idamin_batched
   :outline:
.. doxygenfunction:: rocblas_icamin_batched
   :outline:
.. doxygenfunction:: rocblas_izamin_batched

.. doxygenfunction:: rocblas_isamin_strided_batched
   :outline:
.. doxygenfunction:: rocblas_idamin_strided_batched
   :outline:
.. doxygenfunction:: rocblas_icamin_strided_batched
   :outline:
.. doxygenfunction:: rocblas_izamin_strided_batched

rocblas_Xasum + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sasum
   :outline:
.. doxygenfunction:: rocblas_dasum
   :outline:
.. doxygenfunction:: rocblas_scasum
   :outline:
.. doxygenfunction:: rocblas_dzasum

.. doxygenfunction:: rocblas_sasum_batched
   :outline:
.. doxygenfunction:: rocblas_dasum_batched
   :outline:
.. doxygenfunction:: rocblas_scasum_batched
   :outline:
.. doxygenfunction:: rocblas_dzasum_batched

.. doxygenfunction:: rocblas_sasum_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dasum_strided_batched
   :outline:
.. doxygenfunction:: rocblas_scasum_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dzasum_strided_batched

rocblas_Xaxpy + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_saxpy
   :outline:
.. doxygenfunction:: rocblas_daxpy
   :outline:
.. doxygenfunction:: rocblas_haxpy
   :outline:
.. doxygenfunction:: rocblas_caxpy
   :outline:
.. doxygenfunction:: rocblas_zaxpy

.. doxygenfunction:: rocblas_saxpy_batched
   :outline:
.. doxygenfunction:: rocblas_daxpy_batched
   :outline:
.. doxygenfunction:: rocblas_haxpy_batched
   :outline:
.. doxygenfunction:: rocblas_caxpy_batched
   :outline:
.. doxygenfunction:: rocblas_zaxpy_batched

.. doxygenfunction:: rocblas_saxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_daxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_haxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_caxpy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zaxpy_strided_batched

rocblas_Xcopy + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_scopy
   :outline:
.. doxygenfunction:: rocblas_dcopy
   :outline:
.. doxygenfunction:: rocblas_ccopy
   :outline:
.. doxygenfunction:: rocblas_zcopy

.. doxygenfunction:: rocblas_scopy_batched
   :outline:
.. doxygenfunction:: rocblas_dcopy_batched
   :outline:
.. doxygenfunction:: rocblas_ccopy_batched
   :outline:
.. doxygenfunction:: rocblas_zcopy_batched

.. doxygenfunction:: rocblas_scopy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dcopy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ccopy_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zcopy_strided_batched

rocblas_Xdot + batched, strided_batched
---------------------------------------
.. doxygenfunction:: rocblas_sdot
   :outline:
.. doxygenfunction:: rocblas_ddot
   :outline:
.. doxygenfunction:: rocblas_hdot
   :outline:
.. doxygenfunction:: rocblas_bfdot
   :outline:
.. doxygenfunction:: rocblas_cdotu
   :outline:
.. doxygenfunction:: rocblas_cdotc
   :outline:
.. doxygenfunction:: rocblas_zdotu
   :outline:
.. doxygenfunction:: rocblas_zdotc

.. doxygenfunction:: rocblas_sdot_batched
   :outline:
.. doxygenfunction:: rocblas_ddot_batched
   :outline:
.. doxygenfunction:: rocblas_hdot_batched
   :outline:
.. doxygenfunction:: rocblas_bfdot_batched
   :outline:
.. doxygenfunction:: rocblas_cdotu_batched
   :outline:
.. doxygenfunction:: rocblas_cdotc_batched
   :outline:
.. doxygenfunction:: rocblas_zdotu_batched
   :outline:
.. doxygenfunction:: rocblas_zdotc_batched

.. doxygenfunction:: rocblas_sdot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ddot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_hdot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_bfdot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cdotu_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cdotc_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdotu_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdotc_strided_batched

rocblas_Xnrm2 + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_snrm2
   :outline:
.. doxygenfunction:: rocblas_dnrm2
   :outline:
.. doxygenfunction:: rocblas_scnrm2
   :outline:
.. doxygenfunction:: rocblas_dznrm2

.. doxygenfunction:: rocblas_snrm2_batched
   :outline:
.. doxygenfunction:: rocblas_dnrm2_batched
   :outline:
.. doxygenfunction:: rocblas_scnrm2_batched
   :outline:
.. doxygenfunction:: rocblas_dznrm2_batched

.. doxygenfunction:: rocblas_snrm2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dnrm2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_scnrm2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dznrm2_strided_batched

rocblas_Xrot + batched, strided_batched
---------------------------------------
.. doxygenfunction:: rocblas_srot
   :outline:
.. doxygenfunction:: rocblas_drot
   :outline:
.. doxygenfunction:: rocblas_crot
   :outline:
.. doxygenfunction:: rocblas_csrot
   :outline:
.. doxygenfunction:: rocblas_zrot
   :outline:
.. doxygenfunction:: rocblas_zdrot

.. doxygenfunction:: rocblas_srot_batched
   :outline:
.. doxygenfunction:: rocblas_drot_batched
   :outline:
.. doxygenfunction:: rocblas_crot_batched
   :outline:
.. doxygenfunction:: rocblas_csrot_batched
   :outline:
.. doxygenfunction:: rocblas_zrot_batched
   :outline:
.. doxygenfunction:: rocblas_zdrot_batched

.. doxygenfunction:: rocblas_srot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_crot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csrot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zrot_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdrot_strided_batched

rocblas_Xrotg + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_srotg
   :outline:
.. doxygenfunction:: rocblas_drotg
   :outline:
.. doxygenfunction:: rocblas_crotg
   :outline:
.. doxygenfunction:: rocblas_zrotg

.. doxygenfunction:: rocblas_srotg_batched
   :outline:
.. doxygenfunction:: rocblas_drotg_batched
   :outline:
.. doxygenfunction:: rocblas_crotg_batched
   :outline:
.. doxygenfunction:: rocblas_zrotg_batched

.. doxygenfunction:: rocblas_srotg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drotg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_crotg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zrotg_strided_batched

rocblas_Xrotm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_srotm
   :outline:
.. doxygenfunction:: rocblas_drotm

.. doxygenfunction:: rocblas_srotm_batched
   :outline:
.. doxygenfunction:: rocblas_drotm_batched

.. doxygenfunction:: rocblas_srotm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drotm_strided_batched

rocblas_Xrotmg + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_srotmg
   :outline:
.. doxygenfunction:: rocblas_drotmg

.. doxygenfunction:: rocblas_srotmg_batched
   :outline:
.. doxygenfunction:: rocblas_drotmg_batched

.. doxygenfunction:: rocblas_srotmg_strided_batched
   :outline:
.. doxygenfunction:: rocblas_drotmg_strided_batched

rocblas_Xscal + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sscal
   :outline:
.. doxygenfunction:: rocblas_dscal
   :outline:
.. doxygenfunction:: rocblas_cscal
   :outline:
.. doxygenfunction:: rocblas_zscal
   :outline:
.. doxygenfunction:: rocblas_csscal
   :outline:
.. doxygenfunction:: rocblas_zdscal

.. doxygenfunction:: rocblas_sscal_batched
   :outline:
.. doxygenfunction:: rocblas_dscal_batched
   :outline:
.. doxygenfunction:: rocblas_cscal_batched
   :outline:
.. doxygenfunction:: rocblas_zscal_batched
   :outline:
.. doxygenfunction:: rocblas_csscal_batched
   :outline:
.. doxygenfunction:: rocblas_zdscal_batched

.. doxygenfunction:: rocblas_sscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csscal_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdscal_strided_batched

rocblas_Xswap + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sswap
   :outline:
.. doxygenfunction:: rocblas_dswap
   :outline:
.. doxygenfunction:: rocblas_cswap
   :outline:
.. doxygenfunction:: rocblas_zswap

.. doxygenfunction:: rocblas_sswap_batched
   :outline:
.. doxygenfunction:: rocblas_dswap_batched
   :outline:
.. doxygenfunction:: rocblas_cswap_batched
   :outline:
.. doxygenfunction:: rocblas_zswap_batched

.. doxygenfunction:: rocblas_sswap_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dswap_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cswap_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zswap_strided_batched


Level 2 BLAS
============
rocblas_Xgbmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sgbmv
   :outline:
.. doxygenfunction:: rocblas_dgbmv
   :outline:
.. doxygenfunction:: rocblas_cgbmv
   :outline:
.. doxygenfunction:: rocblas_zgbmv

.. doxygenfunction:: rocblas_sgbmv_batched
   :outline:
.. doxygenfunction:: rocblas_dgbmv_batched
   :outline:
.. doxygenfunction:: rocblas_cgbmv_batched
   :outline:
.. doxygenfunction:: rocblas_zgbmv_batched

.. doxygenfunction:: rocblas_sgbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgbmv_strided_batched

rocblas_Xgemv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sgemv
   :outline:
.. doxygenfunction:: rocblas_dgemv
   :outline:
.. doxygenfunction:: rocblas_cgemv
   :outline:
.. doxygenfunction:: rocblas_zgemv

.. doxygenfunction:: rocblas_sgemv_batched
   :outline:
.. doxygenfunction:: rocblas_dgemv_batched
   :outline:
.. doxygenfunction:: rocblas_cgemv_batched
   :outline:
.. doxygenfunction:: rocblas_zgemv_batched

.. doxygenfunction:: rocblas_sgemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgemv_strided_batched

rocblas_Xger + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sger
   :outline:
.. doxygenfunction:: rocblas_dger
   :outline:
.. doxygenfunction:: rocblas_cgeru
   :outline:
.. doxygenfunction:: rocblas_zgeru
   :outline:
.. doxygenfunction:: rocblas_cgerc
   :outline:
.. doxygenfunction:: rocblas_zgerc

.. doxygenfunction:: rocblas_sger_batched
   :outline:
.. doxygenfunction:: rocblas_dger_batched
   :outline:
.. doxygenfunction:: rocblas_cgeru_batched
   :outline:
.. doxygenfunction:: rocblas_zgeru_batched
   :outline:
.. doxygenfunction:: rocblas_cgerc_batched
   :outline:
.. doxygenfunction:: rocblas_zgerc_batched

.. doxygenfunction:: rocblas_sger_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dger_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgeru_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgeru_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgerc_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgerc_strided_batched

rocblas_Xsbmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_ssbmv
   :outline:
.. doxygenfunction:: rocblas_dsbmv

.. doxygenfunction:: rocblas_ssbmv_batched
   :outline:
.. doxygenfunction:: rocblas_dsbmv_batched

.. doxygenfunction:: rocblas_ssbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsbmv_strided_batched

rocblas_Xspmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sspmv
   :outline:
.. doxygenfunction:: rocblas_dspmv

.. doxygenfunction:: rocblas_sspmv_batched
   :outline:
.. doxygenfunction:: rocblas_dspmv_batched

.. doxygenfunction:: rocblas_sspmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dspmv_strided_batched

rocblas_Xspr + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sspr
   :outline:
.. doxygenfunction:: rocblas_dspr
   :outline:
.. doxygenfunction:: rocblas_cspr
   :outline:
.. doxygenfunction:: rocblas_zspr

.. doxygenfunction:: rocblas_sspr_batched
   :outline:
.. doxygenfunction:: rocblas_dspr_batched
   :outline:
.. doxygenfunction:: rocblas_cspr_batched
   :outline:
.. doxygenfunction:: rocblas_zspr_batched

.. doxygenfunction:: rocblas_sspr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dspr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cspr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zspr_strided_batched

rocblas_Xspr2 + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sspr2
   :outline:
.. doxygenfunction:: rocblas_dspr2

.. doxygenfunction:: rocblas_sspr2_batched
   :outline:
.. doxygenfunction:: rocblas_dspr2_batched

.. doxygenfunction:: rocblas_sspr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dspr2_strided_batched

rocblas_Xsymv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_ssymv
   :outline:
.. doxygenfunction:: rocblas_dsymv
   :outline:
.. doxygenfunction:: rocblas_csymv
   :outline:
.. doxygenfunction:: rocblas_zsymv

.. doxygenfunction:: rocblas_ssymv_batched
   :outline:
.. doxygenfunction:: rocblas_dsymv_batched
   :outline:
.. doxygenfunction:: rocblas_csymv_batched
   :outline:
.. doxygenfunction:: rocblas_zsymv_batched

.. doxygenfunction:: rocblas_ssymv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsymv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csymv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsymv_strided_batched

rocblas_Xsyr + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_ssyr
   :outline:
.. doxygenfunction:: rocblas_dsyr
   :outline:
.. doxygenfunction:: rocblas_csyr
   :outline:
.. doxygenfunction:: rocblas_zsyr

.. doxygenfunction:: rocblas_ssyr_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr_batched
   :outline:
.. doxygenfunction:: rocblas_csyr_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr_batched

.. doxygenfunction:: rocblas_ssyr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr_strided_batched

rocblas_Xsyr2 + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_ssyr2
   :outline:
.. doxygenfunction:: rocblas_dsyr2
   :outline:
.. doxygenfunction:: rocblas_csyr2
   :outline:
.. doxygenfunction:: rocblas_zsyr2

.. doxygenfunction:: rocblas_ssyr2_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2_batched

.. doxygenfunction:: rocblas_ssyr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2_strided_batched

rocblas_Xtbmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_stbmv
   :outline:
.. doxygenfunction:: rocblas_dtbmv
   :outline:
.. doxygenfunction:: rocblas_ctbmv
   :outline:
.. doxygenfunction:: rocblas_ztbmv

.. doxygenfunction:: rocblas_stbmv_batched
   :outline:
.. doxygenfunction:: rocblas_dtbmv_batched
   :outline:
.. doxygenfunction:: rocblas_ctbmv_batched
   :outline:
.. doxygenfunction:: rocblas_ztbmv_batched

.. doxygenfunction:: rocblas_stbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztbmv_strided_batched

rocblas_Xtbsv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_stbsv
   :outline:
.. doxygenfunction:: rocblas_dtbsv
   :outline:
.. doxygenfunction:: rocblas_ctbsv
   :outline:
.. doxygenfunction:: rocblas_ztbsv

.. doxygenfunction:: rocblas_stbsv_batched
   :outline:
.. doxygenfunction:: rocblas_dtbsv_batched
   :outline:
.. doxygenfunction:: rocblas_ctbsv_batched
   :outline:
.. doxygenfunction:: rocblas_ztbsv_batched

.. doxygenfunction:: rocblas_stbsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtbsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctbsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztbsv_strided_batched

rocblas_Xtpmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_stpmv
   :outline:
.. doxygenfunction:: rocblas_dtpmv
   :outline:
.. doxygenfunction:: rocblas_ctpmv
   :outline:
.. doxygenfunction:: rocblas_ztpmv

.. doxygenfunction:: rocblas_stpmv_batched
   :outline:
.. doxygenfunction:: rocblas_dtpmv_batched
   :outline:
.. doxygenfunction:: rocblas_ctpmv_batched
   :outline:
.. doxygenfunction:: rocblas_ztpmv_batched

.. doxygenfunction:: rocblas_stpmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtpmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctpmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztpmv_strided_batched

rocblas_Xtpsv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_stpsv
   :outline:
.. doxygenfunction:: rocblas_dtpsv
   :outline:
.. doxygenfunction:: rocblas_ctpsv
   :outline:
.. doxygenfunction:: rocblas_ztpsv

.. doxygenfunction:: rocblas_stpsv_batched
   :outline:
.. doxygenfunction:: rocblas_dtpsv_batched
   :outline:
.. doxygenfunction:: rocblas_ctpsv_batched
   :outline:
.. doxygenfunction:: rocblas_ztpsv_batched

.. doxygenfunction:: rocblas_stpsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtpsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctpsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztpsv_strided_batched

rocblas_Xtrmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_strmv
   :outline:
.. doxygenfunction:: rocblas_dtrmv
   :outline:
.. doxygenfunction:: rocblas_ctrmv
   :outline:
.. doxygenfunction:: rocblas_ztrmv

.. doxygenfunction:: rocblas_strmv_batched
   :outline:
.. doxygenfunction:: rocblas_dtrmv_batched
   :outline:
.. doxygenfunction:: rocblas_ctrmv_batched
   :outline:
.. doxygenfunction:: rocblas_ztrmv_batched

.. doxygenfunction:: rocblas_strmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctrmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztrmv_strided_batched

rocblas_Xtrsv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_strsv
   :outline:
.. doxygenfunction:: rocblas_dtrsv
   :outline:
.. doxygenfunction:: rocblas_ctrsv
   :outline:
.. doxygenfunction:: rocblas_ztrsv

.. doxygenfunction:: rocblas_strsv_batched
   :outline:
.. doxygenfunction:: rocblas_dtrsv_batched
   :outline:
.. doxygenfunction:: rocblas_ctrsv_batched
   :outline:
.. doxygenfunction:: rocblas_ztrsv_batched

.. doxygenfunction:: rocblas_strsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctrsv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztrsv_strided_batched

rocblas_Xhemv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_chemv
   :outline:
.. doxygenfunction:: rocblas_zhemv

.. doxygenfunction:: rocblas_chemv_batched
   :outline:
.. doxygenfunction:: rocblas_zhemv_batched

.. doxygenfunction:: rocblas_chemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhemv_strided_batched

rocblas_Xhbmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_chbmv
   :outline:
.. doxygenfunction:: rocblas_zhbmv

.. doxygenfunction:: rocblas_chbmv_batched
   :outline:
.. doxygenfunction:: rocblas_zhbmv_batched

.. doxygenfunction:: rocblas_chbmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhbmv_strided_batched

rocblas_Xhpmv + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_chpmv
   :outline:
.. doxygenfunction:: rocblas_zhpmv

.. doxygenfunction:: rocblas_chpmv_batched
   :outline:
.. doxygenfunction:: rocblas_zhpmv_batched

.. doxygenfunction:: rocblas_chpmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhpmv_strided_batched

rocblas_Xher + batched, strided_batched
---------------------------------------
.. doxygenfunction:: rocblas_cher
   :outline:
.. doxygenfunction:: rocblas_zher

.. doxygenfunction:: rocblas_cher_batched
   :outline:
.. doxygenfunction:: rocblas_zher_batched

.. doxygenfunction:: rocblas_cher_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zher_strided_batched

rocblas_Xher2 + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_cher2
   :outline:
.. doxygenfunction:: rocblas_zher2

.. doxygenfunction:: rocblas_cher2_batched
   :outline:
.. doxygenfunction:: rocblas_zher2_batched

.. doxygenfunction:: rocblas_cher2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zher2_strided_batched

rocblas_Xhpr + batched, strided_batched
---------------------------------------
.. doxygenfunction:: rocblas_chpr
   :outline:
.. doxygenfunction:: rocblas_zhpr

.. doxygenfunction:: rocblas_chpr_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr_batched

.. doxygenfunction:: rocblas_chpr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr_strided_batched

rocblas_Xhpr2 + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_chpr2
   :outline:
.. doxygenfunction:: rocblas_zhpr2

.. doxygenfunction:: rocblas_chpr2_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr2_batched

.. doxygenfunction:: rocblas_chpr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr2_strided_batched

Level 3 BLAS
============

rocblas_Xgemm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sgemm
   :outline:
.. doxygenfunction:: rocblas_dgemm
   :outline:
.. doxygenfunction:: rocblas_hgemm
   :outline:
.. doxygenfunction:: rocblas_cgemm
   :outline:
.. doxygenfunction:: rocblas_zgemm

.. doxygenfunction:: rocblas_sgemm_batched
   :outline:
.. doxygenfunction:: rocblas_dgemm_batched
   :outline:
.. doxygenfunction:: rocblas_hgemm_batched
   :outline:
.. doxygenfunction:: rocblas_cgemm_batched
   :outline:
.. doxygenfunction:: rocblas_zgemm_batched

.. doxygenfunction:: rocblas_sgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_hgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgemm_strided_batched

rocblas_Xsymm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_ssymm
   :outline:
.. doxygenfunction:: rocblas_dsymm
   :outline:
.. doxygenfunction:: rocblas_csymm
   :outline:
.. doxygenfunction:: rocblas_zsymm

.. doxygenfunction:: rocblas_ssymm_batched
   :outline:
.. doxygenfunction:: rocblas_dsymm_batched
   :outline:
.. doxygenfunction:: rocblas_csymm_batched
   :outline:
.. doxygenfunction:: rocblas_zsymm_batched

.. doxygenfunction:: rocblas_ssymm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsymm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csymm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsymm_strided_batched

rocblas_Xsyrk + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_ssyrk
   :outline:
.. doxygenfunction:: rocblas_dsyrk
   :outline:
.. doxygenfunction:: rocblas_csyrk
   :outline:
.. doxygenfunction:: rocblas_zsyrk

.. doxygenfunction:: rocblas_ssyrk_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrk_batched
   :outline:
.. doxygenfunction:: rocblas_csyrk_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrk_batched

.. doxygenfunction:: rocblas_ssyrk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyrk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrk_strided_batched

rocblas_Xsyr2k + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_ssyr2k
   :outline:
.. doxygenfunction:: rocblas_dsyr2k
   :outline:
.. doxygenfunction:: rocblas_csyr2k
   :outline:
.. doxygenfunction:: rocblas_zsyr2k

.. doxygenfunction:: rocblas_ssyr2k_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2k_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2k_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2k_batched

.. doxygenfunction:: rocblas_ssyr2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2k_strided_batched

rocblas_Xsyrkx + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_ssyrkx
   :outline:
.. doxygenfunction:: rocblas_dsyrkx
   :outline:
.. doxygenfunction:: rocblas_csyrkx
   :outline:
.. doxygenfunction:: rocblas_zsyrkx

.. doxygenfunction:: rocblas_ssyrkx_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrkx_batched
   :outline:
.. doxygenfunction:: rocblas_csyrkx_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrkx_batched

.. doxygenfunction:: rocblas_ssyrkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyrkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyrkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyrkx_strided_batched

rocblas_Xtrmm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_strmm
   :outline:
.. doxygenfunction:: rocblas_dtrmm
   :outline:
.. doxygenfunction:: rocblas_ctrmm
   :outline:
.. doxygenfunction:: rocblas_ztrmm

.. doxygenfunction:: rocblas_strmm_batched
   :outline:
.. doxygenfunction:: rocblas_dtrmm_batched
   :outline:
.. doxygenfunction:: rocblas_ctrmm_batched
   :outline:
.. doxygenfunction:: rocblas_ztrmm_batched

.. doxygenfunction:: rocblas_strmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctrmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztrmm_strided_batched


rocblas_Xtrsm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_strsm
   :outline:
.. doxygenfunction:: rocblas_dtrsm
   :outline:
.. doxygenfunction:: rocblas_ctrsm
   :outline:
.. doxygenfunction:: rocblas_ztrsm

.. doxygenfunction:: rocblas_strsm_batched
   :outline:
.. doxygenfunction:: rocblas_dtrsm_batched
   :outline:
.. doxygenfunction:: rocblas_ctrsm_batched
   :outline:
.. doxygenfunction:: rocblas_ztrsm_batched

.. doxygenfunction:: rocblas_strsm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrsm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ctrsm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ztrsm_strided_batched

rocblas_Xhemm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_chemm
   :outline:
.. doxygenfunction:: rocblas_zhemm

.. doxygenfunction:: rocblas_chemm_batched
   :outline:
.. doxygenfunction:: rocblas_zhemm_batched

.. doxygenfunction:: rocblas_chemm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhemm_strided_batched

rocblas_Xherk + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_cherk
   :outline:
.. doxygenfunction:: rocblas_zherk

.. doxygenfunction:: rocblas_cherk_batched
   :outline:
.. doxygenfunction:: rocblas_zherk_batched

.. doxygenfunction:: rocblas_cherk_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zherk_strided_batched

rocblas_Xher2k + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_cher2k
   :outline:
.. doxygenfunction:: rocblas_zher2k

.. doxygenfunction:: rocblas_cher2k_batched
   :outline:
.. doxygenfunction:: rocblas_zher2k_batched

.. doxygenfunction:: rocblas_cher2k_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zher2k_strided_batched

rocblas_Xherkx + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_cherkx
   :outline:
.. doxygenfunction:: rocblas_zherkx

.. doxygenfunction:: rocblas_cherkx_batched
   :outline:
.. doxygenfunction:: rocblas_zherkx_batched

.. doxygenfunction:: rocblas_cherkx_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zherkx_strided_batched

rocblas_Xtrtri + batched, strided_batched
-----------------------------------------
.. doxygenfunction:: rocblas_strtri
   :outline:
.. doxygenfunction:: rocblas_dtrtri

.. doxygenfunction:: rocblas_strtri_batched
   :outline:
.. doxygenfunction:: rocblas_dtrtri_batched

.. doxygenfunction:: rocblas_strtri_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dtrtri_strided_batched


BLAS Extensions
===============

rocblas_axpy_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_axpy_ex
.. doxygenfunction:: rocblas_axpy_batched_ex
.. doxygenfunction:: rocblas_axpy_strided_batched_ex

rocblas_dot_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_dot_ex
.. doxygenfunction:: rocblas_dot_batched_ex
.. doxygenfunction:: rocblas_dot_strided_batched_ex

rocblas_dotc_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_dotc_ex
.. doxygenfunction:: rocblas_dotc_batched_ex
.. doxygenfunction:: rocblas_dotc_strided_batched_ex

rocblas_nrm2_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_nrm2_ex
.. doxygenfunction:: rocblas_nrm2_batched_ex
.. doxygenfunction:: rocblas_nrm2_strided_batched_ex

rocblas_rot_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_rot_ex
.. doxygenfunction:: rocblas_rot_batched_ex
.. doxygenfunction:: rocblas_rot_strided_batched_ex

rocblas_scal_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_scal_ex
.. doxygenfunction:: rocblas_scal_batched_ex
.. doxygenfunction:: rocblas_scal_strided_batched_ex

rocblas_gemm_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_gemm_ex
.. doxygenfunction:: rocblas_gemm_batched_ex
.. doxygenfunction:: rocblas_gemm_strided_batched_ex

rocblas_gemm_ext2
-----------------
.. doxygenfunction:: rocblas_gemm_ext2

rocblas_trsm_ex + batched, strided_batched
------------------------------------------
.. doxygenfunction:: rocblas_trsm_ex
.. doxygenfunction:: rocblas_trsm_batched_ex
.. doxygenfunction:: rocblas_trsm_strided_batched_ex

rocblas_Xgeam + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sgeam
   :outline:
.. doxygenfunction:: rocblas_dgeam
   :outline:
.. doxygenfunction:: rocblas_cgeam
   :outline:
.. doxygenfunction:: rocblas_zgeam

.. doxygenfunction:: rocblas_sgeam_batched
   :outline:
.. doxygenfunction:: rocblas_dgeam_batched
   :outline:
.. doxygenfunction:: rocblas_cgeam_batched
   :outline:
.. doxygenfunction:: rocblas_zgeam_batched

.. doxygenfunction:: rocblas_sgeam_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgeam_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgeam_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgeam_strided_batched


rocblas_Xdgmm + batched, strided_batched
----------------------------------------
.. doxygenfunction:: rocblas_sdgmm
   :outline:
.. doxygenfunction:: rocblas_ddgmm
   :outline:
.. doxygenfunction:: rocblas_cdgmm
   :outline:
.. doxygenfunction:: rocblas_zdgmm

.. doxygenfunction:: rocblas_sdgmm_batched
   :outline:
.. doxygenfunction:: rocblas_ddgmm_batched
   :outline:
.. doxygenfunction:: rocblas_cdgmm_batched
   :outline:
.. doxygenfunction:: rocblas_zdgmm_batched

.. doxygenfunction:: rocblas_sdgmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_ddgmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cdgmm_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zdgmm_strided_batched


Auxiliary
=========

rocblas_pointer_to_mode
-----------------------
.. doxygenfunction:: rocblas_pointer_to_mode

rocblas_create_handle
---------------------
.. doxygenfunction:: rocblas_create_handle

rocblas_destroy_handle
----------------------
.. doxygenfunction:: rocblas_destroy_handle

rocblas_set_stream
------------------
.. doxygenfunction:: rocblas_set_stream

rocblas_get_stream
------------------
.. doxygenfunction:: rocblas_get_stream

rocblas_set_pointer_mode
------------------------
.. doxygenfunction:: rocblas_set_pointer_mode

rocblas_get_pointer_mode
------------------------
.. doxygenfunction:: rocblas_get_pointer_mode

rocblas_set_atomics_mode
------------------------
.. doxygenfunction:: rocblas_set_atomics_mode

rocblas_get_atomics_mode
------------------------
.. doxygenfunction:: rocblas_get_atomics_mode

rocblas_set_vector
------------------
.. doxygenfunction:: rocblas_set_vector

rocblas_set_vector_async
------------------------
.. doxygenfunction:: rocblas_set_vector_async

rocblas_get_vector
------------------
.. doxygenfunction:: rocblas_get_vector

rocblas_get_vector_async
------------------------
.. doxygenfunction:: rocblas_get_vector_async

rocblas_set_matrix
------------------
.. doxygenfunction:: rocblas_set_matrix

rocblas_set_matrix_async
------------------------
.. doxygenfunction:: rocblas_set_matrix_async

rocblas_get_matrix
------------------
.. doxygenfunction:: rocblas_get_matrix

rocblas_get_matrix_async
------------------------
.. doxygenfunction:: rocblas_get_matrix_async

rocblas_initialize
------------------------
.. doxygenfunction:: rocblas_initialize

Device Memory functions
=======================

rocblas_start_device_memory_size_query
--------------------------------------
.. doxygenfunction:: rocblas_start_device_memory_size_query

rocblas_stop_device_memory_size_query
-------------------------------------
.. doxygenfunction:: rocblas_stop_device_memory_size_query

rocblas_get_device_memory_size
------------------------------
.. doxygenfunction:: rocblas_get_device_memory_size

rocblas_set_device_memory_size
------------------------------
.. doxygenfunction:: rocblas_set_device_memory_size

rocblas_set_workspace
---------------------
.. doxygenfunction:: rocblas_set_workspace

rocblas_is_managing_device_memory
---------------------------------
.. doxygenfunction:: rocblas_is_managing_device_memory

rocblas_is_user_managing_device_memory
--------------------------------------
.. doxygenfunction:: rocblas_is_user_managing_device_memory


Build Information
=================

rocblas_get_version_string
----------------------------
.. doxygenfunction:: rocblas_get_version_string

rocblas_get_version_string_size
-------------------------------
.. doxygenfunction:: rocblas_get_version_string_size

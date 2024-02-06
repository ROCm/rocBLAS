.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _level-2:

********************************************************************
rocBLAS Level-2 functions
********************************************************************

rocBLAS Level-2 functions perform matrix-vector operations. [Level2]_

rocblas_Xgbmv + batched, strided_batched
========================================

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
========================================

.. doxygenfunction:: rocblas_sgemv
   :outline:
.. doxygenfunction:: rocblas_dgemv
   :outline:
.. doxygenfunction:: rocblas_cgemv
   :outline:
.. doxygenfunction:: rocblas_zgemv

gemv functions have an implementation which uses atomic operations. See section :ref:`Atomic Operations` for more information.

.. doxygenfunction:: rocblas_sgemv_batched
   :outline:
.. doxygenfunction:: rocblas_dgemv_batched
   :outline:
.. doxygenfunction:: rocblas_cgemv_batched
   :outline:
.. doxygenfunction:: rocblas_zgemv_batched

gemv_batched functions have an implementation which uses atomic operations. See section :ref:`Atomic Operations` for more information.

.. doxygenfunction:: rocblas_sgemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dgemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cgemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zgemv_strided_batched

gemv_strided_batched functions have an implementation which uses atomic operations. See section :ref:`Atomic Operations` for more information.

rocblas_Xger + batched, strided_batched
========================================

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

The ger, geru, and gerc functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

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

The ger, geru, and gerc_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

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

The ger, geru, and gerc_strided_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

rocblas_Xsbmv + batched, strided_batched
========================================

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
========================================

.. doxygenfunction:: rocblas_sspmv
   :outline:
.. doxygenfunction:: rocblas_dspmv

The spmv functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sspmv_batched
   :outline:
.. doxygenfunction:: rocblas_dspmv_batched

The spmv_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sspmv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dspmv_strided_batched

The spmv_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xspr + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_sspr
   :outline:
.. doxygenfunction:: rocblas_dspr
   :outline:
.. doxygenfunction:: rocblas_cspr
   :outline:
.. doxygenfunction:: rocblas_zspr

The spr functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sspr_batched
   :outline:
.. doxygenfunction:: rocblas_dspr_batched
   :outline:
.. doxygenfunction:: rocblas_cspr_batched
   :outline:
.. doxygenfunction:: rocblas_zspr_batched

The spr_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sspr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dspr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_cspr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zspr_strided_batched

The spr_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xspr2 + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_sspr2
   :outline:
.. doxygenfunction:: rocblas_dspr2

The spr2 functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sspr2_batched
   :outline:
.. doxygenfunction:: rocblas_dspr2_batched

The spr2_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_sspr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dspr2_strided_batched

The spr2_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xsymv + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_ssymv
   :outline:
.. doxygenfunction:: rocblas_dsymv
   :outline:
.. doxygenfunction:: rocblas_csymv
   :outline:
.. doxygenfunction:: rocblas_zsymv

The symv functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssymv_batched
   :outline:
.. doxygenfunction:: rocblas_dsymv_batched
   :outline:
.. doxygenfunction:: rocblas_csymv_batched
   :outline:
.. doxygenfunction:: rocblas_zsymv_batched

The symv_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssymv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsymv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csymv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsymv_strided_batched

The symv_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xsyr + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_ssyr
   :outline:
.. doxygenfunction:: rocblas_dsyr
   :outline:
.. doxygenfunction:: rocblas_csyr
   :outline:
.. doxygenfunction:: rocblas_zsyr

The syr functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyr_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr_batched
   :outline:
.. doxygenfunction:: rocblas_csyr_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr_batched

The syr_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr_strided_batched

The syr_strided_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

rocblas_Xsyr2 + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_ssyr2
   :outline:
.. doxygenfunction:: rocblas_dsyr2
   :outline:
.. doxygenfunction:: rocblas_csyr2
   :outline:
.. doxygenfunction:: rocblas_zsyr2

The syr2 functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyr2_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2_batched

The syr2_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_ssyr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_dsyr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_csyr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zsyr2_strided_batched

The syr2_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xtbmv + batched, strided_batched
========================================

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
========================================

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
========================================

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
========================================

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
========================================

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
========================================

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
========================================

.. doxygenfunction:: rocblas_chemv
   :outline:
.. doxygenfunction:: rocblas_zhemv

The hemv functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_chemv_batched
   :outline:
.. doxygenfunction:: rocblas_zhemv_batched

The hemv_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_chemv_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhemv_strided_batched

The hemv_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xhbmv + batched, strided_batched
========================================

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
========================================

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
========================================

.. doxygenfunction:: rocblas_cher
   :outline:
.. doxygenfunction:: rocblas_zher

The her functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cher_batched
   :outline:
.. doxygenfunction:: rocblas_zher_batched

The her_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cher_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zher_strided_batched

The her_strided_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

rocblas_Xher2 + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_cher2
   :outline:
.. doxygenfunction:: rocblas_zher2

The her2 functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cher2_batched
   :outline:
.. doxygenfunction:: rocblas_zher2_batched

The her2_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_cher2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zher2_strided_batched

The her2_strided_batched functions support the _64 interface.  Refer to section :ref:`ILP64 API`.

rocblas_Xhpr + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_chpr
   :outline:
.. doxygenfunction:: rocblas_zhpr

The hpr functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_chpr_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr_batched

The hpr_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

.. doxygenfunction:: rocblas_chpr_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr_strided_batched

The hpr_strided_batched functions support the _64 interface. Refer to section :ref:`ILP64 API`.

rocblas_Xhpr2 + batched, strided_batched
========================================

.. doxygenfunction:: rocblas_chpr2
   :outline:
.. doxygenfunction:: rocblas_zhpr2

.. doxygenfunction:: rocblas_chpr2_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr2_batched

.. doxygenfunction:: rocblas_chpr2_strided_batched
   :outline:
.. doxygenfunction:: rocblas_zhpr2_strided_batched



.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _data-types-support:

********************************************************************
Data type support
********************************************************************

The rocBLAS library supports various data types for different levels of BLAS
operations :ref:`Level 1 <level-1>`, :ref:`2 <level-2>`, and :ref:`3 <level-3>`. 

Level 1 functions - Vector operations
=====================================

Level-1 functions support the ILP64 API.  For more information on these ``_64``
functions, see the :ref:`ILP64 API` section.

.. tab-set::

  .. tab-item:: Float types
    :sync: float-type

    .. list-table::
        :header-rows: 1

        * 
          - Function
          - Description
          - float16
          - bfloat16
          - float
          - double

        *
          - :ref:`AMax <rocblas_amax>`, :ref:`AMin <rocblas_amin>`, :ref:`ASum <rocblas_asum>`
          - Finds the first index of the element of minimum or maximum magnitude of a vector x or computes the sum of the magnitudes of elements of a real vector x.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`AXPY <rocblas_axpy>`
          - Scales a vector and adds it to another: :math:`y = \alpha x + y`
          - ✅
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Copy <rocblas_copy>`
          - Copies vector x to y: :math:`y = x`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Dot <rocblas_dot>`
          - Computes the dot product: :math:`result = x^T y`
          - ✅
          - ✅
          - ✅
          - ✅    

        *
          - :ref:`NRM2 <rocblas_nrm2>`
          - Computes the Euclidean norm of a vector.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Rot <rocblas_rot>`, :ref:`Rotg <rocblas_rotg>`
          - Applies the givens rotation matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Rotm <rocblas_rotm>`, :ref:`Rotmg <rocblas_rotmg>`
          - Applies the givens rotation matrix.
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Scal <rocblas_scal>`
          - Scales a vector by a scalar: :math:`x = \alpha x`
          - ❌
          - ❌
          - ✅
          - ✅

        *
          - :ref:`Swap <rocblas_swap>`
          - Interchanges vectors x_i and y_i for i=1, ... , batch_count.
          - ❌
          - ❌
          - ✅
          - ✅

  .. tab-item:: Complex types
    :sync: complex-type

    .. list-table::
        :header-rows: 1

        * 
          - Function
          - Description
          - complex
          - double complex

        *
          - :ref:`AMax <rocblas_amax>`, :ref:`AMin <rocblas_amin>`, :ref:`ASum <rocblas_asum>`
          - Finds the first index of the element of minimum or maximum magnitude of a vector x or computes the sum of the magnitudes of elements of a real vector x.
          - ✅
          - ✅

        *
          - :ref:`AXPY <rocblas_axpy>`
          - Scales a vector and adds it to another: :math:`y = \alpha x + y`
          - ✅
          - ✅

        *
          - :ref:`Copy <rocblas_copy>`
          - Copies vector x to y: :math:`y = x`
          - ✅
          - ✅

        *
          - :ref:`Dot <rocblas_dot>`
          - Computes the dot product: :math:`result = x^T y`
          - ✅
          - ✅      

        *
          - :ref:`NRM2 <rocblas_nrm2>`
          - Computes the Euclidean norm of a vector.
          - ✅
          - ✅

        *
          - :ref:`Rot <rocblas_rot>`, :ref:`Rotg <rocblas_rotg>`
          - Applies the givens rotation matrix.
          - ✅
          - ✅

        *
          - :ref:`Rotm <rocblas_rotm>`, :ref:`Rotmg <rocblas_rotmg>`
          - Applies the givens rotation matrix.
          - ❌
          - ❌

        *
          - :ref:`Scal <rocblas_scal>`
          - Scales a vector by a scalar: :math:`x = \alpha x`
          - ✅
          - ✅

        *
          - :ref:`Swap <rocblas_swap>`
          - Interchanges vectors x_i and y_i for i=1, ... , batch_count.
          - ✅
          - ✅

Level 2 functions - Matrix-Vector operations
============================================

.. list-table::
    :header-rows: 1

    * 
      - Function
      - Description
      - Supported data types

    *
      - :ref:`GBMV <rocblas_gbmv>`
      - General Matrix-Vector multiplication: :math:`y = \alpha A x + \beta y`
      - float, double, complex, double complex

    *
      - :ref:`GEMV <rocblas_gemv>`
      - General Matrix-Vector multiplication: :math:`y = \alpha A x + \beta y`
      - float16, bloat16, float, double, complex, double complex

    *
      - :ref:`GER <rocblas_ger>`
      - Rank-1 update: :math:`A = \alpha x y^T + A`
      - float16, float, double, complex, double complex

    *
      - :ref:`HER <rocblas_her>`
      - Hermitian rank-1 update: :math:`A = \alpha x x^H + A`
      - complex, double complex

    *
      - :ref:`TRSV <rocblas_trsv>`
      - Solves triangular system: :math:`A x = b`
      - float16, float, double, complex

Level 3 functions - Matrix-Matrix operations
============================================

rocBLAS calls the AMD :doc:`Tensile <tensile:src/index>` and
:doc:`hipBLASLt <hipblaslt:index>` libraries for Level-3 GEMMs (matrix matrix
multiplication).

.. list-table::
    :header-rows: 1

    * 
      - Function
      - Description
      - Supported data types

    * 
      - :ref:`GEMM <rocblas_gemm>`
      - General matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
      - float16, bloat16, float, double, complex

    * 
      - :ref:`SYMM <rocblas_symm>`
      - Symmetric matrix-matrix multiplication: :math:`C = \alpha A B + \beta C`
      - float, double, complex, complex double

    * 
      - :ref:`TRSM <rocblas_trsm>`
      - Solves triangular matrix equation: :math:`AX = B`
      - float, double, complex, complex double

    * 
      - :ref:`HEMM <rocblas_hemm>`
      - Hermitian matrix-matrix multiplication.
      - float, double, complex, complex double

Extensions
==========


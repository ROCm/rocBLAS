=================================================
rocBLAS order of argument checking and logging
=================================================

Legacy BLAS
===========
Legacy BLAS has two types of argument checking:

1. Error-return for incorrect argument (Legacy BLAS implement this with a call to the function ``XERBLA``).

2. Quick-return-success when an argument allows for the subprogram to be a no-operation or a constant result.

Level 2 and Level 3 BLAS subprograms have both error-return and quick-return-success. Level 1 BLAS subprograms have only quick-return-success.

rocBLAS
=======
rocBLAS has 5 types of argument checking:

1. ``rocblas_status_invalid_handle`` if the handle is a NULL pointer

2. ``rocblas_status_invalid_size`` for invalid size, increment or leading dimension argument

3. ``rocblas_status_invalid_value`` for unsupported enum value

4. ``rocblas_status_success`` for quick-return-success

5. ``rocblas_status_invalid_pointer`` for NULL argument pointers


rocBLAS has the following differences when compared to Legacy BLAS
==================================================================

- It is a C API, returning a ``rocblas_status`` type indicating the success of the call.

- In legacy BLAS the following functions return a scalar result: dot, nrm2, asum, amax and amin. In rocBLAS a pointers to scalar return value  is passed as the last argument.

- The first argument is a ``rocblas_handle`` argument, an opaque pointer to rocBLAS resources, corresponding to a single HIP stream.

- Scalar arguments like alpha and beta are pointers on either the host or device, controlled by the rocBLAS handle's pointer mode.

- Vector and matrix arguments are always pointers to device memory.

- The ``ROCBLAS_LAYER`` environment variable controls the option to log argument values.

- There is added functionality like

  - batched

  - strided_batched

  - mixed precision in gemm_ex, gemm_batched_ex, and gemm_strided_batched_ex

To accommodate the additions
============================

- See Logging, below.

- For batched and strided_batched L2 and L3 functions there is a quick-return-success for ``batch_count == 0``, and an invalid size error for ``batch_count < 0``.

- For batched and strided_batched L1 functions there is a quick-return-success for ``batch_count <= 0``

- When ``rocblas_pointer_mode == rocblas_pointer_mode_device`` do not copy alpha and or beta from device to host for quick-return-success checks. In this case, omit the quick-return-success checks for alpha and or beta.

- For vectors and matrices with batched stride, there is no argument checking for stride. To access elements in a strided_batched_matrix, for example the C matrix in gemm, the zero based index is calculated as ``i1 + i2 * ldc + i3 * stride_c``, where ``i1 = 0, 1, 2, ..., m-1``; ``i2 = 0, 1, 2, ..., n-1``; ``i3 = 0, 1, 2, ..., batch_count -1``. An incorrect stride can result in a core dump due a segmentation fault. It can also produce an indeterminate result if there is a memory overlap in the output matrix between different values of ``i3``.


Device Memory Size Queries
==========================

- When ``handle->is_device_memory_size_query()`` is true, the call is not a normal call, but it is a device memory size query.

- No logging should be performed during device memory size queries.

- If the rocBLAS kernel requires no temporary device memory, the macro ``RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)`` can be called after checking that ``handle != nullptr``.

- If the rocBLAS kernel requires temporary device memory, then it should be set, and the kernel returned, by calling ``return handle->set_optimal_device_memory_size(size...)``, where ``size...`` is a list of one or more sizes for different sub-problems. The sizes are rounded up and added.

Logging
-------

- There is logging before a quick-return-success or error-return, except:

  - when ``handle == nullptr``, return ``rocblas_status_invalid_handle``
  - when ``handle->is_device_memory_size_query()`` returns ``true``

- Vectors and matrices are logged with their addresses, and are always on device memory.

- Scalar values in device memory are logged as their addresses. Scalar values in host memory are logged as their values, with a ``nullptr`` logged as ``NaN`` (``std::numeric_limits<T>::quiet_NaN()``).

rocBLAS control flow:
=====================

1. If ``handle == nullptr``, return ``rocblas_status_invalid_handle``.

2. If the function does not require temporary device memory, call the macro ``RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);``.

3. If the function requires temporary device memory, and ``handle->is_device_memory_size_query()`` is ``true``, then validate any pointers and arguments required to determine the optimal size of temporary device memory, returning ``rocblas_status_invalid_pointer`` or ``rocblas_status_invalid_size`` if the arguments are invalid, and otherwise ``return handle->set_optimal_device_memory_size(size...);``, where ``size...`` is a list of one or more sizes of temporary buffers, which are allocated with ``handle->device_malloc(size...)`` later.

4. Perform logging if enabled, taking care not to dereference ``nullptr`` arguments.

5. Check for unsupported enum value. Return ``rocblas_status_invalid_value`` if enum value is invalid.

6. Check for invalid sizes. Return ``rocblas_status_invalid_size`` if size arguments are invalid.

7. Return ``rocblas_status_invalid_pointer`` if any pointers used to determine quick return conditions are NULL.

8. If quick return conditions are met:

   - if there is no return value

     - Return ``rocblas_status_success``

   - If there is a return value

     - If the return value pointer argument is nullptr, return ``rocblas_status_invalid_pointer``

     - Else, return ``rocblas_status_success``

9. Return ``rocblas_status_invalid_pointer`` if any pointers not checked in #7 are NULL.

10. (Optional.) Allocate device memory, returning ``rocblas_status_memory_error`` if the allocation fails.

11. If all checks above pass, launch the kernel and return ``rocblas_status_success``.


Legacy L1 BLAS "single vector"
==============================

Below are four code snippets from NETLIB for "single vector" legacy L1 BLAS. They have quick-return-success for (n <= 0) || (incx <= 0)

.. code-block:: bash

      DOUBLE PRECISION FUNCTION DASUM(N,DX,INCX)
      IF (N.LE.0 .OR. INCX.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX)
      IF (N.LT.1 .OR. INCX.LT.1) THEN
          return = ZERO

      SUBROUTINE DSCAL(N,DA,DX,INCX)
      IF (N.LE.0 .OR. INCX.LE.0) RETURN

      INTEGER FUNCTION IDAMAX(N,DX,INCX)
      IDAMAX = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      IDAMAX = 1
      IF (N.EQ.1) RETURN

Legacy L1 BLAS "two vector"
===========================

Below are seven legacy L1 BLAS codes from NETLIB. There is quick-return-success for (n <= 0). In addition, for DAXPY, there is quick-return-success for (alpha == 0)

.. code-block::

      SUBROUTINE DAXPY(N,alpha,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN
      IF (alpha.EQ.0.0d0) RETURN

      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      SUBROUTINE DROT(N,DX,INCX,DY,INCY,C,S)
      IF (N.LE.0) RETURN

      SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DSDOT(N,SX,INCX,SY,INCY)
      IF (N.LE.0) RETURN

      SUBROUTINE DROTM(N,DX,INCX,DY,INCY,DPARAM)
      DFLAG = DPARAM(1)
      IF (N.LE.0 .OR. (DFLAG+TWO.EQ.ZERO)) RETURN

Legacy L2 BLAS
==============
Below are code snippets from NETLIB for legacy L2 BLAS. They have both argument checking and quick-return-success.

.. code-block::

      SUBROUTINE DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
      INFO = 0
      IF (M.LT.0) THEN
          INFO = 1
      ELSE IF (N.LT.0) THEN
          INFO = 2
      ELSE IF (INCX.EQ.0) THEN
          INFO = 5
      ELSE IF (INCY.EQ.0) THEN
          INFO = 7
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 9
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGER  ',INFO)
          RETURN
      END IF

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (ALPHA.EQ.ZERO)) RETURN

.. code-block::

      SUBROUTINE DSYR(UPLO,N,ALPHA,X,INCX,A,LDA)

      INFO = 0
      IF (.NOT.LSAME(UPLO,'U') .AND. .NOT.LSAME(UPLO,'L')) THEN
          INFO = 1
      ELSE IF (N.LT.0) THEN
          INFO = 2
      ELSE IF (INCX.EQ.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,N)) THEN
          INFO = 7
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DSYR  ',INFO)
          RETURN
      END IF

      IF ((N.EQ.0) .OR. (ALPHA.EQ.ZERO)) RETURN

.. code-block::

      SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)

      INFO = 0
      IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND. .NOT.LSAME(TRANS,'C')) THEN
          INFO = 1
      ELSE IF (M.LT.0) THEN
          INFO = 2
      ELSE IF (N.LT.0) THEN
          INFO = 3
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      ELSE IF (INCY.EQ.0) THEN
          INFO = 11
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMV ',INFO)
          RETURN
      END IF

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. ((ALPHA.EQ.ZERO).AND. (BETA.EQ.ONE))) RETURN

.. code-block::

      SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)

      INFO = 0
      IF (.NOT.LSAME(UPLO,'U') .AND. .NOT.LSAME(UPLO,'L')) THEN
          INFO = 1
      ELSE IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND. .NOT.LSAME(TRANS,'C')) THEN
          INFO = 2
      ELSE IF (.NOT.LSAME(DIAG,'U') .AND. .NOT.LSAME(DIAG,'N')) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (LDA.LT.MAX(1,N)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DTRSV ',INFO)
          RETURN
      END IF

      IF (N.EQ.0) RETURN

Legacy L3 BLAS
==============

Below is a code snippet from NETLIB for legacy L3 BLAS dgemm. It has both argument checking and quick-return-success.

.. code-block::

      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)

      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NROWA = M
          NCOLA = K
      ELSE
          NROWA = K
          NCOLA = M
      END IF
      IF (NOTB) THEN
          NROWB = K
      ELSE
          NROWB = N
      END IF

  //  Test the input parameters.

      INFO = 0
      IF ((.NOT.NOTA) .AND. (.NOT.LSAME(TRANSA,'C')) .AND.
     +    (.NOT.LSAME(TRANSA,'T'))) THEN
          INFO = 1
      ELSE IF ((.NOT.NOTB) .AND. (.NOT.LSAME(TRANSB,'C')) .AND.
     +         (.NOT.LSAME(TRANSB,'T'))) THEN
          INFO = 2
      ELSE IF (M.LT.0) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (K.LT.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
          INFO = 8
      ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
          INFO = 10
      ELSE IF (LDC.LT.MAX(1,M)) THEN
          INFO = 13
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMM ',INFO)
          RETURN
      END IF

  //  Quick return if possible.

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (((ALPHA.EQ.ZERO).OR. (K.EQ.0)).AND. (BETA.EQ.ONE))) RETURN


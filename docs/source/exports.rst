
.. toctree::
   :maxdepth: 4
   :caption: Contents:


***********************
Exported BLAS functions
***********************

Auxiliary functions
==================================================

+--------------------------+
|Function name             |
+--------------------------+
| rocblas_create_handle    |
+--------------------------+
| rocblas_destroy_handle   |
+--------------------------+
| rocblas_add_stream       |
+--------------------------+
| rocblas_set_stream       |
+--------------------------+
| rocblas_get_stream       |
+--------------------------+
| rocblas_set_pointer_mode |
+--------------------------+
| rocblas_get_pointer_mode |
+--------------------------+
| rocblas_set_vector       |
+--------------------------+
| rocblas_get_vector       |
+--------------------------+
| rocblas_set_matrix       |
+--------------------------+
| rocblas_get_matrix       |
+--------------------------+

BLAS functions
==========================================================

Level 1
-------

============================== ====== ====== ============== ==============
Function                       single double single complex double complex
============================== ====== ====== ============== ==============
**rocblas_Xscal**              x      x      x              x
rocblas_Xscal_batched          x      x      x              x
rocblas_Xscal_strided_batched  x      x      x              x
**rocblas_Xcopy**              x      x      x              x
rocblas_Xcopy_batched          x      x      x              x
rocblas_Xcopy_strided_batched  x      x      x              x
**rocblas_Xdot**               x      x      x              x
rocblas_Xdot_batched           x      x      x              x
rocblas_Xdot_strided_batched   x      x      x              x
**rocblas_Xdotc**                            x              x
rocblas_Xdotc_batched                        x              x
rocblas_Xdotc_strided_batched                x              x
**rocblas_Xswap**              x      x      x              x
rocblas_Xswap_batched          x      x      x              x
rocblas_Xswap_strided_batched  x      x      x              x
**rocblas_Xaxpy**              x      x      x              x
rocblas_Xaxpy_batched          x      x      x              x
rocblas_Xaxpy_strided_batched  x      x      x              x
**rocblas_Xasum**              x      x      x              x
rocblas_Xasum_batched          x      x      x              x
rocblas_Xasum_strided_batched  x      x      x              x
**rocblas_Xnrm2**              x      x      x              x
rocblas_Xnrm2_batched          x      x      x              x
rocblas_Xnrm2_strided_batched  x      x      x              x
**rocblas_iXamax**             x      x      x              x
rocblas_iXamax_batched         x      x      x              x
rocblas_iXamax_strided_batched x      x      x              x
**rocblas_iXamin**             x      x      x              x
rocblas_iXamin_batched         x      x      x              x
rocblas_iXamin_strided_batched x      x      x              x
**rocblas_Xrot**               x      x      x              x
rocblas_Xrot_batched           x      x      x              x
rocblas_Xrot_strided_batched   x      x      x              x
**rocblas_Xrotg**              x      x      x              x
rocblas_Xrotg_batched          x      x      x              x
rocblas_Xrotg_strided_batched  x      x      x              x
**rocblas_Xrotm**              x      x
rocblas_Xrotm_batched          x      x
rocblas_Xrotm_strided_batched  x      x
**rocblas_Xrotmg**             x      x
rocblas_Xrotmg_batched         x      x
rocblas_Xrotmg_strided_batched x      x
============================== ====== ====== ============== ==============

Level 2
-------

============================= ====== ====== ============== ==============
Function                      single double single complex double complex
============================= ====== ====== ============== ==============
**rocblas_Xgemv**             x      x      x              x
rocblas_Xgemv_batched         x      x      x              x
rocblas_Xgemv_strided_batched x      x      x              x
**rocblas_Xger**              x      x
rocblas_Xger_batched          x      x
rocblas_Xger_strided_batched  x      x
**rocblas_Xsyr**              x      x
rocblas_Xsyr_batched          x      x
rocblas_Xsyr_strided_batched  x      x
**rocblas_Xtrsv**             x      x
rocblas_Xtrsv_batched         x      x
rocblas_Xtrsv_strided_batched x      x
============================= ====== ====== ============== ==============

Level 3
-------

============================== ====== ====== ============== ==============
Function                       single double single complex double complex
============================== ====== ====== ============== ==============
**rocblas_Xtrtri**             x      x
rocblas_Xtrtri_batched         x      x
rocblas_Xtrtri_strided_batched x      x
**rocblas_Xtrsm**              x      x
rocblas_Xtrsm_batched          x      x
rocblas_Xtrsm_strided_batched  x      x
**rocblas_Xtrmm**              x      x
**rocblas_Xgemm**              x      x      x              x
rocblas_Xgemm_batched          x      x      x              x
rocblas_Xgemm_strided_batched  x      x      x              x
**rocblas_Xgeam**              x      x
============================== ====== ====== ============== ==============

BLAS extensions
---------------

See :ref:`api_label` for more information on these extended BLAS functions.

+---------------------------------+
| Function name                   |
+---------------------------------+
| **rocblas_gemm_ex**             |
+---------------------------------+
| rocblas_gemm_batched_ex         |
+---------------------------------+
| rocblas_gemm_strided_batched_ex |
+---------------------------------+
| **rocblas_trsm_ex**             |
+---------------------------------+
| rocblas_trsm_batched_ex         |
+---------------------------------+
| rocblas_trsm_strided_batched_ex |
+---------------------------------+

Rules for obtaining the rocBLAS API from Legacy BLAS
====================================================

1. The Legacy BLAS routine name is changed to lower case, and prefixed
   by rocblas\_.

2. A first argument rocblas_handle handle is added to all rocBlas
   functions.

3. Input arguments are declared with the const modifier.

4. Character arguments are replaced with enumerated types defined in
   rocblas_types.h. They are passed by value on the host.

5. Array arguments are passed by reference on the device.

6. Scalar arguments are passed by value on the host with the following
   two exceptions:

-  Scalar values alpha and beta are passed by reference on either the
   host or the device. The rocBLAS functions will check to see it the
   value is on the device. If this is true, it is used, else the value
   on the host is used.

-  Where Legacy BLAS functions have return values, the return value is
   instead added as the last function argument. It is returned by
   reference on either the host or the device. The rocBLAS functions
   will check to see it the value is on the device. If this is true, it
   is used, else the value is returned on the host. This applies to the
   following functions: xDOT, xDOTU, xNRM2, xASUM, IxAMAX, IxAMIN.

7. The return value of all functions is rocblas_status, defined in
   rocblas_types.h. It is used to check for errors.

rocBLAS interface examples
==========================

In general, the rocBLAS interface is compatible with CPU oriented
`Netlib BLAS <http://www.netlib.org/blas/>`__ and the cuBLAS-v2 API, with the explicit exception that
traditional BLAS interfaces do not accept handles. The cuBLAS’
cublasHandle_t is replaced with rocblas_handle everywhere. Thus, porting
a CUDA application which originally calls the cuBLAS API to a HIP
application calling rocBLAS API should be relatively straightforward.
For example, the rocBLAS SGEMV interface is:

GEMV API
--------

.. code:: c

   rocblas_status
   rocblas_sgemv(rocblas_handle handle,
                 rocblas_operation trans,
                 rocblas_int m, rocblas_int n,
                 const float* alpha,
                 const float* A, rocblas_int lda,
                 const float* x, rocblas_int incx,
                 const float* beta,
                 float* y, rocblas_int incy);

LP64 interface
==============

The rocBLAS library is LP64, so rocblas_int arguments are 32 bit and
rocblas_long arguments are 64 bit.

Column-major storage and 1 based indexing
=========================================

rocBLAS uses column-major storage for 2D arrays, and 1 based indexing
for the functions xMAX and xMIN. This is the same as Legacy BLAS and
cuBLAS.

If you need row-major and 0 based indexing (used in C language arrays)
download the `CBLAS <http://www.netlib.org/blas/#_cblas>`__ file
cblas.tgz. Look at the CBLAS functions that provide a thin interface to
Legacy BLAS. They convert from row-major, 0 based, to column-major, 1
based. This is done by swapping the order of function arguments. It is
not necessary to transpose matrices.

Pointer mode
============

The auxiliary functions rocblas_set_pointer and rocblas_get_pointer are
used to set and get the value of the state variable
rocblas_pointer_mode. If rocblas_pointer_mode ==
rocblas_pointer_mode_host then scalar parameters must be allocated on
the host. If rocblas_pointer_mode == rocblas_pointer_mode_device, then
scalar parameters must be allocated on the device.

There are two types of scalar parameter: 1. scaling parameters like
alpha and beta used in functions like axpy, gemv, gemm 2. scalar results
from functions amax, amin, asum, dot, nrm2

For scalar parameters like alpha and beta when rocblas_pointer_mode ==
rocblas_pointer_mode_host they can be allocated on the host heap or
stack. The kernel launch is asynchronous, and if they are on the heap
they can be freed after the return from the kernel launch. When
rocblas_pointer_mode == rocblas_pointer_mode_device they must not be
changed till the kernel completes.

For scalar results, when rocblas_pointer_mode ==
rocblas_pointer_mode_host then the function blocks the CPU till the GPU
has copied the result back to the host. When rocblas_pointer_mode ==
rocblas_pointer_mode_device the function will return after the
asynchronous launch. Similarly to vector and matrix results, the scalar
result is only available when the kernel has completed execution.

Asynchronous API
================

Except a functions having memory allocation inside preventing
asynchronicity, most of the rocBLAS functions are configured to operate
in non-blocking fashion with respect to CPU, meaning these library
functions return immediately.

hipBLAS
=======

hipBLAS is a BLAS marshalling library, with multiple supported backends. It sits between the application and a 'worker' BLAS library, marshalling inputs
into the backend library and marshalling results back to the application. hipBLAS exports an interface that does not require the client to change,
regardless of the chosen backend. Currently hipBLAS supports rocBLAS and cuBLAS as backends.

hipBLAS focuses on convenience and portability. If performance outweighs these factors then using rocBLAS itself is recommended.

hipBLAS can be found on github `here <https://github.com/ROCmSoftwarePlatform/hipBLAS/>`__.

.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _what-is-rocblas:

********************************************************************
What is rocBLAS
********************************************************************

Introduction
============

rocBLAS is the AMD library for Basic Linear Algebra Subprograms (BLAS) on the :doc:`ROCm platform <rocm:index>`.
It is implemented in the :doc:`HIP programming language <hip:index>` and optimized for AMD GPUs.

The aim of rocBLAS is to provide:

- Functionality similar to legacy BLAS, adapted to run on GPUs
- High-performance robust implementation

rocBLAS is written in C++17 and HIP, and uses the AMD ROCm runtime to run on GPU devices.

The rocBLAS API is a thin C99 API using the Hourglass Pattern. It contains:

- :ref:`level-1`, :ref:`level-2`, and :ref:`level-3` with batched and strided_batched versions
- Extensions to legacy BLAS, including functions for mixed precision
- Auxiliary functions
- Device Memory functions

.. note::
  - The official rocBLAS API is the C99 API defined in ``rocblas.h``, therefore the use of any other public symbols is discouraged. All other C/C++ interfaces may not follow a deprecation model and so can change without warning from one release to the next.
  - rocBLAS array storage format is column major and one based. This is to maintain compatibility with the Legacy BLAS code, which is written in Fortran.
  - rocBLAS calls the AMD Tensile and hipBLASLt libraries for Level 3 GEMMs (matrix matrix multiplication).

Use of Tensile and hipBLASLt
==============

The rocBLAS library internally uses
`Tensile <https://github.com/ROCm/Tensile>`__ and `hipBLASLt <https://github.com/ROCm/hipBLASLt>`__, which
supply the high-performance implementation of GEMM. They are installed as part of the rocBLAS package.
rocBLAS uses CMake for build automation, and CMake downloads Tensile and hipBLASLt during library configuration and automatically
configures them as part of the build, so no further action is required by the
user to set it up.  No external facing API for Tensile or hipBLASLt are provided.

The choice of whether to use Tensile or hipBLASLt is handled automatically based on architecture and data types
The environment variable ``ROCBLAS_USE_HIPBLASLT`` is provided to manually control which GEMM backend is used in the following ways:

- ``ROCBLAS_USE_HIPBLASLT is not set``: the GEMM backend is automatically selected.
- ``ROCBLAS_USE_HIPBLASLT=0``: **Tensile** is always used as the GEMM backend.
- ``ROCBLAS_USE_HIPBLASLT=1``: **hipBLASLt** is preferred as the GEMM backend, but will fallback to **Tensile** on problems for which **hipBLASLt** does not provide a solution or when errors are encountered using the **hipBLASLt* backend.

Note that hipBLASLt in rocBLAS is not currently supported in Windows builds or static builds, and will not be included if building without Tensile.

rocBLAS API and legacy BLAS functions
=====================================

rocBLAS is initialized by calling ``rocblas_create_handle``, and it is terminated by calling ``rocblas_destroy_handle``. The rocblas_handle is persistent and contains:

- HIP stream
- Temporary device workspace
- Mode for enabling or disabling logging (default is logging disabled)

rocBLAS functions run on the host, and they call HIP to launch rocBLAS kernels that run on the device in a HIP stream. The kernels are asynchronous unless:

- The function returns a scalar result from device to host
- Temporary device memory is allocated

In both cases above, the launch can be made asynchronous by:

- Use ``rocblas_pointer_mode_device`` to keep the scalar result on the device. Note that only the following Level1 BLAS functions that return a scalar result: ``Xdot``, ``Xdotu``, ``Xnrm2``, ``Xasum``, ``iXamax``, ``iXamin``.
- Use the provided device memory functions to allocate device memory that persists in the handle. Note that most rocBLAS functions do not allocate temporary device memory.

Before calling a rocBLAS function, arrays must be copied to the device. Integer scalars like m, n, k are stored on the host. Floating point scalars like alpha and beta can be on host or device.

Error handling is by returning a ``rocblas_status``. Functions conform to the legacy BLAS argument checking.


Rules for obtaining rocBLAS API from legacy BLAS functions
----------------------------------------------------------

1. The legacy BLAS routine name is changed to lowercase and prefixed by ``rocblas_<function>``.
   For example the legacy BLAS routine ``SSCAL`` which scales a vector by a constant value, is replaced with ``rocblas_sscal``.

2. A first argument ``rocblas_handle`` handle is added to all rocBLAS functions.

3. Input arguments are declared with the ``const`` modifier.

4. Character arguments are replaced with enumerated types defined in
   ``rocblas_types.h``. They are passed by value on the host.

5. Array arguments are passed by reference on the device.

6. Scalar arguments are passed by value on the host with the following
   exceptions. See the section :ref:`pointer-mode` for more information on
   these exceptions:

   -  Scalar values alpha and beta are passed by reference on either the
      host or the device.
   -  Where Legacy BLAS functions have return values, the return value is
      instead added as the last function argument. It is returned by
      reference on either the host or the device. This applies to the
      following functions: ``xDOT``, ``xDOTU``, ``xNRM2``, ``xASUM``, ``IxAMAX``, ``IxAMIN``.

7. The return value of all functions is ``rocblas_status``, defined in
   ``rocblas_types.h``. It is used to check for errors.


rocBLAS Example Code
====================

Below is a simple example for calling function ``rocblas_sscal``:

.. code-block:: c++

   #include <iostream>
   #include <vector>
   #include "hip/hip_runtime_api.h"
   #include "rocblas.h"

   using namespace std;

   int main()
   {
       rocblas_int n = 10240;
       float alpha = 10.0;

       vector<float> hx(n);
       vector<float> hz(n);
       float* dx;

       rocblas_handle handle;
       rocblas_create_handle(&handle);

       // allocate memory on device
       hipMalloc(&dx, n * sizeof(float));

       // Initial Data on CPU,
       srand(1);
       for( int i = 0; i < n; ++i )
       {
           hx[i] = rand() % 10 + 1;  //generate a integer number between [1, 10]
       }

       // copy array from host memory to device memory
       hipMemcpy(dx, hx.data(), sizeof(float) * n, hipMemcpyHostToDevice);

       // call rocBLAS function
       rocblas_status status = rocblas_sscal(handle, n, &alpha, dx, 1);

       // check status for errors
       if(status == rocblas_status_success)
       {
           cout << "status == rocblas_status_success" << endl;
       }
       else
       {
           cout << "rocblas failure: status = " << status << endl;
       }

       // copy output from device memory to host memory
       hipMemcpy(hx.data(), dx, sizeof(float) * n, hipMemcpyDeviceToHost);

       hipFree(dx);
       rocblas_destroy_handle(handle);
       return 0;
   }


LP64 Interface
--------------

The rocBLAS library default implementations are LP64, so ``rocblas_int`` arguments are 32 bit and
``rocblas_stride`` arguments are 64 bit.

.. _ILP64 API:

ILP64 Interface
---------------

The rocBLAS library Level-1 functions are also provided with ILP64 interfaces. With these interfaces all ``rocblas_int`` arguments are replaced by the typename
``int64_t``.  These ILP64 function names all end with a suffix ``_64``. The only output arguments that change are for the
``xMAX`` and ``xMIN`` for which the index is now ``int64_t``.  Performance should match the LP64 API when problem sizes don't require the additional
precision.  Function level documentation is not repeated for these API as they are identical in behavior to the LP64 versions,
however functions which support this alternate API include the line:
``This function supports the 64-bit integer interface (ILP64)``.

Column-major Storage and 1 Based Indexing
-----------------------------------------

rocBLAS uses column-major storage for 2D arrays, and 1-based indexing
for the functions ``xMAX`` and ``xMIN``. This is the same as legacy BLAS and
cuBLAS.

If you need row-major and 0-based indexing (used in C language arrays), download the file ``cblas.tgz`` from the Netlib Repository.
Look at the CBLAS functions that provide a thin interface to legacy BLAS. They convert from row-major, 0 based, to column-major, 1
based. This is done by swapping the order of function arguments. It is not necessary to transpose matrices.

.. _pointer-mode:

Pointer Mode
------------

The auxiliary functions ``rocblas_set_pointer`` and ``rocblas_get_pointer`` are
used to set and get the value of the state variable
``rocblas_pointer_mode``. This variable is stored in ``rocblas_handle``. If ``rocblas_pointer_mode ==
rocblas_pointer_mode_host``, then scalar parameters must be allocated on
the host. If ``rocblas_pointer_mode == rocblas_pointer_mode_device``, then
scalar parameters must be allocated on the device.

There are two types of scalar parameter:

* Scaling parameters like alpha and beta used in functions like ``axpy``, ``gemv``, ``gemm 2``
* Scalar results from functions ``amax``, ``amin``, ``asum``, ``dot``, ``nrm2``

For scalar parameters like alpha and beta when ``rocblas_pointer_mode ==
rocblas_pointer_mode_host``, they can be allocated on the host heap or
stack. The kernel launch is asynchronous, and if they are on the heap,
they can be freed after the return from the kernel launch. When
``rocblas_pointer_mode == rocblas_pointer_mode_device`` they must not be
changed till the kernel completes.

For scalar results, when ``rocblas_pointer_mode ==
rocblas_pointer_mode_host``, then the function blocks the CPU till the GPU
has copied the result back to the host. When ``rocblas_pointer_mode ==
rocblas_pointer_mode_device`` the function will return after the
asynchronous launch. Similarly to vector and matrix results, the scalar
result is only available when the kernel has completed execution.

Asynchronous API
----------------

rocBLAS functions will be asynchronous unless:

* The function needs to allocate device memory
* The function returns a scalar result from GPU to CPU

The order of operations in the asynchronous functions is as in the figure
below. The argument checking, calculation of process grid, and kernel
launch take very little time. The asynchronous kernel running on the GPU
does not block the CPU. After the kernel launch, the CPU keeps processing
the next instructions.

.. asynch_blocks
.. figure:: ../data/asynch_function.PNG
   :alt: code blocks in asynch function call
   :align: center

   Order of operations in asynchronous functions


The above order of operations will change if there is logging or the
function is synchronous. Logging requires system calls, and the program
must wait for them to complete before executing the next instruction.
See the Logging section for more information.

.. note::
   The default is no logging.

If the CPU needs to allocate device memory, it must wait until memory allocation is complete before
executing the next instruction. For more detailed information, refer to sections :ref:`Device Memory Allocation Usage` and :ref:`Device Memory allocation in detail`.

.. note::
   Memory can be pre-allocated. This will make the function asynchronous, as it removes the need for the function to allocate memory.

The following functions copy a scalar result from GPU to CPU if
``rocblas_pointer_mode == rocblas_pointer_mode_host``: ``asum``, ``dot``, ``max``, ``min``, ``nrm2``.

This makes the function synchronous, as the program must wait
for the copy before executing the next instruction. See :ref:`pointer-mode` for more information.

.. note::
   Set ``rocblas_pointer_mode == rocblas_pointer_mode_device`` makes the function asynchronous by keeping the result on the GPU.

The order of operations with logging, device memory allocation, and return of a scalar
result is as in the figure below:

.. asynch_blocks
.. figure:: ../data/synchronous_function.PNG
   :alt: code blocks in synchronous function call
   :align: center

   Code blocks in synchronous function call

Kernel launch status error checking
-----------------------------------

The function ``hipPeekAtLastError()`` is called before and after rocblas kernel launches. This will detect if launch parameters are incorrect, for example
invalid work-group or thread block sizes. It will also detect if the kernel code can not run on the current GPU device (returns ``rocblas_status_arch_mismatch``).
Note that ``hipPeekAtLastError()`` does not flush the last error. Reporting only a change in ``hipPeekAtLastError()`` as a detection system has the disadvantage
that if the previous last error from another kernel launch or hip call is the same as the error from the current kernel, then no error is reported.
Only the first error would be reported in this case.  You can avoid this behaviour by flushing any previous hip error before calling a rocBLAS function
by calling ``hipGetLastError()``. Note that both ``hipPeekAtLastError()`` and ``hipGetLastError()`` run synchronously on the CPU and they only check the kernel
launch, not the asynchronous work done by the kernel.  We do not clear the last error in case the caller was relying on it for detecting errors in
a batch of hip and rocBLAS function calls.

Complex Number Data Types
-------------------------

Data types for rocBLAS complex numbers in the API are a special case.  For C compiler users, gcc, and other non-amdclang compiler users, these types
are exposed as a struct with x and y components and identical memory layout to std::complex for float and double precision.   Internally a templated
C++ class is defined, but it should be considered deprecated for external use.   For simplified usage with Hipified code there is an option
to interpret the API as using hipFloatComplex and hipDoubleComplex types (i.e. typedef hipFloatComplex rocblas_float_complex).  This is provided
for users to avoid casting when using the hip complex types in their code.  As the memory layout is consistent across all three types,
it is safe to cast arguments to API calls between the 3 types: hipFloatComplex, std::complex<float>, and rocblas_float_complex, as well as for
the double precision variants. To expose the API as using the hip defined complex types, user can use either a compiler define or inlined
#define ROCM_MATHLIBS_API_USE_HIP_COMPLEX before including the header file <rocblas.h>.  Thus the API is compatible with both forms, but
recompilation is required to avoid casting if switching to pass in the hip complex types.  Most device memory pointers are passed with void*
types to hip utility functions (e.g. hipMemcpy), so uploading memory from std::complex arrays or hipFloatComplex arrays requires no changes
regardless of complex data type API choice.

.. _Atomic Operations:

Atomic Operations
-----------------

Some functions within the rocBLAS library such as gemv, symv, trsv, trsm, and gemm may use atomic operations to increase performance.
By using atomics, functions may not give bit-wise reproducible results. Differences between multiple runs should not be significant and will
remain accurate, but if users require identical results across multiple runs, atomics should be turned off. See :any:`rocblas_atomics_mode`,
:any:`rocblas_set_atomics_mode`, and :any:`rocblas_get_atomics_mode`.

In addition to the above API, rocBLAS also provides an environment variable ``ROCBLAS_DEFAULT_ATOMICS_MODE``, which allows users to set the default atomics mode during the creation of ``rocblas_handle``.
:any:`rocblas_set_atomics_mode` has higher precedence and users can use the API in the application to override the configuration set via the environment variable.

* ``ROCBLAS_DEFAULT_ATOMICS_MODE = 0`` : To set the default to be :any:`rocblas_atomics_not_allowed`
* ``ROCBLAS_DEFAULT_ATOMICS_MODE = 1`` : To set the atomics to be :any:`rocblas_atomics_allowed`

Bitwise Reproducibility
-----------------------

Bitwise reproducible results in rocBLAS can be obtained under the following conditions:

* Identical GFX target ISA
* Single HIP stream active per rocBLAS handle
* Identical ROCm versions
* Disabled atomic operations ( for more infromation, see :ref:`Atomic Operations`)


By default rocBLAS may use atomic operations to achieve better performance in some functions.
To ensure bitwise reproducible results, where users require identical results across multiple runs, the following functions require atomics to be disabled

=================================
Functions using atomic operations
=================================

 :any:`rocblas_sgemv`
 :any:`rocblas_dgemv`

 :any:`rocblas_ssymv`
 :any:`rocblas_dsymv`

 :any:`rocblas_strsv`
 :any:`rocblas_dtrsv`
 :any:`rocblas_ztrsv`
 :any:`rocblas_ctrsv`

 :any:`rocblas_strsm`
 :any:`rocblas_dtrsm`
 :any:`rocblas_ztrsm`
 :any:`rocblas_ctrsm`

 :any:`rocblas_sgemm`
 :any:`rocblas_dgemm`
 :any:`rocblas_hgemm`
 :any:`rocblas_zgemm`
 :any:`rocblas_cgemm`

=======================

.. note::

   Functions such as GEMV and TRSM uses temporary device memory to use optimized kernels to achieve higher performance.
   If device memory is unavailable, the functions will proceed to use an unoptimized kernel, this could also produce variable results.
   Users will be notified if the kernel used is unoptimized by returning :any:`rocblas_status_perf_degraded` status.

All other functions except the above-mentioned are bitwise reproducible by default.

MI100 (gfx908) Considerations
-----------------------------

On nodes with the MI100 (gfx908), MFMA (Matrix-Fused-Multiply-Add)
instructions are available to substantially speed up matrix operations.
This hardware feature is used in all gemm and gemm-based functions in
rocBLAS with 32-bit or shorter base datatypes with an associated 32-bit
compute_type (f32_r, i32_r, or f32_c as appropriate).

Specifically, rocBLAS takes advantage of MI100's MFMA instructions for
three real base types f16_r, bf16_r, and f32_r with compute_type f32_r,
one integral base type i8_r with compute_type i32_r, and one complex
base type f32_c with compute_type f32_c.  In summary, all GEMM APIs and
APIs for GEMM-based functions using these five base types and their
associated compute_type (explicit or implicit) take advantage of MI100's
MFMA instructions.

.. note::
   The use of MI100's MFMA instructions is automatic.  There is no user control for on/off.

   Not all problem sizes may select MFMA-based kernels; additional tuning may be needed to get good performance.

MI200 (gfx90a) Considerations
-----------------------------

On nodes with the MI200 (gfx90a), MFMA_F64 instructions are available to
substantially speed up double precision matrix operations.  This
hardware feature is used in all GEMM and GEMM-based functions in
rocBLAS with 64-bit floating-point datatype, namely ``DGEMM``, ``ZGEMM``,
``DTRSM``, ``ZTRSM``, ``DTRMM``, ``ZTRMM``, ``DSYRKX``, and ``ZSYRKX``.

The MI200 ``MFMA_F16``, ``MFMA_BF16`` and ``MFMA_BF16_1K`` instructions
flush subnormal input/output data ("denorms") to zero. It is observed that
certain use cases utilizing the HPA (High Precision Accumulate) HGEMM
kernels where ``a_type=b_type=c_type=d_type=f16_r`` and ``compute_type=f32_r``
do not tolerate the MI200's flush-denorms-to-zero behavior well
due to F16's limited exponent range. An alternate implementation of the
HPA HGEMM kernel utilizing the MFMA_BF16_1K instruction is provided which,
takes advantage of BF16's much larger exponent range, albeit with reduced
accuracy.  To select the alternate implementation of HPA HGEMM with the
``gemm_ex``/``gemm_strided_batched_ex`` functions, for the flags argument, use
the enum value of ``rocblas_gemm_flags_fp16_alt_impl``.

.. note::
   The use of MI200's MFMA instructions (including MFMA_F64) is automatic.  There is no user control for on/off.

   Not all problem sizes may select MFMA-based kernels; additional tuning may be needed to get good performance.


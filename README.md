# rocBLAS
A BLAS implementation on top of AMD's Radeon Open Compute [ROCm][] runtime and toolchains.  rocBLAS is implemented in
the [HIP][] programming language, optimized for AMD's latest discrete GPUs and allowing it to run on CUDA enabled GPUs.

## Migrating libraries to ROCm from OpenCL
[clBLAS][] demonstrated significant performance benefits of data parallel (GPU) computation when applied to solving dense
linear algebra problems, but OpenCL primarily remains in the domain of expert programmers. The ROCm model introduces a
single source paradigm for integrating device and host code together in a single source file, thereby simplifying the
entire development process for heterogeneous computing.  Compilers will get smarter, catching errors at compile/build time
and native profilers/debuggers will better integrate into the development process.  As AMD simplifies the
programming model with ROCm (using HCC and HIP), it is the intent of this library to expose that simplified programming
model to end users.  

## rocBLAS interface
In general, the rocBLAS interface is compatible with [Netlib BLAS][] and the cuBLAS-v2 API, with the explicit exception that
Netlib BLAS does not have handle.  The cuBLAS' cublasHandle_t is replaced with rocblas_handle everywhere. Thus, porting a
CUDA application which originally calls the cuBLAS API to a HIP application calling rocBLAS API should be relatively
straightforward.  For example, the rocBLAS SGEMV interface is

```c
rocblas_status
rocblas_sgemv(rocblas_handle handle,
              rocblas_operation trans,
              rocblas_int m, rocblas_int n,
              const float* alpha,
              const float* A, rocblas_int lda,
              const float* x, rocblas_int incx,
              const float* beta,
              float* y, rocblas_int incy);

```

rocBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data.  Users are
responsible for copying data from/to the host and device memory.  HIP provides memcpy style API's to facilitate data
management.

##Asynchronous API
Except a few routines (like TRSM) having memory allocation inside preventing asynchronicity, most of the library routines
(like BLAS-1 SCAL, BLAS-2 GEMV, BLAS-3 GEMM) are configured to operate in asynchronous fashion with respect to CPU,
meaning that these library function calls return immediately.

##Batched and strided GEMM API
rocBLAS GEMM can process matrices in batches with regular strides.  There are several permutations of these API's, the
following is an example that takes everything

```c
rocblas_status
rocblas_sgemm_strided_batched(
    rocblas_handle handle,
    rocblas_order order,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float* alpha,
    const float* A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const float* B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const float* beta,
          float* C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count )
```

##rocBLAS Wiki

The [wiki][] has helpful information about building the rocblas library, samples and tests.

[wiki]: https://github.com/RadeonOpenCompute/rocBLAS/wiki

[ROCm]: https://radeonopencompute.github.io/

[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/

[Netlib BLAS]: http://www.netlib.org/blas/

[clBLAS]: https://github.com/clMathLibraries/clBLAS

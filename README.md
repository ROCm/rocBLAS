# rocBLAS
A BLAS implementation on top of AMD's Radeon Open Compute [ROCm][] runtime and toolchains.  rocBLAS is implemented in the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## Installing pre-built packages
Download pre-built packages either from [ROCm's package servers](https://rocm.github.io/install.html#installing-from-amd-rocm-repositories) or by clicking the github releases tab and manually downloading, which could be newer.  Release notes are available for each release on the releases tab.
* `sudo apt update && sudo apt install rocblas`

## Quickstart rocBLAS build

#### Bash helper build script (Ubuntu only)
The root of this repository has a helper bash script `install.sh` to build and install rocBLAS on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.  A few commands in the script need sudo access, so it may prompt you for a password.
*  `./install -h` -- shows help
*  `./install -id` -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

## Manual build (all supported platforms)
If you use a distro other than Ubuntu, or would like more control over the build process, the [rocblas build wiki](https://github.com/RadeonOpenCompute/rocBLAS/wiki/Build) has helpful information on how to configure cmake and manually build.

### Functions supported
A list of [exported functions](https://github.com/RadeonOpenCompute/rocBLAS/wiki/exported-functions) from rocblas can be found on the wiki

## rocBLAS interface examples
In general, the rocBLAS interface is compatible with CPU oriented [Netlib BLAS][] and the cuBLAS-v2 API, with
the explicit exception that traditional BLAS interfaces do not accept handles.  The cuBLAS' cublasHandle_t is replaced
with rocblas_handle everywhere. Thus, porting a CUDA application which originally calls the cuBLAS API
to a HIP application calling rocBLAS API should be relatively straightforward. For example, the rocBLAS
SGEMV interface is

### GEMV API

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

### Batched and strided GEMM API
rocBLAS GEMM can process matrices in batches with regular strides.  There are several permutations of these API's, the
following is an example that takes everything

```c
rocblas_status
rocblas_sgemm_strided_batched(
    rocblas_handle handle,
    rocblas_operation transa, rocblas_operation transb,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float* alpha,
    const float* A, rocblas_int ls_a, rocblas_int ld_a, rocblas_int bs_a,
    const float* B, rocblas_int ls_b, rocblas_int ld_b, rocblas_int bs_b,
    const float* beta,
          float* C, rocblas_int ls_c, rocblas_int ld_c, rocblas_int bs_c,
    rocblas_int batch_count )
```

rocBLAS assumes matrices A and vectors x, y are allocated in GPU memory space filled with data.  Users are
responsible for copying data from/to the host and device memory.  HIP provides memcpy style API's to facilitate data
management.

## Asynchronous API
Except a few routines (like TRSM) having memory allocation inside preventing asynchronicity, most of the library routines
(like BLAS-1 SCAL, BLAS-2 GEMV, BLAS-3 GEMM) are configured to operate in asynchronous fashion with respect to CPU,
meaning these library functions return immediately.

[ROCm]: https://github.com/RadeonOpenCompute/ROCm

[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/

[Netlib BLAS]: http://www.netlib.org/blas/

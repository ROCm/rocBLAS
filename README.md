# rocBLAS
Radeon Open Compute BLAS implementation on top of AMD [ROCm][] runtime. 
rocBLAS is implemented with the [HIP][] programming language and optimized for AMD latest discrete GPUs.
It can also run on Nvidia GPUs as long as the CUDA enviroment is configured correctly.

## Migrating libraries to ROCm from OpenCL
A substantial investment has been made by AMD in developing and promoting OpenCL libraries to accelerate common math domains, such as [clBLAS][], [clFFT][], clRNG and clSparse.  These libraries have demonstrated significant performance benefits of data parallel (GPU) computation, but primarily remain in the domain of expert programmers. As AMD simplifies the programming model with ROCm, it would be beneficial to leverage the performance and learning present in the OpenCL libraries and carry that forward.

## rocBLAS interface
In general, rocBLAS interface is compatible with [Netlib BLAS][] and cuBLAS-v2 API except Netlib BLAS does not have handle and cuBLAS' cublasHandle_t is replaced with rocblas_handle everywhere. Thus porting a CUDA application calling cuBLAS API to a HIP application calling rocBLAS API is straightforward. 
For example, the rocBLAS SGEMV interface is

```c
rocblas_status
rocblas_sgemv(rocblas_handle handle,
              rocblas_operation trans,
              rocblas_int m, rocblas_int n,
              const float *alpha,
              const float *A, rocblas_int lda,
              const float *x, rocblas_int incx,
              const float *beta,
              float *y, rocblas_int incy);

```
where rocblas_int is an alias of int, rocblas_operation is a rocBLAS defined enum to specify the non/transpose operation in BLAS. rocBLAS assumed the required matrices A and vectors x, y are already allocated in the GPU memory space filled with data.
Users are repsonsible for copying the data from/to the host CPU memory to/from the GPU memory. HIP provides API to offload and retrieve data from the GPU.

##Asynchronous API
Except a few routines (like TRSM) having memory allocation inside preventing asynchronicity, most of the library routines (like BLAS-1 SCAL, BLAS-2 GEMV, BLAS-3 GEMM) are configured to operate in an asynchronous fashion to CPU, meaning that these library function calls will return to users immediately. 

##rocBLAS Wiki

The [wiki][] has helpful information about building rocblas library, samples and testing files. 

[wiki]: https://github.com/RadeonOpenCompute/rocBLAS/wiki

[ROCm]: https://radeonopencompute.github.io/

[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/

[Netlib BLAS]: http://www.netlib.org/blas/

[clBLAS]: https://github.com/clMathLibraries/clBLAS

[clFFT]: https://github.com/clMathLibraries/clFFT


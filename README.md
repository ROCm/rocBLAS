# rocBLAS
Radeon Open Compute BLAS implementation

# Migrating libraries to ROC
A substantial investment has been made by AMD in developing and promoting OpenCL libraries to accelerate common math domains, such as BLAS, FFT, RNG and Sparse.  These libraries have demonstrated significant performance benefits of data parallel (GPU) computation, but primarily remain in the domain of expert programmers.  OpenCL is not a simple API/runtime to develop for.  As AMD simplifies the programming model with ROC, it would be beneficial to leverage the performance and learning present in the OpenCL libraries and carry that forward.

## Do we still need libraries?
The ROC model introduces a single source paradigm for integrating device and host code together in a single source file, thereby simplifying the entire development process for heterogeneous computing.  Compilers will get smarter, catching errors at compile/build time and native profilers/debuggers will better integrate into the development process.  However, achieving high performance code is still a challenging task, and requires expert knowledge of both problem domain and compute architecture.  Finding experts with the intersection of these skills is hard, and it's useful to leverage their expertise in a shared module that can be 'developed by few, used by many'.  As demonstrated by the well established x86 market, compute libraries are still very important even though a majority of domain developers can program in high level languages like Python or C++.

## Simplicity and performance; can we have both?
The best case scenario for libraries moving forward, which I believe is possible to develop
 - provide an intuitive and familiar  interface, callable from ROC compatible compilers like Kalmar or python using programming paradigms familiar to those programming for a CPU
 - provide an advanced interface (callable from ROC compilers) that expose performance primitives upon request, like queue's and events which enable more advanced asynchronous operations
 - provide OpenCL compatible interfaces for users with existing OpenCL code and who value the platform agnostic nature of OpenCL

I believe it is possible to provide interface(s) that satisfy all bullet points above.  By default, we can have the libraries configured to operate in a synchronous fashion, meaning that library function calls do not return to user until the library operation is complete.  This models familiar behavior to user's accustomed to CPU style programming, in which work is done in the calling thread and it doesn't return until that work is done.  However, the library state can be set to operate asynchronously, in which it is up to the user to manage 'event' objects and appropriately 'wait' on them as they determine.  This model is for power users who wish to control and manage scheduling decisions for themselves, potentially increasing application performance.  The decision to use OpenCL as a management runtime (for portability) or not can be configured as a build time option, in which if the user elects to build libraries with OpenCL (they are all open source), the API's reconfigure themselves to accept and return OpenCL state objects.  

## rocBLAS interface
A new interface for rocBLAS is proposed, and documented with doxygen annotations.  The name remains as rocBLAS for now, but I recognize that might not make sense for upcoming ROC platforms.  This proposed interface does not necessarily impose that the library name should stay the same.  hcBLAS is another possibility.

Porting rocBLAS to ROC necessarily introduces changes to the API and library, but also presents an opportunity to improve our API's at the same time.  The following is a list of improvements to the old API which the new API's address.

- Simplify the rocBLAS API from the tremendous complexity of BLAS plus OpenCL
  - Remove from the API's any explicit mention of OpenCL types
  - Establish synchronous API operation as the default behavior, with asynchronous as optional
- Introduce matrix, vector and scalar types to BLAS to encapsulate device data and abstract differences in runtime
- Batched processing to rocBLAS operations are possible, without complicating the API (performance for small problems)
  - BLAS types include information for batches of matrices, vectors and scalars
- Add new striding to the API (inspired by BLIS), which specify row and column strides separately
  - BLAS types include strides
  - Removes the transpose flags as redundant
  - Removes the row/column order flag as redundant
- BLAS types contain a precision flag, allowing us the option to support mixed mode operations for BLAS
  - For example, destination matrix C is double precision, but source matrices A and B are single precision

## Examples

The following example API's assume single GPU operation.  A separate, but similar API will be created to support multi-GPU operation, which would be designed as a layer above the single GPU API.  

Example of proposed GEMM v3 call<sup>[1](#single-GPU)</sup>:
```c
ROCBLAS_EXPORT rocblasStatus
  rocblasGemm( const rocblasScalar* alpha,
              const rocblasMatrix* a,
              const rocblasMatrix* b,
              const rocblasScalar* beta,
              rocblasMatrix* c,
              rocblasControl control );
```

The following is the interface for GEMM v2:
```c
ROCBLAS_DEPRECATED clblasStatus
clblasSgemm(
    clblasOrder order,
    clblasTranspose transA,
    clblasTranspose transB,
    size_t M,
    size_t N,
    size_t K,
    cl_float alpha,
    const cl_mem A,
    size_t offA,
    size_t lda,
    const cl_mem B,
    size_t offB,
    size_t ldb,
    cl_float beta,
    cl_mem C,
    size_t offC,
    size_t ldc,
    cl_uint numCommandQueues,
    cl_command_queue \*commandQueues,
    cl_uint numEventsInWaitList,
    const cl_event \*eventWaitList,
    cl_event \*events);
```

## Foot-notes
<a name="single-GPU">[1]</a>: This API is designed for single GPU operation.  A separate API will be designed that enables multi-GPU operation on it's input parameters

# rocBLAS
A BLAS implementation on top of AMD's Radeon Open Compute [ROCm][] runtime and toolchains.  rocBLAS is implemented in
the [HIP][] programming language and optimized for AMD's latest discrete GPUs.

## rocBLAS Wiki
The [wiki][] has helpful information about building the rocblas library, samples and tests.

## Building rocBLAS
The build infrastructure for rocBLAS is based on [Cmake](https://cmake.org/) v3.5.  This is the version of cmake available on ROCm supported platforms.  Examples of installing cmake:
* Ubuntu: `sudo apt install cmake-qt-gui`
* Fedora: `sudo dnf install cmake-gui`

### Library
The rocBLAS library has one dependency named [Tensile](https://github.com/ROCmSoftwarePlatform/Tensile), which supplies the high-performance implementation of xGEMM.  Tensile is downloaded by cmake during library configuration and automatically configured as part of the build, so no further action is required by the user to set it up.  Tensile is predominately written in python2.7 (not python3), so it does bring python dependencies which can easily be installed with distro package managers.  The rocBLAS library contains both host and device code, so the HCC compiler must be specified during cmake configuration to properly initialize build tools.  Example steps to build rocblas:

#### (One time only)
* Ubuntu: `sudo apt install python2.7 python-yaml`
* Fedora: `sudo dnf install python PyYAML`

#### Configure and build steps
```bash
mkdir -p [ROCBLAS_BUILD_DIR]/release
cd [ROCBLAS_BUILD_DIR]/release
# Default install location is in /opt/rocm, define -DCMAKE_INSTALL_PREFIX=<path> to specify other
# Default build config is 'Release', define -DCMAKE_BUILD_TYPE=<config> to specify other
CXX=/opt/rocm/bin/hcc ccmake [ROCBLAS_SOURCE]
make -j$(nproc)
sudo make install # sudo required if installing into system directory such as /opt/rocm
```

### rocBLAS clients
The repository contains source for clients that serve as samples, tests and benchmarks.  Clients source can be found in the clients subdir.

### Dependencies (only necessary for rocblas clients)
The rocBLAS samples have no external dependencies, but our unit test and benchmarking applications do.  These clients introduce the following dependencies:
1.  [boost](http://www.boost.org/)
2.  [lapack](https://github.com/Reference-LAPACK/lapack-release)
  * lapack itself brings a dependency on a fortran compiler
3.  [googletest](https://github.com/google/googletest)

Linux distros typically have an easy installation mechanism for boost through the native package manager.

* Ubuntu: `sudo apt install libboost-program-options-dev`
* Fedora: `sudo dnf install boost-program-options`

Unfortunately, googletest and lapack are not as easy to install.  Many distros do not provide a googletest package with pre-compiled libraries, and the lapack packages do not have the necessary cmake config files for cmake to configure linking the cblas library.  rocBLAS provide a cmake script that builds the above dependencies from source.  This is an optional step; users can provide their own builds of these dependencies and help cmake find them by setting the CMAKE_PREFIX_PATH definition.  The following is a sequence of steps to build dependencies and install them to the cmake default /usr/local.

#### (optional, one time only)
```bash
mkdir -p [ROCBLAS_BUILD_DIR]/release/deps
cd [ROCBLAS_BUILD_DIR]/release/deps
ccmake -DBUILD_BOOST=OFF [ROCBLAS_SOURCE]/deps   # assuming boost is installed through package manager as above
make -j$(nproc) install
```

Once dependencies are available on the system, it is possible to configure the clients to build.  This requires a few extra cmake flags to the library cmake configure script. If the dependencies are not installed into system defaults (like /usr/local ), you should pass the CMAKE_PREFIX_PATH to cmake to help find them.
* `-DCMAKE_PREFIX_PATH="<semicolon separated paths>"`
```bash
# Default install location is in /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to specify other
CXX=/opt/rocm/bin/hcc ccmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON [ROCBLAS_SOURCE]
make -j$(nproc)
sudo make install   # sudo required if installing into system directory such as /opt/rocm
```


#### CUDA build errata
Since rocBLAS is written with HiP kernels, it should build and run on CUDA platforms.  However, currently the build

## Migrating libraries to ROCm from OpenCL
[clBLAS][] demonstrated significant performance benefits of data parallel (GPU) computation when applied to solving dense
linear algebra problems, but OpenCL primarily remains in the domain of expert programmers. The ROCm model introduces a
single source paradigm for integrating device and host code together in a single source file, thereby simplifying the
entire development process for heterogeneous computing.  Compilers will get smarter, catching errors at compile/build time
and native profilers/debuggers will better integrate into the development process.  As AMD simplifies the
programming model with ROCm (using HCC and HIP), it is the intent of this library to expose that simplified programming
model to end users.

## rocBLAS interface
In general, the rocBLAS interface is compatible with Legacy [Netlib BLAS][] and the cuBLAS-v2 API, with
the explicit exception that Legacy BLAS does not have handle.  The cuBLAS' cublasHandle_t is replaced
with rocblas_handle everywhere. Thus, porting a CUDA application which originally calls the cuBLAS API
to a HIP application calling rocBLAS API should be relatively straightforward. For example, the rocBLAS
SGEMV interface is

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

## Rules for obtaining the rocBLAS API from Legacy BLAS
1. The Legacy BLAS routine name is changed to lower case, and prefixed by rocblas_.

2. A first argument rocblas_handle handle is added to all rocBlas functions.

3. Input arguments are declared with the const modifier.

4. Character arguments are replaced with enumerated types defined in rocblas_types.h.
   They are passed by value on the host.

5. Array arguments are passed by reference on the device.

6. Scalar arguments are passed by value on the host with the following two exceptions:

  * Scalar values alpha and beta are passed by reference on either the host or
    the device. The rocBLAS functions will check to see it the value is on
    the device. If this is true, it is used, else the value on the host is
    used.

  * Where Legacy BLAS functions have return values, the return value is instead
    added as the last function argument. It is returned by reference on either
    the host or the device. The rocBLAS functions will check to see it the value
    is on the device. If this is true, it is used, else the value is returned
    on the host. This applies to the following functions: xDOT, xDOTU, xNRM2,
    xASUM, IxAMAX, IxAMIN.

7. The return value of all functions is rocblas_status, defined in rocblas_types.h. It is
   used to check for errors.


#### Additional notes

  * The rocBLAS library is LP64, so rocblas_int arguments are 32 bit and rocblas_long
    arguments are 64 bit.

  * rocBLAS uses column-major storage for 2D arrays, and 1 based indexing for
    the functions xMAX and xMIN. This is the same as Legacy BLAS and cuBLAS.
    If you need row-major and 0 based indexing (used in C language arrays)
    download the [CBLAS](http://www.netlib.org/blas/#_cblas) file cblas.tgz.
    Look at the CBLAS functions that provide a thin interface to Legacy BLAS. They
    convert from row-major, 0 based, to column-major, 1 based. This is done by
    swapping the order of function arguments. It is not necessary to transpose
    matrices.

  * The auxiliary functions rocblas_set_pointer and rocblas_get_pointer are used
    to set and get the value of the state variable rocblas_pointer_mode. This
    variable is not used, it is added for compatibility with cuBLAS. rocBLAS
    will check if your scalar argument passed by reference is on the device.
    If this is true it will pass by reference on the device, else it passes
    by reference on the host.

## Asynchronous API
Except a few routines (like TRSM) having memory allocation inside preventing asynchronicity, most of the library routines
(like BLAS-1 SCAL, BLAS-2 GEMV, BLAS-3 GEMM) are configured to operate in asynchronous fashion with respect to CPU,
meaning that these library function calls return immediately.

## Batched and strided GEMM API
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



[wiki]: https://github.com/RadeonOpenCompute/rocBLAS/wiki

[ROCm]: https://github.com/RadeonOpenCompute/ROCm

[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/

[Netlib BLAS]: http://www.netlib.org/blas/

[clBLAS]: https://github.com/clMathLibraries/clBLAS

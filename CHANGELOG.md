# Change Log for rocBLAS

Full documentation for rocBLAS is available at [rocblas.readthedocs.io](https://rocblas.readthedocs.io/en/latest/).

## (Unreleased) rocBLAS 2.41.0
### Optimizations
- Improved performance of non-batched and batched syr for all sizes and data types
- Improved performance of non-batched and batched hemv for all sizes and data types
- Improved performance of non-batched and batched symv for all sizes and data types

### Changed
- Update from C++14 to C++17.
- Packaging split into runtime package called rocblas and development packages called rocblas-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except Centos 7 to aid in the transition. The suggests feaure in packaging is introduced as a deprecated feature and will be removed in a future rocm release. 

### Fixed
- For function geam avoid overflow in offset calculation.
- For function syr avoid overflow in offset calculation.
- For function gemv (Transpose-case) avoid overflow in offset calculation.
- For functions ssyrk and dsyrk, allow conjugate-transpose case to match legacy BLAS. Behavior is the same as the transpose case.

## [rocBLAS 2.40.0 for ROCm 4.4.0]
### Optimizations
- Improved performance of non-batched and batched dot, dotc, and dot_ex for small n. e.g. sdot n <= 31000.
- Improved performance of non-batched and batched trmv for all sizes and matrix types.
- Improved performance of non-batched and batched gemv transpose case for all sizes and datatypes.
- Improved performance of sger and dger for all sizes, in particular the larger dger sizes.
- Improved performance of syrkx for for large size including those in rocBLAS Issue #1184.

## [rocBLAS 2.39.0 for ROCm 4.3.0]
### Optimizations
- Improved performance of non-batched and batched rocblas_Xgemv for gfx908 when m <= 15000 and n <= 15000
- Improved performance of non-batched and batched rocblas_sgemv and rocblas_dgemv for gfx906 when m <= 6000 and n <= 6000
- Improved the overall performance of non-batched and batched rocblas_cgemv for gfx906
- Improved the overall performance of rocblas_Xtrsv

### Changed
- Internal use only APIs prefixed with rocblas_internal_ and deprecated to discourage use

## [rocBLAS 2.38.0 for ROCm 4.2.0]
### Added
- Added option to install script to build only rocBLAS clients with a pre-built rocBLAS library
- Supported gemm ext for unpacked int8 input layout on gfx908 GPUs
  - Added new flags rocblas_gemm_flags::rocblas_gemm_flags_pack_int8x4 to specify if using the packed layout
    - Set the rocblas_gemm_flags_pack_int8x4 when using packed int8x4, this should be always set on GPUs before gfx908.
    - For gfx908 GPUs, unpacked int8 is supported so no need to set this flag.
    - Notice the default flags 0 uses unpacked int8, this somehow changes the behaviour of int8 gemm from ROCm 4.1.0
- Added a query function rocblas_query_int8_layout_flag to get the preferable layout of int8 for gemm by device

### Optimizations
- Improved performance of single precision copy, swap, and scal when incx == 1 and incy == 1.
- Improved performance of single precision axpy when incx == 1, incy == 1 and batch_count =< 8192.
- Improved performance of trmm.

### Changed
- Change cmake_minimum_required to VERSION 3.16.8

## [rocBLAS 2.36.0 for ROCm 4.1.0]
### Added
- Added Numerical checking helper function to detect zero/NaN/Inf in the input and the output vectors of rocBLAS level 1 and 2 functions.
- Added Numerical checking helper function to detect zero/NaN/Inf in the input and the output general matrices of rocBLAS level 2 and 3 functions.
### Fixed
- Fixed complex unit test bug caused by incorrect caxpy and zaxpy function signatures.
- Make functions compliant with Legacy Blas for special values alpha == 0, k == 0, beta == 1, beta == 0.
### Optimizations
- Improved performance of single precision axpy_batched and axpy_strided_batched: batch_count >= 8192.
- Improved performance of trmm.

## [rocBLAS 2.34.0 for ROCm 4.0.0]
### Added
- Add changelog.
- Improved performance of gemm_batched for small m, n, k and NT, NC, TN, TT, TC, CN, CT, CC.
- Improved performance of gemv, gemv_batched, gemv_strided_batched: small n large m.
- Removed support for legacy hcc compiler.
- Add rot_ex, rot_batched_ex, and rot_strided_batched_ex.

### Fixed
- Removed `-DUSE_TENSILE_HOST` from `roc::rocblas` CMake usage requirements. This
  is a rocblas internal variable, and does not need to be defined in user code.


## [rocBLAS 2.32.0 for ROCm 3.10.0]
### Added
- Improved performance of gemm_batched for NN, general m, n, k, small m, n, k.


## [rocBLAS 2.30.0 for ROCm 3.9.0]
### Added
- Slight improvements to FP16 Megatron BERT performance on MI50.
- Improvements to FP16 Transformer performance on MI50.
- Slight improvements to FP32 Transformer performance on MI50.
- Improvements to FP32 DLRM Terabyte performance on gfx908.


## [rocBLAS 2.28.0 for ROCm 3.8.0]
### Added
- added two functions:
  - rocblas_status rocblas_set_atomics_mode(rocblas_atomics_mode mode)
  - rocblas_status rocblas_get_atomics_mode(rocblas_atomics_mode mode)
- added enum rocblas_atomics_mode. It can have two values
  - rocblas_atomics_allowed
  - rocblas_atomics_not_allowed
  The default is rocblas_atomics_not_allowed
- function rocblas_Xdgmm algorithm corrected and incx=0 support added
- dependencies:
  - rocblas-tensile internal component requires msgpack instead of LLVM
- Moved the following files from /opt/rocm/include to /opt/rocm/include/internal:
  - rocblas-auxillary.h
  - rocblas-complex-types.h
  - rocblas-functions.h
  - rocblas-types.h
  - rocblas-version.h
  - rocblas_bfloat16.h

  These files should NOT be included directly as this may lead to errors. Instead, /opt/rocm/include/rocblas.h should be included directly. /opt/rocm/include/rocblas_module.f90 can also be direcly used.


## [rocBLAS 2.26.0 for ROCm 3.7.0]
### Added
- Improvements to rocblas_Xgemm_batched performance for small m, n, k.
- Improvements to rocblas_Xgemv_batched  and rocblas_Xgemv_strided_batched performance for small m (QMCPACK use).
- Improvements to rocblas_Xdot (batched and non-batched) performance when both incx and incy are 1.
- Improvements to FP32 ONNX BERT performance for MI50.
- Significant improvements to FP32 Resnext, Inception Convolution performance for gfx908.
- Slight improvements to FP32 DLRM Terabyte performance for gfx908.
- Significant improvements to FP32 BDAS performance for gfx908.
- Significant improvements to FP32 BDAS performance for MI50 and MI60.
- Added substitution method for small trsm sizes with m <= 64 && n <= 64. Increases performance drastically for small batched trsm.


## [rocBLAS 2.24.0 for ROCm 3.6.0]
### Added
- Improvements to User Guide and Design Document.
- L1 dot function optimized to utilize shuffle instructions ( improvements on bf16, f16, f32 data types ).
- L1 dot function added x dot x optimized kernel.
- Standardization of L1 rocblas-bench to use device pointer mode to focus on GPU memory bandwidth.
- Adjustments for hipcc (hip-clang) compiler as standard build compiler and Centos8 support.
- Added Fortran interface for all rocBLAS functions.


## [rocBLAS 2.22.0 for ROCm 3.5.0]
### Added
- add geam complex, geam_batched, and geam_strided_batched.
- add dgmm, dgmm_batched, and dgmm_strided_batched.
- Optimized performance
  - ger
    - rocblas_sger, rocblas_dger,
    - rocblas_sger_batched, rocblas_dger_batched
    - rocblas_sger_strided_batched, rocblas_dger_strided_batched
  - geru
    - rocblas_cgeru, rocblas_zgeru
    - rocblas_cgeru_batched, rocblas_zgeru_batched
    - rocblas_cgeru_strided_batched, rocblas_zgeru_strided_batched
  - gerc
    - rocblas_cgerc, rocblas_zgerc
    - rocblas_cgerc_batched, rocblas_zgerc_batched
    - rocblas_cgerc_strided_batched, rocblas_zgerc_strided_batched
  - symv
    - rocblas_ssymv, rocblas_dsymv, rocblas_csymv, rocblas_zsymv
    - rocblas_ssymv_batched, rocblas_dsymv_batched, rocblas_csymv_batched, rocblas_zsymv_batched
    - rocblas_ssymv_strided_batched, rocblas_dsymv_strided_batched, rocblas_csymv_strided_batched, rocblas_zsymv_strided_batched
  - sbmv
    - rocblas_ssbmv, rocblas_dsbmv
    - rocblas_ssbmv_batched, rocblas_dsbmv_batched
    - rocblas_ssbmv_strided_batched, rocblas_dsbmv_strided_batched
  - spmv
    - rocblas_sspmv, rocblas_dspmv
    - rocblas_sspmv_batched, rocblas_dspmv_batched
    - rocblas_sspmv_strided_batched, rocblas_dspmv_strided_batched
- improved documentation.
- Fix argument checking in functions to match legacy BLAS.
- Fixed conjugate-transpose version of geam.

### Known Issues
- Compilation for GPU Targets:
When using the install.sh script for "all" GPU Targets, which is the default, you must first set an environment variable HCC_AMDGPU_TARGET listing the GPU targets, e.g.  HCC_AMDGPU_TARGET=gfx803,gfx900,gfx906,gfx908
If building for a specific architecture(s) using the  -a | --architecture flag, you should also set the environment variable HCC_AMDGPU_TARGET to match.
Mismatching the environment variable to the -a flag architectures creates builds that may result in SEGFAULTS when running on GPUs which weren't specified.

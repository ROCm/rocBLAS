# Change Log for rocBLAS

Full documentation for rocBLAS is available at [rocblas.readthedocs.io](https://rocblas.readthedocs.io/en/latest/).

## (Unreleased) rocBLAS 3.0.0
### Optimizations
- Improved performance of Level 2 rocBLAS GEMV on gfx90a GPU for non-transposed problems having small matrices and larger batch counts. Performance enhanced for problem sizes when m and n <= 32 and batch_count >= 256.
- Improved performance of rocBLAS syr2k for single, double, and double-complex precision, and her2k for double-complex precision. Slightly improved performance for general sizes on gfx90a.
### Added
- Added bf16 inputs and f32 compute support to Level 1 rocBLAS Extension functions axpy_ex, scal_ex and nrm2_ex.
### Deprecated
- trmm inplace is deprecated. It will be replaced by trmm that has both inplace and out-of-place functionality
- rocblas_query_int8_layout_flag() is deprecated and will be removed in a future release
- rocblas_gemm_flags_pack_int8x4 enum is deprecated and will be removed in a future release
- rocblas_set_device_memory_size() is deprecated and will be replaced by a future function rocblas_increase_device_memory_size()
- rocblas_is_user_managing_device_memory() is deprecated and will be removed in a future release
### Removed
- is_complex helper was deprecated and now removed.  Use rocblas_is_complex instead.
- The enum truncate_t and the value truncate was deprecated and now removed from. It was replaced by rocblas_truncate_t and rocblas_truncate, respectively.
- rocblas_set_int8_type_for_hipblas was deprecated and is now removed.
- rocblas_get_int8_type_for_hipblas was deprecated and is now removed.
### Dependencies
- build only dependency on python joblib added as used by Tensile build
- fix for cmake install on some OS when performed by install.sh -d --cmake_install
### Fixed
- make trsm offset calculations 64 bit safe
### Changed
- refactor rotg test code

## (Unreleased) rocBLAS 2.47.0
### Added
- added functionality rocblas_geam_ex for matrix-matrix minimum operations
- added HIP Graph support as beta feature for rocBLAS Level 1, Level 2, and Level 3(pointer mode host) functions
- added beta features API. Exposed using compiler define ROCBLAS_BETA_FEATURES_API
- added support for vector initialization in the rocBLAS test framework with negative increments
- added windows build documentation for forthcoming support using ROCm HIP SDK
- added scripts to plot performance for multiple functions
### Optimizations
- improved performance of Level 2 rocBLAS GEMV for float and double precision. Performance enhanced by 150-200% for certain problem sizes when (m==n) measured on a gfx90a GPU.
- improved performance of Level 2 rocBLAS GER for float, double and complex float precisions. Performance enhanced by 5-7% for certain problem sizes measured on a gfx90a GPU.
- improved performance of Level 2 rocBLAS SYMV for float and double precisions. Performance enhanced by 120-150% for certain problem sizes measured on both gfx908 and gfx90a GPUs.
### Fixed
- fixed setting of executable mode on client script rocblas_gentest.py to avoid potential permission errors with clients rocblas-test and rocblas-bench
- fixed deprecated API compatibility with Visual Studio compiler
- fixed test framework memory exception handling for Level 2 functions when the host memory allocation exceeds the available memory
### Changed
- install.sh internally runs rmake.py (also used on windows) and rmake.py may be used directly by developers on linux (use --help)
- rocblas client executables all now begin with rocblas- prefix
### Removed
- install.sh removed options -o --cov as now Tensile will use the default COV format, set by cmake define Tensile_CODE_OBJECT_VERSION=default

## rocBLAS 2.46.0 for ROCm 5.4.0
### Added
- client smoke test dataset added for quick validation using command rocblas-test --yaml rocblas_smoke.yaml
- Added stream order device memory allocation as a non-default beta option.

### Optimized
- Improved trsm performance for small sizes by using a substitution method technique
- Improved syr2k and her2k performance significantly by using a block-recursive algorithm

### Changed
- Level 2, Level 1, and Extension functions: argument checking when the handle is set to rocblas_pointer_mode_host now returns the status of rocblas_status_invalid_pointer only for pointers that must be dereferenced based on the alpha and beta argument values.  With handle mode rocblas_pointer_mode_device only pointers that are always dereferenced regardless of alpha and beta values are checked and so may lead to a return status of rocblas_status_invalid_pointer.   This improves consistency with legacy BLAS behaviour.
- Add variable to turn on/off ieee16/ieee32 tests for mixed precision gemm
- Allow hipBLAS to select int8 datatype
- Disallow B == C && ldb != ldc in rocblas_xtrmm_outofplace

### Fixed
- FORTRAN interfaces generalized for FORTRAN compilers other than gfortran
- fix for trsm_strided_batched rocblas-bench performance gathering
- Fix for rocm-smi path in commandrunner.py script to match ROCm 5.2 and above

## rocBLAS 2.45.0 for ROCm 5.3.0
### Added
- install.sh option --upgrade_tensile_venv_pip to upgrade Pip in Tensile Virtual Environment. The corresponding CMake option is TENSILE_VENV_UPGRADE_PIP.
- install.sh option --relocatable or -r adds rpath and removes ldconf entry on rocBLAS build.
- install.sh option --lazy-library-loading to enable on-demand loading of tensile library files at runtime to speedup rocBLAS initialization.
- Support for RHEL9 and CS9.
- Added Numerical checking routine for symmetric, Hermitian, and triangular matrices, so that they could be checked for any numerical abnormalities such as NaN, Zero, infinity and denormal value.

### Optimizations
- trmm_outofplace performance improvements for all sizes and data types using block-recursive algorithm.
- herkx performance improvements for all sizes and data types using block-recursive algorithm.
- syrk/herk performance improvements by utilising optimised syrkx/herkx code.
- symm/hemm performance improvements for all sizes and datatypes using block-recursive algorithm.

### Changed
- Unifying library logic file names: affects HBH (->HHS_BH), BBH (->BBS_BH), 4xi8BH (->4xi8II_BH). All HPA types are using the new naming convention now.
- Level 3 function argument checking when the handle is set to rocblas_pointer_mode_host now returns the status of rocblas_status_invalid_pointer only for pointers that must be dereferenced based on the alpha and beta argument values. With handle mode rocblas_pointer_mode_device only pointers that are always dereferenced regardless of alpha and beta values are checked and so may lead to a return status of rocblas_status_invalid_pointer. This improves consistency with legacy BLAS behaviour.
- Level 1, 2, and 3 function argument checking for enums is now more rigorously matching legacy BLAS so returns rocblas_status_invalid_value if arguments do not match the accepted subset.
- Add quick-return for internal trmm and gemm template functions.
- Moved function block sizes to a shared header file.
- Level 1, 2, and 3 functions use rocblas_stride datatype for offset.
- Modified the matrix and vector memory allocation in our test infrastructure for all Level 1, 2, 3 and BLAS_EX functions.
- Added specific initialization for symmetric, Hermitian, and triangular matrix types in our test infrastructure.
- Added NaN tests to the test infrastructure for the rest of Level 3, BLAS_EX functions.

### Fixed
- Improved logic to #include <filesystem> vs <experimental/filesystem>.
- install.sh -s option to build rocblas as a static library.
- dot function now sets the device results asynchronously for N <= 0

### Deprecated
- is_complex helper is now deprecated.  Use rocblas_is_complex instead.
- The enum truncate_t and the value truncate is now deprecated and will removed from the ROCm release 6.0. It is replaced by rocblas_truncate_t and rocblas_truncate, respectively. The new enum rocblas_truncate_t and the value rocblas_truncate could be used from this ROCm release for an easy transition.

### Removed
- install.sh options  --hip-clang , --no-hip-clang, --merge-files, --no-merge-files are removed.

## rocBLAS 2.44.0 for ROCm 5.2.0
### Added
- Packages for test and benchmark executables on all supported OSes using CPack.
- Added Denormal number detection to the Numerical checking helper function to detect denormal/subnormal numbers in the input and the output vectors of rocBLAS level 1 and 2 functions.
- Added Denormal number detection to the Numerical checking helper function to detect denormal/subnormal numbers in the input and the output general matrices of rocBLAS level 2 and 3 functions.
- Added NaN initialization tests to the yaml files of Level 2 rocBLAS batched and strided-batched functions for testing purposes.
- Added memory allocation check to avoid disk swapping during rocblas-test runs by skipping tests.

### Optimizations
- Improved performance of non-batched and batched her2 for all sizes and data types.
- Improved performance of non-batched and batched amin for all data types using shuffle reductions.
- Improved performance of non-batched and batched amax for all data types using shuffle reductions.
- Improved performance of trsv for all sizes and data types.

### Changed
- Modifying gemm_ex for HBH (High-precision F16). The alpha/beta data type remains as F32 without narrowing to F16 and expanding back to F32 in the kernel. This change prevents rounding errors due to alpha/beta conversion in situations where alpha/beta are not exactly represented as an F16.
- Modified non-batched and batched asum, nrm2 functions to use shuffle instruction based reductions.
- For gemm, gemm_ex, gemm_ex2 internal API use rocblas_stride datatype for offset.
- For symm, hemm, syrk, herk, dgmm, geam internal API use rocblas_stride datatype for offset.
-  AMD copyright year for all rocBLAS files.
- For gemv (transpose-case), typecasted the 'lda'(offset) datatype to size_t during offset calculation to avoid overflow and remove duplicate template functions.

### Fixed
- For function her2 avoid overflow in offset calculation.
- For trsm when alpha == 0 and on host, allow A to be nullptr.
- Fixed memory access issue in trsv.
- Fixed git pre-commit script to update only AMD copyright year.
- Fixed dgmm, geam test functions to set correct stride values.
- For functions ssyr2k and dsyr2k allow trans == rocblas_operation_conjugate_transpose.
- Fixed compilation error for clients-only build.

### Removed
- Remove Navi12 (gfx1011) from fat binary.

## rocBLAS 2.43.0 for ROCm 5.1.0
### Added
- Option to install script for number of jobs to use for rocBLAS and Tensile compilation (-j, --jobs)
- Option to install script to build clients without using any Fortran (--clients_no_fortran)
- rocblas_client_initialize function, to perform rocBLAS initialize for clients(benchmark/test) and report the execution time.
- Added tests for output of reduction functions when given bad input
- Added user specified initialization (rand_int/trig_float/hpl) for initializing matrices and vectors in rocblas-bench

### Optimizations
- Improved performance of trsm with side == left and n == 1
- Improved performance of trsm with side == left and m <= 32 along with side == right and n <= 32

### Changed
- For syrkx and trmm internal API use rocblas_stride datatype for offset
- For non-batched and batched gemm_ex functions if the C matrix pointer equals the D matrix pointer (aliased) their respective type and leading dimension arguments must now match
- Test client dependencies updated to GTest 1.11
- non-global false positives reported by cppcheck from file based suppression to inline suppression. File based suppression will only be used for global false positives.
- Help menu messages in install.sh
- For ger function, typecast the 'lda'(offset) datatype to size_t during offset calculation to avoid overflow and remove duplicate template functions.
- Modified default initialization from rand_int to hpl for initializing matrices and vectors in rocblas-bench

### Fixed
- For function trmv (non-transposed cases) avoid overflow in offset calculation
- Fixed cppcheck errors/warnings
- Fixed doxygen warnings

## rocBLAS 2.42.0 for ROCm 5.0.0
### Added
- Added rocblas_get_version_string_size convenience function
- Added rocblas_xtrmm_outofplace, an out-of-place version of rocblas_xtrmm
- Added hpl and trig initialization for gemm_ex to rocblas-bench
- Added source code gemm. It can be used as an alternative to Tensile for debugging and development
- Added option ROCM_MATHLIBS_API_USE_HIP_COMPLEX to opt-in to use hipFloatComplex and hipDoubleComplex

### Optimizations
- Improved performance of non-batched and batched single-precision GER for size m > 1024. Performance enhanced by 5-10% measured on a MI100 (gfx908) GPU.
- Improved performance of non-batched and batched HER for all sizes and data types. Performance enhanced by 2-17% measured on a MI100 (gfx908) GPU.

### Changed
- Instantiate templated rocBLAS functions to reduce size of librocblas.so
- Removed static library dependency on msgpack
- Removed boost dependencies for clients

### Fixed
- Option to install script to build only rocBLAS clients with a pre-built rocBLAS library
- Correctly set output of nrm2_batched_ex and nrm2_strided_batched_ex when given bad input
- Fix for dgmm with side == rocblas_side_left and a negative incx
- Fixed out-of-bounds read for small trsm
- Fixed numerical checking for tbmv_strided_batched

## rocBLAS 2.41.0 for ROCm 4.5.0
### Optimizations
- Improved performance of non-batched and batched syr for all sizes and data types
- Improved performance of non-batched and batched hemv for all sizes and data types
- Improved performance of non-batched and batched symv for all sizes and data types
- Improved memory utilization in rocblas-bench, rocblas-test gemm functions, increasing possible runtime sizes.
- Improved performance of non-batched and batched dot, dotc, and dot_ex for small n. e.g. sdot n <= 31000.
- Improved performance of non-batched and batched trmv for all sizes and matrix types.
- Improved performance of non-batched and batched gemv transpose case for all sizes and datatypes.
- Improved performance of sger and dger for all sizes, in particular the larger dger sizes.
- Improved performance of syrkx for for large size including those in rocBLAS Issue #1184.

### Changed
- Update from C++14 to C++17.
- Packaging split into a runtime package (called rocblas) and a development package (called rocblas-dev for `.deb` packages, and rocblas-devel for `.rpm` packages). The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

### Fixed
- For function geam avoid overflow in offset calculation.
- For function syr avoid overflow in offset calculation.
- For function gemv (Transpose-case) avoid overflow in offset calculation.
- For functions ssyrk and dsyrk, allow conjugate-transpose case to match legacy BLAS. Behavior is the same as the transpose case.

## rocBLAS 2.39.0 for ROCm 4.3.0
### Optimizations
- Improved performance of non-batched and batched rocblas_Xgemv for gfx908 when m <= 15000 and n <= 15000
- Improved performance of non-batched and batched rocblas_sgemv and rocblas_dgemv for gfx906 when m <= 6000 and n <= 6000
- Improved the overall performance of non-batched and batched rocblas_cgemv for gfx906
- Improved the overall performance of rocblas_Xtrsv

### Changed
- Internal use only APIs prefixed with rocblas_internal_ and deprecated to discourage use

## rocBLAS 2.38.0 for ROCm 4.2.0
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

## rocBLAS 2.36.0 for ROCm 4.1.0
### Added
- Added Numerical checking helper function to detect zero/NaN/Inf in the input and the output vectors of rocBLAS level 1 and 2 functions.
- Added Numerical checking helper function to detect zero/NaN/Inf in the input and the output general matrices of rocBLAS level 2 and 3 functions.
### Fixed
- Fixed complex unit test bug caused by incorrect caxpy and zaxpy function signatures.
- Make functions compliant with Legacy Blas for special values alpha == 0, k == 0, beta == 1, beta == 0.
### Optimizations
- Improved performance of single precision axpy_batched and axpy_strided_batched: batch_count >= 8192.
- Improved performance of trmm.

## rocBLAS 2.34.0 for ROCm 4.0.0
### Added
- Add changelog.
- Improved performance of gemm_batched for small m, n, k and NT, NC, TN, TT, TC, CN, CT, CC.
- Improved performance of gemv, gemv_batched, gemv_strided_batched: small n large m.
- Removed support for legacy hcc compiler.
- Add rot_ex, rot_batched_ex, and rot_strided_batched_ex.

### Fixed
- Removed `-DUSE_TENSILE_HOST` from `roc::rocblas` CMake usage requirements. This
  is a rocblas internal variable, and does not need to be defined in user code.


## rocBLAS 2.32.0 for ROCm 3.10.0
### Added
- Improved performance of gemm_batched for NN, general m, n, k, small m, n, k.


## rocBLAS 2.30.0 for ROCm 3.9.0
### Added
- Slight improvements to FP16 Megatron BERT performance on MI50.
- Improvements to FP16 Transformer performance on MI50.
- Slight improvements to FP32 Transformer performance on MI50.
- Improvements to FP32 DLRM Terabyte performance on gfx908.


## rocBLAS 2.28.0 for ROCm 3.8.0
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

  These files should NOT be included directly as this may lead to errors. Instead, /opt/rocm/include/rocblas.h should be included directly. /opt/rocm/include/rocblas_module.f90 can also be directly used.


## rocBLAS 2.26.0 for ROCm 3.7.0
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


## rocBLAS 2.24.0 for ROCm 3.6.0
### Added
- Improvements to User Guide and Design Document.
- L1 dot function optimized to utilize shuffle instructions ( improvements on bf16, f16, f32 data types ).
- L1 dot function added x dot x optimized kernel.
- Standardization of L1 rocblas-bench to use device pointer mode to focus on GPU memory bandwidth.
- Adjustments for hipcc (hip-clang) compiler as standard build compiler and Centos8 support.
- Added Fortran interface for all rocBLAS functions.


## rocBLAS 2.22.0 for ROCm 3.5.0
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

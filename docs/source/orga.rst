********************************
Library Source Code Organization
********************************

The rocBLAS code is split into three major parts:

- The `library` directory contains all source code for the library.
- The `clients` directory contains all test code and code to build clients.
- Infrastructure

The `library` directory
-----------------------

library/include
```````````````
Contains C98 include files for the external API. These files also contain Doxygen
comments that document the API.

library/src/blas[1,2,3]
```````````````````````
Source code for Level 1, 2 and 3 BLAS functions in '.cpp' and '.hpp' files.

- The '.cpp' files contain

  - external C functions that call templated functions with an '_impl' extension.
  - The _impl functions have argument checking and logging, and they in turn call functions with a '_template' extension.

- The '.hpp' files contain

  - '_template' functions that set up the workgroup and call HIP launch to run '_kernel' functions.
  - '_kernel' functions that run on the device.

library/src/blas3/Tensile
`````````````````````````
Code for calling Tensile from rocBLAS, and YAML files with Tensile tuning configurations.

library/src/blas_ex
```````````````````
Source code for mixed precision BLAS.

library/src/include
```````````````````
Internal include files for:

- handle code
- device memory allocation
- logging
- utility code


The `clients` directory
-----------------------

clients/gtest
`````````````
Code for client rocblas-test. This client is used to test rocBLAS.

clients/benchmarks
``````````````````
Code for client rocblas-benchmark. This client is used to benchmark rocBLAS functions.

clients/include
```````````````
Code for testing and benchmarking individual rocBLAS functions, and utility code for testing.

clients/common
``````````````
Common code used by both rocblas-benchmark and rocblas-test.

clients/samples
```````````````
Sample code for calling rocBLAS functions.


Infrastructure
--------------

- CMake is used to build and package rocBLAS. There are CMakeLists.txt files throughout the code.
- Doxygen/Breathe/Sphinx/ReadTheDocs are used to produce documentation. Content for the documentation is from:

  - Doxygen comments in include files in the directory library/include
  - files in the directory docs/source.

- Jenkins is used to automate Continuous Integration testing.
- clang-format is used to format C++ code.



============
Clients
============

rocBLAS builds 2 clients in the directory rocBLAS/build/release/clients/staging

1. rocblas-test: used rocBLAS correctness tests.

2. rocblas-benchmark: used to benchmark rocBLAS functions.

rocblas-test
============

rocblas-test uses Googletest. The tests are in 4 categories:

- quick
- pre_checkin
- nightly
- known_bug

To run the quick tests:

.. code-block:: bash

   ./rocblas-test --gtest_filter=*quick*

The number of lines of output can be reduced with:

.. code-block:: bash

   GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*quick*

gtest_filter can also be used to run tests for a particular function, and a particular set of input parameters. For example, to run all quick tests for the function rocblas_saxpy:

.. code-block:: bash

   ./rocblas-test --gtest_filter=*quick*axpy*f32_r*

The pattern for ``--gtest_filter`` is:

.. code-block:: bash

   --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]


rocblas-bench
=============

rocblas-bench uses a command line interface to run performance benchmarks on individual functions. For more information:

.. code-block:: bash

   ./rocblas-bench --help

As an example, to time sgemm:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096

On a vega20 machine this outputs a performance of 11941.5 Gflops below:

.. code-block:: bash

   transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us
   N,N,4096,4096,4096,1,4096,4096,0,4096,11941.5,11509.4

rocBLAS logging can be turned on by setting environment variable ROCBLAS_LAYER=2. If you run any code that calls rocBLAS functions with logging turned on it will output a line each time a rocBLAS function is called. The line that it outputs is the line you need to run rocblas-bench. For example if you run:

.. code-block:: bash

   ROCBLAS_LAYER=2 ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096

it will output the following 12 times:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096

The reason for the 12 times is because it calls rocblas_sgemm 12 times to measure its performance.



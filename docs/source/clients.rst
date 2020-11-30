============
Clients
============

The rocBLAS consists of two clients. They are,

1. rocblas-bench

2. rocblas-test

These two clients can be built by following the instructions at `Building and Installing rocBLAS github page <https://github.com/ROCmSoftwarePlatform/rocBLAS-internal/blob/develop/docs/source/install.rst>`_. After building the rocBLAS clients, they can be found in the directory ``rocBLAS/build/release/clients/staging``.

The next two sections will cover a brief explanation and the usage of each rocBLAS client.

rocblas-bench
=============
rocblas-bench is used to measure performance and to verify the correctness of rocBLAS functions.

It has a command line interface. For more information:

.. code-block:: bash

   ./rocblas-bench --help

For example, to measure performance of sgemm:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096

On a vega20 machine the above command outputs a performance of 11941.5 Gflops below:

.. code-block:: bash

   transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us
   N,N,4096,4096,4096,1,4096,4096,0,4096,11941.5,11509.4

A useful way of finding the parameters that can be used with ``./rocblas-bench -f gemm`` is to turn on logging
by setting environment variable ``ROCBLAS_LAYER=2``. For example if the user runs:

.. code-block:: bash

   ROCBLAS_LAYER=2 ./rocblas-bench -f gemm -i 1 -j 0

The above command will log:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 128 -n 128 -k 128 --alpha 1 --lda 128 --ldb 128 --beta 0 --ldc 128

The user can copy and change the above command. For example, to change the datatype to IEEE-64 bit and the size to 2048:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f64_r --transposeA N --transposeB N -m 2048 -n 2048 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048

Logging affects performance, so only use it to log the command to copy and change, then run the command without logging to measure performance.

Note that rocblas-bench also has the flag ``-v 1`` for correctness checks.

rocblas-test
============

rocblas-test is used in performing rocBLAS unit tests and it uses Googletest framework.

The tests are in 4 categories:

- quick
- pre_checkin
- nightly
- known_bug

To run the quick tests:

.. code-block:: bash

   ./rocblas-test --gtest_filter=*quick*

The other tests can also be run using the above command by replacing ``*quick*`` with ``*pre_checkin*``, ``*nightly*``, and ``*known_bug*``.

The pattern for ``--gtest_filter`` is:

.. code-block:: bash

   --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]

gtest_filter can also be used to run tests for a particular function, and a particular set of input parameters. For example, to run all quick tests for the function rocblas_saxpy:

.. code-block:: bash

   ./rocblas-test --gtest_filter=*quick*axpy*f32_r*

The number of lines of output can be reduced with:

.. code-block:: bash

   GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*quick*


Useful environment variables
============================

In bash:

- ``export AMD_LOG_LEVEL=3`` (reset by =0). When the user runs the application it will log every HIP kernel, including rocBLAS kernels.

- ``export HIP_LAUNCH_BLOCKING = 0``: make HIP APIs host-synchronous so they are blocked until any kernel launches or data-copy commands are complete (an alias is CUDA_LAUNCH_BLOCKING)

For more profiling tools, see `Profiling and Debugging HIP Code <https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/blob/master/docs/markdown/hip_profiling.md#profiling-hip-apis>`_ .

The IR and ISA can be dumped by setting the following environment variable before building the app:

.. code-block:: bash

    export KMDUMPISA=1

    export KMDUMPLLVM=1

    export KMDUMPDIR=/path/to/dump

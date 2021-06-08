==========================
rocBLAS Numerical Checking
==========================

**Note that performance will degrade when numerical checking is enabled.**

rocBLAS provides the environment variable ``ROCBLAS_CHECK_NUMERICS`` which allows user to debug numerical abnormalities. Setting a value of ``ROCBLAS_CHECK_NUMERICS`` enables checks on the input and the output vectors/matrices
of the rocBLAS functions for (not-a-number) NaN's, zeros, and infinities. Numerical checking has been added to all level 1 functions. In level 2, all functions include the check for the input and the output vectors for numerical abnormalities.
But, only the general (ge) type input and the output matrix are checked in level 2 functions. In level 3, GEMM is the only function to have numerical checking.


``ROCBLAS_CHECK_NUMERICS`` is a bitwise OR of zero or more bit masks as follows:

* ``ROCBLAS_CHECK_NUMERICS = 0``: Is not set, then there is no numerical checking

* ``ROCBLAS_CHECK_NUMERICS = 1``: Fully informative message, print's the results of numerical checking whether the input and the output Matrices / Vectors have NaN's / zeros / infinities to the console

* ``ROCBLAS_CHECK_NUMERICS = 2``: Print's result of numerical checking only if the input and the output Matrices / Vectors has a NaN/infinity

* ``ROCBLAS_CHECK_NUMERICS = 4``: Return ``rocblas_status_check_numeric_fail`` status if there is a NaN / infinity

An example usage of ``ROCBLAS_CHECK_NUMERICS`` is shown below,

.. code-block:: bash

    ROCBLAS_CHECK_NUMERICS=4 ./rocblas-bench -f gemm -i 1 -j 0

The above command will return a ``rocblas_status_check_numeric_fail``, if the input and the ouptut matrices of BLAS level 3 GEMM funtion has a NaN or infinity.
If there is no numerical abnormalities then ``rocblas_status_success`` is returned .

.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _logging:

********************************************************************
Logging in rocBLAS
********************************************************************

**Note that performance will degrade when logging is enabled.**

User can set four environment variables to control logging:

* ``ROCBLAS_LAYER``

* ``ROCBLAS_LOG_TRACE_PATH``

* ``ROCBLAS_LOG_BENCH_PATH``

* ``ROCBLAS_LOG_PROFILE_PATH``

``ROCBLAS_LAYER`` is a bitwise OR of zero or more bit masks as follows:

*  If ``ROCBLAS_LAYER`` is not set, then there is no logging.

*  If ``(ROCBLAS_LAYER & 1) != 0``, then there is trace logging.

*  If ``(ROCBLAS_LAYER & 2) != 0``, then there is bench logging.

*  If ``(ROCBLAS_LAYER & 4) != 0``, then there is profile logging.

Trace logging outputs a line each time a rocBLAS function is called. The
line contains the function name and the values of arguments.

Bench logging outputs a line each time a rocBLAS function is called. The
line can be used with the executable ``rocblas-bench`` to call the
function with the same arguments.

Profile logging, at the end of program execution, outputs a YAML
description of each rocBLAS function called, the values of its
performance-critical arguments, and the number of times it was called
with those arguments (the ``call_count``). Some arguments, such as
``alpha`` and ``beta`` in GEMM, are recorded with a value representing
the category that the argument falls in, such as ``-1``, ``0``, ``1``,
or ``2``. The number of categories, and the values representing them,
may change over time, depending on how many categories are needed to
adequately represent all the values that can affect the performance
of the function.

The default stream for logging output is standard error. Three
environment variables can set the full path name for a log file:

* ``ROCBLAS_LOG_TRACE_PATH`` sets the full path name for trace logging.
* ``ROCBLAS_LOG_BENCH_PATH`` sets the full path name for bench logging.
* ``ROCBLAS_LOG_PROFILE_PATH`` sets the full path name for profile logging.

For example, in Bash shell, to output bench logging to the file
``bench_logging.txt`` in your present working directory:

* ``export ROCBLAS_LOG_BENCH_PATH=$PWD/bench_logging.txt``

Note that a full path is required, not a relative path. In the above
command $PWD expands to the full path of your present working directory.
If paths are not set, then the logging output is streamed to standard error.

When profile logging is enabled, memory usage increases. If the
program exits abnormally, then it is possible that profile logging will
not be outputted before the program exits.

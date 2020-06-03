
.. toctree::
   :maxdepth: 4
   :caption: Contents:

=======
Logging
=======

**Note that performance will degrade when logging is enabled.**

Four environment variables can be set to control logging:

* ``ROCBLAS_LAYER``

* ``ROCBLAS_LOG_TRACE_PATH``

* ``ROCBLAS_LOG_BENCH_PATH``

* ``ROCBLAS_LOG_PROFILE_PATH``

``ROCBLAS_LAYER`` is a bitwise OR of zero or more bit masks as follows:

*  If ``ROCBLAS_LAYER`` is not set, then there is no logging

*  If ``(ROCBLAS_LAYER & 1) != 0``, then there is trace logging

*  If ``(ROCBLAS_LAYER & 2) != 0``, then there is bench logging

*  If ``(ROCBLAS_LAYER & 4) != 0``, then there is profile logging

Trace logging outputs a line each time a rocBLAS function is called. The
line contains the function name and the values of arguments.

Bench logging outputs a line each time a rocBLAS function is called. The
line can be used with the executable ``rocblas-bench`` to call the
function with the same arguments.

Profile logging, at the end of program execution, outputs a YAML
description of each rocBLAS function called, the values of its
arguments, and the number of times it was called with those arguments.

The default stream for logging output is standard error. Three
environment variables can set the full path name for a log file:

* ``ROCBLAS_LOG_TRACE_PATH`` sets the full path name for trace logging
* ``ROCBLAS_LOG_BENCH_PATH`` sets the full path name for bench logging
* ``ROCBLAS_LOG_PROFILE_PATH`` sets the full path name for profile logging

If a path name cannot be opened, then the corresponding logging output
is streamed to standard error.

When profile logging is enabled, memory usage will increase. If the
program exits abnormally, then it is possible that profile logging will
not be outputted before the program exits.

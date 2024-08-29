.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _programmers-guide:

********************************************************************
Programmers guide
********************************************************************

================================
Source Code Organization
================================

The rocBLAS code can be found at https://github.com/ROCm/rocBLAS. It is split into three major parts:

- The ``library`` directory contains all source code for the library.
- The ``clients`` directory contains all test code and code to build clients.
- Infrastructure such as ``docs`` and ``cmake`` to support the library.

The `library` Directory
-----------------------

The ``library`` directory contains the following structure and content for rocBLAS.

library/include
''''''''''''''''

Contains C98 include files for the external API. These files also contain Doxygen
comments that document the API.

library/src/blas[1,2,3]
'''''''''''''''''''''''''

Source code for Level 1, 2, and 3 BLAS functions in `.cpp` and `.hpp` files.

- The `*.cpp` files contain

  - External C functions that call or instantiate templated functions with an `_impl` extension
  - The `_impl` functions have argument checking and logging, and they in turn call functions with a `_template` extension

- The `*_imp.hpp` files contain
  - `_template` functions that may be exported to rocSOLVER and usually call the `_launcher` functions
  - API implementations that can be instantiated in two ways: once for the original APIs with integer args using rocblas_int and
again for the ILP64 API with integer arguments as int64_t.

- The `*_kernels.cpp` files contain
  - `_launcher` functions that invoke or launch kernels with ROCBLAS_LAUNCH_KERNEL or related macros
  - `_kernel` functions that run on the device

library/src/blas_ex
''''''''''''''''''''

Source code for mixed precision BLAS

library/src/src64
'''''''''''''''''

This directory contains the ILP64 source code for Level 1, 2, and 3 BLAS and mixed precision functions in blas_ex.
Files should normally end with _64 before the file type extension (e.g. _64.cpp).
The API integers are int64_t instead of rocblas_int.
Function behaviour is kept identical at the higher level detail by instantiable macros and C++ templates.  Only at the kernel dispatch level does the code diverge by
providing a _64 version for which invocation is controlled by the ROCBLAS_API macro.
The directory structure mirrors the level organization used for the parent directory library/src.

device kernel code
''''''''''''''''''

Most BLAS device functions (kernels) are C++ templated functions based on data type.  In C++ host code any duplicate instantiations of templates can be handled by the linker
and the duplicates will be ignored.  LLVM device code instantiations, however, are not handled in this way; therefore we must avoid duplicate instantiations in multiple code units.
Thus kernel templates should only be provided as C++ template prototypes in the include files unless they must be instantiated.  We should try to instantiate all forms in a single
unit (e.g. a .cpp file) and expose a launcher C++ interface to invoke the device calls, where possible.  This is especially important for ILP64 implementations where we want to
reuse the LP64 instantiations without any duplication to avoid bloating the library size.

library/src/blas3/Tensile
'''''''''''''''''''''''''

Code for calling Tensile from rocBLAS, and YAML files with Tensile tuning configurations

library/src/include
'''''''''''''''''''

Internal include files for:

- Handle code
- Device memory allocation
- Logging
- Numerical checking
- Utility code




The `clients` Directory
-----------------------

The ``clients`` directory contains all test code and code to build clients.

clients/gtest
'''''''''''''

Code for client rocblas-test. This client is used to test rocBLAS.

clients/benchmarks
''''''''''''''''''

Code for client rocblas-benchmark. This client is used to benchmark rocBLAS functions.

clients/include
'''''''''''''''

Code for testing and benchmarking individual rocBLAS functions, and utility code for testing.
Test harness functions are templated by data type and are defined in separate files for each function form: non-batched, batched, strided_batched.
When a function also supports the ILP64 API then both forms can be tested by the same template and is controlled the Arguments api member variable.
This follows the pattern for FORTRAN API testing and includes FORTRAN_64 for the ILP64 form.

clients/common
''''''''''''''

Common code used by both rocblas-benchmark and rocblas-test

clients/samples
'''''''''''''''

Sample code for calling rocBLAS functions


Infrastructure
--------------

- CMake is used to build and package rocBLAS. There are ``CMakeLists.txt`` files throughout the code.
- Doxygen/Breathe/Sphinx/ReadTheDocs are used to produce documentation. Content for the documentation is from:

  - Doxygen comments in include files in the directory ``library/include``
  - Files in the ``docs`` folder.

- Jenkins is used to automate Continuous Integration testing.
- clang-format is used to format C++ code.


=====================================
Handle, Stream, and Device Management
=====================================

Handle
-------

A ``rocBLAS_handle`` must be created as shown before calling other rocBLAS functions:

::

    rocblas_handle handle;
    if(rocblas_create_handle(&handle) != rocblas_status_success) return EXIT_FAILURE;

The created handle should be destroyed as shown when the users have completed calling rocBLAS functions:

::

    if(rocblas_destroy_handle(handle) != rocblas_status_success) return EXIT_FAILURE;

The above-created handle will use the default stream and the default device. If the user wants the non-default
stream and the non-default device, then call:

::

    int deviceId = non_default_device_id;
    if(hipSetDevice(deviceId) != hipSuccess) return EXIT_FAILURE;

    //optional call to rocblas_initialize
    rocblas_initialize();

    // note the order, call hipSetDevice before hipStreamCreate
    hipStream_t stream;
    if(hipStreamCreate(&stream) != hipSuccess) return EXIT_FAILURE;

    rocblas_handle handle;
    if(rocblas_create_handle(&handle) != rocblas_status_success) return EXIT_FAILURE;

    if(rocblas_set_stream(handle, stream) != rocblas_status_success) return EXIT_FAILURE;


For the library to use a non-default device within a host thread, the device must be set using ``hipSetDevice()`` before creating the handle.

The device in the host thread should not be changed between ``hipStreamCreate`` and ``hipStreamDestroy``. If the device in the host thread is changed between creating and destroying the stream, then the behavior is undefined.

If the user created a non-default stream, it is the user's responsibility to synchronize the non-default stream before destroying it:

::

    // Synchronize the non-default stream before destroying it
    if(hipStreamSynchronize(stream) != hipSuccess) return EXIT_FAILURE;

    if(hipStreamDestroy(stream) != hipSuccess) return EXIT_FAILURE;

When a user switches from one non-default stream to another, they must complete all rocblas operations previously submitted with this handle on the old stream using ``hipStreamSynchronize(old_stream)`` API before setting the new stream.
::

    // Synchronize the old stream
    if(hipStreamSynchronize(old_stream) != hipSuccess) return EXIT_FAILURE;

    // Destroy the old stream (this step is optional but must come after synchronization)
    if(hipStreamDestroy(old_stream) != hipSuccess) return EXIT_FAILURE;

    // Create a new stream (this step can be done before the steps above)
    if(hipStreamCreate(&new_stream) != hipSuccess) return EXIT_FAILURE;

    // Set the handle to use the new stream (must come after synchronization)
    if(rocblas_set_stream(handle, new_stream) != rocblas_status_success) return EXIT_FAILURE;

The above ``hipStreamSynchronize`` is necessary because the ``rocBLAS_handle`` contains allocated device
memory that must not be shared by multiple asynchronous streams at the same time.

If either the old or new stream is the default (NULL) stream, it is not necessary to
synchronize the old stream before destroying it, or before setting the new stream,
because the synchronization is implicit.

.. note::
  A user can switch from one non-default stream to another without calling ``hipStreamSynchronize()`` by enabling stream-order memory allocation.
  Refer to section :ref:`stream order alloc`.

Creating the handle will incur a startup cost. There is an additional startup cost for
gemm functions to load gemm kernels for a specific device. Users can shift the
gemm startup cost to occur after setting the device by calling ``rocblas_initialize()``
after calling ``hipSetDevice()``. This action needs to be done once for each device.
If the user has two rocBLAS handles which use the same device, then the user only needs to call ``rocblas_initialize()``
once. If ``rocblas_initialize()`` is not called, then the first gemm call will have
the startup cost.

The ``rocBLAS_handle`` stores the following:

- Stream
- Logging mode
- Pointer mode
- Atomics mode

Stream and Device Management
-----------------------------

HIP kernels are launched in a queue. This queue is otherwise known as a stream. A stream is a queue of
work on a particular device.

A ``rocBLAS_handle`` always has one stream, and a stream is always associated with one device. The ``rocBLAS_handle`` is passed as an argument to all rocBLAS functions that launch kernels, and these kernels are
launched in that handle's stream to run on that stream's device.

If the user does not create a stream, then the ``rocBLAS_handle`` uses the default (NULL)
stream, maintained by the system. Users cannot create or destroy the default
stream. However, users can create a new non-default stream and bind it to the ``rocBLAS_handle`` with the
two commands: ``hipStreamCreate()`` and ``rocblas_set_stream()``.

rocBLAS supports use of non-blocking stream for functions requiring synchronization to guarantee results on the host.
For functions like ``rocblas_Xnrm2``, scalar result is copied from device to host when ``rocblas_pointer_mode == rocblas_pointer_mode_host``.
This is done using ``hipMemcpyAsync()`` followed by ``hipStreamSynchronize()``. The stream that is synchronized is the stream in the ``rocBLAS_handle``.

.. note::
  Exception to the above pattern are the following rocBLAS functions, :any:`rocblas_set_vector`, :any:`rocblas_get_vector`, :any:`rocblas_set_matrix`, :any:`rocblas_get_matrix` which block on default stream.

If the user creates a stream, they are responsible for destroying it with ``hipStreamDestroy()``. If the handle
is switching from one non-default stream to another, then the old stream needs to be synchronized. Next, the user needs to create and set the new non-default stream using ``hipStreamCreate()`` and ``rocblas_set_stream()``, respectively. Then the user can optionally destroy the old stream.

HIP has two important device management functions, ``hipSetDevice()``, and ``hipGetDevice()``.

- ``hipSetDevice()``: Set default device to be used for subsequent hip API calls from this thread.
- ``hipGetDevice()``: Return the default device id for the calling host thread.

The device which was set using ``hipSetDevice()`` at the time of calling
``hipStreamCreate()`` is the one that is associated with a stream. But, if the device was not set using ``hipSetDevice()``, then, the default device will be used.

Users cannot switch the device in a stream between ``hipStreamCreate()`` and ``hipStreamDestroy()``.
If users want to use another device, they should create another stream.

rocBLAS never sets a device, it only queries using ``hipGetDevice()``. If rocBLAS does not see a
valid device, it returns an error message to users.

Multiple Streams and Multiple Devices
-------------------------------------

If a machine has ``num`` GPU devices, they will have deviceID numbers 0, 1, 2, ... (``num`` - 1). The
default device has ``deviceID == 0``. Each ``rocBLAS_handle`` can only be used with a single device, but users can run `<num>` handles on `<num>` devices concurrently.


.. _Device Memory allocation in detail:

========================
Device Memory Allocation
========================

Requirements
-------------

- Some rocBLAS functions need temporary device memory.
- Allocating and deallocating device memory is expensive and synchronizing.
- Temporary device memory should be recycled across multiple rocBLAS function calls using the same ``rocblas_handle``.
- The following schemes need to be supported:

  - **Default** Functions allocate required device memory automatically. This has the disadvantage that allocation is a synchronizing event.
  - **Preallocate** Query all the functions called using a ``rocblas_handle`` to find out how much device memory is needed. Preallocate the required device memory when the ``rocblas_handle`` is created, and there are no more synchronizing allocations or deallocations.
  - **Manual** Query a function to find out how much device memory is required. Allocate and deallocate the device memory before and after function calls. This allows the user to control where the synchronizing allocation and deallocation occur.

In all above schemes, temporary device memory needs to be held by the ``rocblas_handle`` and recycled if a subsequent function using the handle needs it.

Design
------

- rocBLAS uses per-handle device memory allocation with out-of-band management.
- The state of the device memory is stored in the ``rocblas_handle``.
- For the user of rocBLAS:

  - Functions are provided to query how much device memory a function needs.
  - An environment variable is provided to preallocate when the ``rocblas_handle`` is created.
  - Functions are provided to manually allocate and deallocate after the ``rocblas_handle`` is created.
  - The following two values are added to the ``rocblas_status`` enum to indicate how a rocBLAS function is changing the state of the temporary device memory in the ``rocblas_handle`` :

     - rocblas_status_size_unchanged
     - rocblas_status_size_increased

- For the rocBLAS developer:

  - Functions are provided to answer device memory size queries.
  - Functions are provided to allocate temporary device memory.
  - Opaque RAII objects are used to hold the temporary device memory, and allocated memory is returned to the handle automatically when it is no longer needed.

The functions for the rocBLAS user are described in the :ref:`api-reference-guide`. The functions for the rocBLAS developer are described below.


Answering device memory size queries in functions that need memory
------------------------------------------------------------------

Example
'''''''

Functions should contain code like below to answer a query on how much temporary device memory is required. In this case, ``m * n * sizeof(T)`` bytes of memory is required:

.. code-block:: c++

    rocblas_status rocblas_function(rocblas_handle handle, ...)
    {
        if(!handle) return rocblas_status_invalid_handle;

        if (handle->is_device_memory_size_query())
        {
            size_t size = m * n * sizeof(T);
            return handle->set_optimal_device_memory_size(size);
        }

        //  rest of function
    }


Function
'''''''''

.. code-block:: c++

    bool _rocblas_handle::is_device_memory_size_query() const

Indicates if the current function call is collecting information about the optimal device memory allocation size

return value:

- **true** if information is being collected
- **false** if information is not being collected

Function
''''''''

.. code-block:: c++

    rocblas_status _rocblas_handle::set_optimal_device_memory_size(size...)

Sets the optimal size(s) of device memory buffer(s) in bytes for this function. The sizes are rounded up to the next multiple of 64 (or some other chunk size), and the running maximum is updated.

return value:

- **rocblas_status_size_unchanged** If the maximum optimal device memory size did not change, this is the case where the function does not use device memory.
- **rocblas_satus_size_increased** If the maximum optimal device memory size increased.
- **rocblas_status_internal_error** If this function is not supposed to be collecting size information.

Function
''''''''

.. code-block:: c++

    size_t rocblas_sizeof_datatype(rocblas_datatype type)

Returns size of a rocBLAS runtime data type


Answering device memory size queries in functions that do not need memory
--------------------------------------------------------------------------

Example
'''''''

.. code-block:: c++

    rocblas_status rocblas_function(rocblas_handle handle, ...)
    {
        if(!handle) return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    //  rest of function
    }

Macro
'''''

.. code-block:: c++

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)

A convenience macro that returns ``rocblas_status_size_unchanged`` if the function call is a memory size query


rocBLAS Kernel device memory allocation
-----------------------------------------

Example
'''''''

Device memory can be allocated for `n` floats using ``device_malloc`` as follows:

.. code-block:: c++

     auto workspace = handle->device_malloc(n * sizeof(float));
     if (!workspace) return rocblas_status_memory_error;
     float* ptr = static_cast<float*>(workspace);

Example
'''''''

To allocate multiple buffers:

.. code-block:: c++

    size_t size1 = m * n;
    size_t size2 = m * k;

    auto workspace = handle->device_malloc(size1, size2);
    if (!workspace) return rocblas_status_memory_error;

    void * w_buf1, * w_buf2;
    w_buf1 = workspace[0];
    w_buf2 = workspace[1];


Function
'''''''''

.. code-block:: c++

    auto workspace = handle->device_malloc(size...)

- Returns an opaque RAII object lending allocated device memory to a particular rocBLAS function.
- The object returned is convertible to ``void *`` or other pointer types if only one size is specified.
- The individual pointers can be accessed with the subscript ``operator[]``.
- The lifetime of the returned object is the lifetime of the borrowed device memory (RAII).
- To simplify and optimize the code, only one successful allocation object can be alive at a time.
- If the handle's device memory is currently being managed by rocBLAS, as in the default scheme, it is expanded in size as necessary.
- If the user allocated (or pre-allocated) an explicit size of device memory, then that size is used as the limit, and no resizing or synchronization ever occurs.

Parameters:

- **size** size in bytes of memory to be allocated

return value:

- **On success**, returns an opaque RAII object that evaluates to ``true`` when converted to ``bool``
- **On failure**, returns an opaque RAII object that evaluates to ``false`` when converted to ``bool``


Performance Degrade
--------------------

The ``rocblas_status`` enum value ``rocblas_status_perf_degraded`` is used to indicate that a slower algorithm was used because of insufficient device memory for the optimal algorithm.

Example
'''''''

.. code-block:: c++

    rocblas_status ret = rocblas_status_success;
    size_t size_for_optimal_algorithm = m + n + k;
    size_t size_for_degraded_algorithm = m;
    auto workspace_optimal = handle->device_malloc(size_for_optimal_algorithm);
    if (workspace_optimal)
    {
        // Algorithm using larger optimal memory
    }
    else
    {
        auto workspace_degraded = handle->device_malloc(size_for_degraded_algorithm);
        if (workspace_degraded)
        {
            // Algorithm using smaller degraded memory
            ret = rocblas_status_perf_degraded;
        }
        else
        {
            // Not enough device memory for either optimal or degraded algorithm
            ret = rocblas_status_memory_error;
        }
    }
    return ret;


===================
Thread Safe Logging
===================

rocBLAS has thread safe logging. This prevents garbled output when multiple threads are writing to the same file.

Thread safe logging is obtained from using ``rocblas_internal_ostream``, a class that can be used similarly to ``std::ostream``. It provides standardized methods for formatted output to either strings or files. The default constructor of ``rocblas_internal_ostream`` writes to strings, which are thread-safe because they are owned by the calling thread. There are also ``rocblas_internal_ostream`` constructors for writing to files. The ``rocblas_internal_ostream::yaml_on`` and ``rocblas_internal_ostream::yaml_off`` IO modifiers turn YAML formatting mode on and off.

``rocblas_cout`` and ``rocblas_cerr`` are the thread-safe versions of ``std::cout`` and ``std::cerr``.

Many output identifiers have been marked "poisoned" in rocblas-test and rocblas-bench, to catch the use of non-thread-safe IO. These include ``std::cout``, ``std::cerr``, ``printf``, ``fprintf``, ``fputs``, ``puts``, and others. The poisoning is not turned on in the library itself or in the samples, because we cannot impose restrictions on the use of these symbols on outside users.

``rocblas_handle`` contains three ``rocblas_internal_ostream`` pointers for logging output:

- static rocblas_internal_ostream* log_trace_os
- static rocblas_internal_ostream* log_bench_os
- static rocblas_internal_ostream* log_profile_os

The user can also create ``rocblas_internal_ostream`` pointers/objects outside the handle.

Each ``rocblas_internal_ostream`` associated with a file points to a single ``rocblas_internal_ostream::worker`` with a ``std::shared_ptr``, for writing to the file. The worker is mapped from the device id and inode corresponding to the file. More than one ``rocblas_internal_ostream`` can point to the same worker.

This means if more than one ``rocblas_internal_ostream`` is writing to a single output file, they will share the same ``rocblas_internal_ostream::worker``.

The ``<<`` operator for ``rocblas_internal_ostream`` is overloaded. Output is first accumulated in ``rocblas_internal_ostream::os``, a ``std::ostringstream`` buffer. Each ``rocblas_internal_ostream`` has its own os ``std::ostringstream`` buffer, so strings in os will not be garbled.

When ``rocblas_internal_ostream.os`` is flushed with either a ``std::endl`` or an explicit flush of ``rocblas_internal_ostream``, then ``rocblas_internal_ostream::worker::send`` pushes the string contents of ``rocblas_internal_ostream.os`` and a promise, the pair being called a task, onto ``rocblas_internal_ostream.worker.queue``.

The send function uses promise/future to asynchronously transfer data from ``rocblas_internal_ostream.os`` to ``rocblas_internal_ostream.worker.queue``, and to wait for the worker to finish writing the string to the file. It also locks a mutex to make sure the push of the task onto the queue is atomic.

The ``ostream.worker.queue`` will contain a number of tasks. When ``rocblas_internal_ostream`` is destroyed, all the ``tasks.string`` in ``rocblas_internal_ostream.worker.queue`` are printed to the ``rocblas_internal_ostream`` file, the ``std::shared_ptr`` to the ``ostream.worker`` is destroyed, and if the reference count to the worker becomes 0, the worker's thread is sent a 0-length string to tell it to exit.


===========================
rocBLAS Numerical Checking
===========================

.. note::
  Performance will degrade when numerical checking is enabled.

rocBLAS provides the environment variable ``ROCBLAS_CHECK_NUMERICS``, which allows users to debug numerical abnormalities. Setting a value of ``ROCBLAS_CHECK_NUMERICS`` enables checks on the input and the output vectors/matrices
of the rocBLAS functions for (not-a-number) NaN's, zeros, infinities, and denormal/subnormal values. Numerical checking is available to check the input and the output vectors for all level 1 and 2 functions.
In level 2 functions, only the general (ge) type input and the output matrix can be checked for numerical abnormalities. In level 3, GEMM is the only function to have numerical checking.


``ROCBLAS_CHECK_NUMERICS`` is a bitwise OR of zero or more bit masks as follows:

* ``ROCBLAS_CHECK_NUMERICS = 0``: is not set, then there is no numerical checking

* ``ROCBLAS_CHECK_NUMERICS = 1``: fully informative message, prints the results of numerical checking whether the input and the output Matrices/Vectors have NaN/zero/infinity/denormal values to the console

* ``ROCBLAS_CHECK_NUMERICS = 2``: prints result of numerical checking only if the input and the output Matrices/Vectors has a NaN/infinity/denormal value

* ``ROCBLAS_CHECK_NUMERICS = 4``: return ``rocblas_status_check_numeric_fail`` status if there is a NaN/infinity/denormal value

* ``ROCBLAS_CHECK_NUMERICS = 8``: ignore denormal values if there are no NaN/infinity values present

An example usage of ``ROCBLAS_CHECK_NUMERICS`` is shown below,

.. code-block:: bash

    ROCBLAS_CHECK_NUMERICS=4 ./rocblas-bench -f gemm -i 1 -j 0

The above command will return a ``rocblas_status_check_numeric_fail`` if the input and the output matrices of BLAS level 3 GEMM function has a NaN/infinity/denormal value.
If there are no numerical abnormalities, then ``rocblas_status_success`` is returned.

.. note::
  In stream capture mode all numerical checking will be skipped and ``rocblas_status_success`` is returned.

===============================================
rocBLAS Order of Argument Checking and Logging
===============================================

Legacy BLAS
-------------

Legacy BLAS has two types of argument checking:

- Error-return for incorrect argument (Legacy BLAS implement this with a call to the function ``XERBLA``)
- Quick-return-success when an argument allows for the subprogram to be a no-operation or a constant result

Level 2 and Level 3 BLAS subprograms have both error-return and quick-return-success. Level 1 BLAS subprograms have only quick-return-success

rocBLAS
--------

rocBLAS has 5 types of argument checking:

- ``rocblas_status_invalid_handle`` if the handle is a NULL pointer
- ``rocblas_status_invalid_size`` for invalid size, increment or leading dimension argument
- ``rocblas_status_invalid_value`` for unsupported enum value
- ``rocblas_status_success`` for quick-return-success
- ``rocblas_status_invalid_pointer`` for NULL argument pointers


rocBLAS has the Following Differences When Compared To Legacy BLAS
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

- It is a C API, returning a ``rocblas_status`` type indicating the success of the call.
- In legacy BLAS, the following functions return a scalar result: ``dot``, ``nrm2``, ``asum``, ``amax``, and ``amin``. In rocBLAS, a pointers to scalar return value is passed as the last argument.
- The first argument is a ``rocblas_handle`` argument, an opaque pointer to rocBLAS resources, corresponding to a single HIP stream.
- Scalar arguments like alpha and beta are pointers on either the host or device, controlled by the rocBLAS handle's pointer mode.  In cases where the other arguments do not dictate an early return, if the alpha and beta pointers are NULL the function will return ``rocblas_status_invalid_pointer``.
- Vector and matrix arguments are always pointers to device memory.
- When ``rocblas_pointer_mode == rocblas_pointer_mode_host`` alpha and beta values are inspected and based on their values it is determined which vector and matrix pointers must be dereferenced.  If these pointers will be dereferenced a NULL pointer will lead to a return value ``rocblas_status_invalid_pointer``.
- Otherwise if ``rocblas_pointer_mode == rocblas_pointer_mode_device`` we do NOT check if these vector or matrix pointers will dereference a NULL pointer as we do not want to slow execution to fetch and inspect alpha and beta values.
- The ``ROCBLAS_LAYER`` environment variable controls the option to log argument values.
- There is added functionality like
  - batched
  - strided_batched
  - mixed precision in gemm_ex, gemm_batched_ex, and gemm_strided_batched_ex

To Accommodate the Additions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- See Logging below.
- For batched and strided_batched L2 and L3 functions, there is a quick-return-success for ``batch_count == 0``, and an invalid size error for ``batch_count < 0``.
- For batched and strided_batched L1 functions, there is a quick-return-success for ``batch_count <= 0``
- When ``rocblas_pointer_mode == rocblas_pointer_mode_device`` alpha and beta are not copied from device to host for quick-return-success checks. In this case, the quick-return-success checks are omitted. This will still give a correct result, but the operation will be slower.
- For strided_batched functions there is no argument checking for stride. To access elements in a strided_batched_matrix, for example the C matrix in gemm, the zero based index is calculated as ``i1 + i2 * ldc + i3 * stride_c``, where ``i1 = 0, 1, 2, ..., m-1``; ``i2 = 0, 1, 2, ..., n-1``; ``i3 = 0, 1, 2, ..., batch_count -1``. An incorrect stride can result in a core dump due a segmentation fault. It can also produce an indeterminate result if there is a memory overlap in the output matrix between different values of ``i3``.


Device Memory Size Queries
--------------------------

- When ``handle->is_device_memory_size_query()`` is true, the call is not a normal call, but it is a device memory size query.

- No logging should be performed during device memory size queries.

- If the rocBLAS kernel requires no temporary device memory, the macro ``RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)`` can be called after checking that ``handle != nullptr``.

- If the rocBLAS kernel requires temporary device memory, then it should be set, and the kernel returned, by calling ``return handle->set_optimal_device_memory_size(size...)``, where ``size...`` is a list of one or more sizes for different sub-problems. The sizes are rounded up and added.

Logging
--------

- There is logging before a quick-return-success or error-return, except:
  - When ``handle == nullptr``, return ``rocblas_status_invalid_handle``.
  - When ``handle->is_device_memory_size_query()`` returns ``true``.

- Vectors and matrices are logged with their addresses and are always on device memory.
- Scalar values in device memory are logged as their addresses. Scalar values in host memory are logged as their values, with a ``nullptr`` logged as ``NaN`` (``std::numeric_limits<T>::quiet_NaN()``).

rocBLAS Control Flow
--------------------

1. If ``handle == nullptr``, then return ``rocblas_status_invalid_handle``.

2. If the function does not require temporary device memory, then call the macro ``RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);``.

3. If the function requires temporary device memory, and ``handle->is_device_memory_size_query()`` is ``true``, then validate any pointers and arguments required to determine the optimal size of temporary device memory, returning ``rocblas_status_invalid_pointer`` or ``rocblas_status_invalid_size`` if the arguments are invalid, and otherwise ``return handle->set_optimal_device_memory_size(size...);``, where ``size...`` is a list of one or more sizes of temporary buffers, which are allocated with ``handle->device_malloc(size...)`` later.

4. Perform logging if enabled, taking care not to dereference ``nullptr`` arguments.

5. Check for unsupported enum value. Return ``rocblas_status_invalid_value`` if enum value is invalid.

6. Check for invalid sizes. Return ``rocblas_status_invalid_size`` if size arguments are invalid.

7. Return ``rocblas_status_invalid_pointer`` if any pointers used to determine quick return conditions are NULL.

8. If quick return conditions are met:

   - If there is no return value
     - Return ``rocblas_status_success``
   - If there is a return value
     - If the return value pointer argument is nullptr, return ``rocblas_status_invalid_pointer``
     - Else, return ``rocblas_status_success``

9. If any pointers not checked in #7 are NULL and MUST be dereferenced return ``rocblas_status_invalid_pointer``; only when in ``rocblas_pointer_mode == rocblas_pointer_mode_host`` can it be determined efficiently if some vector/matrix arguments must be dereferenced.

10. (Optional.) Allocate device memory, returning ``rocblas_status_memory_error`` if the allocation fails.

11. If all checks above pass, launch the kernel and return ``rocblas_status_success``.


Legacy L1 BLAS "single vector"
-------------------------------

Below are four code snippets from NETLIB for "single vector" legacy L1 BLAS. They have quick-return-success for (n <= 0) || (incx <= 0):

.. code-block:: bash

      DOUBLE PRECISION FUNCTION DASUM(N,DX,INCX)
      IF (N.LE.0 .OR. INCX.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX)
      IF (N.LT.1 .OR. INCX.LT.1) THEN
          return = ZERO

      SUBROUTINE DSCAL(N,DA,DX,INCX)
      IF (N.LE.0 .OR. INCX.LE.0) RETURN

      INTEGER FUNCTION IDAMAX(N,DX,INCX)
      IDAMAX = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      IDAMAX = 1
      IF (N.EQ.1) RETURN

Legacy L1 BLAS "two vector"
---------------------------

Below are seven legacy L1 BLAS codes from NETLIB. There is quick-return-success for (n <= 0). In addition, for DAXPY, there is quick-return-success for (alpha == 0):

.. code-block::

      SUBROUTINE DAXPY(N,alpha,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN
      IF (alpha.EQ.0.0d0) RETURN

      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      SUBROUTINE DROT(N,DX,INCX,DY,INCY,C,S)
      IF (N.LE.0) RETURN

      SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DSDOT(N,SX,INCX,SY,INCY)
      IF (N.LE.0) RETURN

      SUBROUTINE DROTM(N,DX,INCX,DY,INCY,DPARAM)
      DFLAG = DPARAM(1)
      IF (N.LE.0 .OR. (DFLAG+TWO.EQ.ZERO)) RETURN

Legacy L2 BLAS
-----------------

Below are code snippets from NETLIB for legacy L2 BLAS. They have both argument checking and quick-return-success:

.. code-block::

      SUBROUTINE DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
      INFO = 0
      IF (M.LT.0) THEN
          INFO = 1
      ELSE IF (N.LT.0) THEN
          INFO = 2
      ELSE IF (INCX.EQ.0) THEN
          INFO = 5
      ELSE IF (INCY.EQ.0) THEN
          INFO = 7
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 9
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGER  ',INFO)
          RETURN
      END IF

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (ALPHA.EQ.ZERO)) RETURN

.. code-block::

      SUBROUTINE DSYR(UPLO,N,ALPHA,X,INCX,A,LDA)

      INFO = 0
      IF (.NOT.LSAME(UPLO,'U') .AND. .NOT.LSAME(UPLO,'L')) THEN
          INFO = 1
      ELSE IF (N.LT.0) THEN
          INFO = 2
      ELSE IF (INCX.EQ.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,N)) THEN
          INFO = 7
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DSYR  ',INFO)
          RETURN
      END IF

      IF ((N.EQ.0) .OR. (ALPHA.EQ.ZERO)) RETURN

.. code-block::

      SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)

      INFO = 0
      IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND. .NOT.LSAME(TRANS,'C')) THEN
          INFO = 1
      ELSE IF (M.LT.0) THEN
          INFO = 2
      ELSE IF (N.LT.0) THEN
          INFO = 3
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      ELSE IF (INCY.EQ.0) THEN
          INFO = 11
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMV ',INFO)
          RETURN
      END IF

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. ((ALPHA.EQ.ZERO).AND. (BETA.EQ.ONE))) RETURN

.. code-block::

      SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)

      INFO = 0
      IF (.NOT.LSAME(UPLO,'U') .AND. .NOT.LSAME(UPLO,'L')) THEN
          INFO = 1
      ELSE IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND. .NOT.LSAME(TRANS,'C')) THEN
          INFO = 2
      ELSE IF (.NOT.LSAME(DIAG,'U') .AND. .NOT.LSAME(DIAG,'N')) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (LDA.LT.MAX(1,N)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DTRSV ',INFO)
          RETURN
      END IF

      IF (N.EQ.0) RETURN

Legacy L3 BLAS
----------------

Below is a code snippet from NETLIB for legacy L3 BLAS dgemm. It has both argument checking and quick-return-success:

.. code-block::

      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)

      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NROWA = M
          NCOLA = K
      ELSE
          NROWA = K
          NCOLA = M
      END IF
      IF (NOTB) THEN
          NROWB = K
      ELSE
          NROWB = N
      END IF

  //  Test the input parameters.

      INFO = 0
      IF ((.NOT.NOTA) .AND. (.NOT.LSAME(TRANSA,'C')) .AND.
     +    (.NOT.LSAME(TRANSA,'T'))) THEN
          INFO = 1
      ELSE IF ((.NOT.NOTB) .AND. (.NOT.LSAME(TRANSB,'C')) .AND.
     +         (.NOT.LSAME(TRANSB,'T'))) THEN
          INFO = 2
      ELSE IF (M.LT.0) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (K.LT.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
          INFO = 8
      ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
          INFO = 10
      ELSE IF (LDC.LT.MAX(1,M)) THEN
          INFO = 13
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMM ',INFO)
          RETURN
      END IF

  //  Quick return if possible.

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (((ALPHA.EQ.ZERO).OR. (K.EQ.0)).AND. (BETA.EQ.ONE))) RETURN

.. raw:: latex

    \newpage

=================================
rocBLAS Benchmarking and Testing
=================================

There are three client executables that can be used with rocBLAS. They are:

- rocblas-bench
- rocblas-gemm-tune
- rocblas-test

These three clients can be built by following the instructions in the Building and Installing section of the User Guide. After building the rocBLAS clients, they can be found in the directory ``rocBLAS/build/release/clients/staging``.

.. note::
  The ``rocblas-bench`` and ``rocblas-test`` executables use AMD's ILP64 version of AOCL-BLAS 4.2 as the host reference BLAS to verify correctness. However, there is a known issue with AOCL-BLAS that can cause these executables to hang. This problem can arise because the AOCL-BLAS library launches multiple threads to perform computations. If the number of threads matches the total number of CPU threads, it can lead to thread oversubscription, causing the program to hang.
  To prevent this issue, we recommend limiting the number of threads that the AOCL-BLAS library uses to fewer than the available CPU cores. You can do this by setting the ``OMP_NUM_THREADS`` environment variable.

  For example, on a server with 32 cores, you can limit the number of threads to 28 by setting ``export OMP_NUM_THREADS=28``

The next three sections will provide a brief explanation and the usage of each rocBLAS client.

rocblas-bench
--------------

rocblas-bench is used to measure performance and verify the correctness of rocBLAS functions.

It has a command line interface. For more information:

.. code-block:: bash

   rocBLAS/build/release/clients/staging/rocblas-bench --help

The following table shows all the data types in rocBLAS:

.. list-table:: Data types in rocBLAS
   :widths: 25 25
   :header-rows: 1

   * - Data type
     - accronym
   * - real 16 bit Brain Floating Point
     - bf16_r
   * - real half
     - f16_r (h)
   * - real float
     - f32_r (s)
   * - real double
     - f64_r (d)
   * - Complex float
     - f32_c (c)
   * - Complex double
     - f64_c (z)
   * - Integer 32
     - i32_r
   * - Integer 8
     - i8_r


All options for problem types in rocBLAS for gemm are shown here:

- N: not transposed
- T: transposed
- C: complex conjugate (for real data type C is the same as T)


.. list-table:: various matrix operations
   :widths: 25 25 25
   :header-rows: 1

   * - Problem Types
     - problem_type
     - data type
   * - NN
     - Cijk_Ailk_Bljk
     - real/complex
   * - NT
     - Cijk_Ailk_Bjlk
     - real/complex
   * - TN
     - Cijk_Alik_Bljk
     - real/complex
   * - TT
     - Cijk_Alik_Bjlk
     - real/complex
   * - NC
     - Cijk_Ailk_BjlkC
     - complex
   * - CN
     - Cijk_AlikC_Bljk
     - complex
   * - CC
     - Cijk_AlikC_BjlkC
     - complex
   * - TC
     - Cijk_Alik_BjlkC
     - complex
   * - CT
     - Cijk_AlikC_Bjlk
     - complex


For example, NT means A * B\ :sup:`T`\.

Gemm functions can be divided into two main categories:

#. HPA functions (HighPrecisionAccumulate) where the compute data type is different from the input data type (A/B). All HPA functions must be called using *gemm_ex* API in rocblas-bench (and not gemm). gemm_ex function name consists of three letters: A/B data type, C/D data type, compute data type.

#. Non-HPA functions where the input (A/B), output (C/D), and compute data types are all the same. Non-HPA cases can be called using *gemm* or *gemm_ex*. But using *gemm* is recommended.

The following table shows all possible gemm functions in rocBLAS.

.. list-table:: all gemm functions in rocBLAS
   :widths: 20 30 10 10 10
   :header-rows: 1

   * - function
     - Kernel name
     - A/B data type
     - C/D data type
     - compute data type
   * - hgemm
     - <arch>_<problem_type>_HB
     - f16_r
     - f16_r
     - f16_r
   * - hgemm_batched
     - <arch>_<problem_type>_HB_GB
     - f16_r
     - f16_r
     - f16_r
   * - hgemm_strided_batched
     - <arch>_<problem_type>_HB
     - f16_r
     - f16_r
     - f16_r
   * - sgemm
     - <arch>_<problem_type>_SB
     - f32_r
     - f32_r
     - f32_r
   * - sgemm_batched
     - <arch>_<problem_type>_SB_GB
     - f32_r
     - f32_r
     - f32_r
   * - sgemm_strided_batched
     - <arch>_<problem_type>_SB
     - f32_r
     - f32_r
     - f32_r
   * - dgemm
     - <arch>_<problem_type>_DB
     - f64_r
     - f64_r
     - f64_r
   * - dgemm_batched
     - <arch>_<problem_type>_DB_GB
     - f64_r
     - f64_r
     - f64_r
   * - dgemm_strided_batched
     - <arch>_<problem_type>_DB
     - f64_r
     - f64_r
     - f64_r
   * - cgemm
     - <arch>_<problem_type>_CB
     - f32_c
     - f32_c
     - f32_c
   * - cgemm_batched
     - <arch>_<problem_type>_CB_GB
     - f32_c
     - f32_c
     - f32_c
   * - cgemm_strided_batched
     - <arch>_<problem_type>_CB
     - f32_c
     - f32_c
     - f32_c
   * - zgemm
     - <arch>_<problem_type>_ZB
     - f64_c
     - f64_c
     - f64_c
   * - zgemm_batched
     - <arch>_<problem_type>_ZB_GB
     - f64_c
     - f64_c
     - f64_c
   * - zgemm_strided_batched
     - <arch>_<problem_type>_ZB
     - f64_c
     - f64_c
     - f64_c
   * - HHS
     - <arch>_<problem_type>_HHS_BH
     - f16_r
     - f16_r
     - f32_r
   * - HHS_batched
     - <arch>_<problem_type>_HHS_BH_GB
     - f16_r
     - f16_r
     - f32_r
   * - HHS_strided_batched
     - <arch>_<problem_type>_HHS_BH
     - f16_r
     - f16_r
     - f32_r
   * - HSS
     - <arch>_<problem_type>_HSS_BH
     - f16_r
     - f32_r
     - f32_r
   * - HSS_batched
     - <arch>_<problem_type>_HSS_BH_GB
     - f16_r
     - f32_r
     - f32_r
   * - HSS_strided_batched
     - <arch>_<problem_type>_HSS_BH
     - f16_r
     - f32_r
     - f32_r
   * - BBS
     - <arch>_<problem_type>_BBS_BH
     - bf16_r
     - bf16_r
     - f32_r
   * - BBS_batched
     - <arch>_<problem_type>_BBS_BH_GB
     - bf16_r
     - bf16_r
     - f32_r
   * - BBS_strided_batched
     - <arch>_<problem_type>_BBS_BH
     - bf16_r
     - bf16_r
     - f32_r
   * - BSS
     - <arch>_<problem_type>_BSS_BH
     - bf16_r
     - f32_r
     - f32_r
   * - BSS_batched
     - <arch>_<problem_type>_BSS_BH_GB
     - bf16_r
     - f32_r
     - f32_r
   * - BSS_strided_batched
     - <arch>_<problem_type>_BSS_BH
     - bf16_r
     - f32_r
     - f32_r
   * - I8II
     - <arch>_<problem_type>_I8II_BH
     - I8
     - I
     - I
   * - I8II_batched
     - <arch>_<problem_type>_I8II_BH_GB
     - I8
     - I
     - I
   * - I8II_strided_batched
     - <arch>_<problem_type>_I8II_BH
     - I8
     - I
     - I


.. raw:: latex

    \newpage

How to benchmark the performance of a gemm function using rocblas-bench
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

This method is good only if you want to test a few sizes, otherwise, refer to the next section. The following listing shows how to configure rocblas-bench to call each of the gemm functions:


Non-HPA cases (gemm)

.. code-block:: bash

   #dgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r d --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1.0
   # dgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r d --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # dgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r d --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # sgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r s --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # sgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r s --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # sgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r s --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # hgemm (this function is not really very fast. Use HHS instead, which is faster and more accurate)
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r h --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # hgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r h --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # hgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r h --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # cgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # cgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # cgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r c --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # zgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r z --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # zgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r z --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # zgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r z --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # cgemm (NC)
   $ ./rocblas-bench -f gemm --transposeA N --transposeB C -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # cgemm batched (NC)
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB C -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # cgemm strided batched (NC)
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB C -m 1024 -n 2048 -k 512 -r c --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5



.. raw:: latex

    \newpage

HPA cases (gemm_ex)

.. code-block:: bash

   # HHS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type h --ldc 1024 --d_type h --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # HHS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type h --ldc 1024 --d_type h --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # HHS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --stride_a 4096 --b_type h --ldb 2048 --stride_b 4096 --c_type h --ldc 1024 --stride_c 2097152 --d_type h --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # HSS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # HSS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # HSS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --stride_a 4096 --b_type h --ldb 2048 --stride_b 4096 --c_type s --ldc 1024 --stride_c 2097152 --d_type s --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # BBS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type bf16_r --ldc 1024 --d_type bf16_r --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # BBS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type bf16_r --ldc 1024 --d_type bf16_r --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # BBS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --stride_a 4096 --b_type bf16_r --ldb 2048 --stride_b 4096 --c_type bf16_r --ldc 1024 --stride_c 2097152 --d_type bf16_r --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # BSS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # BSS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # BSS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --stride_a 4096 --b_type bf16_r --ldb 2048 --stride_b 4096 --c_type s --ldc 1024 --stride_c 2097152 --d_type s --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # I8II
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --b_type i8_r --ldb 2048 --c_type i32_r --ldc 1024 --d_type i32_r --ldd 1024 --compute_type i32_r --alpha 1.1 --beta 1
   # I8II batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --b_type i8_r --ldb 2048 --c_type i32_r --ldc 1024 --d_type i32_r --ldd 1024 --compute_type i32_r --alpha 1.1 --beta 1 --batch_count 5
   # I8II strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --stride_a 4096 --b_type i8_r --ldb 2048 --stride_b 4096 --c_type i32_r --ldc 1024 --stride_c 2097152 --d_type i32_r --ldd 1024 --stride_d 2097152 --compute_type i32_r --alpha 1.1 --beta 1 --batch_count 5

.. raw:: latex

    \newpage

How to set rocblas-bench parameters in a yaml file
''''''''''''''''''''''''''''''''''''''''''''''''''

If you want to benchmark many sizes, it is recommended to use rocblas-bench with the batch call to eliminate the latency in loading the GEMM library which rocblas links to.  The batch call takes a yaml file with a list of all problem sizes. You can have multiple sizes of different types in one yaml file. The benchmark setting is different from the direct call to the rocblas-bench. A sample setting for each function is listed below. Once you have the yaml file, you can benchmark the sizes as follows:

.. code-block:: bash

  rocBLAS/build/release/clients/staging/rocblas-bench --yaml problem-sizes.yaml


Here are the configurations for each function:


Non-HPA cases (gemm)

.. code-block:: bash

    # dgemm
    - { rocblas_function: "rocblas_dgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # dgemm batched
    - { rocblas_function: "rocblas_dgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # dgemm strided batched
    - { rocblas_function: "rocblas_dgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # sgemm
    - { rocblas_function: "rocblas_sgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # sgemm batched
    - { rocblas_function: "rocblas_sgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # sgemm strided batched
    - { rocblas_function: "rocblas_sgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # hgemm
    - { rocblas_function: "rocblas_hgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # hgemm batched
    - { rocblas_function: "rocblas_hgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # hgemm strided batched
    - { rocblas_function: "rocblas_hgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # cgemm
    - { rocblas_function: "rocblas_cgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # cgemm batched
    - { rocblas_function: "rocblas_cgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # cgemm strided batched
    - { rocblas_function: "rocblas_cgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # zgemm
    - { rocblas_function: "rocblas_zgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # zgemm batched
    - { rocblas_function: "rocblas_zgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # zgemm strided batched
    - { rocblas_function: "rocblas_zgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # cgemm
    - { rocblas_function: "rocblas_cgemm",         transA: "N", transB: "C", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # cgemm batched
    - { rocblas_function: "rocblas_cgemm_batched", transA: "N", transB: "C", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # cgemm strided batched
    - { rocblas_function: "rocblas_cgemm_strided_batched", transA: "N", transB: "C", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

.. raw:: latex

    \newpage

HPA cases (gemm_ex)

.. code-block:: bash

    # HHS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # HHS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # HHS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # HSS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # HSS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # HSS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # BBS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # BBS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # BBS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # BSS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # BSS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # BSS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # I8II
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: i8_r, b_type: i8_r, c_type: i32_r, d_type: i32_r, compute_type: i32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # I8II batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: i8_r, b_type: i8_r, c_type: i32_r, d_type: i32_r, compute_type: i32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # I8II strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: i8_r, b_type: i8_r, c_type: i32_r, d_type: i32_r, compute_type: i32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }


For example, the performance of sgemm using rocblas-bench on a vega20 machine returns:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096
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

To measure performance on the ILP64 API functions, when they exist, add the argument ``--api 1`` rather than changing the function name set in ``-f``.
Logging affects performance, so only use it to log the command to copy and change, then run the command without logging to measure performance.

Note that rocblas-bench also has the flag ``-v 1`` for correctness checks.

How to benchmark the performance of special case gemv_batched and gemv_strided_batched functions for mixed precision (HSH, HSS, TST, TSS) using rocblas-bench
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

The command to execute rocblas-bench for ``rocblas_hshgemv_batched`` with half-precision input, single precision compute, and half-precision output (HSH):

.. code-block:: bash

   ./rocblas-bench -f gemv_batched --a_type f16_r --c_type f16_r --compute_type f32_r --transposeA N -m 128 -n 128 --alpha 1  --lda 128  --incx 1 --beta 1 --incy 1  --batch_count 2

For the above command, instead of using the ``-r`` to specify the precision, we need to pass three additional arguments (``a_type``, ``c_type``, and ``compute_type``) to resolve the ambiguity of using mixed precision compute.

This mixed-precision support is only available for gemv_batched, gemv_strided_batched, and rocBLAS extension functions (e.g, ``axpy_ex``, ``scal_ex``, ``gemm_ex``, etc.). For further information, refer to the :ref:`api-reference-guide`.

rocblas-gemm-tune
-----------------

rocblas-gemm-tune is used to find the best performing GEMM kernel for each of a given set of GEMM problems.

It has a command line interface, which mimics the ``--yaml`` input used by rocblas-bench (see above section for details).

To generate the expected ``--yaml`` input, profile logging can be used, by setting environment variable ``ROCBLAS_LAYER=4``.

For more information on rocBLAS logging, see ``Logging in rocBLAS``, in the :ref:`api-reference-guide`.

An example input file:

.. code-block:: bash

    - {'rocblas_function': 'gemm_ex', 'transA': 'N', 'transB': 'N', 'M': 320, 'N': 588, 'K': 4096, 'alpha': 1, 'a_type': 'f32_r', 'lda': 320, 'b_type': 'f32_r', 'ldb': 6144, 'beta': 0, 'c_type': 'f32_r', 'ldc': 320, 'd_type': 'f32_r', 'ldd': 320, 'compute_type': 'f32_r', 'device': 0}
    - {'rocblas_function': 'gemm_ex', 'transA': 'N', 'transB': 'N', 'M': 512, 'N': 3096, 'K': 512, 'alpha': 1, 'a_type': 'f16_r', 'lda': 512, 'b_type': 'f16_r', 'ldb': 512, 'beta': 0, 'c_type': 'f16_r', 'ldc': 512, 'd_type': 'f16_r', 'ldd': 512, 'compute_type': 'f32_r', 'device': 0}

Expected output (note selected GEMM idx may differ):

.. code-block:: bash

    transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,input_type,output_type,compute_type,solution_index
    N,N,320,588,1,4096,1,0,320,6144,320,f32_r,f32_r,f32_r,3788
    N,N,512,3096,1,512,1,0,512,512,512,f16_r,f16_r,f32_r,4546

Where the far right values (``solution_index``) are the indices of the best performing kernels for those GEMMs in the rocBLAS kernel library. These indices can be directly used in future GEMM calls, but please note that these indices cannot be reused across library releases or across different device architectures.

See `example_user_driven_tuning.cpp <https://github.com/ROCm/rocBLAS/blob/develop/clients/samples/example_user_driven_tuning.cpp>`_ for sample code of directly using kernels via their indices.

If the output is stored in a file, the results can be used to override default kernel selection with the kernels found, by setting the environment variable ``ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=<path>``, where ``<path>`` points to the stored file.

rocblas-test
-------------

rocblas-test is used in performing rocBLAS unit tests and it uses Googletest framework.

The tests are in five categories:

- quick
- pre_checkin
- nightly
- stress
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

The default verbosity shows test category totals and specific test failure details, matching an implicit environment variable setting of GTEST_LISTENER=NO_PASS_LINE_IN_LOG.
To get an output listing of each individual test that is run, use:

.. code-block:: bash

   GTEST_LISTENER=PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*quick*

``rocblas-test`` can be driven by tests specified in a yaml file using the ``--yaml`` argument.
As the test categories pre_checkin and nightly can require hours to run, a short smoke test set is provided in a yaml file.
This ``rocblas_smoke.yaml`` test set should only require a few minutes to test a few small problem sizes for every function:

.. code-block:: bash

   ./rocblas-test --yaml rocblas_smoke.yaml

* yaml extension for lock step multiple variable scanning

Both rocblas-test and rocblas-bench can use an extension added to scan over multiple variables in lock step implemented by the Arguments class.  For this purpose set the Arugments member variable
``scan`` to the range to scan over and use ``*c_scan_value`` to retrieve the values. This can be used to avoid all combinations of yaml variable values that are normally generated.
For example, ``- { scan: [32..256..32], M: *c_scan_value, N: *c_scan_value, lda: *c_scan_value }``

* large memory tests (stress category)

Some tests in the stress category may attempt to allocate more RAM than available.  While these tests should automatically get skipped, in some cases, such
as running in a docker container, they may instead result in process termination.  You can limit the peak RAM allocations in GB using the environment variable:

.. code-block:: bash

   ROCBLAS_CLIENT_RAM_GB_LIMIT=32 ./rocblas-test --gtest_filter=*stress*

* long-running tests

The rocblas-test process will be terminated if a single test takes longer than a timeout. Change the timeout with the environment variable ROCBLAS_TEST_TIMEOUT,
whose value is in seconds (default is 600 seconds):

  .. code-block:: bash

   ROCBLAS_TEST_TIMEOUT=900 ./rocblas-test --gtest_filter=*stress*

* debugging rocblas-test

The rocblas-test process will catch signals internally which may interfere with debugger use.  To defeat this set the environment variable ROCBLAS_TEST_NO_SIGACTION:

  .. code-block:: bash

   ROCBLAS_TEST_NO_SIGACTION=1 rocgdb ./rocblas-test --gtest_filter=*stress*


Add New rocBLAS Unit Test
--------------------------

To add new data-driven tests to the rocBLAS Google Test Framework:

**I**. Create a C++ header file with the name ``testing_<function>.hpp`` in the
``include`` subdirectory, with templated functions for a specific rocBLAS
routine. Examples:

.. code-block::

   testing_gemm.hpp
   testing_gemm_ex.hpp

In this ``testing_*.hpp`` file, create a templated function which returns ``void``
and accepts a ``const Arguments&`` parameter. Example:

.. code-block::

   template<typename Ti, typename To, typename Tc>
   void testing_gemm_ex(const Arguments& arg)
   {
   // ...
   }

This function is used for yaml file driven argument testing.  It will be invoked by the dispatch code for each permutation of the yaml driven parameters.
Additionally a template function for bad argument handling tests should be created.  Example:

.. code-block::

  template <typename T>
  void testing_gemv_bad_arg(const Arguments& arg)
  {
  // ...
  }

These ``bad_arg`` test function templates should be used to set arguments programmatically where it is simpler than the yaml approach, for example to pass NULL pointers.
It is expected that member variable values in the Arguments parameter will not be utilized with the common exception of ``api`` member variable of Arguments which can drive selection of C, FORTRAN,
C_64, or FORTRAN_64 API bad argument tests.

All functions should be generalized with template parameters as much as possible,
to avoid copy-and-paste code.

In this function, use the following macros and functions to check results:

.. code-block::

   HIP_CHECK_ERROR             Verifies that a HIP call returns success
   ROCBLAS_CHECK_ERROR         Verifies that a rocBLAS call returns success
   EXPECT_ROCBLAS_STATUS       Verifies that a rocBLAS call returns a certain status
   unit_check_general          Check that two answers agree (see unit.hpp)
   near_check_general          Check that two answers are close (see near.hpp)

.. code-block::

   DAPI_CHECK                  Verifies either LP64 or ILP64 function form returns success (based on Arguments member variable api)
   DAPI_EXPECT                 Verifies either LP64 or ILP64 function form returns a certain status
   DAPI_DISPATCH               Invoke either LP64 or ILP64 function form

In addition, you can use Google Test Macros such as the below, as long as they are
guarded by ``#ifdef GOOGLE_TEST``\ :

.. code-block::

   EXPECT_EQ
   ASSERT_EQ
   EXPECT_TRUE
   ASSERT_TRUE
   ...

Note: The ``device_vector`` template allocates memory on the device. You must check whether
converting the ``device_vector`` to ``bool`` returns ``false``\ , and if so, report a HIP memory
error and then exit the current function. Example:

.. code-block::

   // allocate memory on device
   device_vector<T> dx(size_x);
   device_vector<T> dy(size_y);
   if(!dx || !dy)
   {
       CHECK_HIP_ERROR(hipErrorOutOfMemory);
       return;
   }

The general outline of the function should be:

#. Convert any scalar arguments (e.g., ``alpha`` and ``beta``\ ) to ``double``.
#. If the problem size arguments are invalid, use a ``safe_size`` to allocate arrays,
   call the rocBLAS routine with the original arguments, and verify that it returns
   ``rocblas_status_invalid_size``. Return.
#. Set up host and device arrays (see ``rocblas_vector.hpp`` and ``rocblas_init.hpp``\ ).
#. Call a CBLAS or other reference implementation on the host arrays.
#. Call rocBLAS using both device pointer mode and host pointer mode, verifying that
   every rocBLAS call is successful by wrapping it in ``ROCBLAS_CHECK_ERROR()``.
#. If ``arg.unit_check`` is enabled, use ``unit_check_general`` or ``near_check_general`` to validate results.
#. (Deprecated) If ``arg.norm_check`` is enabled, calculate and print out norms.
#. If ``arg.timing`` is enabled, perform benchmarking (currently under refactoring).

**II**. Create a C++ file with the name ``<function>_gtest.cpp`` in the ``gtest``
subdirectory, where ``<function>`` is a non-type-specific shorthand for the
function(s) being tested. Example:

.. code-block::

   gemm_gtest.cpp
   trsm_gtest.cpp
   blas1_gtest.cpp

In the C++ file, follow these steps:

A. Include the header files related to the tests, as well as ``type_dispatch.hpp``.
Example:

.. code-block:: c++

   #include "testing_syr.hpp"
   #include "type_dispatch.hpp"

B. Wrap the body with an anonymous namespace, to minimize namespace collisions:

.. code-block:: c++

   namespace {

C. Create a templated class which accepts any number of type parameters followed by one anonymous trailing type parameter defaulted to ``void`` (to be used with ``enable_if``\ ).

Choose the number of type parameters based on how likely in the future that
the function will support a mixture of that many different types, e.g. Input
type (\ ``Ti``\ ), Output type (\ ``To``\ ), Compute type (\ ``Tc``\ ). If the function will
never support more than 1-2 type parameters, then that many can be used. But
if the function may be expanded later to support mixed types, then those
should be planned for ahead of time and placed in the template parameters.

Unless the number of type parameters is greater than one and is always
fixed, then later type parameters should default to earlier ones, so that
a subset of type arguments can used, and so that code which works for
functions which take one type parameter may be used for functions which
take one or more type parameters. Example:

.. code-block:: c++

   template< typename Ti, typename To = Ti, typename Tc = To, typename = void>

Make the primary definition of this class template derive from the ``rocblas_test_invalid`` class. Example:

.. code-block:: c++

    template <typename T, typename = void>
    struct syr_testing : rocblas_test_invalid
    {
    };

D. Create one or more partial specializations of the class template conditionally enabled by the type parameters matching legal combinations of types.

If the first type argument is ``void``\ , then these partial specializations must not apply, so that the default based on ``rocblas_test_invalid`` can perform the correct behavior when ``void`` is passed to indicate failure.

In the partial specialization(s), derive from the ``rocblas_test_valid`` class.

In the partial specialization(s), create a functional ``operator()`` which takes a ``const Arguments&`` parameter and calls templated test functions (usually in ``include/testing_*.hpp``\ ) with the specialization's template arguments when the ``arg.function`` string matches the function name. If ``arg.function`` does not match any function related to this test, mark it as a test failure. Example:

.. code-block:: c++

    template <typename T>
    struct syr_testing<T,
                      std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>
                      > : rocblas_test_valid
   {
       void operator()(const Arguments& arg)
       {
           if(!strcmp(arg.function, "syr"))
               testing_syr<T>(arg);
           else
               FAIL() << "Internal error: Test called with unknown function: "
                      << arg.function;
       }
   };

E. If necessary, create a type dispatch function for this function (or group of functions it belongs to) in ``include/type_dispatch.hpp``. If possible, use one of the existing dispatch functions, even if it covers a superset of allowable types. The purpose of ``type_dispatch.hpp`` is to perform runtime type dispatch in a single place, rather than copying it across several test files.

The type dispatch function takes a ``template`` template parameter of ``template<typename...> class`` and a function parameter of type ``const Arguments&``. It looks at the runtime type values in ``Arguments``\ , and instantiates the template with one or more static type arguments, corresponding to the dynamic runtime type arguments.

It treats the passed template as a functor, passing the Arguments argument to a particular instantiation of it.

The combinations of types handled by this "runtime type to template type instantiation mapping" function can be general, because the type combinations which do not apply to a particular test case will have the template argument set to derive from ``rocblas_test_invalid``\ , which will not create any unresolved instantiations. If unresolved instantiation compile or link errors occur, then the ``enable_if<>`` condition in step D needs to be refined to be ``false`` for type combinations which do not apply.

The return type of this function needs to be ``auto``\ , picking up the return type of the functor.

If the runtime type combinations do not apply, then this function should return ``TEST<void>{}(arg)``\ , where ``TEST`` is the template parameter. However, this is less important than step D above in excluding invalid type
combinations with ``enable_if``\ , since this only excludes them at run-time, and they need to be excluded by step D at compile-time in order to avoid unresolved references or invalid instantiations. Example:

.. code-block:: c++

   template <template <typename...> class TEST>
   auto rocblas_simple_dispatch(const Arguments& arg)
   {
       switch(arg.a_type)
       {
         case rocblas_datatype_f16_r: return TEST<rocblas_half>{}(arg);
         case rocblas_datatype_f32_r: return TEST<float>{}(arg);
         case rocblas_datatype_f64_r: return TEST<double>{}(arg);
         case rocblas_datatype_bf16_r: return TEST<rocblas_bfloat16>{}(arg);
         case rocblas_datatype_f16_c: return TEST<rocblas_half_complex>{}(arg);
         case rocblas_datatype_f32_c: return TEST<rocblas_float_complex>{}(arg);
         case rocblas_datatype_f64_c: return TEST<rocblas_double_complex>{}(arg);
         default: return TEST<void>{}(arg);
       }
   }

F. Create a (possibly-templated) test implementation class which derives from the ``RocBLAS_Test`` template class, passing itself to ``RocBLAS_Test`` (the CRTP pattern) as well as the template class defined above. Example:

.. code-block:: c++

   struct syr : RocBLAS_Test<syr, syr_testing>
   {
       // ...
   };

In this class, implement three static functions:

 ``static bool type_filter(const Arguments& arg)`` returns ``true`` if the types described by ``*_type`` in the ``Arguments`` structure, match a valid type combination.

This is usually implemented simply by calling the dispatch function in step E, passing it the helper ``type_filter_functor`` template class defined in ``RocBLAS_Test``. This functor uses the same runtime type checks as are used to instantiate test functions with particular type arguments, but instead, this returns ``true`` or ``false`` depending on whether a function would have been called. It is used to filter out tests whose runtime parameters do not match a valid test.

Since ``RocBLAS_Test`` is a dependent base class if this test implementation class is templated, you may need to use a fully-qualified name (\ ``A::B``\ ) to resolve ``type_filter_functor``\ , and in the last part of this name, the keyword ``template`` needs to precede ``type_filter_functor``. The first half of the fullyqualified name can be this class itself, or the full instantation of ``RocBLAS_Test<...>``. Example:

.. code-block:: c++

   static bool type_filter(const Arguments& arg)
   {
       return rocblas_blas1_dispatch<
           blas1_test_template::template type_filter_functor>(arg);
   }


``static bool function_filter(const Arguments& arg)`` returns ``true`` if the function name in ``Arguments`` matches one of the functions handled by this test. Example:

.. code-block:: c++

   // Filter for which functions apply to this suite
   static bool function_filter(const Arguments& arg)
   {
     return !strcmp(arg.function, "ger") || !strcmp(arg.function, "ger_bad_arg");
   }


``static std::string name_suffix(const Arguments& arg)`` returns a string which will be used as the Google Test name's suffix. It will provide an alphanumeric representation of the test's arguments.

Use the ``RocBLAS_TestName`` helper class template to create the name. It accepts ostream output (like ``std::cout``\ ), and can be automatically converted to ``std::string`` after all of the text of the name has been streamed to it.

The ``RocBLAS_TestName`` helper class constructor accepts a string argument which will be included in the test name. It is generally passed the ``Arguments`` structure's ``name`` member.

The ``RocBLAS_TestName`` helper class template should be passed the name of this test implementation class (including any implicit template arguments) as a template argument, so that every instantiation of this test implementation class creates a unique instantiation of ``RocBLAS_TestName``. ``RocBLAS_TestName`` has some static data that needs to be kept local to each test.

 ``RocBLAS_TestName`` converts non-alphanumeric characters into suitable replacements, and disambiguates test names when the same arguments appear more than once.

 Since the conversion of the stream into a ``std::string`` is a destructive one-time operation, the ``RocBLAS_TestName`` value converted to ``std::string`` needs to be an rvalue. Example:

.. code-block:: c++

   static std::string name_suffix(const Arguments& arg)
   {
       // Okay: rvalue RocBLAS_TestName object streamed to and returned
       return RocBLAS_TestName<syr>() << rocblas_datatype2string(arg.a_type)
           << '_' << (char) std::toupper(arg.uplo) << '_' << arg.N
           << '_' << arg.alpha << '_' << arg.incx << '_' << arg.lda;
   }

   static std::string name_suffix(const Arguments& arg)
   {
       RocBLAS_TestName<gemm_test_template> name;
       name << rocblas_datatype2string(arg.a_type);
       if(GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
           name << rocblas_datatype2string(arg.b_type)
                << rocblas_datatype2string(arg.c_type)
                << rocblas_datatype2string(arg.d_type)
                << rocblas_datatype2string(arg.compute_type);
       name << '_' << (char) std::toupper(arg.transA)
                   << (char) std::toupper(arg.transB) << '_' << arg.M
                   << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                   << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_'
                   << arg.ldc;
       // name is an lvalue: Must use std::move to convert it to rvalue.
       // name cannot be used after it's converted to a string, which is
       // why it must be "moved" to a string.
       return std::move(name);
   }

G. Choose a non-type-specific shorthand name for the test, which will be displayed as part of the test name in the Google Tests output (and hence will be stringified). Create a type alias for this name, unless the name is already the name of the class defined in step F, and it is not templated. For example, for a templated class defined in step F, create an alias for one of its instantiations:

.. code-block:: c++

   using gemm = gemm_test_template<gemm_testing, GEMM>;

H. Pass the name created in step G to the ``TEST_P`` macro, along with a broad test category name that this test belongs to (so that Google Test filtering can be used to select all tests in a category). The broad test category suffix should be _tensile if it requires Tensile.

In the body following this ``TEST_P`` macro, call the dispatch function from step E, passing it the class from step C as a template template argument, passing the result of ``GetParam()`` as an ``Arguments`` structure, and wrapping the call in the ``CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES()`` macro. Example:

.. code-block:: c++

   TEST_P(gemm, blas3_tensile) { CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam())); }

The ``CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES()`` macro detects signals such as ``SIGSEGV`` and uncaught C++ exceptions returned from rocBLAS C APIs as failures, without terminating the test program.

I. Call the ``INSTANTIATE_TEST_CATEGORIES`` macro which instantiates the Google Tests across all test categories (\ ``quick``\ , ``pre_checkin``\ , ``nightly``\ , ``known_bug``\ ), passing it the same test name as in steps G and H. Example:

.. code-block:: c++

   INSTANTIATE_TEST_CATEGORIES(gemm);

J. Don't forget to close the anonymous namespace:

.. code-block:: c++

   } // namespace

**III.** Create a ``<function>.yaml`` file with the same name as the C++ file, just with a ``.yaml`` extension.

   In the YAML file, define tests with combinations of parameters.

   The YAML files are organized as files which ``include:`` each other (an extension to YAML), define anchors for data types and data structures, list of test parameters or subsets thereof, and ``Tests`` which describe a combination of parameters including ``category`` and ``function``.

   ``category`` must be one of ``quick``\ , ``pre_checkin``\ , ``nightly``\ , or ``known_bug``. The category is automatically changed to ``known_bug`` if the test matches a test in ``known_bugs.yaml``.

   ``function`` must be one of the functions tested for and recognized in steps D-F.

   The syntax and idioms of the YAML files is best described by looking at the
   existing ``*_gtest.yaml`` files as examples.

**IV.** Add the YAML file to ``rocblas_gtest.yaml``\ , to be included. Examnple:

.. code-block:: yaml

   include: blas1_gtest.yaml

**V.** Add the YAML file to the list of dependencies for ``rocblas_gtest.data`` in ``CMakeLists.txt``.  Example:

.. code-block:: cmake

   add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                       COMMAND ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                       DEPENDS ../common/rocblas_gentest.py rocblas_gtest.yaml ../include/rocblas_common.yaml known_bugs.yaml blas1_gtest.yaml gemm_gtest.yaml gemm_batched_gtest.yaml gemm_strided_batched_gtest.yaml gemv_gtest.yaml symv_gtest.yaml syr_gtest.yaml ger_gtest.yaml trsm_gtest.yaml trtri_gtest.yaml geam_gtest.yaml set_get_vector_gtest.yaml set_get_matrix_gtest.yaml
                       WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

**VI.** Add the ``.cpp`` file to the list of sources for ``rocblas-test`` in ``CMakeLists.txt``. Example:

.. code-block:: c++

   set(rocblas_test_source
       rocblas_gtest_main.cpp
       ${Tensile_TEST_SRC}
       set_get_pointer_mode_gtest.cpp
       logging_mode_gtest.cpp
       set_get_vector_gtest.cpp
       set_get_matrix_gtest.cpp
       blas1_gtest.cpp
       gemv_gtest.cpp
       ger_gtest.cpp
       syr_gtest.cpp
       symv_gtest.cpp
       geam_gtest.cpp
       trtri_gtest.cpp
      )

**VII.** Aim for a function to have tests in each of the categories: quick, pre_checkin, nightly. Aim for tests for each function to have runtime in the table below:

+---------+-------------------+--------------------+-----------------------+
|         |   quick           | pre_checkin        | nightly               |
+=========+===================+====================+=======================+
|         |                   |                    |                       |
| Level 1 |   2 - 12 sec      |  20 - 36 sec       |   70 - 200 sec        |
|         |                   |                    |                       |
+---------+-------------------+--------------------+-----------------------+
|         |                   |                    |                       |
| Level 2 |   6 - 36 sec      |  35 - 100 sec      |   200 - 650 sec       |
|         |                   |                    |                       |
+---------+-------------------+--------------------+-----------------------+
|         |                   |                    |                       |
| Level 3 |   20 sec - 2 min  |  2 - 6 min         |   12 - 24 min         |
|         |                   |                    |                       |
+---------+-------------------+--------------------+-----------------------+


Many examples are available in ``gtest/*_gtest.{cpp,yaml}``

Testing During Development
--------------------------

ILP64 APIs require such large problem sizes that getting code coverage during tests is cost-prohibitive.
Therefore there are some hooks to help with early developer testing using smaller sizes.
You can compile with ``-DROCBLAS_DEV_TEST_ILP64`` to test ILP64 code when otherwise it would not be invoked.
For example, a ``scal`` implementation may call the original 32-bit API code when ``N`` and ``incx`` are less than ``c_ILP64_i32_max``.
``c_ILP64_i32_max`` is usually defined as ``std::numeric_limits<int32_t>::max()``,
but with ``ROCBLAS_DEV_TEST_ILP64`` defined then ``c_ILP64_i32_max`` is defined as zero.
Thus for small sizes it will branch and use ILP64 support code instead of using the 32-bit original API.
The specifics vary for each implementation and require yaml configuration to test C_64 APIs with small sizes.
It is intended as a by-pass for when early detection of small sizes invokes the 32-bit APIs.
This is for developer testing only. This should not be used for production code.

Test coverage during development should be much more exhaustive than final versions of test sets.
We limit our test times so a trade-off between coverage and test duration must be made.
During development it is expected problem space will be covered in more depth to look for potential anomalies.
Any special cases should be analyzed, reduced in scope, and represented in the final test category.

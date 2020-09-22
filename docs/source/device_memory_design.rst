
.. toctree::
   :maxdepth: 4
   :caption: Contents:

========================
Device Memory Allocation
========================

Requirements
============
- Some rocBLAS functions need temporary device memory.
- Allocating and deallocating device memory is expensive and synchronizing.
- Temporary device memory should be recycled across multiple rocBLAS function calls using the same rocblas_handle.
- The following schemes need to be supported:

  - **Default** Functions allocate required device memory automatically. This has the disadvantage that allocation is a synchronizing event.
  - **Preallocate** Query all the functions called using a rocblas_handle to find out how much device memory is needed. Preallocate the required device memory when the rocblas_handle is created, and there are no more synchronizing allocations or deallocations.
  - **Manual** Query a function to find out how much device memory is required. Allocate and deallocate the device memory before and after function calls. This allows the user to control where the synchronizing allocation and deallocation occur.

In all above schemes, temporary device memory needs to be held by the rocblas_handle and recycled if a subsequent function using the handle needs it.

Design
======

- rocBLAS uses per-handle device memory allocation with out-of-band management
- The state of the device memory is stored in the rocblas_handle
- For the user of rocBLAS:

  - Functions are provided to query how much device memory a function needs
  - An environment variable is provided to pre-allocate when the rocblas_handle is created
  - Functions are provided to manually allocate and deallocate after the rocblas_handle is created
  - The following two values are added to the rocblas_status enum to indicate how a rocBLAS function is changing the state of the temporary device memory in the rocBLAS handle

     - rocblas_status_size_unchanged
     - rocblas_status_size_increased

- For the rocBLAS developer:

  - Functions are provided to answer device memory size queries
  - Functions are provided to allocate temporary device memory
  - opaque RAII objects are used to hold the temorary device memory, and allocated memory is returned to the handle automatically when it is no longer needed.

The functions for the rocBLAS user are described in the User Guide. The functions for the rocBLAS developer are described below.


Answering Device Memory Size Queries In Function That Needs Memory
===================================================================

Example
-------

Functions should contain code like below to answer a query on how much temporary device memory is required. In this case ``m * n * sizeof(T)`` bytes of memory is required:

::

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
--------

::

    bool _rocblas_handle::is_device_memory_size_query() const

Indicates if the current function call is collecting information about the optimal device memory allocation size.

return value:

- **true** if information is being collected
- **false** if information is not being collected

Function
--------

::

    rocblas_status _rocblas_handle::set_optimal_device_memory_size(size...)

Sets the optimal size(s) of device memory buffer(s) in bytes for this function. The sizes are rounded up to the next multiple of 64 (or some other chunk size), and the running maximum is updated.

return value:

- **rocblas_status_size_unchanged** if he maximum optimal device memory size did not change, this is the case where the function does not use device memory
- **rocblas_satus_size_increased** if the maximum optimal device memory size increased
- **rocblas_status_internal_error** if this function is not suposed to be collecting size information

Function
--------

::

    size_t rocblas_sizeof_datatype(rocblas_datatype type)

Returns size of a rocBLAS runtime data type.


Answering Device Memory Size Queries In Function That Does Not Needs Memory
============================================================================

Example
-------

::

    rocblas_status rocblas_function(rocblas_handle handle, ...)
    {
        if(!handle) return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    //  rest of function
    }

Macro
-----

::

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)

A convenience macro that returns rocblas_status_size_unchanged if the function call is a memory size query.


rocBLAS Kernel Device Memory Allocation
=======================================

Example
-------

Device memory can be allocated for n floats using device_malloc as follows:

::

     float* mem = (float*)handle->device_malloc(n * sizeof(float));
     if (!mem) return rocblas_status_memory_error;

Example
-------

To allocate multiple buffers

::

    size_t size1 = m * n;
    size_t size2 = m * k;

    auto mem = handle->device_malloc(size1, size2);
    if (!mem) return rocblas_status_memory_error;

    void * buf1, * buf2;
    std::tie(buf1, buf2) = mem;


Function
--------

::

    auto mem = handle->device_malloc(size...)

- Returns an opaque RAII object lending allocated device memory to a particular rocBLAS function.
- The object returned is convertible to void * or other pointer types if only one size is specified
- The object can be assigned to std::tie(ptr1, ptr2, ...), if more than one size is specified
- The lifetime of the returned object is the lifetime of the borrowed device memory (RAII)
- To simplify and optimize the code, only one successful allocation object can be alive at a time
- If the handle's device memory is currently being managed by rocBLAS as in the default scheme, it is expanded in size as necessary
- If the user allocated (or pre-allocated) an explicit size of device memory, then that size is used as the limit, and no resizing or synchronization ever occurs

Parameters:

- **size** size in bytes of memory to be allocated

return value:

- **On success**, returns an opaque RAII object
- **On failure**, returns a null pointer


Performance degrade
===================
The rocblas_status enum value ``rocblas_status_perf_degraded`` is used to indicate that a slower algorithm was used because of insufficient device memory for the optimal algorithm.

Example
-------

::

    rocblas_status ret = rocblas_status_success;
    size_t size_for_optimal_algorithm = m + n + k;
    size_t size_for_degraded_algorithm = m;
    auto mem_optimal = handle->device_malloc(size_for_optimal_algorithm);
    if (mem1)
    {
        // Algorithm using larger optimal memory
    }
    else
    {
        auto mem_degraded = handle->device_malloc(size_for_degraded_algorithm);
        if (mem_degraded)
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


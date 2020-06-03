
.. toctree::
   :maxdepth: 4
   :caption: Contents:

========================
Device Memory Allocation
========================

The following computational functions use temporary device memory.

+------------------------------------+------------------------------------------+
|Function                            |use of temporary device memory            |
+====================================+==========================================+
|L1 reduction functions              |reduction array                           |
| - rocblas_Xdot                     |                                          |
| - rocblas_Xmax                     |                                          |
| - rocblas_Xmin                     |                                          |
| - rocblas_Xnrm2                    |                                          |
+------------------------------------+------------------------------------------+
|L3 gemm based functions             |block of matrix                           |
| - rocblas_Xtrsm                    |                                          |
| - rocblas_Xtrmm                    |                                          |
+------------------------------------+------------------------------------------+
|auxilliary                          |buffer to compress noncontiguous arrays   |
| - rocblas_set_vector               |                                          |
| - rocblas_get_vector               |                                          |
| - rocblas_set_matrix               |                                          |
| - rocblas_get_matrix               |                                          |
+------------------------------------+------------------------------------------+


For temporary device memory rocBLAS uses a per-handle memory allocation with out-of-band management. This allows for recycling temporary device memory across multiple computational functions that use the same handle. There are helper functions to get and set the number of bytes allocated. There are helper functions to measure how much memory will be required by a section of code. These functions allow for 3 schemes to be used:

#. **Default**: Computational functions allocate the memory they require. Note that any memory allocated persists in the handle, so it is available for later computational functions that use the handle. This has the disadvantage that allocation is a synchronizing event.
#. **Preallocate**:  Set an environment variable to preallocate required memory when handle is created, and thereafter there are no more synchronizing allocations or deallocations. This requires the use of helper functions to measure the memory use between the handle creation and destruction.
#. **Manual**: Manually allocate and deallocate memory throughout the program. The user will then be controlling where the synchronizing allocation and deallocation occur.


Environment Variable for Preallocating
======================================
The environment variable ROCBLAS_DEVICE_MEMORY_SIZE is used to set how much memory to preallocate:

- if > 0, sets the default handle device memory size to the specified size (in bytes)
- if == 0 or unset, lets rocBLAS manage device memory, using a default size (like 1MB), and expanding it when necessary

Functions for manually allocating
=================================
The following helper functions can be used to manually allocate and deallocate. See the API section for information on the functions.

- rocblas_set_device_memory_size
- rocblas_get_device_memory_size
- rocblas_is_managing_device_memory
- rocblas_start_device_memory_size_query
- rocblas_stop_device_memory_size_query

rocBLAS Function Return Values for insufficient device memory
=============================================================
If the user preallocates or manually allocates, then that size is used as the limit, and no resizing or synchronizing ever occurs. The following two function return values indicate insufficient memory:

- rocblas_status == rocblas_status_memory_error: indicates there is not sufficient device memory for a rocBLAS function
- rocblas_status == rocblas_status_perf_degraded: indicates that a slower algorthm was used because of insufficient device memory for the optimal algorithm

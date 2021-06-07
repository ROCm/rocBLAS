========================
Device Memory Allocation
========================

The following computational functions use temporary device memory.

+------------------------------------+------------------------------------------------+
|Function                            |use of temporary device memory                  |
+====================================+================================================+
|L1 reduction functions              |reduction array                                 |
| - rocblas_Xdot                     |                                                |
| - rocblas_Xmax                     |                                                |
| - rocblas_Xmin                     |                                                |
| - rocblas_Xnrm2                    |                                                |
| - rocblas_dot_ex                   |                                                |
| - rocblas_nrm2_ex                  |                                                |
+------------------------------------+------------------------------------------------+
|L2 functions                        |result array before overwriting input           |
| - rocblas_Xtbmv                    |                                                |
| - rocblas_Xtpmv                    |                                                |
| - rocblas_Xtrmv                    |                                                |
| - rocblas_Xtrsv                    |                                                |
| - rocblas_Xgemv (optional)         |column reductions of skinny transposed matrices |
+------------------------------------+------------------------------------------------+
|L3 gemm based functions             |block of matrix                                 |
| - rocblas_Xtrsm                    |                                                |
| - rocblas_Xgemm                    |                                                |
| - rocblas_Xtrtri                   |                                                |
+------------------------------------+------------------------------------------------+
|auxiliary                           |buffer to compress noncontiguous arrays         |
| - rocblas_set_vector               |                                                |
| - rocblas_get_vector               |                                                |
| - rocblas_set_matrix               |                                                |
| - rocblas_get_matrix               |                                                |
+------------------------------------+------------------------------------------------+


For temporary device memory rocBLAS uses a per-handle memory allocation with out-of-band management. The temporary device memory is stored in the handle. This allows for recycling temporary device memory across multiple computational kernels that use the same handle. Each handle has a single stream, and kernels execute in order in the stream, with each kernel completing before the next kernel in the stream starts. There are 4 schemes for temporary device memory:

#. **rocBLAS_managed**: This is the default scheme. If there is not enough memory in the handle, computational functions allocate the memory they require. Note that any memory allocated persists in the handle, so it is available for later computational functions that use the handle.
#. **user_managed, preallocate**: An environment variable is set before the rocBLAS handle is created and thereafter there are no more allocations or deallocations.
#. **user_managed, manual**:  The user calls helper functions to get or set memory size throughout the program, thereby controlling when allocation and deallocation occur.
#. **user_owned**:  User allocates workspace and calls a helper function to allow rocBLAS to access the workspace.

The default scheme has the disadvantage that allocation is synchronizing, so if there is not enough memory in the handle, a synchronizing deallocation and allocation occurs.

Environment Variable for Preallocating
======================================
The environment variable ROCBLAS_DEVICE_MEMORY_SIZE is used to set how much memory to preallocate:

- if > 0, sets the default handle device memory size to the specified size (in bytes)
- if == 0 or unset, lets rocBLAS manage device memory, using a default size (like 32MB), and expanding it when necessary

Functions for manually setting memory size
==========================================

- rocblas_set_device_memory_size
- rocblas_get_device_memory_size
- rocblas_is_user_managing_device_memory

Function for setting user owned workspace
=========================================

- rocblas_set_workspace

Functions for finding how much memory is required
=================================================

- rocblas_start_device_memory_size_query
- rocblas_stop_device_memory_size_query
- rocblas_is_managing_device_memory

See the API section for information on the above functions.

rocBLAS Function Return Values for insufficient device memory
=============================================================
If the user preallocates or manually allocates, then that size is used as the limit, and no resizing or synchronizing ever occurs. The following two function return values indicate insufficient memory:

- rocblas_status == rocblas_status_memory_error: indicates there is not sufficient device memory for a rocBLAS function
- rocblas_status == rocblas_status_perf_degraded: indicates that a slower algorthm was used because of insufficient device memory for the optimal algorithm

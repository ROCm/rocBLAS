
.. toctree::
   :maxdepth: 4
   :caption: Contents:

Device Memory Allocation
------------------------

The following rocBLAS kernels use temporary device memory.

+------------------------------------+------------------------------------------+
|Kernel                              |use of temporary device memory            |
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


rocBLAS uses a per-handle device memory allocation with
out-of-band management. This allows for recycling temporary device memory across multiple rocBLAS kernel
calls within the same stream (handle). There are client functions to get and set the number of bytes allocated
in the handle. There are client functions to measure how much memory will be required by a section of code.
These functions allow for 3 schemes to be used:

#. The default scheme: Kernels allocate the memory they require. Note that any memory allocate persists in the handle,
   so it is available for later functions that use the handle. This has the disadvantage that allocation is a synchronizing event.
#. Preallocate required memory when handle is created, and thereafter there are no more synchronizing allocations or deallocations.
   This requires the use of client functions to measure the memory use between the handle creation and destruction.
#. Manually allocate and deallocate memory throughout the program. The user will then be controlling where the
   synchronizing allocation and deallocation occur.


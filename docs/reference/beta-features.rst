.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _beta-features:

********************************************************************
rocBLAS Beta Features
********************************************************************

To allow for future growth and changes, the features in this section are not subject to the same
level of backwards compatibility and support as the normal rocBLAS API. These features are subject
to change and/or removal in future release of rocBLAS.

To use the following beta API features ``ROCBLAS_BETA_FEATURES_API`` must be defined before including ``rocblas.h``.

rocblas_gemm_ex_get_solutions + batched, strided_batched
=========================================================

.. doxygenfunction:: rocblas_gemm_ex_get_solutions
.. doxygenfunction:: rocblas_gemm_ex_get_solutions_by_type
.. doxygenfunction:: rocblas_gemm_batched_ex_get_solutions
.. doxygenfunction:: rocblas_gemm_batched_ex_get_solutions_by_type
.. doxygenfunction:: rocblas_gemm_strided_batched_ex_get_solutions

rocblas_gemm_ex3 + batched, strided_batched
=========================================================

.. doxygenfunction:: rocblas_gemm_ex3
.. doxygenfunction:: rocblas_gemm_batched_ex3
.. doxygenfunction:: rocblas_gemm_strided_batched_ex3

Graph Support for rocBLAS
=========================================================

Most of the rocBLAS functions can be captured into a graph node via Graph Management HIP APIs, except those listed in :ref:`Functions Unsupported with Graph Capture`.
For a list of graph related HIP APIs, refer to `Graph Management HIP API <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___graph.html#graph-management>`_.

.. code-block:: c++

      CHECK_HIP_ERROR((hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
      rocblas_<function>(<arguments>);
      CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));

The above code will create a graph with ``rocblas_function()`` as graph node. The captured graph can be launched as shown below:

.. code-block:: c++

      CHECK_HIP_ERROR(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));
      CHECK_HIP_ERROR(hipGraphLaunch(instance, stream));


Graph support requires Asynchronous HIP APIs, hence, users must enable stream-order memory allocation. For more details refer to section :ref:`stream order alloc`.

During stream capture, rocBLAS stores the allocated host and device memory in the handle and the allocated memory will be freed when the handle is destroyed.

.. _Functions Unsupported with Graph Capture:

Functions Unsupported with Graph Capture
=========================================================

- The following Level-1 functions place results into host buffers (in pointer mode host) which enforces synchronization.

      - `dot`
      - `asum`
      - `nrm2`
      - `imax`
      - `imin`

- BLAS Level-3 and BLAS-EX functions in pointer mode device do not support HIP Graph. Support will be added in future releases.

HIP Graph Known Issues in rocBLAS
=========================================================
- On Windows platform, batched functions (Level-1, Level-2 and Level-3) produce incorrect results.


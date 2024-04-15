.. meta::
  :description: rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation

.. _helper-functions:

********************************************************************
rocBLAS Helper Functions
********************************************************************

Auxiliary Functions
===================

.. doxygenfunction:: rocblas_create_handle
.. doxygenfunction:: rocblas_destroy_handle
.. doxygenfunction:: rocblas_set_stream
.. doxygenfunction:: rocblas_get_stream
.. doxygenfunction:: rocblas_set_pointer_mode
.. doxygenfunction:: rocblas_get_pointer_mode
.. doxygenfunction:: rocblas_set_atomics_mode
.. doxygenfunction:: rocblas_get_atomics_mode
.. doxygenfunction:: rocblas_pointer_to_mode
.. doxygenfunction:: rocblas_set_vector
.. doxygenfunction:: rocblas_get_vector
.. doxygenfunction:: rocblas_set_matrix
.. doxygenfunction:: rocblas_get_matrix
.. doxygenfunction:: rocblas_set_vector_async
.. doxygenfunction:: rocblas_set_matrix_async
.. doxygenfunction:: rocblas_get_matrix_async
.. doxygenfunction:: rocblas_initialize
.. doxygenfunction:: rocblas_status_to_string

Device Memory Allocation Functions
==================================

.. doxygenfunction:: rocblas_start_device_memory_size_query
.. doxygenfunction:: rocblas_stop_device_memory_size_query
.. doxygenfunction:: rocblas_get_device_memory_size
.. doxygenfunction:: rocblas_set_device_memory_size
.. doxygenfunction:: rocblas_set_workspace
.. doxygenfunction:: rocblas_is_managing_device_memory
.. doxygenfunction:: rocblas_is_user_managing_device_memory

For more detailed information, refer to sections :ref:`Device Memory Allocation Usage` and :ref:`Device Memory allocation in detail`.

Build Information Functions
===========================

.. doxygenfunction:: rocblas_get_version_string_size
.. doxygenfunction:: rocblas_get_version_string


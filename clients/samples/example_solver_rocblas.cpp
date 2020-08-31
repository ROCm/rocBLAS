/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include "rocblas_device_malloc.hpp"
#include "utility.hpp"

int main()
{
    rocblas_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_create_handle(&handle));

    {
        size_t size1 = 512, size2 = 1024, size3;

        if(rocblas_is_device_memory_size_query(handle))
        {
            rocblas_cerr << "rocblas_is_device_memory_size_query() incorrectly returned true"
                         << std::endl;
            return 1;
        }

        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));

        if(!rocblas_is_device_memory_size_query(handle))
        {
            rocblas_cerr << "rocblas_is_device_memory_size_query() incorrectly returned false"
                         << std::endl;
            return 1;
        }

        CHECK_ALLOC_QUERY(rocblas_set_optimal_device_memory_size(handle, size1));
        CHECK_ALLOC_QUERY(rocblas_set_optimal_device_memory_size(handle, size2));
        CHECK_ALLOC_QUERY(rocblas_set_optimal_device_memory_size(handle, size1));

        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size3));

        if(size2 != size3)
        {
            rocblas_cerr << "size returned in query differs from expected" << std::endl;
            return 1;
        }

        rocblas_device_malloc mem(handle, 1024);

        if(!mem)
        {
            rocblas_cerr << "rocblas_device_malloc failed" << std::endl;
            return 1;
        }

        void* ptr = static_cast<void*>(mem);
        if(!ptr)
        {
            rocblas_cerr << "nullptr returned from rocblas_device_malloc" << std::endl;
            return 1;
        }

        rocblas_device_malloc_base& ref = mem;

        rocblas_device_malloc mem2 = rocblas_device_malloc(handle, 0);

        rocblas_device_malloc_base& ref2 = mem2;

        rocblas_device_malloc mem3 = std::move(mem);
    }
    // We must destroy all rocblas_device_malloc() objects before the handle is destroyed

    CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(handle));

    rocblas_cout << "All tests passed" << std::endl;
}

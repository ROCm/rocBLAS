/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_nrm2_ex.hpp"
#include "../blas1/rocblas_reduction_impl.hpp"

namespace
{
    // allocate workspace inside this API
    template <rocblas_int NB>
    rocblas_status rocblas_nrm2_ex_impl(rocblas_handle   handle,
                                        rocblas_int      n,
                                        const void*      x,
                                        rocblas_datatype x_type,
                                        rocblas_int      incx,
                                        void*            results,
                                        rocblas_datatype result_type,
                                        rocblas_datatype execution_type)
    {
        if(!handle)
        {
            return rocblas_status_invalid_handle;
        }

        size_t dev_bytes = rocblas_reduction_kernel_workspace_size<NB>(n, 1, execution_type);

        if(handle->is_device_memory_size_query())
        {
            if(n <= 0 || incx <= 0)
            {
                return rocblas_status_size_unchanged;
            }
            else
            {
                return handle->set_optimal_device_memory_size(dev_bytes);
            }
        }

        auto x_type_str      = rocblas_datatype_string(x_type);
        auto result_type_str = rocblas_datatype_string(result_type);
        auto ex_type_str     = rocblas_datatype_string(execution_type);
        auto layer_mode      = handle->layer_mode;
        if(layer_mode & rocblas_layer_mode_log_trace)
        {
            log_trace(handle, "nrm2_ex", n, x, x_type_str, incx, result_type_str, ex_type_str);
        }

        if(layer_mode & rocblas_layer_mode_log_bench)
        {
            log_bench(handle,
                      "./rocblas-bench",
                      "-f",
                      "nrm2_ex",
                      "-n",
                      n,
                      "--incx",
                      incx,
                      log_bench_ex_precisions(x_type, result_type, execution_type));
        }

        if(layer_mode & rocblas_layer_mode_log_profile)
        {
            log_profile(handle,
                        "nrm2_ex",
                        "N",
                        n,
                        "a_type",
                        x_type_str,
                        "incx",
                        incx,
                        "b_type",
                        result_type_str,
                        "compute_type",
                        ex_type_str);
        }

        if(!results)
        {
            return rocblas_status_invalid_pointer;
        }

        // Quick return if possible.
        if(n <= 0 || incx <= 0)
        {
            if(rocblas_pointer_mode_device == handle->pointer_mode)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    results, 0, rocblas_sizeof_datatype(result_type), handle->get_stream()));
            }
            else
            {
                memset(results, 0, rocblas_sizeof_datatype(result_type));
            }
            return rocblas_status_success;
        }

        if(!x)
        {
            return rocblas_status_invalid_pointer;
        }

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
        {
            return rocblas_status_memory_error;
        }

        static constexpr bool           isbatched     = false;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr rocblas_int    shiftx_0      = 0;

        return rocblas_nrm2_ex_template<NB, isbatched>(handle,
                                                       n,
                                                       x,
                                                       x_type,
                                                       shiftx_0,
                                                       incx,
                                                       stridex_0,
                                                       batch_count_1,
                                                       results,
                                                       result_type,
                                                       execution_type,
                                                       (void*)w_mem);
    }

} // namespace

/* ============================================================================================ */

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

rocblas_status rocblas_nrm2_ex(rocblas_handle   handle,
                               rocblas_int      n,
                               const void*      x,
                               rocblas_datatype x_type,
                               rocblas_int      incx,
                               void*            results,
                               rocblas_datatype result_type,
                               rocblas_datatype execution_type)
{
    try
    {
        constexpr rocblas_int NB = 512;
        return rocblas_nrm2_ex_impl<NB>(
            handle, n, x, x_type, incx, results, result_type, execution_type);
    }
    catch(...)
    {
        return exception_to_rocblas_status();
    }
}

} // extern "C"

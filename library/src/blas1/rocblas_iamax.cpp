/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_iamax.hpp"
#include "check_numerics_vector.hpp"
#include "rocblas_reduction_impl.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_iamax_name[] = "unknown";
    template <>
    constexpr char rocblas_iamax_name<float>[] = "rocblas_isamax";
    template <>
    constexpr char rocblas_iamax_name<double>[] = "rocblas_idamax";
    template <>
    constexpr char rocblas_iamax_name<rocblas_float_complex>[] = "rocblas_icamax";
    template <>
    constexpr char rocblas_iamax_name<rocblas_double_complex>[] = "rocblas_izamax";

    // allocate workspace inside this API
    template <typename S, typename T>
    rocblas_status rocblas_iamax_impl(
        rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result)
    {
        static constexpr bool           isbatched     = false;
        static constexpr rocblas_stride shiftx_0      = 0;
        static constexpr rocblas_stride stridex_0     = 0;
        static constexpr rocblas_int    batch_count_1 = 1;
        static constexpr int            NB            = 1024;

        size_t         dev_bytes = 0;
        rocblas_status checks_status
            = rocblas_reduction_setup<NB, isbatched, rocblas_index_value_t<S>>(
                handle,
                n,
                x,
                incx,
                stridex_0,
                batch_count_1,
                result,
                rocblas_iamax_name<T>,
                "iamax",
                dev_bytes);
        if(checks_status != rocblas_status_continue)
        {
            return checks_status;
        }

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status check_numerics_status
                = rocblas_internal_check_numerics_vector_template(rocblas_iamax_name<T>,
                                                                  handle,
                                                                  n,
                                                                  x,
                                                                  0,
                                                                  incx,
                                                                  stridex_0,
                                                                  batch_count_1,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
        {
            return rocblas_status_memory_error;
        }
        rocblas_status status
            = rocblas_internal_iamax_template<NB, isbatched>(handle,
                                                             n,
                                                             x,
                                                             shiftx_0,
                                                             incx,
                                                             stridex_0,
                                                             batch_count_1,
                                                             result,
                                                             (rocblas_index_value_t<S>*)w_mem);
        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status check_numerics_status
                = rocblas_internal_check_numerics_vector_template(rocblas_iamax_name<T>,
                                                                  handle,
                                                                  n,
                                                                  x,
                                                                  0,
                                                                  incx,
                                                                  stridex_0,
                                                                  batch_count_1,
                                                                  check_numerics,
                                                                  is_input);
            if(check_numerics_status != rocblas_status_success)
                return check_numerics_status;
        }

        return status;
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

#ifdef IMPL
#error IMPL IS ALREADY DEFINED
#endif

#define IMPL(name_, typei_, typew_)                                     \
    rocblas_status name_(rocblas_handle handle,                         \
                         rocblas_int    n,                              \
                         const typei_*  x,                              \
                         rocblas_int    incx,                           \
                         rocblas_int*   results)                        \
    try                                                                 \
    {                                                                   \
        return rocblas_iamax_impl<typew_>(handle, n, x, incx, results); \
    }                                                                   \
    catch(...)                                                          \
    {                                                                   \
        return exception_to_rocblas_status();                           \
    }

IMPL(rocblas_isamax, float, float);
IMPL(rocblas_idamax, double, double);
IMPL(rocblas_icamax, rocblas_float_complex, float);
IMPL(rocblas_izamax, rocblas_double_complex, double);

#undef IMPL

} // extern "C"

/* ************************************************************************
 * Copyright 2018-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_reduction_impl.hpp"

#ifndef MAX_MIN
#define MAX_MIN max
#endif

#ifndef AMAX_AMIN_REDUCTION
#define AMAX_AMIN_REDUCTION rocblas_reduce_amax
#endif

#define QUOTE2(S) #S
#define QUOTE(S) QUOTE2(S)

#define JOIN2(A, B) A##B
#define JOIN(A, B) JOIN2(A, B)

// pair of value and index
template <typename T>
struct index_value_t
{
    // important: index must come first, so that index_value_t* can be cast to rocblas_int*
    rocblas_int index;
    T           value;
};

// Specialization of default_value for index_value_t<T>
template <typename T>
struct rocblas_default_value<index_value_t<T>>
{
    __forceinline__ __host__ __device__ constexpr auto operator()() const
    {
        index_value_t<T> x;
        x.index = -1;
        return x;
    }
};

// Fetch absolute value
template <typename To>
struct rocblas_fetch_amax_amin
{
    template <typename Ti>
    __forceinline__ __host__ __device__ index_value_t<To> operator()(Ti x, rocblas_int index)
    {
        return {index, fetch_asum(x)};
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amax
{
    template <typename To>
    __forceinline__ __host__ __device__ void operator()(index_value_t<To>& __restrict__ x,
                                                        const index_value_t<To>& __restrict__ y)
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value > x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

// Replaces x with y if y.value < x.value or y.value == x.value and y.index < x.index
struct rocblas_reduce_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ void operator()(index_value_t<To>& __restrict__ x,
                                                        const index_value_t<To>& __restrict__ y)
    {
        // If y.index == -1 then y.value is invalid and should not be compared
        if(y.index != -1)
        {
            if(x.index == -1 || y.value < x.value)
                x = y; // if larger or smaller, update max/min and index
            else if(y.index < x.index && x.value == y.value)
                x.index = y.index; // if equal, choose smaller index
        }
    }
};

struct rocblas_finalize_amax_amin
{
    template <typename To>
    __forceinline__ __host__ __device__ auto operator()(const index_value_t<To>& x)
    {
        return x.index + 1;
    }
};

template <typename>
static constexpr char rocblas_iamaxmin_name[] = "unknown";
template <>
static constexpr char rocblas_iamaxmin_name<float>[] = "rocblas_isa" QUOTE(MAX_MIN);
template <>
static constexpr char rocblas_iamaxmin_name<double>[] = "rocblas_ida" QUOTE(MAX_MIN);
template <>
static constexpr char rocblas_iamaxmin_name<rocblas_float_complex>[] = "rocblas_ica" QUOTE(MAX_MIN);
template <>
static constexpr char rocblas_iamaxmin_name<rocblas_double_complex>[]
    = "rocblas_iza" QUOTE(MAX_MIN);

// allocate workspace inside this API
template <typename To, typename Ti>
static rocblas_status rocblas_iamaxmin(
    rocblas_handle handle, rocblas_int n, const Ti* x, rocblas_int incx, rocblas_int* result)
{

    static constexpr bool           isbatched     = false;
    static constexpr rocblas_stride stridex_0     = 0;
    static constexpr rocblas_int    batch_count_1 = 1;
    static constexpr int            NB            = 1024;

    return rocblas_reduction_impl<NB,
                                  isbatched,
                                  rocblas_fetch_amax_amin<To>,
                                  AMAX_AMIN_REDUCTION,
                                  rocblas_finalize_amax_amin,
                                  index_value_t<To>>(handle,
                                                     n,
                                                     x,
                                                     incx,
                                                     stridex_0,
                                                     batch_count_1,
                                                     result,
                                                     rocblas_iamaxmin_name<Ti>,
                                                     "ia" QUOTE(MAX_MIN));
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

#define IMPL(name_, typei_, typew_)                                   \
    rocblas_status name_(rocblas_handle handle,                       \
                         rocblas_int    n,                            \
                         const typei_*  x,                            \
                         rocblas_int    incx,                         \
                         rocblas_int*   results)                      \
    {                                                                 \
        return rocblas_iamaxmin<typew_>(handle, n, x, incx, results); \
    }

IMPL(JOIN(rocblas_isa, MAX_MIN), float, float);
IMPL(JOIN(rocblas_ida, MAX_MIN), double, double);
IMPL(JOIN(rocblas_ica, MAX_MIN), rocblas_float_complex, float);
IMPL(JOIN(rocblas_iza, MAX_MIN), rocblas_double_complex, double);

#undef IMPL

} // extern "C"

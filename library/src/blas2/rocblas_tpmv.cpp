/* ************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocblas_tpmv.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "utility.hpp"

namespace
{
    template <typename>
    constexpr char rocblas_tpmv_name[] = "unknown";
    template <>
    constexpr char rocblas_tpmv_name<float>[] = "rocblas_stpmv";
    template <>
    constexpr char rocblas_tpmv_name<double>[] = "rocblas_dtpmv";
    template <>
    constexpr char rocblas_tpmv_name<rocblas_float_complex>[] = "rocblas_ctpmv";
    template <>
    constexpr char rocblas_tpmv_name<rocblas_double_complex>[] = "rocblas_ztpmv";

    template <typename T>
    rocblas_status rocblas_tpmv_impl(rocblas_handle    handle,
                                     rocblas_fill      uplo,
                                     rocblas_operation transA,
                                     rocblas_diagonal  diag,
                                     rocblas_int       m,
                                     const T*          A,
                                     T*                x,
                                     rocblas_int       incx)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        if(!handle->is_device_memory_size_query())
        {
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto uplo_letter   = rocblas_fill_letter(uplo);
                auto transA_letter = rocblas_transpose_letter(transA);
                auto diag_letter   = rocblas_diag_letter(diag);
                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    log_trace(handle, rocblas_tpmv_name<T>, uplo, transA, diag, m, A, x, incx);
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    log_bench(handle,
                              "./rocblas-bench",
                              "-f",
                              "tpmv",
                              "-r",
                              rocblas_precision_string<T>,
                              "--uplo",
                              uplo_letter,
                              "--transposeA",
                              transA_letter,
                              "--diag",
                              diag_letter,
                              "-m",
                              m,
                              "--incx",
                              incx);
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    log_profile(handle,
                                rocblas_tpmv_name<T>,
                                "uplo",
                                uplo_letter,
                                "transA",
                                transA_letter,
                                "diag",
                                diag_letter,
                                "M",
                                m,
                                "incx",
                                incx);
                }
            }
        }

        size_t         dev_bytes;
        rocblas_status arg_status
            = rocblas_tpmv_arg_check<T>(handle, uplo, transA, diag, m, A, x, incx, 1, dev_bytes);
        if(arg_status != rocblas_status_continue)
            return arg_status;

        auto w_mem = handle->device_malloc(dev_bytes);
        if(!w_mem)
            return rocblas_status_memory_error;

        auto check_numerics = handle->check_numerics;
        if(check_numerics)
        {
            bool           is_input = true;
            rocblas_status tpmv_check_numerics_status
                = rocblas_tpmv_check_numerics(rocblas_tpmv_name<T>,
                                              handle,
                                              m,
                                              A,
                                              0,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(tpmv_check_numerics_status != rocblas_status_success)
                return tpmv_check_numerics_status;
        }

        static constexpr rocblas_int    NB          = ROCBLAS_TPMV_NB;
        static constexpr rocblas_int    batch_count = 1;
        static constexpr rocblas_stride offset_a    = 0;
        static constexpr rocblas_stride offset_x    = 0;
        static constexpr rocblas_stride stride_x    = 0;
        static constexpr rocblas_stride stride_a    = 0;
        static constexpr rocblas_stride stride_w    = 0;

        rocblas_status status = rocblas_tpmv_template<NB>(handle,
                                                          uplo,
                                                          transA,
                                                          diag,
                                                          m,
                                                          A,
                                                          offset_a,
                                                          stride_a,
                                                          x,
                                                          offset_x,
                                                          incx,
                                                          stride_x,
                                                          (T*)w_mem,
                                                          stride_w,
                                                          batch_count);

        if(status != rocblas_status_success)
            return status;

        if(check_numerics)
        {
            bool           is_input = false;
            rocblas_status tpmv_check_numerics_status
                = rocblas_tpmv_check_numerics(rocblas_tpmv_name<T>,
                                              handle,
                                              m,
                                              A,
                                              0,
                                              0,
                                              x,
                                              0,
                                              incx,
                                              0,
                                              1,
                                              check_numerics,
                                              is_input);
            if(tpmv_check_numerics_status != rocblas_status_success)
                return tpmv_check_numerics_status;
        }
        return status;
    }

} // namespace

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" {

#ifdef IMPL
#error IMPL ALREADY DEFINED
#endif

#define IMPL(routine_name_, T_)                                              \
    rocblas_status routine_name_(rocblas_handle    handle,                   \
                                 rocblas_fill      uplo,                     \
                                 rocblas_operation transA,                   \
                                 rocblas_diagonal  diag,                     \
                                 rocblas_int       m,                        \
                                 const T_*         A,                        \
                                 T_*               x,                        \
                                 rocblas_int       incx)                     \
    try                                                                      \
    {                                                                        \
        return rocblas_tpmv_impl(handle, uplo, transA, diag, m, A, x, incx); \
    }                                                                        \
    catch(...)                                                               \
    {                                                                        \
        return exception_to_rocblas_status();                                \
    }

IMPL(rocblas_stpmv, float);
IMPL(rocblas_dtpmv, double);
IMPL(rocblas_ctpmv, rocblas_float_complex);
IMPL(rocblas_ztpmv, rocblas_double_complex);

#undef IMPL

} // extern "C"

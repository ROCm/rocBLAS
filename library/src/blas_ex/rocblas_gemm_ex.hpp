/* ************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#ifdef BUILD_WITH_TENSILE
#include "../blas3/Tensile/gemm_tensile.hpp"
#endif

#include "gemm.hpp"
#include "handle.hpp"
#include "logging.hpp"

/////////////////
// Device Side //
/////////////////
template <typename To>
rocblas_status device_strided_batched_matrix_copy(rocblas_handle handle,
                                                  const To*      src,
                                                  rocblas_stride ld_src,
                                                  rocblas_stride stride_src,
                                                  To*            dst,
                                                  rocblas_stride ld_dst,
                                                  rocblas_stride stride_dst,
                                                  rocblas_int    n1,
                                                  rocblas_int    n2,
                                                  rocblas_int    batch_count)
{
    if(rocblas_internal_tensile_debug_skip_launch())
        return rocblas_status_success;

    if(src == dst && ld_src == ld_dst && stride_src == stride_dst)
        return rocblas_status_success; // no copy if src matrix == dst matrix

    if(n1 == ld_src && n1 == ld_dst && stride_src == n2 * ld_src && stride_dst == n2 * ld_dst)
    {
        // src and dst batch matrices are contiguous, use single copy
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(dst,
                                           src,
                                           sizeof(To) * n1 * n2 * batch_count,
                                           hipMemcpyDeviceToDevice,
                                           handle->get_stream()));
    }
    else if(n1 == ld_src && n1 == ld_dst)
    {
        // individual matrices in batch matrix are contiguous, one copy for each matrix
        for(size_t i3 = 0; i3 < batch_count; i3++)
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(dst + i3 * stride_dst,
                                               src + i3 * stride_src,
                                               sizeof(To) * n1 * n2,
                                               hipMemcpyDeviceToDevice,
                                               handle->get_stream()));
    }
    else
    {
        // individual matrices not contiguous, one copy for each contiguous column
        for(int i3 = 0; i3 < batch_count; i3++)
            for(int i2 = 0; i2 < n2; i2++)
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(dst + i2 * ld_dst + i3 * stride_dst,
                                                   src + i2 * ld_src + i3 * stride_src,
                                                   sizeof(To) * n1,
                                                   hipMemcpyDeviceToDevice,
                                                   handle->get_stream()));
    }
    return rocblas_status_success;
}

//------------------------------------------------------------------------------

///////////////
// Host Side //
///////////////
template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_batched_template(rocblas_handle     handle,
                                        rocblas_operation  trans_a,
                                        rocblas_operation  trans_b,
                                        rocblas_int        m,
                                        rocblas_int        n,
                                        rocblas_int        k,
                                        const Tc*          alpha,
                                        const Ti*          a[],
                                        rocblas_int        offset_a,
                                        rocblas_int        lda,
                                        rocblas_stride     stride_a,
                                        const Ti*          b[],
                                        rocblas_int        offset_b,
                                        rocblas_int        ldb,
                                        rocblas_stride     stride_b,
                                        const Tc*          beta,
                                        const To*          c[],
                                        rocblas_int        offset_c,
                                        rocblas_int        ldc,
                                        rocblas_stride     stride_c,
                                        To*                d[],
                                        rocblas_int        offset_d,
                                        rocblas_int        ldd,
                                        rocblas_stride     stride_d,
                                        rocblas_int        batch_count,
                                        rocblas_gemm_flags flags)
{
    RocblasContractionProblem<Ti, To, Tc> problem{
        handle,   trans_a, trans_b,  m,        n,           k,        alpha,    nullptr,
        a,        lda,     stride_a, offset_a, nullptr,     b,        ldb,      stride_b,
        offset_b, beta,    nullptr,  c,        ldc,         stride_c, offset_c, nullptr,
        d,        ldd,     stride_d, offset_d, batch_count, false,    flags};

    return runContractionProblem(problem);
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_batched_template(rocblas_handle     handle,
                                        rocblas_operation  trans_a,
                                        rocblas_operation  trans_b,
                                        rocblas_int        m,
                                        rocblas_int        n,
                                        rocblas_int        k,
                                        const Tc*          alpha,
                                        const Ti*          a,
                                        rocblas_int        offset_a,
                                        rocblas_int        lda,
                                        rocblas_stride     stride_a,
                                        const Ti*          b,
                                        rocblas_int        offset_b,
                                        rocblas_int        ldb,
                                        rocblas_stride     stride_b,
                                        const Tc*          beta,
                                        const To*          c,
                                        rocblas_int        offset_c,
                                        rocblas_int        ldc,
                                        rocblas_stride     stride_c,
                                        To*                d,
                                        rocblas_int        offset_d,
                                        rocblas_int        ldd,
                                        rocblas_stride     stride_d,
                                        rocblas_int        batch_count,
                                        rocblas_gemm_flags flags)
{
    RocblasContractionProblem<Ti, To, Tc> problem{
        handle,   trans_a, trans_b,  m,        n,           k,        alpha,    a,
        nullptr,  lda,     stride_a, offset_a, b,           nullptr,  ldb,      stride_b,
        offset_b, beta,    c,        nullptr,  ldc,         stride_c, offset_c, d,
        nullptr,  ldd,     stride_d, offset_d, batch_count, true,     flags};

    return runContractionProblem(problem);
}

template <bool BATCHED, typename Ti, typename To = Ti, typename Tc = To>
rocblas_status gemm_ex_typecasting(rocblas_handle     handle,
                                   rocblas_operation  trans_a,
                                   rocblas_operation  trans_b,
                                   rocblas_int        m,
                                   rocblas_int        n,
                                   rocblas_int        k,
                                   const void*        alpha,
                                   const void*        a,
                                   rocblas_int        offsetAin,
                                   rocblas_int        lda,
                                   rocblas_stride     stride_a,
                                   const void*        b,
                                   rocblas_int        offsetBin,
                                   rocblas_int        ldb,
                                   rocblas_stride     stride_b,
                                   const void*        beta,
                                   const void*        c,
                                   rocblas_int        offsetCin,
                                   rocblas_int        ldc,
                                   rocblas_stride     stride_c,
                                   void*              d,
                                   rocblas_int        offsetDin,
                                   rocblas_int        ldd,
                                   rocblas_stride     stride_d,
                                   rocblas_int        batch_count,
                                   rocblas_gemm_flags flags)
{
    Tc alpha_h, beta_h;
    RETURN_IF_ROCBLAS_ERROR(
        copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h, beta_h, k));

    // check alignment of pointers before casting
    if(BATCHED)
    {
        if(!isAligned(a, sizeof(Ti*)) || !isAligned(b, sizeof(Ti*)) || !isAligned(c, sizeof(To*))
           || !isAligned(d, sizeof(To*)))
            return rocblas_status_invalid_size;

        // Pass alpha and beta as simple array (stride of 1)
        // since Tensile does not have gemm_batched, we will have to iterate
        // over batches either way
        return gemm_ex_batched_template(handle,
                                        trans_a,
                                        trans_b,
                                        m,
                                        n,
                                        k,
                                        (const Tc*)alpha,
                                        (const Ti**)a,
                                        offsetAin,
                                        lda,
                                        stride_a,
                                        (const Ti**)b,
                                        offsetBin,
                                        ldb,
                                        stride_b,
                                        (const Tc*)beta,
                                        (const To**)c,
                                        offsetCin,
                                        ldc,
                                        stride_c,
                                        (To**)d,
                                        offsetDin,
                                        ldd,
                                        stride_d,
                                        batch_count,
                                        flags);
    }
    else
    {
        if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
           || !isAligned(d, sizeof(To)))
            return rocblas_status_invalid_size;

        return gemm_ex_batched_template(handle,
                                        trans_a,
                                        trans_b,
                                        m,
                                        n,
                                        k,
                                        (const Tc*)alpha,
                                        (const Ti*)a,
                                        offsetAin,
                                        lda,
                                        stride_a,
                                        (const Ti*)b,
                                        offsetBin,
                                        ldb,
                                        stride_b,
                                        (const Tc*)beta,
                                        (const To*)c,
                                        offsetCin,
                                        ldc,
                                        stride_c,
                                        (To*)d,
                                        offsetDin,
                                        ldd,
                                        stride_d,
                                        batch_count,
                                        flags);
    }
}

template <typename T>
inline rocblas_status validateArgs(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const void*       a,
                                   rocblas_int       ld_a,
                                   const void*       b,
                                   rocblas_int       ld_b,
                                   const T*          beta,
                                   const void*       c,
                                   rocblas_int       ld_c,
                                   const void*       d,
                                   rocblas_int       ld_d,
                                   rocblas_datatype  compute_type,
                                   rocblas_int       batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // leading dimensions must be valid
    if(ld_c < m || ld_d < m || ld_a < (trans_a == rocblas_operation_none ? m : k)
       || ld_b < (trans_b == rocblas_operation_none ? k : n))
        return rocblas_status_invalid_size;

    // quick return
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // pointers must be valid
    if((k && !alpha) || !beta || !d)
        return rocblas_status_invalid_pointer;

    // If C is nullptr, beta must be zero
    if(!c)
    {
        switch(compute_type)
        {
        case rocblas_datatype_f16_r:
            if(*(const rocblas_half*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_r:
            if(*(const float*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_r:
            if(*(const double*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_i32_r:
            if(*(const int32_t*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_c:
            if(*(const rocblas_float_complex*)beta)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_c:
            if(*(const rocblas_double_complex*)beta)
                return rocblas_status_invalid_pointer;
            break;
        default:
            break;
        }
    }

    // If k != 0 and either A or B is nullptr, alpha must be zero
    if(k && (!a || !b))
    {
        switch(compute_type)
        {
        case rocblas_datatype_f16_r:
            if(*(const rocblas_half*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_r:
            if(*(const float*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_r:
            if(*(const double*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_i32_r:
            if(*(const int32_t*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f32_c:
            if(*(const rocblas_float_complex*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        case rocblas_datatype_f64_c:
            if(*(const rocblas_double_complex*)alpha)
                return rocblas_status_invalid_pointer;
            break;
        default:
            break;
        }
    }

    return rocblas_status_continue;
}

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_template(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        rocblas_int       m,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const void*       alpha,
                                        const void*       a,
                                        rocblas_datatype  a_type,
                                        rocblas_int       offsetAin,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_a,
                                        const void*       b,
                                        rocblas_datatype  b_type,
                                        rocblas_int       offsetBin,
                                        rocblas_int       ldb,
                                        rocblas_stride    stride_b,
                                        const void*       beta,
                                        const void*       c,
                                        rocblas_datatype  c_type,
                                        rocblas_int       offsetCin,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_c,
                                        void*             d,
                                        rocblas_datatype  d_type,
                                        rocblas_int       offsetDin,
                                        rocblas_int       ldd,
                                        rocblas_stride    stride_d,
                                        rocblas_int       batch_count,
                                        rocblas_datatype  compute_type,
                                        uint32_t          flags)
{
    // Note: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(BATCHED)
    {
        stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
        stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
        stride_c = rocblas_stride(ldc) * n;
        stride_d = rocblas_stride(ldd) * n;
    }

    rocblas_status rb_status = rocblas_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                    \
    handle, trans_a, trans_b, m, n, k, alpha, a, offsetAin, lda, stride_a, b, offsetBin, ldb,  \
        stride_b, beta, c, offsetCin, ldc, stride_c, d, offsetDin, ldd, stride_d, batch_count, \
        rocblas_gemm_flags(flags)

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ex_typecasting<BATCHED, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ex_typecasting<BATCHED, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r)
    {
        if(c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r)
        {
            if(compute_type == rocblas_datatype_f16_r)
            {
                rb_status = gemm_ex_typecasting<BATCHED, rocblas_half>(EX_TYPECASTING_PARM);
            }
            else if(compute_type == rocblas_datatype_f32_r)
            {
                rb_status = gemm_ex_typecasting<BATCHED, rocblas_half, rocblas_half, float>(
                    EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
                && compute_type == rocblas_datatype_f32_r)
        {
            rb_status
                = gemm_ex_typecasting<BATCHED, rocblas_half, float, float>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        if(c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r)
        {
            rb_status = gemm_ex_typecasting<BATCHED, rocblas_bfloat16, rocblas_bfloat16, float>(
                EX_TYPECASTING_PARM);
        }
        else if(c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r)
        {
            rb_status
                = gemm_ex_typecasting<BATCHED, rocblas_bfloat16, float, float>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        bool useInt8x4 = flags & rocblas_gemm_flags_pack_int8x4;

        // Here is point where we decide to branch to real int8 or rocblas_int8x4
        // MatrixInstruction kernel uses general int8 (unless rocblas_gemm_flags_pack_int8x4 is set)
        if(!useInt8x4)
        {
            rb_status = gemm_ex_typecasting<BATCHED, int8_t, int32_t>(EX_TYPECASTING_PARM);
        }
        // Else, we check if we can pack 4 int8:
        else
        {
            // For now, K must be a multiple of 4
            if(k % 4 != 0 || ((trans_a == rocblas_operation_transpose) && (lda % 4 != 0))
               || ((trans_b == rocblas_operation_none) && (ldb % 4 != 0))
               || (batch_count > 1 && (stride_a % 4 != 0 || stride_b % 4 != 0)))
            {
                rb_status = rocblas_status_invalid_size;
            }
            else
            {
                // adjust by 4 for Tensile
                lda = (trans_a == rocblas_operation_none) ? lda : lda / 4;
                ldb = (trans_b == rocblas_operation_none) ? ldb / 4 : ldb;
                k   = k / 4;
                if(!BATCHED)
                {
                    stride_a = stride_a / 4;
                    stride_b = stride_b / 4;
                }
                rb_status
                    = gemm_ex_typecasting<BATCHED, rocblas_int8x4, int32_t>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ex_typecasting<BATCHED,
                                        rocblas_float_complex,
                                        rocblas_float_complex,
                                        rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ex_typecasting<BATCHED,
                                        rocblas_double_complex,
                                        rocblas_double_complex,
                                        rocblas_double_complex>(EX_TYPECASTING_PARM);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}

#undef EX_TYPECASTING_PARM

// Copy alpha and beta to host if on device
template <typename T>
rocblas_status copy_alpha_beta_to_host_if_on_device(rocblas_handle   handle,
                                                    const T*&        alpha,
                                                    const T*&        beta,
                                                    rocblas_union_t& alpha_h,
                                                    rocblas_union_t& beta_h,
                                                    rocblas_int      k,
                                                    rocblas_datatype compute_type)
{
    switch(compute_type)
    {
    case rocblas_datatype_f16_r:
        return copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h.h, beta_h.h, k);
    case rocblas_datatype_f32_r:
        return copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h.s, beta_h.s, k);
    case rocblas_datatype_f64_r:
        return copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h.d, beta_h.d, k);
    case rocblas_datatype_i32_r:
        return copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h.i, beta_h.i, k);
    case rocblas_datatype_f32_c:
        return copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h.c, beta_h.c, k);
    case rocblas_datatype_f64_c:
        return copy_alpha_beta_to_host_if_on_device(handle, alpha, beta, alpha_h.z, beta_h.z, k);
    default:
        return rocblas_status_not_implemented;
    }
}

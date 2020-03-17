/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GEMM_EX_HPP
#define __ROCBLAS_GEMM_EX_HPP

#ifndef USE_TENSILE_HOST
#include "Tensile.h"
#include "TensileTypes.h"
#endif
#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

/////////////////
// Device Side //
/////////////////
template <typename To>
static rocblas_status device_strided_batched_matrix_copy(const To*      src,
                                                         rocblas_stride ld_src,
                                                         rocblas_stride stride_src,
                                                         To*            dst,
                                                         rocblas_stride ld_dst,
                                                         rocblas_stride stride_dst,
                                                         rocblas_int    n1,
                                                         rocblas_int    n2,
                                                         rocblas_int    batch_count)
{
    if(src == dst && ld_src == ld_dst && stride_src == stride_dst)
        return rocblas_status_success; // no copy if src matrix == dst matrix

    if(n1 == ld_src && n1 == ld_dst && stride_src == n2 * ld_src && stride_dst == n2 * ld_dst)
    {
        // src and dst batch matrices are contiguous, use single copy
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dst, src, sizeof(To) * n1 * n2 * batch_count, hipMemcpyDeviceToDevice));
    }
    else if(n1 == ld_src && n1 == ld_dst)
    {
        // individual matrices in batch matrix are contiguous, one copy for each matrix
        for(size_t i3 = 0; i3 < batch_count; i3++)
            RETURN_IF_HIP_ERROR(hipMemcpy(dst + i3 * stride_dst,
                                          src + i3 * stride_src,
                                          sizeof(To) * n1 * n2,
                                          hipMemcpyDeviceToDevice));
    }
    else
    {
        // individual matrices not contiguous, one copy for each contiguous column
        for(int i3 = 0; i3 < batch_count; i3++)
            for(int i2 = 0; i2 < n2; i2++)
                RETURN_IF_HIP_ERROR(hipMemcpy(dst + i2 * ld_dst + i3 * stride_dst,
                                              src + i2 * ld_src + i3 * stride_src,
                                              sizeof(To) * n1,
                                              hipMemcpyDeviceToDevice));
    }
    return rocblas_status_success;
}

//------------------------------------------------------------------------------

///////////////
// Host Side //
///////////////
template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_batched_template(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        rocblas_int       m,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const Tc*         alpha,
                                        const Ti*         a[],
                                        size_t            offset_a,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_a,
                                        const Ti*         b[],
                                        size_t            offset_b,
                                        rocblas_int       ldb,
                                        rocblas_stride    stride_b,
                                        const Tc*         beta,
                                        const To*         c[],
                                        size_t            offset_c,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_c,
                                        To*               d[],
                                        size_t            offset_d,
                                        rocblas_int       ldd,
                                        rocblas_stride    stride_d,
                                        rocblas_int       batch_count,
                                        hipEvent_t*       startEvent = nullptr,
                                        hipEvent_t*       stopEvent  = nullptr)
{
    // BATCHED VERSION
    // Host arrays of device pointers.
    Ti* hostA[batch_count];
    Ti* hostB[batch_count];
    To* hostC[batch_count];
    To* hostD[batch_count];

    RETURN_IF_HIP_ERROR(hipMemcpy(hostA, a, sizeof(hostA), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(hostB, b, sizeof(hostB), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(hostC, c, sizeof(hostC), hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(hostD, d, sizeof(hostD), hipMemcpyDeviceToHost));

    stride_a = rocblas_stride(lda) * (trans_a == rocblas_operation_none ? k : m);
    stride_b = rocblas_stride(ldb) * (trans_b == rocblas_operation_none ? n : k);
    stride_c = rocblas_stride(ldc) * n;
    stride_d = rocblas_stride(ldd) * n;

    rocblas_status status = rocblas_status_success;
    for(rocblas_int bi = 0; bi < batch_count; bi++)
    {
        // Tensile does not support batched gemm_ex yet, must do naive version
        status = gemm_ex_batched_template(handle,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          hostA[bi],
                                          offset_a,
                                          lda,
                                          stride_a,
                                          hostB[bi],
                                          offset_b,
                                          ldb,
                                          stride_b,
                                          beta,
                                          hostC[bi],
                                          offset_c,
                                          ldc,
                                          stride_c,
                                          hostD[bi],
                                          offset_d,
                                          ldd,
                                          stride_d,
                                          1,
                                          startEvent,
                                          stopEvent);
        if(status != rocblas_status_success)
            break;
    }
    return status;
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_batched_template(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        rocblas_int       m,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const Tc*         alpha,
                                        const Ti*         a,
                                        size_t            offset_a,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_a,
                                        const Ti*         b,
                                        size_t            offset_b,
                                        rocblas_int       ldb,
                                        rocblas_stride    stride_b,
                                        const Tc*         beta,
                                        const To*         c,
                                        size_t            offset_c,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_c,
                                        To*               d,
                                        size_t            offset_d,
                                        rocblas_int       ldd,
                                        rocblas_stride    stride_d,
                                        rocblas_int       batch_count,
                                        hipEvent_t*       startEvent = nullptr,
                                        hipEvent_t*       stopEvent  = nullptr)
{
    a += offset_a;
    b += offset_b;
    c += offset_c;
    d += offset_d;

    static const bool arch_lt906 = handle->device_arch_id() < 906;
    const To*         c_in;
    rocblas_int       ldi;
    rocblas_stride    stride_i;

    if(!arch_lt906 && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{})
       && ((ldc >= ldd && (stride_c >= stride_d || batch_count == 1) && m == ldd)
           || (ldc == ldd && (stride_c == stride_d || batch_count == 1))))
    {
        c_in     = c;
        ldi      = ldc;
        stride_i = stride_c;
    }
    else
    {
        device_strided_batched_matrix_copy(c, ldc, stride_c, d, ldd, stride_d, m, n, batch_count);
        c_in     = d;
        ldi      = ldd;
        stride_i = stride_d;
    }

#ifdef USE_TENSILE_HOST

    RocblasContractionProblem<Ti, To, Tc> problem{
        handle, trans_a,  trans_b, m,    n,   k,        alpha, a,   lda,      stride_a,   b,
        ldb,    stride_b, beta,    c_in, ldi, stride_i, d,     ldd, stride_d, batch_count};

    if(startEvent && stopEvent)
        return handle->host->runContractionProblem(problem, startEvent, stopEvent);
    else
        return handle->host->runContractionProblem(problem);

#else // USE_TENSILE_HOST

    TensileStatus  t_status;
    rocblas_status rb_status;

    t_status = call_tensile_ex<Ti, To, Tc>(d,
                                           c_in,
                                           a,
                                           b,
                                           *alpha,
                                           *beta,
                                           ldd,
                                           stride_d,
                                           ldi,
                                           stride_i,
                                           lda,
                                           stride_a,
                                           ldb,
                                           stride_b,
                                           m,
                                           n,
                                           batch_count,
                                           k,
                                           handle->rocblas_stream,
                                           GetTransposeMode(trans_a, trans_b),
                                           startEvent,
                                           stopEvent);

    rb_status = (t_status == tensileStatusSuccess) ? rocblas_status_success
                                                   : rocblas_status_internal_error;
    return rb_status;

#endif // USE_TENSILE_HOST
}

template <bool BATCHED, typename Ti, typename To = Ti, typename Tc = To>
rocblas_status gemm_ex_typecasting(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const void*       alpha,
                                   const void*       a,
                                   rocblas_int       offsetAin,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const void*       b,
                                   rocblas_int       offsetBin,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   const void*       beta,
                                   const void*       c,
                                   rocblas_int       offsetCin,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   void*             d,
                                   rocblas_int       offsetDin,
                                   rocblas_int       ldd,
                                   rocblas_stride    stride_d,
                                   rocblas_int       batch_count,
                                   void*             startEvent = nullptr,
                                   void*             stopEvent  = nullptr)
{
    Tc alpha_h, beta_h;

    // Right now Tensile requires alpha and beta to be passed by value on host.
    // If in device pointer mode, copy alpha and beta to host.
    // TODO: Make this asynchronous, putting synchronization in closer to Tensile call.
    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        RETURN_IF_HIP_ERROR(hipMemcpy(&alpha_h, alpha, sizeof(Tc), hipMemcpyDeviceToHost));
        RETURN_IF_HIP_ERROR(hipMemcpy(&beta_h, beta, sizeof(Tc), hipMemcpyDeviceToHost));
        alpha = &alpha_h;
        beta  = &beta_h;
    }

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
                                        (hipEvent_t*)startEvent,
                                        (hipEvent_t*)stopEvent);
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
                                        (hipEvent_t*)startEvent,
                                        (hipEvent_t*)stopEvent);
    }
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
                                        void*             startEvent = nullptr,
                                        void*             stopEvent  = nullptr)
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
        startEvent, stopEvent

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
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ex_typecasting<BATCHED, rocblas_half>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status
            = gemm_ex_typecasting<BATCHED, rocblas_half, rocblas_half, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
#ifdef USE_TENSILE_HOST
        rb_status = gemm_ex_typecasting<BATCHED, rocblas_bfloat16, rocblas_bfloat16, float>(
            EX_TYPECASTING_PARM);
#else
        rb_status = gemm_ex_typecasting<BATCHED, tensile_bfloat16, tensile_bfloat16, float>(
            EX_TYPECASTING_PARM);
#endif
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        // For now, K must be a multiple of 4
        if(k % 4 != 0 || ((trans_a == rocblas_operation_transpose) && (lda % 4 != 0))
           || ((trans_b == rocblas_operation_none) && (ldb % 4 != 0)) || stride_a % 4 != 0
           || stride_b % 4 != 0)
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

#ifdef USE_TENSILE_HOST
            rb_status = gemm_ex_typecasting<BATCHED, int8_t, int32_t>(EX_TYPECASTING_PARM);
#else
            rb_status
                = gemm_ex_typecasting<BATCHED, TensileInt8x4, TensileInt32>(EX_TYPECASTING_PARM);
#endif
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

#endif

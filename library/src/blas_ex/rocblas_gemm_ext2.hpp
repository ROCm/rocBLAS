/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GEMM_EXT2_HPP
#define __ROCBLAS_GEMM_EXT2_HPP

// This functionality is only availble when using the new Tensile client
#ifdef USE_TENSILE_HOST

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
                                                         rocblas_stride row_stride_src,
                                                         rocblas_stride col_stride_src,
                                                         rocblas_stride batch_stride_src,
                                                         To*            dst,
                                                         rocblas_stride row_stride_dst,
                                                         rocblas_stride col_stride_dst,
                                                         rocblas_stride batch_stride_dst,
                                                         rocblas_int    n1,
                                                         rocblas_int    n2,
                                                         rocblas_int    batch_count)
{
    if(src == dst && col_stride_src == col_stride_dst && batch_stride_src == batch_stride_dst)
        return rocblas_status_success; // no copy if src matrix == dst matrix

    if(n1 == col_stride_src && n1 == col_stride_dst && batch_stride_src == n2 * col_stride_src
       && batch_stride_dst == n2 * col_stride_dst)
    {
        // src and dst batch matrices are contiguous, use single copy
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dst, src, sizeof(To) * n1 * n2 * batch_count, hipMemcpyDeviceToDevice));
    }
    else if(n1 == col_stride_src && n1 == col_stride_dst)
    {
        // individual matrices in batch matrix are contiguous, one copy for each matrix
        for(size_t i3 = 0; i3 < batch_count; i3++)
            RETURN_IF_HIP_ERROR(hipMemcpy(dst + i3 * batch_stride_dst,
                                          src + i3 * batch_stride_src,
                                          sizeof(To) * n1 * n2,
                                          hipMemcpyDeviceToDevice));
    }
    else
    {
        // individual matrices not contiguous, one copy for each contiguous column
        for(int i3 = 0; i3 < batch_count; i3++)
            for(int i2 = 0; i2 < n2; i2++)
                RETURN_IF_HIP_ERROR(hipMemcpy(dst + i2 * col_stride_dst + i3 * batch_stride_dst,
                                              src + i2 * col_stride_src + i3 * batch_stride_src,
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
rocblas_status gemm_ext2_batched_template(rocblas_handle    handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const Tc*         alpha,
                                          const Ti*         a[],
                                          size_t            offset_a,
                                          rocblas_int       row_stride_a,
                                          rocblas_int       col_stride_a,
                                          rocblas_stride    batch_stride_a,
                                          const Ti*         b[],
                                          size_t            offset_b,
                                          rocblas_int       row_stride_b,
                                          rocblas_int       col_stride_b,
                                          rocblas_stride    batch_stride_b,
                                          const Tc*         beta,
                                          const To*         c[],
                                          size_t            offset_c,
                                          rocblas_int       row_stride_c,
                                          rocblas_int       col_stride_c,
                                          rocblas_stride    batch_stride_c,
                                          To*               d[],
                                          size_t            offset_d,
                                          rocblas_int       row_stride_d,
                                          rocblas_int       col_stride_d,
                                          rocblas_stride    batch_stride_d,
                                          rocblas_int       batch_count = 1)
{
    // BATCHED VERSION
    // Host arrays of device pointers.
    auto hostA = std::make_unique<Ti*[]>(batch_count);
    auto hostB = std::make_unique<Ti*[]>(batch_count);
    auto hostC = std::make_unique<To*[]>(batch_count);
    auto hostD = std::make_unique<To*[]>(batch_count);

    RETURN_IF_HIP_ERROR(hipMemcpy(&hostA[0], a, sizeof(Ti*) * batch_count, hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(&hostB[0], b, sizeof(Ti*) * batch_count, hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(&hostC[0], c, sizeof(To*) * batch_count, hipMemcpyDeviceToHost));
    RETURN_IF_HIP_ERROR(hipMemcpy(&hostD[0], d, sizeof(To*) * batch_count, hipMemcpyDeviceToHost));

    batch_stride_a = rocblas_stride(col_stride_a) * (trans_a == rocblas_operation_none ? k : m);
    batch_stride_b = rocblas_stride(col_stride_b) * (trans_b == rocblas_operation_none ? n : k);
    batch_stride_c = rocblas_stride(col_stride_c) * n;
    batch_stride_d = rocblas_stride(col_stride_c) * n;

    rocblas_status status = rocblas_status_success;
    for(rocblas_int bi = 0; bi < batch_count; bi++)
    {
        status = gemm_ext2_batched_template(handle,
                                            trans_a,
                                            trans_b,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            hostA[bi],
                                            offset_a,
                                            row_stride_a,
                                            col_stride_a,
                                            batch_stride_a,
                                            hostB[bi],
                                            offset_b,
                                            row_stride_b,
                                            col_stride_b,
                                            batch_stride_b,
                                            beta,
                                            hostC[bi],
                                            offset_c,
                                            row_stride_c,
                                            col_stride_c,
                                            batch_stride_c,
                                            hostD[bi],
                                            offset_d,
                                            row_stride_d,
                                            col_stride_d,
                                            batch_stride_d);

        if(status != rocblas_status_success)
            break;
    }
    return status;
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ext2_batched_template(rocblas_handle    handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const Tc*         alpha,
                                          const Ti*         a,
                                          size_t            offset_a,
                                          rocblas_int       row_stride_a,
                                          rocblas_int       col_stride_a,
                                          rocblas_stride    batch_stride_a,
                                          const Ti*         b,
                                          size_t            offset_b,
                                          rocblas_int       row_stride_b,
                                          rocblas_int       col_stride_b,
                                          rocblas_stride    batch_stride_b,
                                          const Tc*         beta,
                                          const To*         c,
                                          size_t            offset_c,
                                          rocblas_int       row_stride_c,
                                          rocblas_int       col_stride_c,
                                          rocblas_stride    batch_stride_c,
                                          To*               d,
                                          size_t            offset_d,
                                          rocblas_int       row_stride_d,
                                          rocblas_int       col_stride_d,
                                          rocblas_stride    batch_stride_d,
                                          rocblas_int       batch_count = 1)
{
    a += offset_a;
    b += offset_b;
    c += offset_c;
    d += offset_d;

    static const bool arch_lt906 = handle->device_arch_id() < 906;
    const To*         c_in;
    rocblas_int       row_stride_i, col_stride_i;
    rocblas_stride    batch_stride_i;

    if(!arch_lt906 && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{})
       && ((col_stride_c >= col_stride_d && (batch_stride_c >= batch_stride_d || batch_count == 1)
            && m == col_stride_d)
           || (col_stride_c == col_stride_d && row_stride_c == row_stride_d
               && (batch_count == 1 || batch_stride_c == batch_stride_d))))
    {
        c_in           = c;
        row_stride_i   = row_stride_c;
        col_stride_i   = col_stride_c;
        batch_stride_i = batch_stride_c;
    }
    else
    {
        device_strided_batched_matrix_copy(c,
                                           row_stride_c,
                                           col_stride_c,
                                           batch_stride_c,
                                           d,
                                           row_stride_d,
                                           col_stride_d,
                                           batch_stride_d,
                                           m,
                                           n,
                                           batch_count);
        c_in           = d;
        row_stride_i   = row_stride_d;
        col_stride_i   = col_stride_d;
        batch_stride_i = batch_stride_d;
    }

    RocblasContractionProblem<Ti, To, Tc> problem{handle,
                                                  trans_a,
                                                  trans_b,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  a,
                                                  row_stride_a,
                                                  col_stride_a,
                                                  batch_stride_a,
                                                  b,
                                                  row_stride_b,
                                                  col_stride_b,
                                                  batch_stride_b,
                                                  beta,
                                                  c_in,
                                                  row_stride_i,
                                                  col_stride_i,
                                                  batch_stride_i,
                                                  d,
                                                  row_stride_d,
                                                  col_stride_d,
                                                  batch_stride_d,
                                                  batch_count};

    return handle->host->runContractionProblem(problem);
}

template <bool BATCHED, typename Ti, typename To = Ti, typename Tc = To>
rocblas_status gemm_ext2_typecasting(rocblas_handle    handle,
                                     rocblas_operation trans_a,
                                     rocblas_operation trans_b,
                                     rocblas_int       m,
                                     rocblas_int       n,
                                     rocblas_int       k,
                                     const void*       alpha,
                                     const void*       a,
                                     rocblas_int       offsetAin,
                                     rocblas_int       row_stride_a,
                                     rocblas_int       col_stride_a,
                                     rocblas_stride    batch_stride_a,
                                     const void*       b,
                                     rocblas_int       offsetBin,
                                     rocblas_int       row_stride_b,
                                     rocblas_int       col_stride_b,
                                     rocblas_stride    batch_stride_b,
                                     const void*       beta,
                                     const void*       c,
                                     rocblas_int       offsetCin,
                                     rocblas_int       row_stride_c,
                                     rocblas_int       col_stride_c,
                                     rocblas_stride    batch_stride_c,
                                     void*             d,
                                     rocblas_int       offsetDin,
                                     rocblas_int       row_stride_d,
                                     rocblas_int       col_stride_d,
                                     rocblas_stride    batch_stride_d,
                                     rocblas_int       batch_count)
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
    // TODO: replace with C++17 constexpr if
    if(BATCHED)
    {
        if(!isAligned(a, sizeof(Ti*)) || !isAligned(b, sizeof(Ti*)) || !isAligned(c, sizeof(To*))
           || !isAligned(d, sizeof(To*)))
            return rocblas_status_invalid_size;

        // Pass alpha and beta as simple array (stride of 1)
        // since Tensile does not have gemm_batched, we will have to iterate
        // over batches either way
        return gemm_ext2_batched_template(handle,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          (const Tc*)alpha,
                                          (const Ti**)a,
                                          offsetAin,
                                          row_stride_a,
                                          col_stride_a,
                                          batch_stride_a,
                                          (const Ti**)b,
                                          offsetBin,
                                          row_stride_b,
                                          col_stride_b,
                                          batch_stride_b,
                                          (const Tc*)beta,
                                          (const To**)c,
                                          offsetCin,
                                          row_stride_c,
                                          col_stride_c,
                                          batch_stride_c,
                                          (To**)d,
                                          offsetDin,
                                          row_stride_d,
                                          col_stride_d,
                                          batch_stride_d,
                                          batch_count);
    }
    else
    {
        if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
           || !isAligned(d, sizeof(To)))
            return rocblas_status_invalid_size;

        return gemm_ext2_batched_template(handle,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          (const Tc*)alpha,
                                          (const Ti*)a,
                                          offsetAin,
                                          row_stride_a,
                                          col_stride_a,
                                          batch_stride_a,
                                          (const Ti*)b,
                                          offsetBin,
                                          row_stride_b,
                                          col_stride_b,
                                          batch_stride_b,
                                          (const Tc*)beta,
                                          (const To*)c,
                                          offsetCin,
                                          row_stride_c,
                                          col_stride_c,
                                          batch_stride_c,
                                          (To*)d,
                                          offsetDin,
                                          row_stride_d,
                                          col_stride_d,
                                          batch_stride_d,
                                          batch_count);
    }
}

template <bool BATCHED>
rocblas_status rocblas_gemm_ext2_template(rocblas_handle    handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const void*       alpha,
                                          const void*       a,
                                          rocblas_datatype  a_type,
                                          rocblas_int       offsetAin,
                                          rocblas_int       row_stride_a,
                                          rocblas_int       col_stride_a,
                                          rocblas_stride    batch_stride_a,
                                          const void*       b,
                                          rocblas_datatype  b_type,
                                          rocblas_int       offsetBin,
                                          rocblas_int       row_stride_b,
                                          rocblas_int       col_stride_b,
                                          rocblas_stride    batch_stride_b,
                                          const void*       beta,
                                          const void*       c,
                                          rocblas_datatype  c_type,
                                          rocblas_int       offsetCin,
                                          rocblas_int       row_stride_c,
                                          rocblas_int       col_stride_c,
                                          rocblas_stride    batch_stride_c,
                                          void*             d,
                                          rocblas_datatype  d_type,
                                          rocblas_int       offsetDin,
                                          rocblas_int       row_stride_d,
                                          rocblas_int       col_stride_d,
                                          rocblas_stride    batch_stride_d,
                                          rocblas_int       batch_count,
                                          rocblas_datatype  compute_type)
{
    // Note: k==0 is not an early exit, since C still needs to be multiplied by beta
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(BATCHED)
    {
        batch_stride_a = rocblas_stride(col_stride_a) * (trans_a == rocblas_operation_none ? k : m);
        batch_stride_b = rocblas_stride(col_stride_b) * (trans_b == rocblas_operation_none ? n : k);
        batch_stride_c = rocblas_stride(col_stride_c) * n;
        batch_stride_d = rocblas_stride(col_stride_d) * n;
    }

    rocblas_status rb_status = rocblas_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                \
    handle, trans_a, trans_b, m, n, k, alpha, a, offsetAin, row_stride_a, col_stride_a,    \
        batch_stride_a, b, offsetBin, row_stride_b, col_stride_b, batch_stride_b, beta, c, \
        offsetCin, row_stride_c, col_stride_c, batch_stride_c, d, offsetDin, row_stride_d, \
        col_stride_d, batch_stride_d, batch_count

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r
       && c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r
       && compute_type == rocblas_datatype_f64_r)
    {
        rb_status = gemm_ext2_typecasting<BATCHED, double>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r
            && c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ext2_typecasting<BATCHED, float>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f16_r)
    {
        rb_status = gemm_ext2_typecasting<BATCHED, rocblas_half>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r
            && c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ext2_typecasting<BATCHED, rocblas_half, rocblas_half, float>(
            EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_bf16_r && b_type == rocblas_datatype_bf16_r
            && c_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_bf16_r
            && compute_type == rocblas_datatype_f32_r)
    {
        rb_status = gemm_ext2_typecasting<BATCHED, rocblas_bfloat16, rocblas_bfloat16, float>(
            EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_i8_r && b_type == rocblas_datatype_i8_r
            && c_type == rocblas_datatype_i32_r && d_type == rocblas_datatype_i32_r
            && compute_type == rocblas_datatype_i32_r)
    {
        // For now, K must be a multiple of 4
        if(k % 4 != 0 || ((trans_a == rocblas_operation_transpose) && (col_stride_a % 4 != 0))
           || ((trans_b == rocblas_operation_none) && (col_stride_b % 4 != 0))
           || batch_stride_a % 4 != 0 || batch_stride_b % 4 != 0)
        {
            rb_status = rocblas_status_invalid_size;
        }
        else
        {
            // adjust by 4 for Tensile
            col_stride_a = (trans_a == rocblas_operation_none) ? col_stride_a : col_stride_a / 4;
            col_stride_b = (trans_b == rocblas_operation_none) ? col_stride_b / 4 : col_stride_b;
            k            = k / 4;
            if(!BATCHED)
            {
                batch_stride_a /= 4;
                batch_stride_b /= 4;
            }

            rb_status = gemm_ext2_typecasting<BATCHED, int8_t, int32_t>(EX_TYPECASTING_PARM);
        }
    }
    else if(a_type == rocblas_datatype_f32_c && b_type == rocblas_datatype_f32_c
            && c_type == rocblas_datatype_f32_c && d_type == rocblas_datatype_f32_c
            && compute_type == rocblas_datatype_f32_c)
    {
        rb_status = gemm_ext2_typecasting<BATCHED,
                                          rocblas_float_complex,
                                          rocblas_float_complex,
                                          rocblas_float_complex>(EX_TYPECASTING_PARM);
    }
    else if(a_type == rocblas_datatype_f64_c && b_type == rocblas_datatype_f64_c
            && c_type == rocblas_datatype_f64_c && d_type == rocblas_datatype_f64_c
            && compute_type == rocblas_datatype_f64_c)
    {
        rb_status = gemm_ext2_typecasting<BATCHED,
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

#endif // USE_TENSILE_HOST

#endif

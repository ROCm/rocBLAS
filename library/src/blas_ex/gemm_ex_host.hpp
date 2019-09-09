/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __GEMM_EX_HOST__
#define __GEMM_EX_HOST__

#include "Tensile.h"
#include "TensileTypes.h"
// #include "gemm.h"
#include "gemm_ex_device.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"


template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_handle_transpose(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        unsigned int      m,
                                        unsigned int      n,
                                        unsigned int      k,
                                        const Tc          alpha,
                                        const Ti*         a,
                                        unsigned int      lda,
                                        unsigned int      stride_a,
                                        const Ti*         b,
                                        unsigned int      ldb,
                                        unsigned int      stride_b,
                                        const Tc          beta,
                                        const To*         c,
                                        unsigned int      ldc,
                                        unsigned int      stride_c,
                                        To*               d,
                                        unsigned int      ldd,
                                        unsigned int      stride_d,
                                        unsigned int      batch_count)
{
    TensileStatus  t_status;
    rocblas_status rb_status;

    static const bool arch_lt906 = handle->device_arch_id() < 906;
    const To* c_in;
    unsigned int ldi, stride_i;

    if(!arch_lt906 && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{}) &&
       ((ldc >= ldd && stride_c >= stride_d && m == ldd) || (ldc == ldd && stride_c == stride_d)))
    {
        c_in     = c;
        ldi      = ldc;
        stride_i = stride_c;
    }
    else
    {
        device_strided_batched_matrix_copy(
            c, ldc, stride_c, d, ldd, stride_d, m, n, batch_count, sizeof(To));
        c_in     = d;
        ldi      = ldd;
        stride_i = stride_d;
    }

    t_status = call_tensile_ex<Ti,To,Tc>((To*)d,
                                         (const To*)c_in,
                                         (const Ti*)a,
                                         (const Ti*)b,
                                         alpha, beta,
                                         unsigned(ldd), stride_d,
                                         unsigned(ldi), stride_i,
                                         unsigned(lda), stride_a,
                                         unsigned(ldb), stride_b,
                                         unsigned(m),
                                         unsigned(n),
                                         unsigned(batch_count),
                                         unsigned(k),
                                         handle->rocblas_stream, GetTransposeMode(trans_a, trans_b));

    rb_status = (t_status == tensileStatusSuccess) ? rocblas_status_success : rocblas_status_internal_error;
    return rb_status;
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_handle_transpose(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        unsigned int      m,
                                        unsigned int      n,
                                        unsigned int      k,
                                        const Tc          alpha,
                                        const Ti*         a[],
                                        unsigned int      lda,
                                        const Ti*         b[],
                                        unsigned int      ldb,
                                        const Tc          beta,
                                        const To*         c[],
                                        unsigned int      ldc,
                                        To*               d[],
                                        unsigned int      ldd,
                                        unsigned int      batch_count,
                                        rocblas_int       offset_a,
                                        rocblas_int       offset_b,
                                        rocblas_int       offset_c,
                                        rocblas_int       offset_d)
{
    // Host arrays of device pointers.
    Ti* hostA[batch_count];
    Ti* hostB[batch_count];
    To* hostC[batch_count];
    To* hostD[batch_count];

    hipError_t errA = hipMemcpy(hostA, a, batch_count * sizeof(Ti*), hipMemcpyDeviceToHost);
    hipError_t errB = hipMemcpy(hostB, b, batch_count * sizeof(Ti*), hipMemcpyDeviceToHost);
    hipError_t errC = hipMemcpy(hostC, c, batch_count * sizeof(To*), hipMemcpyDeviceToHost);
    hipError_t errD = hipMemcpy(hostD, d, batch_count * sizeof(To*), hipMemcpyDeviceToHost);

    if(get_rocblas_status_for_hip_status(errA) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errA);
    else if(get_rocblas_status_for_hip_status(errB) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errB);
    else if(get_rocblas_status_for_hip_status(errC) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errC);
    else if(get_rocblas_status_for_hip_status(errD) != rocblas_status_success)
        return get_rocblas_status_for_hip_status(errD);

    rocblas_int    stride_a    = trans_a == rocblas_operation_none ? lda * k : lda * m;
    rocblas_int    stride_b    = trans_b == rocblas_operation_none ? ldb * n : ldb * k;
    rocblas_int    stride_c    = ldc * n;
    rocblas_int    stride_d    = ldd * n;

    rocblas_status status = rocblas_status_internal_error;
    for(int bi = 0; bi < batch_count; bi++)
    {
        status = gemm_ex_handle_transpose(handle,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          hostA[bi] + offset_a,
                                          lda,
                                          stride_a,
                                          hostB[bi] + offset_b,
                                          ldb,
                                          stride_b,
                                          beta,
                                          hostC[bi] + offset_c,
                                          ldc,
                                          stride_c,
                                          hostD[bi] + offset_d,
                                          ldd,
                                          stride_d,
                                          1);
        if(status != rocblas_status_success)
            return status;
    }
    return status;
}

#if defined(USE_CHUNKING)
template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_chunking(rocblas_handle    handle,
                                rocblas_operation trans_a,
                                rocblas_operation trans_b,
                                unsigned int      m,
                                unsigned int      n,
                                unsigned int      k,
                                Tc                alpha,
                                const Ti*         a,
                                unsigned int      lda,
                                unsigned int      stride_a,
                                const Ti*         b,
                                unsigned int      ldb,
                                unsigned int      stride_b,
                                Tc                beta,
                                const To*         c,
                                unsigned int      ldc,
                                unsigned int      stride_c,
                                To*               d,
                                unsigned int      ldd,
                                unsigned int      stride_d,
                                unsigned int      batch_count)
{
    unsigned int int_limit    = std::numeric_limits<int>::max() / sizeof(To);
    unsigned int m_chunk_size = m;
    unsigned int n_chunk_size = n;

    unsigned int m_chunk_size_a;
    unsigned int n_chunk_size_b;
    unsigned int n_chunk_size_c = int_limit / ldc;
    unsigned int n_chunk_size_d = int_limit / ldd;

    n_chunk_size = n_chunk_size < n_chunk_size_c ? n_chunk_size : n_chunk_size_c;
    n_chunk_size = n_chunk_size < n_chunk_size_d ? n_chunk_size : n_chunk_size_d;

    if(trans_b == rocblas_operation_none)
    {
        n_chunk_size_b = int_limit / ldb;
        n_chunk_size   = n_chunk_size < n_chunk_size_b ? n_chunk_size : n_chunk_size_b;
    }

    if(trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose)
    {
        m_chunk_size_a = int_limit / lda;
        m_chunk_size   = m_chunk_size < m_chunk_size_a ? m_chunk_size : m_chunk_size_a;
    }

    // if chunk_size < 1 return error because offset for a single row or column is larger than
    // can fit into 32 bit register
    if(m_chunk_size < 1)
        return rocblas_status_invalid_size;
    if(n_chunk_size < 1)
        return rocblas_status_invalid_size;

    unsigned int n_chunk_count = ((n - 1) / n_chunk_size) + 1;
    unsigned int m_chunk_count = ((m - 1) / m_chunk_size) + 1;

    rocblas_status return_status = rocblas_status_success;
    rocblas_status status        = rocblas_status_success;

    for(int n_chunk_iterator = 0; n_chunk_iterator < n_chunk_count; n_chunk_iterator++)
    {
        unsigned int n_chunk_remaining = n - (n_chunk_size * n_chunk_iterator);

        unsigned int n_chunk_size_corrected
            = n_chunk_size < n_chunk_remaining ? n_chunk_size : n_chunk_remaining;

        for(int m_chunk_iterator = 0; m_chunk_iterator < m_chunk_count; m_chunk_iterator++)
        {
            unsigned int m_chunk_remaining = m - (m_chunk_size * m_chunk_iterator);

            unsigned int m_chunk_size_corrected
                = m_chunk_size < m_chunk_remaining ? m_chunk_size : m_chunk_remaining;

            size_t c_offset
                = n_chunk_iterator * n_chunk_size * ldc + m_chunk_iterator * m_chunk_size;
            size_t d_offset
                = n_chunk_iterator * n_chunk_size * ldd + m_chunk_iterator * m_chunk_size;
            size_t a_offset = m_chunk_iterator * m_chunk_size;
            size_t b_offset = n_chunk_iterator * n_chunk_size;

            if(trans_b == rocblas_operation_none)
                b_offset *= ldb;
            if(trans_a != rocblas_operation_none)
                a_offset *= lda;

            status = gemm_ex_handle_transpose<Ti, To, Tc>(handle,
                                                          trans_a,
                                                          trans_b,
                                                          m_chunk_size_corrected,
                                                          n_chunk_size_corrected,
                                                          k,
                                                          alpha,
                                                          a + a_offset,
                                                          lda,
                                                          stride_a,
                                                          b + b_offset,
                                                          ldb,
                                                          stride_b,
                                                          beta,
                                                          c + c_offset,
                                                          ldc,
                                                          stride_c,
                                                          d + d_offset,
                                                          ldd,
                                                          stride_d,
                                                          batch_count);

            if(status != rocblas_status_success)
                return_status = status;
        }
    }
    return return_status;
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_chunking_batched(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        unsigned int      m,
                                        unsigned int      n,
                                        unsigned int      k,
                                        Tc                alpha,
                                        const Ti*         a[],
                                        unsigned int      offsetAin,
                                        unsigned int      lda,
                                        const Ti*         b[],
                                        unsigned int      offsetBin,
                                        unsigned int      ldb,
                                        Tc                beta,
                                        const To*         c[],
                                        unsigned int      offsetCin,
                                        unsigned int      ldc,
                                        To*               d[],
                                        unsigned int      offsetDin,
                                        unsigned int      ldd,
                                        unsigned int      batch_count)
{
    unsigned int int_limit    = std::numeric_limits<int>::max() / sizeof(To);
    unsigned int m_chunk_size = m;
    unsigned int n_chunk_size = n;

    unsigned int m_chunk_size_a;
    unsigned int n_chunk_size_b;
    unsigned int n_chunk_size_c = int_limit / ldc;
    unsigned int n_chunk_size_d = int_limit / ldd;

    n_chunk_size = n_chunk_size < n_chunk_size_c ? n_chunk_size : n_chunk_size_c;
    n_chunk_size = n_chunk_size < n_chunk_size_d ? n_chunk_size : n_chunk_size_d;

    if(trans_b == rocblas_operation_none)
    {
        n_chunk_size_b = int_limit / ldb;
        n_chunk_size   = n_chunk_size < n_chunk_size_b ? n_chunk_size : n_chunk_size_b;
    }

    if(trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose)
    {
        m_chunk_size_a = int_limit / lda;
        m_chunk_size   = m_chunk_size < m_chunk_size_a ? m_chunk_size : m_chunk_size_a;
    }

    // if chunk_size < 1 return error because offset for a single row or column is larger than
    // can fit into 32 bit register
    if(m_chunk_size < 1)
        return rocblas_status_invalid_size;
    if(n_chunk_size < 1)
        return rocblas_status_invalid_size;

    unsigned int n_chunk_count = ((n - 1) / n_chunk_size) + 1;
    unsigned int m_chunk_count = ((m - 1) / m_chunk_size) + 1;

    rocblas_status return_status = rocblas_status_success;
    rocblas_status status        = rocblas_status_success;

    for(int n_chunk_iterator = 0; n_chunk_iterator < n_chunk_count; n_chunk_iterator++)
    {
        unsigned int n_chunk_remaining = n - (n_chunk_size * n_chunk_iterator);

        unsigned int n_chunk_size_corrected
            = n_chunk_size < n_chunk_remaining ? n_chunk_size : n_chunk_remaining;

        for(int m_chunk_iterator = 0; m_chunk_iterator < m_chunk_count; m_chunk_iterator++)
        {
            unsigned int m_chunk_remaining = m - (m_chunk_size * m_chunk_iterator);

            unsigned int m_chunk_size_corrected
                = m_chunk_size < m_chunk_remaining ? m_chunk_size : m_chunk_remaining;

            size_t c_offset
                = n_chunk_iterator * n_chunk_size * ldc + m_chunk_iterator * m_chunk_size;
            size_t d_offset
                = n_chunk_iterator * n_chunk_size * ldd + m_chunk_iterator * m_chunk_size;
            size_t a_offset = m_chunk_iterator * m_chunk_size;
            size_t b_offset = n_chunk_iterator * n_chunk_size;

            if(trans_b == rocblas_operation_none)
                b_offset *= ldb;
            if(trans_a != rocblas_operation_none)
                a_offset *= lda;

            status = gemm_ex_handle_transpose<Ti, To, Tc>(handle,
                                                          trans_a,
                                                          trans_b,
                                                          m_chunk_size_corrected,
                                                          n_chunk_size_corrected,
                                                          k,
                                                          alpha,
                                                          a,
                                                          lda,
                                                          b,
                                                          ldb,
                                                          beta,
                                                          c,
                                                          ldc,
                                                          d,
                                                          ldd,
                                                          batch_count,
                                                          a_offset + offsetAin,
                                                          b_offset + offsetBin,
                                                          c_offset + offsetCin,
                                                          d_offset + offsetDin);

            if(status != rocblas_status_success)
                return_status = status;
        }
    }
    return return_status;
}
#else
#define gemm_ex_chunking gemm_ex_handle_transpose
#endif // defined(USE_CHUNKING)

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_typecasting(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const void*       alpha,
                                   const void*       a,
                                   rocblas_int       lda,
                                   rocblas_int       stride_a,
                                   const void*       b,
                                   rocblas_int       ldb,
                                   rocblas_int       stride_b,
                                   const void*       beta,
                                   const void*       c,
                                   rocblas_int       ldc,
                                   rocblas_int       stride_c,
                                   void*             d,
                                   rocblas_int       ldd,
                                   rocblas_int       stride_d,
                                   rocblas_int       batch_count)
{
    Tc h_alpha;
    Tc h_beta;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // copy alpha and beta from device to host and convert type
        hipMemcpy(&h_alpha, alpha, sizeof(Tc), hipMemcpyDeviceToHost);
        hipMemcpy(&h_beta, beta, sizeof(Tc), hipMemcpyDeviceToHost);
    }
    else
    {
        h_alpha = *((const Tc*)alpha);
        h_beta  = *((const Tc*)beta);
    }

    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
       || !isAligned(d, sizeof(To)))
    {
        return rocblas_status_invalid_size;
    }

    return gemm_ex_chunking<Ti, To, Tc>(handle,
                                        trans_a,
                                        trans_b,
                                        unsigned(m),
                                        unsigned(n),
                                        unsigned(k),
                                        h_alpha,
                                        (const Ti*)a,
                                        unsigned(lda),
                                        unsigned(stride_a),
                                        (const Ti*)b,
                                        unsigned(ldb),
                                        unsigned(stride_b),
                                        h_beta,
                                        (const To*)c,
                                        unsigned(ldc),
                                        unsigned(stride_c),
                                        (To*)d,
                                        unsigned(ldd),
                                        unsigned(stride_d),
                                        unsigned(batch_count));
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_typecasting_batched(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const void*       alpha,
                                           const void*       a,
                                           rocblas_int       offsetAin,
                                           rocblas_int       lda,
                                           const void*       b,
                                           rocblas_int       offsetBin,
                                           rocblas_int       ldb,
                                           const void*       beta,
                                           const void*       c,
                                           rocblas_int       offsetCin,
                                           rocblas_int       ldc,
                                           void*             d,
                                           rocblas_int       offsetDin,
                                           rocblas_int       ldd,
                                           rocblas_int       batch_count)
{
    Tc h_alpha;
    Tc h_beta;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // copy alpha and beta from device to host and convert type
        hipMemcpy(&h_alpha, alpha, sizeof(Tc), hipMemcpyDeviceToHost);
        hipMemcpy(&h_beta, beta, sizeof(Tc), hipMemcpyDeviceToHost);
    }
    else
    {
        h_alpha = *((const Tc*)alpha);
        h_beta  = *((const Tc*)beta);
    }

    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti*)) || !isAligned(b, sizeof(Ti*)) || !isAligned(c, sizeof(To*))
       || !isAligned(d, sizeof(To)))
    {
        return rocblas_status_invalid_size;
    }

    return gemm_ex_chunking_batched<Ti, To, Tc>(handle,
                                                trans_a,
                                                trans_b,
                                                unsigned(m),
                                                unsigned(n),
                                                unsigned(k),
                                                h_alpha,
                                                (const Ti**)a,
                                                unsigned(offsetAin),
                                                unsigned(lda),
                                                (const Ti**)b,
                                                unsigned(offsetBin),
                                                unsigned(ldb),
                                                h_beta,
                                                (const To**)c,
                                                unsigned(offsetCin),
                                                unsigned(ldc),
                                                (To**)d,
                                                unsigned(offsetDin),
                                                unsigned(ldd),
                                                unsigned(batch_count));
}

#endif

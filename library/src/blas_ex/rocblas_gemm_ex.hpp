/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// clang-format off
void device_matrix_copy(const void* src,
                        rocblas_int ld_src,
                        void* dst,
                        rocblas_int ld_dst,
                        rocblas_int n1,
                        rocblas_int n2,
                        size_t elem_size)
{
    if((src != dst) || (ld_src != ld_dst)) // no copy if src matrix == dst matrix
    {
        if((n1 == ld_src) && (n1 == ld_dst))
        {
            // src and dst matrices are contiguous, use single copy
            size_t matrix_size = n1 * n2 * elem_size;
            PRINT_IF_HIP_ERROR(hipMemcpy(dst, src, matrix_size, hipMemcpyDeviceToDevice));
        }
        else
        {
            // matrices not contiguous, one copy for each contiguous column
            size_t column_size = n1 * elem_size;
            const void* src_void;
            void* dst_void;

            for(int i2 = 0; i2 < n2; i2++)
            {
                src_void = static_cast<const void*>(static_cast<const uint8_t*>(src)
                                                    + (i2 * ld_src * elem_size));
                dst_void = static_cast<      void*>(static_cast<uint8_t*>(dst)
                                                    + (i2 * ld_dst * elem_size));
                PRINT_IF_HIP_ERROR(hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice));
            }
        }
    }
}

void device_strided_batched_matrix_copy(const void* src,
                        rocblas_int ld_src,
                        rocblas_int stride_src,
                        void* dst,
                        rocblas_int ld_dst,
                        rocblas_int stride_dst,
                        rocblas_int n1,
                        rocblas_int n2,
                        rocblas_int batch_count,
                        size_t elem_size)
{
    if((src != dst) || (ld_src != ld_dst) || (stride_src != stride_dst)) // no copy if src matrix == dst matrix
    {
        const void* src_void;
        void* dst_void;

        if((n1 == ld_src) && (n1 == ld_dst) && (stride_src == n2 * ld_src) && (stride_dst == n2 * ld_dst))
        {
            // src and dst batch matrices are contiguous, use single copy
            size_t matrix_size = n1 * n2 * batch_count * elem_size;
            PRINT_IF_HIP_ERROR(hipMemcpy(dst, src, matrix_size, hipMemcpyDeviceToDevice));
        }
        else if((n1 == ld_src) && (n1 == ld_dst))
        {
            // individual matrices in batch matrix are contiguous, one copy for each matrix
            size_t matrix_size = n1 * n2 * elem_size;
            for (int i3 = 0; i3 < batch_count; i3++)
            {
                src_void = static_cast<const void*>(static_cast<const uint8_t*>(src)
                                                    + (i3 * stride_src * elem_size));

                dst_void = static_cast<      void*>(static_cast<      uint8_t*>(dst)
                                                    + (i3 * stride_dst * elem_size));

                PRINT_IF_HIP_ERROR(hipMemcpy(dst_void, src_void, matrix_size, hipMemcpyDeviceToDevice));
            }

        }
        else
        {
            // individual matrices not contiguous, one copy for each contiguous column
            size_t column_size = n1 * elem_size;
            const void* src_void;
            void* dst_void;
            for(int i3 = 0; i3 < batch_count; i3++)
            {
                for(int i2 = 0; i2 < n2; i2++)
                {
                    src_void = static_cast<const void*>(static_cast<const uint8_t*>(src)
                                                        + (i2 * ld_src * elem_size)
                                                        + (i3 * stride_src * elem_size));

                    dst_void = static_cast<      void*>(static_cast<uint8_t*>(dst)
                                                        + (i2 * ld_dst * elem_size)
                                                        + (i3 * stride_dst * elem_size));

                    PRINT_IF_HIP_ERROR(hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice));
                }
            }
        }
    }
}
//------------------------------------------------------------------------------
#define TENSILE_IN_ARGS(Ti, To, Tc)                                                                              \
              To* dataD, const To* dataC, const Ti* dataA, const Ti* dataB,                                      \
              Tc alpha, Tc beta,                                                                                 \
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,    \
              unsigned int strideB1J, unsigned int strideB2K,                                                    \
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream

// Ti is typename for input data, To is typename for output data, Tc is typename for compute
template <typename Ti, typename To, typename Tc>
TensileStatus tensile_Cijk_Ailk_Bljk_B(TENSILE_IN_ARGS(Ti, To, Tc));
template <typename Ti, typename To, typename Tc>
TensileStatus tensile_Cijk_Ailk_Bjlk_B(TENSILE_IN_ARGS(Ti, To, Tc));
template <typename Ti, typename To, typename Tc>
TensileStatus tensile_Cijk_Alik_Bljk_B(TENSILE_IN_ARGS(Ti, To, Tc));
template <typename Ti, typename To, typename Tc>
TensileStatus tensile_Cijk_Alik_Bjlk_B(TENSILE_IN_ARGS(Ti, To, Tc));

#define TENSILE_OUT_ARGS_HALF                                             \
    dataD, dataC, dataA, dataB, alpha_half, beta_half,                    \
        strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K, \
        sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr
//---typename_data=TensileHalf-----typename_compute=float---------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf,TensileHalf,float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Ailk_Bljk_HBH(TENSILE_OUT_ARGS_HALF);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf,TensileHalf,float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Ailk_Bjlk_HBH(TENSILE_OUT_ARGS_HALF);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf,TensileHalf,float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Alik_Bljk_HBH(TENSILE_OUT_ARGS_HALF);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf,TensileHalf,float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Alik_Bjlk_HBH(TENSILE_OUT_ARGS_HALF);
}
#undef TENSILE_OUT_ARGS_HALF
#define TENSILE_OUT_ARGS                                                  \
    dataD, dataC, dataA, dataB, alpha, beta,                              \
        strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K, \
        sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr
//---typename_data=TensileHalf-----typename_compute=TensileHalf---------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf,TensileHalf,TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Ailk_Bljk_HB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf,TensileHalf,TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Ailk_Bjlk_HB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf,TensileHalf,TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Alik_Bljk_HB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf,TensileHalf,TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Alik_Bjlk_HB(TENSILE_OUT_ARGS);
}
//---typename_data=float-----------typename_compute=float---------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<float,float,float>(
    TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Ailk_Bljk_SB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<float,float,float>(
    TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Ailk_Bjlk_SB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<float,float,float>(
    TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Alik_Bljk_SB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<float,float,float>(
    TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Alik_Bjlk_SB(TENSILE_OUT_ARGS);
}
//---typename_data=double----------typename_compute=double--------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<double,double,double>(
    TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Ailk_Bljk_DB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<double,double,double>(
    TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Ailk_Bjlk_DB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<double,double,double>(
    TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Alik_Bljk_DB(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<double,double,double>(
    TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Alik_Bjlk_DB(TENSILE_OUT_ARGS);
}
//---typename_input=int8----typename_output=int------typename_compute=int--------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileInt8x4,TensileInt32,TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Ailk_Bljk_4xi8BH(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileInt8x4,TensileInt32,TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Ailk_Bjlk_4xi8BH(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<TensileInt8x4,TensileInt32,TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Alik_Bljk_4xi8BH(TENSILE_OUT_ARGS);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileInt8x4,TensileInt32,TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Alik_Bjlk_4xi8BH(TENSILE_OUT_ARGS);
}
#undef TENSILE_IN_ARGS
#undef TENSILE_OUT_ARGS
//------------------------------------------------------------------------------

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_handle_transpose(rocblas_handle handle,
               rocblas_operation trans_a,
               rocblas_operation trans_b,
               unsigned int m,
               unsigned int n,
               unsigned int k, const Tc alpha,
               const Ti* a, unsigned int lda, unsigned int stride_a,
               const Ti* b, unsigned int ldb, unsigned int stride_b, const Tc beta,
               const To* c, unsigned int ldc, unsigned int stride_c,
                     To* d, unsigned int ldd, unsigned int stride_d, unsigned int batch_count)
{
    TensileStatus t_status;
    rocblas_status rb_status;

    const To* c_in;
    if(ldc == ldd && stride_c == stride_d)
    {
        c_in = c;
    }
    else
    {
        device_strided_batched_matrix_copy(c, ldc, stride_c, d, ldd, stride_d, m, n, batch_count, sizeof(To));
        c_in = d;
    }

    if((trans_a == rocblas_operation_none) && (trans_b == rocblas_operation_none))
    {
        t_status = tensile_Cijk_Ailk_Bljk_B<Ti,To,Tc>(static_cast<To*>(d), 
                                                      static_cast<const To*>(c_in), 
                                                      static_cast<const Ti*>(a), 
                                                      static_cast<const Ti*>(b),
                                                      alpha, beta,
                                                      static_cast<unsigned int>(ldd), stride_d, 
                                                      static_cast<unsigned int>(lda), stride_a, 
                                                      static_cast<unsigned int>(ldb), stride_b,
                                                      static_cast<unsigned int>(m),
                                                      static_cast<unsigned int>(n),
                                                      static_cast<unsigned int>(batch_count),
                                                      static_cast<unsigned int>(k),
                                                      handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_none) &&
            (trans_b == rocblas_operation_transpose || trans_b == rocblas_operation_conjugate_transpose))
    {
        t_status = tensile_Cijk_Ailk_Bjlk_B<Ti,To,Tc>(static_cast<To*>(d), 
                                                      static_cast<const To*>(c_in), 
                                                      static_cast<const Ti*>(a), 
                                                      static_cast<const Ti*>(b),
                                                      alpha, beta,
                                                      static_cast<unsigned int>(ldd), stride_d, 
                                                      static_cast<unsigned int>(lda), stride_a, 
                                                      static_cast<unsigned int>(ldb), stride_b,
                                                      static_cast<unsigned int>(m),
                                                      static_cast<unsigned int>(n),
                                                      static_cast<unsigned int>(batch_count),
                                                      static_cast<unsigned int>(k),
                                                      handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose) &&
            (trans_b == rocblas_operation_none))
    {
        t_status = tensile_Cijk_Alik_Bljk_B<Ti,To,Tc>(static_cast<To*>(d),
                                                      static_cast<const To*>(c_in), 
                                                      static_cast<const Ti*>(a),
                                                      static_cast<const Ti*>(b),
                                                      alpha, beta,
                                                      static_cast<unsigned int>(ldd), stride_d, 
                                                      static_cast<unsigned int>(lda), stride_a, 
                                                      static_cast<unsigned int>(ldb), stride_b,
                                                      static_cast<unsigned int>(m),
                                                      static_cast<unsigned int>(n),
                                                      static_cast<unsigned int>(batch_count),
                                                      static_cast<unsigned int>(k),
                                                      handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose) &&
            (trans_b == rocblas_operation_transpose || trans_b == rocblas_operation_conjugate_transpose))
    {
        t_status = tensile_Cijk_Alik_Bjlk_B<Ti,To,Tc>(static_cast<To*>(d),
                                                      static_cast<const To*>(c_in), 
                                                      static_cast<const Ti*>(a),
                                                      static_cast<const Ti*>(b),
                                                      alpha, beta,
                                                      static_cast<unsigned int>(ldd), stride_d, 
                                                      static_cast<unsigned int>(lda), stride_a, 
                                                      static_cast<unsigned int>(ldb), stride_b,
                                                      static_cast<unsigned int>(m),
                                                      static_cast<unsigned int>(n),
                                                      static_cast<unsigned int>(batch_count),
                                                      static_cast<unsigned int>(k),
                                                      handle->rocblas_stream);
    }
    else
    {
        t_status = tensileStatusFailure;
    }

    if(t_status == tensileStatusSuccess)
    {
        rb_status = rocblas_status_success;
    }
    else
    {
        rb_status = rocblas_status_internal_error;
    }

    return rb_status;
}

#if defined(USE_CHUNKING)
template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_chunking(rocblas_handle handle,
                            rocblas_operation trans_a,
                            rocblas_operation trans_b,
                            unsigned int m,
                            unsigned int n,
                            unsigned int k,
                            Tc alpha,
                            const Ti* a, unsigned int lda, unsigned int stride_a,
                            const Ti* b, unsigned int ldb, unsigned int stride_b,
                            Tc beta,
                            const To* c, unsigned int ldc, unsigned int stride_c,
                            To* d, unsigned int ldd, unsigned int stride_d,
                            unsigned int batch_count)
{
    unsigned int int_limit      = std::numeric_limits<int>::max() / sizeof(To);
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
        n_chunk_size = n_chunk_size < n_chunk_size_b ? n_chunk_size : n_chunk_size_b;
    }

    if(trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose)
    {
        m_chunk_size_a = int_limit / lda;
        m_chunk_size = m_chunk_size < m_chunk_size_a ? m_chunk_size : m_chunk_size_a;
    }

    // if chunk_size < 1 return error because offset for a single row or column is larger than
    // can fit into 32 bit register
    if(m_chunk_size < 1) return rocblas_status_invalid_size;
    if(n_chunk_size < 1) return rocblas_status_invalid_size;

    unsigned int n_chunk_count  = ((n - 1) / n_chunk_size) + 1;
    unsigned int m_chunk_count  = ((m - 1) / m_chunk_size) + 1;

    rocblas_status return_status = rocblas_status_success;
    rocblas_status status = rocblas_status_success;

    for(int n_chunk_iterator = 0; n_chunk_iterator < n_chunk_count; n_chunk_iterator++)
    {
        unsigned int n_chunk_remaining = n - (n_chunk_size * n_chunk_iterator);

        unsigned int n_chunk_size_corrected = n_chunk_size < n_chunk_remaining ? n_chunk_size : n_chunk_remaining;

        for(int m_chunk_iterator = 0; m_chunk_iterator < m_chunk_count; m_chunk_iterator++)
        {
            unsigned int m_chunk_remaining = m - (m_chunk_size * m_chunk_iterator);

            unsigned int m_chunk_size_corrected = m_chunk_size < m_chunk_remaining ? m_chunk_size : m_chunk_remaining;

            size_t c_offset = n_chunk_iterator * n_chunk_size * ldc + m_chunk_iterator * m_chunk_size;
            size_t d_offset = n_chunk_iterator * n_chunk_size * ldd + m_chunk_iterator * m_chunk_size;
            size_t a_offset = m_chunk_iterator * m_chunk_size;
            size_t b_offset = n_chunk_iterator * n_chunk_size;

            if(trans_b == rocblas_operation_none) b_offset *= ldb;
            if(trans_a != rocblas_operation_none) a_offset *= lda;


            status = gemm_ex_handle_transpose<Ti, To, Tc>(
                            handle,
                            trans_a,
                            trans_b,
                            m_chunk_size_corrected,
                            n_chunk_size_corrected,
                            k,
                            alpha,
                            a + a_offset, lda, stride_a,
                            b + b_offset, ldb, stride_b, beta,
                            c + c_offset, ldc, stride_c,
                            d + d_offset, ldd, stride_d, batch_count);

            if(status != rocblas_status_success) return_status = status;
        }
    }
    return return_status;
}
#else
#define gemm_ex_chunking        gemm_ex_handle_transpose
#endif  // defined(USE_CHUNKING)

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_typecasting(rocblas_handle handle,
                                   rocblas_operation trans_a, rocblas_operation trans_b,
                                   rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
                                   const void* a, rocblas_int lda, rocblas_int stride_a,
                                   const void* b, rocblas_int ldb, rocblas_int stride_b, const void* beta,
                                   const void* c, rocblas_int ldc, rocblas_int stride_c,
                                   void* d, rocblas_int ldd, rocblas_int stride_d, rocblas_int batch_count)
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
        h_alpha = *(static_cast<const Tc*>(alpha));
        h_beta = *(static_cast<const Tc*>(beta));
    }

    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) ||
       !isAligned(c, sizeof(To)) || !isAligned(d, sizeof(To)))
    {
        return rocblas_status_invalid_size;
    }

    return gemm_ex_chunking<Ti,To,Tc>(handle,
                                      trans_a,
                                      trans_b,
                                      static_cast<unsigned int>(m),
                                      static_cast<unsigned int>(n),
                                      static_cast<unsigned int>(k),
                                      h_alpha,
                                      static_cast<const Ti*>(a), static_cast<unsigned int>(lda), static_cast<unsigned int>(stride_a),
                                      static_cast<const Ti*>(b), static_cast<unsigned int>(ldb), static_cast<unsigned int>(stride_b),
                                      h_beta,
                                      static_cast<const To*>(c), static_cast<unsigned int>(ldc), static_cast<unsigned int>(stride_c),
                                      static_cast<      To*>(d), static_cast<unsigned int>(ldd), static_cast<unsigned int>(stride_d),
                                      static_cast<unsigned int>(batch_count));
}
// clang-format on

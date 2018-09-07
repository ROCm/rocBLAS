/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#include <hip/hip_runtime.h>

#include "rocblas.h"
#include "Tensile.h"
#include "TensileTypes.h"
#include "status.h"
#include "definitions.h"
#include "handle.h"
#include "logging.h"
#include "utility.h"

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
            // matrices C and D are contiguous, use single copy
            size_t matrix_size = n1 * n2 * elem_size;
            PRINT_IF_HIP_ERROR(hipMemcpy(dst, src, matrix_size, hipMemcpyDeviceToDevice))
        }
        else
        {
            size_t column_size = n1 * elem_size;

            for(int i2 = 0; i2 < n2; i2++)
            {
                const void* src_void = static_cast<const void*>(static_cast<const uint8_t*>(src) +
                                                                (i2 * ld_src * elem_size));
                void* dst_void =
                    static_cast<void*>(static_cast<uint8_t*>(dst) + (i2 * ld_dst * elem_size));

                PRINT_IF_HIP_ERROR(
                    hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice))
            }
        }
    }
}
//------------------------------------------------------------------------------
// clang-format off
// Td is typename for data, Tc is typename for compute
template <typename Td, typename Tc>
TensileStatus tensile_Cijk_Ailk_Bljk_B(Td* dataC, const Td* dataA, const Td* dataB, Tc alpha, Tc beta,
              unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream);
template <typename Td, typename Tc>
TensileStatus tensile_Cijk_Ailk_Bjlk_B(Td* dataC, const Td* dataA, const Td* dataB, Tc alpha, Tc beta,
              unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream);
template <typename Td, typename Tc>
TensileStatus tensile_Cijk_Alik_Bljk_B(Td* dataC, const Td* dataA, const Td* dataB, Tc alpha, Tc beta,
              unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream);
template <typename Td, typename Tc>
TensileStatus tensile_Cijk_Alik_Bjlk_B(Td* dataC, const Td* dataA, const Td* dataB, Tc alpha, Tc beta,
              unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K, 
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream);
//---typename_data=TensileHalf-----typename_compute=float---------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf, float>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Ailk_Bljk_HBH(dataC, dataA, dataB, alpha_half, beta_half, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf, float>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Ailk_Bjlk_HBH(dataC, dataA, dataB, alpha_half, beta_half, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf, float>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Alik_Bljk_HBH(dataC, dataA, dataB, alpha_half, beta_half, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf, float>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half = static_cast<TensileHalf>(alpha);
    TensileHalf beta_half = static_cast<TensileHalf>(beta);
    return tensile_Cijk_Alik_Bjlk_HBH(dataC, dataA, dataB, alpha_half, beta_half, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
//---typename_data=TensileHalf-----typename_compute=TensileHalf---------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf,TensileHalf>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              TensileHalf alpha, TensileHalf beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bljk_HB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf,TensileHalf>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              TensileHalf alpha, TensileHalf beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bjlk_HB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf,TensileHalf>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              TensileHalf alpha, TensileHalf beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Alik_Bljk_HB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf,TensileHalf>(
              TensileHalf* dataC, const TensileHalf* dataA, const TensileHalf* dataB,
              TensileHalf alpha, TensileHalf beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Alik_Bjlk_HB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
//---typename_data=float-----------typename_compute=float---------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<float,float>(float* dataC, const float* dataA, const float* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bljk_SB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<float,float>(float* dataC, const float* dataA, const float* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bjlk_SB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<float,float>(float* dataC, const float* dataA, const float* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Alik_Bljk_SB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<float,float>(float* dataC, const float* dataA, const float* dataB,
              float alpha, float beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Alik_Bjlk_SB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
//---typename_data=double----------typename_compute=double--------------------------
template <>
TensileStatus tensile_Cijk_Ailk_Bljk_B<double,double>(double* dataC, const double* dataA, const double* dataB,
              double alpha, double beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bljk_DB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<double,double>(double* dataC, const double* dataA, const double* dataB,
              double alpha, double beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bjlk_DB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bljk_B<double,double>(double* dataC, const double* dataA, const double* dataB,
              double alpha, double beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Alik_Bljk_DB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
template <>
TensileStatus tensile_Cijk_Alik_Bjlk_B<double,double>(double* dataC, const double* dataA, const double* dataB,
              double alpha, double beta, unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
              unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,
              unsigned int strideB1J, unsigned int strideB2K,
              unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL, hipStream_t stream)
{
    return tensile_Cijk_Alik_Bjlk_DB(dataC, dataA, dataB, alpha, beta, offsetC, offsetA, offsetB,
           strideC1J, strideC2K, strideA1L, strideA2K, strideB1J, strideB2K,
           sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr);
}
// clang-format off
//------------------------------------------------------------------------------

template <typename Td, typename Tc>
rocblas_status tensile_gemm_handle_transpose(rocblas_handle handle,
               rocblas_operation trans_a, rocblas_operation trans_b,
               rocblas_int m, rocblas_int n, rocblas_int k, const Tc alpha,
               const Td* a, rocblas_int lda,
               const Td* b, rocblas_int ldb, const Tc beta,
               const Td* c, rocblas_int ldc,
               Td* d, rocblas_int ldd)
{
    TensileStatus t_status;
    rocblas_status rb_status;

    device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(Td));

    if((trans_a == rocblas_operation_none) && (trans_b == rocblas_operation_none))
    {
        unsigned int const stride_a = static_cast<unsigned int const>(lda * k);
        unsigned int const stride_b = static_cast<unsigned int const>(ldb * n);
        unsigned int const stride_d = static_cast<unsigned int const>(ldd * n);
        t_status = tensile_Cijk_Ailk_Bljk_B<Td,Tc>(static_cast<Td*>(d), 
                                                   static_cast<const Td*>(a), 
                                                   static_cast<const Td*>(b),
                                                   alpha, beta, 0, 0, 0, 
                                                   static_cast<unsigned int>(ldd), stride_d, 
                                                   static_cast<unsigned int>(lda), stride_a, 
                                                   static_cast<unsigned int>(ldb), stride_b,
                                                   static_cast<unsigned int>(m), 
                                                   static_cast<unsigned int>(n), 
                                                   static_cast<unsigned int>(1), 
                                                   static_cast<unsigned int>(k), 
                                                   handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_none) &&
            (trans_b == rocblas_operation_transpose || trans_b == rocblas_operation_conjugate_transpose))
    {
        unsigned int const stride_a = static_cast<unsigned int const>(lda * k);
        unsigned int const stride_b = static_cast<unsigned int const>(ldb * k);
        unsigned int const stride_d = static_cast<unsigned int const>(ldd * n);
        t_status = tensile_Cijk_Ailk_Bjlk_B<Td,Tc>(static_cast<Td*>(d), 
                                                   static_cast<const Td*>(a), 
                                                   static_cast<const Td*>(b),
                                                   alpha, beta, 0, 0, 0, 
                                                   static_cast<unsigned int>(ldd), stride_d, 
                                                   static_cast<unsigned int>(lda), stride_a, 
                                                   static_cast<unsigned int>(ldb), stride_b,
                                                   static_cast<unsigned int>(m), 
                                                   static_cast<unsigned int>(n), 
                                                   static_cast<unsigned int>(1), 
                                                   static_cast<unsigned int>(k), 
                                                   handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose) &&
            (trans_b == rocblas_operation_none))
    {
        unsigned int const stride_a = static_cast<unsigned int const>(lda * m);
        unsigned int const stride_b = static_cast<unsigned int const>(ldb * n);
        unsigned int const stride_d = static_cast<unsigned int const>(ldd * n);
        t_status = tensile_Cijk_Alik_Bljk_B<Td,Tc>(static_cast<Td*>(d),
                                                   static_cast<const Td*>(a),
                                                   static_cast<const Td*>(b),
                                                   alpha, beta, 0, 0, 0, 
                                                   static_cast<unsigned int>(ldd), stride_d, 
                                                   static_cast<unsigned int>(lda), stride_a, 
                                                   static_cast<unsigned int>(ldb), stride_b,
                                                   static_cast<unsigned int>(m), 
                                                   static_cast<unsigned int>(n),
                                                   static_cast<unsigned int>(1), 
                                                   static_cast<unsigned int>(k),
                                                   handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose) &&
            (trans_b == rocblas_operation_transpose || trans_b == rocblas_operation_conjugate_transpose))
    {
        unsigned int const stride_a = static_cast<unsigned int const>(lda * m);
        unsigned int const stride_b = static_cast<unsigned int const>(ldb * k);
        unsigned int const stride_d = static_cast<unsigned int const>(ldd * n);
        t_status = tensile_Cijk_Alik_Bjlk_B<Td,Tc>(static_cast<Td*>(d),
                                                   static_cast<const Td*>(a),
                                                   static_cast<const Td*>(b),
                                                   alpha, beta, 0, 0, 0, 
                                                   static_cast<unsigned int>(ldd), stride_d, 
                                                   static_cast<unsigned int>(lda), stride_a, 
                                                   static_cast<unsigned int>(ldb), stride_b,
                                                   static_cast<unsigned int>(m), 
                                                   static_cast<unsigned int>(n),
                                                   static_cast<unsigned int>(1), 
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

template <typename Td, typename Tc>
rocblas_status tensile_gemm_chunk(rocblas_handle handle,
                            rocblas_operation trans_a,
                            rocblas_operation trans_b,
                            rocblas_int m,
                            rocblas_int n,
                            rocblas_int k,
                            Tc alpha,
                            const Td* a,
                            rocblas_int lda,
                            const Td* b,
                            rocblas_int ldb,
                            Tc beta,
                            const Td* c,
                            rocblas_int ldc,
                            Td* d,
                            rocblas_int ldd)
{
    unsigned int int_limit      = std::numeric_limits<int>::max() / sizeof(Td);
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

            status = tensile_gemm_handle_transpose<Td, Tc>(
                            handle,
                            trans_a,
                            trans_b,
                            m_chunk_size_corrected,
                            n_chunk_size_corrected,
                            k,
                            alpha,
                            a + a_offset,
                            lda,
                            b + b_offset,
                            ldb,
                            beta,
                            c + c_offset,
                            ldc,
                            d + d_offset,
                            ldd);

            if(status != rocblas_status_success) return_status = status;
        }
    }
    return return_status;
}

template <typename Td, typename Tc>
rocblas_status tensile_gemm_typecasting(rocblas_handle handle,
                            rocblas_operation trans_a, rocblas_operation trans_b,
                            rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha,
                            const void* a, rocblas_int lda,
                            const void* b, rocblas_int ldb, const void* beta,
                            const void* c, rocblas_int ldc,
                            void* d, rocblas_int ldd)
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

    return tensile_gemm_chunk<Td,Tc>(handle,
                            trans_a,
                            trans_b,
                            m,
                            n,
                            k,
                            h_alpha,
                            static_cast<const Td*>(a),
                            lda,
                            static_cast<const Td*>(b),
                            ldb,
                            h_beta,
                            static_cast<const Td*>(c),
                            ldc,
                            static_cast<Td*>(d),
                            ldd);
}

/*! \brief BLAS EX API

    \details
    GEMM_EX performs one of the matrix-matrix operations

        D = alpha*op( A )*op( B ) + beta*C,

    where op( X ) is one of

        op( X ) = X      or
        op( X ) = X**T   or
        op( X ) = X**H,

    alpha and beta are scalars, and A, B, C, and D are matrices, with
    op( A ) an m by k matrix, op( B ) a k by n matrix and C and D are m by n matrices.

    @param[in]
    handle    rocblas_handle.
              handle to the rocblas library context queue.
    @param[in]
    transA    rocblas_operation
              specifies the form of op( A )
    @param[in]
    transB    rocblas_operation
              specifies the form of op( B )
    @param[in]
    m         rocblas_int.
              matrix dimension m
    @param[in]
    n         rocblas_int.
              matrix dimension n
    @param[in]
    k         rocblas_int.
              matrix dimension k
    @param[in]
    alpha     const void *
              specifies the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         void *
              pointer storing matrix A on the GPU.
    @param[in]
    a_type    rocblas_datatype
              specifies the datatype of matrix A
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    b         void *
              pointer storing matrix B on the GPU.
    @param[in]
    b_type    rocblas_datatype
              specifies the datatype of matrix B
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    beta      const void *
              specifies the scalar beta. Same datatype as compute_type.
    @param[in]
    c         void *
              pointer storing matrix C on the GPU.
    @param[in]
    c_type    rocblas_datatype
              specifies the datatype of matrix C
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.
    @param[out]
    d         void *
              pointer storing matrix D on the GPU.
    @param[in]
    d_type    rocblas_datatype
              specifies the datatype of matrix D
    @param[in]
    ldd       rocblas_int
              specifies the leading dimension of D.
    @param[in]
    compute_type
              rocblas_datatype
              specifies the datatype of computation
    @param[in]
    algo      rocblas_gemm_algo
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              uint32_t
              reserved for future use
    @param[in]
    flags     uint32_t
              reserved for future use
    @param[in/out]
    workspace_size   
              size_t*
              size of workspace
    @parm[in]
    workspace void*
              workspace

    ********************************************************************/

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          rocblas_int m,
                                          rocblas_int n,
                                          rocblas_int k,
                                          const void* alpha,
                                          const void* a,
                                          rocblas_datatype a_type,
                                          rocblas_int lda,
                                          const void* b,
                                          rocblas_datatype b_type,
                                          rocblas_int ldb,
                                          const void* beta,
                                          const void* c,
                                          rocblas_datatype c_type,
                                          rocblas_int ldc,
                                          void* d,
                                          rocblas_datatype d_type,
                                          rocblas_int ldd,
                                          rocblas_datatype compute_type,
                                          rocblas_gemm_algo algo,
                                          uint32_t solution_index,
                                          uint32_t flags,
                                          size_t* workspace_size,
                                          void* workspace)
{
    // handle, alpha, beta must not be null pointers for logging
    if(nullptr == handle)
    {
        return rocblas_status_invalid_handle;
    }
    if(nullptr == alpha || nullptr == beta)
    {
        return rocblas_status_invalid_pointer;
    }

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {

        double alpha_double;
        double beta_double;
        if(compute_type == rocblas_datatype_f16_r)
        {
            _Float16 alpha_half = *(static_cast<const _Float16*>(alpha));
            _Float16 beta_half  = *(static_cast<const _Float16*>(beta));
            alpha_double = static_cast<const double>(alpha_half);
            beta_double  = static_cast<const double>(beta_half);
        }
        else if(compute_type == rocblas_datatype_f32_r)
        {
            float alpha_float = *(static_cast<const float*>(alpha));
            float beta_float  = *(static_cast<const float*>(beta));
            alpha_double = static_cast<const double>(alpha_float);
            beta_double  = static_cast<const double>(beta_float);
        }
        else if(compute_type == rocblas_datatype_f64_r)
        {
            alpha_double = *(static_cast<const double*>(alpha));
            beta_double  = *(static_cast<const double*>(beta));
        }

        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  alpha_double,
                  (const void*&)a,
                  a_type,
                  lda,
                  (const void*&)b,
                  b_type,
                  ldb,
                  beta_double,
                  (const void*&)c,
                  c_type,
                  ldc,
                  (const void*&)d,
                  d_type,
                  ldd,
                  compute_type,
                  algo,
                  solution_index,
                  flags,
                  workspace_size,
                  (const void*&)workspace);


        std::string trans_a_letter = rocblas_transpose_letter(trans_a);
        std::string trans_b_letter = rocblas_transpose_letter(trans_b);
        log_bench(handle,
                  "./rocblas-bench -f gemm_ex",
                  "--transposeA",
                  trans_a_letter,
                  "--transposeB",
                  trans_b_letter,
                  "-m",
                  m,
                  "-n",
                  n,
                  "-k",
                  k,
                  "--alpha",
                  alpha_double,
                  "--a_type",
                  a_type,
                  "--lda",
                  lda,
                  "--b_type",
                  b_type,
                  "--ldb",
                  ldb,
                  "--beta",
                  beta_double,
                  "--c_type",
                  c_type,
                  "--ldc",
                  ldc,
                  "--d_type",
                  d_type,
                  "--ldd",
                  ldd,
                  "--compute_type",
                  compute_type,
                  "--algo",
                  algo,
                  "--solution_index",
                  solution_index,
                  "--flags",
                  flags,
                  "--workspace_size",
                  workspace_size);
    }
    else
    {
        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  (const void*&)alpha,
                  (const void*&)a,
                  a_type,
                  lda,
                  (const void*&)b,
                  b_type,
                  ldb,
                  (const void*&)beta,
                  (const void*&)c,
                  c_type,
                  ldc,
                  (const void*&)d,
                  d_type,
                  ldd,
                  compute_type,
                  algo,
                  solution_index,
                  flags,
                  "--workspace_size",
                  workspace_size);
    }

    // quick return m,n,k equal to 0 is valid in BLAS
    if(m == 0 || n == 0 || k == 0)
    {
        return rocblas_status_success;
    }

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0)
    {
        return rocblas_status_invalid_size;
    }

    // pointers must be valid
    if(nullptr == a || nullptr == b || nullptr == c || nullptr == d)
    {
        return rocblas_status_invalid_pointer;
    }

    rocblas_int num_rows_a = (trans_a == rocblas_operation_none) ? m : k;
    rocblas_int num_rows_b = (trans_b == rocblas_operation_none) ? k : n;
    rocblas_int num_rows_c = m;
    rocblas_int num_rows_d = m;

    // leading dimensions must be valid
    if(num_rows_a > lda || num_rows_b > ldb || num_rows_c > ldc || num_rows_d > ldd)
    {
        return rocblas_status_invalid_size;
    }

    rocblas_status rb_status = rocblas_status_internal_error;

    if(a_type == rocblas_datatype_f64_r && b_type == rocblas_datatype_f64_r &&
       c_type == rocblas_datatype_f64_r && d_type == rocblas_datatype_f64_r &&
       compute_type == rocblas_datatype_f64_r)
    {
        rb_status = tensile_gemm_typecasting<double, double>(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, d, ldd);
    }
    else if(a_type == rocblas_datatype_f32_r && b_type == rocblas_datatype_f32_r &&
            c_type == rocblas_datatype_f32_r && d_type == rocblas_datatype_f32_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        rb_status = tensile_gemm_typecasting<float, float>(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, d, ldd);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f16_r)
    {
        rb_status = tensile_gemm_typecasting<_Float16, _Float16>(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, d, ldd);
    }
    else if(a_type == rocblas_datatype_f16_r && b_type == rocblas_datatype_f16_r &&
            c_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f16_r &&
            compute_type == rocblas_datatype_f32_r)
    {
        rb_status = tensile_gemm_typecasting<_Float16, float>(
            handle, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, d, ldd);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}

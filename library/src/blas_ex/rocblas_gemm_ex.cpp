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

void device_matrix_copy(void* src,
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
                void* src_void =
                    static_cast<void*>(static_cast<uint8_t*>(src) + (i2 * ld_src * elem_size));
                void* dst_void =
                    static_cast<void*>(static_cast<uint8_t*>(dst) + (i2 * ld_dst * elem_size));

                PRINT_IF_HIP_ERROR(
                    hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice))
            }
        }
    }
}
//------------------------------------------------------------------------------
template <typename T>
TensileStatus tensile_Cijk_Ailk_Bljk_B( T * dataC, const T * dataA, const T * dataB,
    T alpha, T beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream);
template <typename T>
TensileStatus tensile_Cijk_Ailk_Bjlk_B( T * dataC, const T * dataA, const T * dataB,
    T alpha, T beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream);
template <typename T>
TensileStatus tensile_Cijk_Alik_Bljk_B( T * dataC, const T * dataA, const T * dataB,
    T alpha, T beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream);
template <typename T>
TensileStatus tensile_Cijk_Alik_Bjlk_B( T * dataC, const T * dataA, const T * dataB,
    T alpha, T beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB,
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream);
//------------------------------------------------------------------------------
template<>
TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf>(
    TensileHalf * dataC, const TensileHalf * dataA, const TensileHalf * dataB,
    TensileHalf alpha, TensileHalf beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bljk_HB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf>(
    TensileHalf * dataC, const TensileHalf * dataA, const TensileHalf * dataB,
    TensileHalf alpha, TensileHalf beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bjlk_HB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf>(
    TensileHalf * dataC, const TensileHalf * dataA, const TensileHalf * dataB,
    TensileHalf alpha, TensileHalf beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Alik_Bljk_HB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf>(
    TensileHalf * dataC, const TensileHalf * dataA, const TensileHalf * dataB,
    TensileHalf alpha, TensileHalf beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Alik_Bjlk_HB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
//------------------------------------------------------------------------------
template<>
TensileStatus tensile_Cijk_Ailk_Bljk_B<float>(
    float * dataC, const float * dataA, const float * dataB,
    float alpha, float beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bljk_SB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<float>(
    float * dataC, const float * dataA, const float * dataB,
    float alpha, float beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bjlk_SB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Alik_Bljk_B<float>(
    float * dataC, const float * dataA, const float * dataB,
    float alpha, float beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Alik_Bljk_SB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Alik_Bjlk_B<float>(
    float * dataC, const float * dataA, const float * dataB,
    float alpha, float beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Alik_Bjlk_SB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
//------------------------------------------------------------------------------
template<>
TensileStatus tensile_Cijk_Ailk_Bljk_B<double>(
    double * dataC, const double * dataA, const double * dataB,
    double alpha, double beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bljk_DB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Ailk_Bjlk_B<double>(
    double * dataC, const double * dataA, const double * dataB,
    double alpha, double beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Ailk_Bjlk_DB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Alik_Bljk_B<double>(
    double * dataC, const double * dataA, const double * dataB,
    double alpha, double beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Alik_Bljk_DB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
template<>
TensileStatus tensile_Cijk_Alik_Bjlk_B<double>(
    double * dataC, const double * dataA, const double * dataB,
    double alpha, double beta,
    unsigned int offsetC, unsigned int offsetA, unsigned int offsetB, 
    unsigned int strideC1J, unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K, 
    unsigned int strideB1J, unsigned int strideB2K,
    unsigned int sizeI, unsigned int sizeJ, unsigned int sizeK, unsigned int sizeL,
    hipStream_t stream)
{
    return tensile_Cijk_Alik_Bjlk_DB(dataC, dataA, dataB,
           alpha, beta,
           offsetC, offsetA, offsetB,
           strideC1J, strideC2K,  strideA1L, strideA2K,  strideB1J,  strideB2K,
           sizeI, sizeJ, sizeK, sizeL,
           stream, 0, nullptr, nullptr);
}
//------------------------------------------------------------------------------

template <typename T>
rocblas_status tensile_gemm( rocblas_handle handle,
                              rocblas_operation trans_a, rocblas_operation trans_b,
                              int m, int n, int k, const float* alpha,
                              const void* a, int lda,
                              const void* b, int ldb, const float* beta,
                              void* c, int ldc,
                              void* d, int ldd)
{
    T h_alpha;
    T h_beta;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // copy alpha and beta from device to host and convert type
        float h_alpha_float;
        float h_beta_float;
        hipMemcpy(&h_alpha_float, alpha, sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(&h_beta_float, beta, sizeof(float), hipMemcpyDeviceToHost);

        h_alpha = static_cast<T>(h_alpha_float);
        h_beta  = static_cast<T>(h_beta_float);
    }
    else
    {
        // convert type of alpha and beta
        h_alpha = static_cast<T>(*alpha);
        h_beta  = static_cast<T>(*beta);
    }

    TensileStatus t_status;
    rocblas_status rb_status;

    device_matrix_copy(c, ldc, d, ldd, m, n, sizeof(T));

    if((trans_a == rocblas_operation_none) && (trans_b == rocblas_operation_none))
    {
        unsigned int const stride_a = lda * k;
        unsigned int const stride_b = ldb * n;
        unsigned int const stride_d = ldd * n;
        t_status = tensile_Cijk_Ailk_Bljk_B<T>(static_cast<T*>(d),
                                           static_cast<const T*>(a),
                                           static_cast<const T*>(b),
                                             h_alpha,
                                             h_beta,
                                             0,
                                             0,
                                             0,
                                             ldd,
                                             stride_d,
                                             lda,
                                             stride_a,
                                             ldb,
                                             stride_b,
                                             m,
                                             n,
                                             1,
                                             k,
                                             handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_none) && (trans_b == rocblas_operation_transpose || trans_b == rocblas_operation_conjugate_transpose))
    {
        unsigned int const stride_a = lda * k;
        unsigned int const stride_b = ldb * k;
        unsigned int const stride_d = ldd * n;
        t_status = tensile_Cijk_Ailk_Bjlk_B<T>(static_cast<T*>(d),
                                           static_cast<const T*>(a),
                                           static_cast<const T*>(b),
                                             h_alpha,
                                             h_beta,
                                             0,
                                             0,
                                             0,
                                             ldd,
                                             stride_d,
                                             lda,
                                             stride_a,
                                             ldb,
                                             stride_b,
                                             m,
                                             n,
                                             1,
                                             k,
                                             handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose) && (trans_b == rocblas_operation_none))
    {
        unsigned int const stride_a = lda * m;
        unsigned int const stride_b = ldb * n;
        unsigned int const stride_d = ldd * n;
        t_status = tensile_Cijk_Alik_Bljk_B<T>(static_cast<T*>(d),
                                           static_cast<const T*>(a),
                                           static_cast<const T*>(b),
                                             h_alpha,
                                             h_beta,
                                             0,
                                             0,
                                             0,
                                             ldd,
                                             stride_d,
                                             lda,
                                             stride_a,
                                             ldb,
                                             stride_b,
                                             m,
                                             n,
                                             1,
                                             k,
                                             handle->rocblas_stream);
    }
    else if((trans_a == rocblas_operation_transpose || trans_a == rocblas_operation_conjugate_transpose) && (trans_b == rocblas_operation_transpose || trans_b == rocblas_operation_conjugate_transpose))
    {
        unsigned int const stride_a = lda * m;
        unsigned int const stride_b = ldb * k;
        unsigned int const stride_d = ldd * n;
        t_status = tensile_Cijk_Alik_Bjlk_B<T>(static_cast<T*>(d),
                                           static_cast<const T*>(a),
                                           static_cast<const T*>(b),
                                             h_alpha,
                                             h_beta,
                                             0,
                                             0,
                                             0,
                                             ldd,
                                             stride_d,
                                             lda,
                                             stride_a,
                                             ldb,
                                             stride_b,
                                             m,
                                             n,
                                             1,
                                             k,
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
    @param[in]
    n         rocblas_int.
    @param[in]
    k         rocblas_int.
    @param[in]
    alpha     specifies the scalar alpha.
    @param[in]
    A         pointer storing matrix A on the GPU.
    @param[in]
    Atype     rocblas_precision
              specifies the datatype of matrix A
    @param[in]
    lda       rocblas_int
              specifies the leading dimension of A.
    @param[in]
    B         pointer storing matrix B on the GPU.
    @param[in]
    Btype     rocblas_precision
              specifies the datatype of matrix B
    @param[in]
    ldb       rocblas_int
              specifies the leading dimension of B.
    @param[in]
    beta      specifies the scalar beta.
    @param[in]
    C         pointer storing matrix C on the GPU.
    @param[in]
    Ctype     rocblas_precision
              specifies the datatype of matrix C
    @param[in]
    ldc       rocblas_int
              specifies the leading dimension of C.
    @param[out]
    D         pointer storing matrix D on the GPU.
    @param[in]
    @param[in]
    Dtype     rocblas_precision
              specifies the datatype of matrix D
    ldd       rocblas_int
              specifies the leading dimension of D.
    @param[in]
    computeType
              specifies the datatype of computation
    @param[in]
    algo      rocblas_gemm_algo
              enumerant specifying the algorithm type.
    @param[in]
    kernel_index
              reserved for future use
    @param[in]
    flags     uint32_t
              reserved for future use


    ********************************************************************/

extern "C" rocblas_status rocblas_gemm_ex(rocblas_handle handle,
                                          rocblas_operation trans_a,
                                          rocblas_operation trans_b,
                                          int m,
                                          int n,
                                          int k,
                                          const float* alpha,
                                          const void* a,
                                          rocblas_precision a_type,
                                          int lda,
                                          const void* b,
                                          rocblas_precision b_type,
                                          int ldb,
                                          const float* beta,
                                          void* c,
                                          rocblas_precision c_type,
                                          int ldc,
                                          void* d,
                                          rocblas_precision d_type,
                                          int ldd,
                                          rocblas_precision compute_type,
                                          rocblas_gemm_algo algo,
                                          uint32_t kernel_index,
                                          uint32_t flags)
{
    if(nullptr == handle)
        return rocblas_status_invalid_handle;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        log_trace(handle,
                  "rocblas_gemm_ex",
                  trans_a,
                  trans_b,
                  m,
                  n,
                  k,
                  *alpha,
                  (const void*&)a,
                  a_type,
                  lda,
                  (const void*&)b,
                  b_type,
                  ldb,
                  *beta,
                  (const void*&)c,
                  c_type,
                  ldc,
                  (const void*&)d,
                  d_type,
                  ldd,
                  compute_type,
                  algo,
                  kernel_index,
                  flags);

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
                  *alpha,
                  "--a_type",
                  a_type,
                  "--lda",
                  lda,
                  "--b_type",
                  b_type,
                  "--ldb",
                  ldb,
                  "--beta",
                  *beta,
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
                  "--kernel_index",
                  kernel_index,
                  "--flags",
                  flags);
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
                  kernel_index,
                  flags);
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
    if(a == nullptr || b == nullptr || c == nullptr || d == nullptr || alpha == nullptr ||
       beta == nullptr)
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
    size_t c_byte_size;
    size_t d_byte_size;

    if(a_type == rocblas_precision_double && b_type == rocblas_precision_double &&
       c_type == rocblas_precision_double && d_type == rocblas_precision_double &&
       compute_type == rocblas_precision_double)
    {
        rb_status = tensile_gemm<double>(  handle,
                                          trans_a, trans_b,
                                          m, n, k,
                                          alpha,
                                          a, lda,
                                          b, ldb, beta,
                                          c, ldc,
                                          d, ldd);
    }
    else if(a_type == rocblas_precision_single && b_type == rocblas_precision_single &&
            c_type == rocblas_precision_single && d_type == rocblas_precision_single &&
            compute_type == rocblas_precision_single)
    {
        rb_status = tensile_gemm<float>(  handle,
                                          trans_a, trans_b,
                                          m, n, k,
                                          alpha,
                                          a, lda,
                                          b, ldb, beta,
                                          c, ldc,
                                          d, ldd);

    }
    else if(a_type == rocblas_precision_half && b_type == rocblas_precision_half &&
            c_type == rocblas_precision_half && d_type == rocblas_precision_half &&
            compute_type == rocblas_precision_half)
    {
        rb_status = tensile_gemm<_Float16>(  handle,
                                          trans_a, trans_b,
                                          m, n, k,
                                          alpha,
                                          a, lda,
                                          b, ldb, beta,
                                          c, ldc,
                                          d, ldd);
    }
    else
    {
        rb_status = rocblas_status_not_implemented;
    }

    return rb_status;
}

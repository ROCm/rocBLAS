/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#ifndef __ROCBLAS_GEMM_EX_HPP
#define __ROCBLAS_GEMM_EX_HPP

#include "Tensile.h"
#include "TensileTypes.h"
#include "gemm.hpp"
#include "handle.h"
#include "logging.h"
#include "rocblas.h"
#include "utility.h"

/////////////////
// Device Side //
/////////////////
void device_matrix_copy(const void* src,
                        rocblas_int ld_src,
                        void*       dst,
                        rocblas_int ld_dst,
                        rocblas_int n1,
                        rocblas_int n2,
                        size_t      elem_size)
{
    if(src != dst || ld_src != ld_dst) // no copy if src matrix == dst matrix
    {
        if(n1 == ld_src && n1 == ld_dst)
        {
            // src and dst matrices are contiguous, use single copy
            size_t matrix_size = n1 * n2 * elem_size;
            PRINT_IF_HIP_ERROR(hipMemcpy(dst, src, matrix_size, hipMemcpyDeviceToDevice));
        }
        else
        {
            // matrices not contiguous, one copy for each contiguous column
            size_t      column_size = n1 * elem_size;
            const void* src_void;
            void*       dst_void;

            for(int i2 = 0; i2 < n2; i2++)
            {
                src_void = static_cast<const void*>(static_cast<const uint8_t*>(src)
                                                    + (i2 * ld_src * elem_size));
                dst_void
                    = static_cast<void*>(static_cast<uint8_t*>(dst) + (i2 * ld_dst * elem_size));
                PRINT_IF_HIP_ERROR(
                    hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice));
            }
        }
    }
}

static void device_strided_batched_matrix_copy(const void*    src,
                                               rocblas_int    ld_src,
                                               rocblas_stride stride_src,
                                               void*          dst,
                                               rocblas_int    ld_dst,
                                               rocblas_stride stride_dst,
                                               rocblas_int    n1,
                                               rocblas_int    n2,
                                               rocblas_int    batch_count,
                                               size_t         elem_size)
{
    if(src != dst || ld_src != ld_dst
       || stride_src != stride_dst) // no copy if src matrix == dst matrix
    {
        const void* src_void;
        void*       dst_void;

        if(n1 == ld_src && n1 == ld_dst && stride_src == n2 * ld_src && stride_dst == n2 * ld_dst)
        {
            // src and dst batch matrices are contiguous, use single copy
            size_t matrix_size = n1 * n2 * batch_count * elem_size;
            PRINT_IF_HIP_ERROR(hipMemcpy(dst, src, matrix_size, hipMemcpyDeviceToDevice));
        }
        else if(n1 == ld_src && n1 == ld_dst)
        {
            // individual matrices in batch matrix are contiguous, one copy for each matrix
            size_t matrix_size = n1 * n2 * elem_size;
            for(int i3 = 0; i3 < batch_count; i3++)
            {
                src_void = static_cast<const void*>(static_cast<const uint8_t*>(src)
                                                    + (i3 * stride_src * elem_size));

                dst_void = static_cast<void*>(static_cast<uint8_t*>(dst)
                                              + (i3 * stride_dst * elem_size));

                PRINT_IF_HIP_ERROR(
                    hipMemcpy(dst_void, src_void, matrix_size, hipMemcpyDeviceToDevice));
            }
        }
        else
        {
            // individual matrices not contiguous, one copy for each contiguous column
            size_t      column_size = n1 * elem_size;
            const void* src_void;
            void*       dst_void;
            for(int i3 = 0; i3 < batch_count; i3++)
            {
                for(int i2 = 0; i2 < n2; i2++)
                {
                    src_void = static_cast<const void*>(static_cast<const uint8_t*>(src)
                                                        + (i2 * ld_src * elem_size)
                                                        + (i3 * stride_src * elem_size));

                    dst_void
                        = static_cast<void*>(static_cast<uint8_t*>(dst) + (i2 * ld_dst * elem_size)
                                             + (i3 * stride_dst * elem_size));

                    PRINT_IF_HIP_ERROR(
                        hipMemcpy(dst_void, src_void, column_size, hipMemcpyDeviceToDevice));
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
#define TENSILE_IN_ARGS(Ti, To, Tc)                                                              \
    To *dataD, const To *dataC, const Ti *dataA, const Ti *dataB, Tc alpha, Tc beta,             \
        unsigned int strideD1J, unsigned int strideD2K, unsigned int strideC1J,                  \
        unsigned int strideC2K, unsigned int strideA1L, unsigned int strideA2K,                  \
        unsigned int strideB1J, unsigned int strideB2K, unsigned int sizeI, unsigned int sizeJ,  \
        unsigned int sizeK, unsigned int sizeL, hipStream_t stream, unsigned int numInputEvents, \
        void *dummy1, void *dummy2

#define TENSILE_OUT_ARGS                                                                   \
    dataD, dataC, dataA, dataB, alpha, beta, strideD1J, strideD2K, strideC1J, strideC2K,   \
        strideA1L, strideA2K, strideB1J, strideB2K, sizeI, sizeJ, sizeK, sizeL, stream, 0, \
        nullptr, nullptr

// Ti is typename for input data, To is typename for output data, Tc is typename for compute
template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_Ailk_Bljk_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensileStatusFailure;
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_Ailk_Bjlk_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensileStatusFailure;
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_Alik_Bljk_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensileStatusFailure;
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_Alik_Bjlk_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensileStatusFailure;
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_Ailk_BjlkC_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensile_Cijk_Ailk_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_AlikC_Bljk_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensile_Cijk_Alik_Bljk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_Alik_BjlkC_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensile_Cijk_Alik_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_AlikC_Bjlk_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensile_Cijk_Alik_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus tensile_Cijk_AlikC_BjlkC_B(TENSILE_IN_ARGS(Ti, To, Tc))
{
    return tensile_Cijk_Alik_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
}

//----- typename_data = tensile_bfloat16 ----- typename_compute = float -----------------------
template <>
inline TensileStatus tensile_Cijk_Ailk_Bljk_B<tensile_bfloat16, tensile_bfloat16, float>(
    TENSILE_IN_ARGS(tensile_bfloat16, tensile_bfloat16, float))
{
    return tensile_Cijk_Ailk_Bljk_BBH(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Ailk_Bjlk_B<tensile_bfloat16, tensile_bfloat16, float>(
    TENSILE_IN_ARGS(tensile_bfloat16, tensile_bfloat16, float))
{
    return tensile_Cijk_Ailk_Bjlk_BBH(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bljk_B<tensile_bfloat16, tensile_bfloat16, float>(
    TENSILE_IN_ARGS(tensile_bfloat16, tensile_bfloat16, float))
{
    return tensile_Cijk_Alik_Bljk_BBH(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bjlk_B<tensile_bfloat16, tensile_bfloat16, float>(
    TENSILE_IN_ARGS(tensile_bfloat16, tensile_bfloat16, float))
{
    return tensile_Cijk_Alik_Bjlk_BBH(TENSILE_OUT_ARGS);
}

//----- typename_data = TensileHalf ----- typename_compute = float---------------------------
#define TENSILE_OUT_ARGS_HALF                                                                      \
    dataD, dataC, dataA, dataB, alpha_half, beta_half, strideD1J, strideD2K, strideC1J, strideC2K, \
        strideA1L, strideA2K, strideB1J, strideB2K, sizeI, sizeJ, sizeK, sizeL, stream, 0,         \
        nullptr, nullptr

template <>
inline TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf, TensileHalf, float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half(alpha);
    TensileHalf beta_half(beta);
    return tensile_Cijk_Ailk_Bljk_HBH(TENSILE_OUT_ARGS_HALF);
}
template <>
inline TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf, TensileHalf, float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half(alpha);
    TensileHalf beta_half(beta);
    return tensile_Cijk_Ailk_Bjlk_HBH(TENSILE_OUT_ARGS_HALF);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf, TensileHalf, float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half(alpha);
    TensileHalf beta_half(beta);
    return tensile_Cijk_Alik_Bljk_HBH(TENSILE_OUT_ARGS_HALF);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf, TensileHalf, float>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, float))
{
    //TODO: alpha and beta need to have precision equal to compute type, not data type
    TensileHalf alpha_half(alpha);
    TensileHalf beta_half(beta);
    return tensile_Cijk_Alik_Bjlk_HBH(TENSILE_OUT_ARGS_HALF);
}
#undef TENSILE_OUT_ARGS_HALF

//----- typename_data = TensileHalf ----- typename_compute = TensileHalf ---------------------
template <>
inline TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileHalf, TensileHalf, TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Ailk_Bljk_HB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileHalf, TensileHalf, TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Ailk_Bjlk_HB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bljk_B<TensileHalf, TensileHalf, TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Alik_Bljk_HB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileHalf, TensileHalf, TensileHalf>(
    TENSILE_IN_ARGS(TensileHalf, TensileHalf, TensileHalf))
{
    return tensile_Cijk_Alik_Bjlk_HB(TENSILE_OUT_ARGS);
}

//----- typename_data = float ----------- typename_compute = float ---------------------------
template <>
inline TensileStatus
    tensile_Cijk_Ailk_Bljk_B<float, float, float>(TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Ailk_Bljk_SB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus
    tensile_Cijk_Ailk_Bjlk_B<float, float, float>(TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Ailk_Bjlk_SB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_Bljk_B<float, float, float>(TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Alik_Bljk_SB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_Bjlk_B<float, float, float>(TENSILE_IN_ARGS(float, float, float))
{
    return tensile_Cijk_Alik_Bjlk_SB(TENSILE_OUT_ARGS);
}

//----- typename_data = double ---------- typename_compute = double --------------------------
template <>
inline TensileStatus
    tensile_Cijk_Ailk_Bljk_B<double, double, double>(TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Ailk_Bljk_DB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus
    tensile_Cijk_Ailk_Bjlk_B<double, double, double>(TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Ailk_Bjlk_DB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_Bljk_B<double, double, double>(TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Alik_Bljk_DB(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_Bjlk_B<double, double, double>(TENSILE_IN_ARGS(double, double, double))
{
    return tensile_Cijk_Alik_Bjlk_DB(TENSILE_OUT_ARGS);
}

//----- typename_input = int8 ---- typename_output = int ------ typename_compute = int ------------------
template <>
inline TensileStatus tensile_Cijk_Ailk_Bljk_B<TensileInt8x4, TensileInt32, TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Ailk_Bljk_4xi8BH(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Ailk_Bjlk_B<TensileInt8x4, TensileInt32, TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Ailk_Bjlk_4xi8BH(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bljk_B<TensileInt8x4, TensileInt32, TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Alik_Bljk_4xi8BH(TENSILE_OUT_ARGS);
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bjlk_B<TensileInt8x4, TensileInt32, TensileInt32>(
    TENSILE_IN_ARGS(TensileInt8x4, TensileInt32, TensileInt32))
{
    return tensile_Cijk_Alik_Bjlk_4xi8BH(TENSILE_OUT_ARGS);
}

//----- typename_data=rocblas_float_complex ---------- typename_compute = rocblas_float_complex --------------------------
#define TENSILE_COMPLEX_OUT_ARGS(Ti, To, Tc)                                             \
    (To*)dataD, (const To*)dataC, (const Ti*)dataA, (const Ti*)dataB, *((Tc*)&alpha),    \
        *((Tc*)&beta), strideD1J, strideD2K, strideC1J, strideC2K, strideA1L, strideA2K, \
        strideB1J, strideB2K, sizeI, sizeJ, sizeK, sizeL, stream, 0, nullptr, nullptr

static_assert(std::is_standard_layout<TensileComplexFloat>{},
              "TensileComplexFloat is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<TensileComplexFloat>{},
              "TensileComplexFloat is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(rocblas_float_complex) == sizeof(TensileComplexFloat),
              "TensileComplexFloat does not match public rocblas_float_complex");
template <>
inline TensileStatus
    tensile_Cijk_Ailk_Bljk_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_Ailk_Bljk_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_Ailk_Bjlk_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_Ailk_Bjlk_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_Bljk_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_Alik_Bljk_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_Bjlk_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_Alik_Bjlk_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
// Complex Conjugate
template <>
inline TensileStatus
    tensile_Cijk_Ailk_BjlkC_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_Ailk_BjlkC_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_AlikC_Bljk_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_AlikC_Bljk_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_Alik_BjlkC_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_Alik_BjlkC_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_AlikC_Bjlk_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_AlikC_Bjlk_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}
template <>
inline TensileStatus
    tensile_Cijk_AlikC_BjlkC_B<rocblas_float_complex, rocblas_float_complex, rocblas_float_complex>(
        TENSILE_IN_ARGS(rocblas_float_complex, rocblas_float_complex, rocblas_float_complex))
{
    return tensile_Cijk_AlikC_BjlkC_CB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat));
}

//----- typename_data = rocblas_double_complex ---------- typename_compute = rocblas_double_complex --------------------------
static_assert(std::is_standard_layout<TensileComplexDouble>{},
              "TensileComplexDouble is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<TensileComplexDouble>{},
              "TensileComplexDouble is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(rocblas_double_complex) == sizeof(TensileComplexDouble),
              "TensileComplexDouble does not match rocblas_double_complex");
template <>
inline TensileStatus tensile_Cijk_Ailk_Bljk_B<rocblas_double_complex,
                                              rocblas_double_complex,
                                              rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_Ailk_Bljk_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_Ailk_Bjlk_B<rocblas_double_complex,
                                              rocblas_double_complex,
                                              rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_Ailk_Bjlk_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bljk_B<rocblas_double_complex,
                                              rocblas_double_complex,
                                              rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_Alik_Bljk_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_Alik_Bjlk_B<rocblas_double_complex,
                                              rocblas_double_complex,
                                              rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_Alik_Bjlk_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
// Complex Conjugate
template <>
inline TensileStatus tensile_Cijk_Ailk_BjlkC_B<rocblas_double_complex,
                                               rocblas_double_complex,
                                               rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_Ailk_BjlkC_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_AlikC_Bljk_B<rocblas_double_complex,
                                               rocblas_double_complex,
                                               rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_AlikC_Bljk_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_Alik_BjlkC_B<rocblas_double_complex,
                                               rocblas_double_complex,
                                               rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_Alik_BjlkC_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_AlikC_Bjlk_B<rocblas_double_complex,
                                               rocblas_double_complex,
                                               rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_AlikC_Bjlk_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}
template <>
inline TensileStatus tensile_Cijk_AlikC_BjlkC_B<rocblas_double_complex,
                                                rocblas_double_complex,
                                                rocblas_double_complex>(
    TENSILE_IN_ARGS(rocblas_double_complex, rocblas_double_complex, rocblas_double_complex))
{
    return tensile_Cijk_AlikC_BjlkC_ZB(
        TENSILE_COMPLEX_OUT_ARGS(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble));
}

template <typename Ti, typename To, typename Tc>
inline TensileStatus call_tensile_ex(To*            dataD,
                                     const To*      dataC,
                                     const Ti*      dataA,
                                     const Ti*      dataB,
                                     Tc             alpha,
                                     Tc             beta,
                                     unsigned int   strideD1J,
                                     unsigned int   strideD2K,
                                     unsigned int   strideC1J,
                                     unsigned int   strideC2K,
                                     unsigned int   strideA1L,
                                     unsigned int   strideA2K,
                                     unsigned int   strideB1J,
                                     unsigned int   strideB2K,
                                     unsigned int   sizeI,
                                     unsigned int   sizeJ,
                                     unsigned int   sizeK,
                                     unsigned int   sizeL,
                                     hipStream_t    stream,
                                     transpose_mode transposeMode)
{
    switch(transposeMode)
    {
    case NN:
        return tensile_Cijk_Ailk_Bljk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case NT:
        return tensile_Cijk_Ailk_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case NC:
        return tensile_Cijk_Ailk_BjlkC_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case TN:
        return tensile_Cijk_Alik_Bljk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case CN:
        return tensile_Cijk_AlikC_Bljk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case TT:
        return tensile_Cijk_Alik_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case TC:
        return tensile_Cijk_Alik_BjlkC_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case CT:
        return tensile_Cijk_AlikC_Bjlk_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    case CC:
        return tensile_Cijk_AlikC_BjlkC_B<Ti, To, Tc>(TENSILE_OUT_ARGS);
    }

    return tensileStatusFailure;
}

#undef TENSILE_COMPLEX_OUT_ARGS
#undef TENSILE_IN_ARGS
#undef TENSILE_OUT_ARGS

//------------------------------------------------------------------------------

///////////////
// Host Side //
///////////////

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_handle_transpose(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        unsigned int      m,
                                        unsigned int      n,
                                        unsigned int      k,
                                        const Tc*         alpha,
                                        rocblas_stride    stride_alpha,
                                        const Ti*         a[],
                                        rocblas_int       offset_a,
                                        unsigned int      lda,
                                        unsigned int      stride_a,
                                        const Ti*         b[],
                                        rocblas_int       offset_b,
                                        unsigned int      ldb,
                                        unsigned int      stride_b,
                                        const Tc*         beta,
                                        rocblas_stride    stride_beta,
                                        const To*         c[],
                                        rocblas_int       offset_c,
                                        unsigned int      ldc,
                                        unsigned int      stride_c,
                                        To*               d[],
                                        rocblas_int       offset_d,
                                        unsigned int      ldd,
                                        unsigned int      stride_d,
                                        unsigned int      batch_count)
{
    // BATCHED VERSION
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
    stride_a = trans_a == rocblas_operation_none ? lda * k : lda * m;
    stride_b = trans_b == rocblas_operation_none ? ldb * n : ldb * k;
    stride_c = ldc * n;
    stride_d = ldd * n;

    rocblas_status status = rocblas_status_internal_error;
    for(int bi = 0; bi < batch_count; bi++)
    {
        // Tensile does not support batched gemm_ex yet, must do naive version
        status = gemm_ex_handle_transpose(handle,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          alpha + bi * stride_alpha,
                                          0, // using single alpha ^
                                          hostA[bi],
                                          offset_a,
                                          lda,
                                          stride_a,
                                          hostB[bi],
                                          offset_b,
                                          ldb,
                                          stride_b,
                                          beta + bi * stride_beta,
                                          0, // see ^
                                          hostC[bi],
                                          offset_c,
                                          ldc,
                                          stride_c,
                                          hostD[bi],
                                          offset_d,
                                          ldd,
                                          stride_d,
                                          1);
        if(status != rocblas_status_success)
            return status;
    }
    return status;
}

template <typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_handle_transpose(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        unsigned int      m,
                                        unsigned int      n,
                                        unsigned int      k,
                                        const Tc*         alpha,
                                        rocblas_stride    stride_alpha,
                                        const Ti*         a,
                                        rocblas_int       offset_a,
                                        unsigned int      lda,
                                        unsigned int      stride_a,
                                        const Ti*         b,
                                        rocblas_int       offset_b,
                                        unsigned int      ldb,
                                        unsigned int      stride_b,
                                        const Tc*         beta,
                                        rocblas_stride    stride_beta,
                                        const To*         c,
                                        rocblas_int       offset_c,
                                        unsigned int      ldc,
                                        unsigned int      stride_c,
                                        To*               d,
                                        rocblas_int       offset_d,
                                        unsigned int      ldd,
                                        unsigned int      stride_d,
                                        unsigned int      batch_count)
{
    a += offset_a;
    b += offset_b;
    c += offset_c;
    d += offset_d;

    TensileStatus  t_status;
    rocblas_status rb_status;

    static const bool arch_lt906 = handle->device_arch_id() < 906;
    const To*         c_in;
    unsigned int      ldi, stride_i;

    if(!arch_lt906 && (std::is_same<Ti, float>{} || std::is_same<Ti, double>{})
       && ((ldc >= ldd && stride_c >= stride_d && m == ldd)
           || (ldc == ldd && stride_c == stride_d)))
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

    if(stride_alpha != 0 || stride_beta != 0)
    {
        for(int bi = 0; bi < batch_count; bi++)
        {
            t_status = call_tensile_ex<Ti, To, Tc>((To*)(d + bi * stride_d),
                                                   (const To*)(c_in + bi * stride_c),
                                                   (const Ti*)(a + bi * stride_a),
                                                   (const Ti*)(b + bi * stride_b),
                                                   alpha[bi * stride_alpha],
                                                   beta[bi * stride_beta],
                                                   unsigned(ldd),
                                                   stride_d,
                                                   unsigned(ldi),
                                                   stride_i,
                                                   unsigned(lda),
                                                   stride_a,
                                                   unsigned(ldb),
                                                   stride_b,
                                                   unsigned(m),
                                                   unsigned(n),
                                                   unsigned(1),
                                                   unsigned(k),
                                                   handle->rocblas_stream,
                                                   GetTransposeMode(trans_a, trans_b));
        }
    }
    else
    {
        // single alpha/beta
        t_status = call_tensile_ex<Ti, To, Tc>((To*)d,
                                               (const To*)c_in,
                                               (const Ti*)a,
                                               (const Ti*)b,
                                               *alpha,
                                               *beta,
                                               unsigned(ldd),
                                               stride_d,
                                               unsigned(ldi),
                                               stride_i,
                                               unsigned(lda),
                                               stride_a,
                                               unsigned(ldb),
                                               stride_b,
                                               unsigned(m),
                                               unsigned(n),
                                               unsigned(batch_count),
                                               unsigned(k),
                                               handle->rocblas_stream,
                                               GetTransposeMode(trans_a, trans_b));
    }

    rb_status = (t_status == tensileStatusSuccess) ? rocblas_status_success
                                                   : rocblas_status_internal_error;
    return rb_status;
}

#if defined(USE_CHUNKING)

template <typename Ti, typename To, typename To2, typename Tc>
rocblas_status gemm_ex_chunking(rocblas_handle               handle,
                                rocblas_operation            trans_a,
                                rocblas_operation            trans_b,
                                unsigned int                 m,
                                unsigned int                 n,
                                unsigned int                 k,
                                Tc*                          alpha,
                                unsigned int stride_alpha Ti a,
                                unsigned int                 offsetAin,
                                unsigned int                 lda,
                                unsigned int                 stride_a,
                                Ti                           b,
                                unsigned int                 offsetBin,
                                unsigned int                 ldb,
                                unsigned int                 stride_b,
                                Tc*                          beta,
                                unsigned int                 stride_beta,
                                To                           c,
                                unsigned int                 offsetCin,
                                unsigned int                 ldc,
                                unsigned int                 stride_c,
                                To2                          d,
                                unsigned int                 offsetDin,
                                unsigned int                 ldd,
                                unsigned int                 stride_d,
                                unsigned int                 batch_count)
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

            status = gemm_ex_handle_transpose(handle,
                                              trans_a,
                                              trans_b,
                                              m_chunk_size_corrected,
                                              n_chunk_size_corrected,
                                              k,
                                              alpha,
                                              stride_alpha,
                                              a,
                                              a_offset + offsetAin,
                                              lda,
                                              stride_a,
                                              b,
                                              b_offset + offsetBin,
                                              ldb,
                                              stride_b,
                                              beta,
                                              stride_beta,
                                              c,
                                              c_offset + offsetCin,
                                              ldc,
                                              stride_c,
                                              d,
                                              d_offset + offsetDin,
                                              ldd,
                                              stride_d,
                                              batch_count);

            if(status != rocblas_status_success)
                return_status = status;
        }
    }
    return return_status;
}
#else
#define gemm_ex_chunking gemm_ex_handle_transpose
#endif // defined(USE_CHUNKING)

template <bool BATCHED, typename Ti, typename To, typename Tc>
rocblas_status gemm_ex_typecasting(rocblas_handle    handle,
                                   rocblas_operation trans_a,
                                   rocblas_operation trans_b,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const void*       alpha,
                                   rocblas_stride    stride_alpha,
                                   const void*       a,
                                   rocblas_int       offsetAin,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const void*       b,
                                   rocblas_int       offsetBin,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   const void*       beta,
                                   rocblas_stride    stride_beta,
                                   const void*       c,
                                   rocblas_int       offsetCin,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_c,
                                   void*             d,
                                   rocblas_int       offsetDin,
                                   rocblas_int       ldd,
                                   rocblas_stride    stride_d,
                                   rocblas_int       batch_count)
{
    Tc h_alpha[batch_count];
    Tc h_beta[batch_count];

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        // copy alpha and beta from device to host and convert type
        for(int b = 0; b < batch_count; b++)
            hipMemcpy(
                &h_alpha[b], (Tc*)alpha + b * stride_alpha, sizeof(Tc), hipMemcpyDeviceToHost);

        for(int b = 0; b < batch_count; b++)
            hipMemcpy(&h_beta[b], (Tc*)beta + b * stride_beta, sizeof(Tc), hipMemcpyDeviceToHost);
    }
    else
    {
        for(int b = 0; b < batch_count; b++)
            h_alpha[b] = *(((const Tc*)alpha) + b * stride_alpha);
        for(int b = 0; b < batch_count; b++)
            h_beta[b] = *(((const Tc*)beta) + b * stride_beta);
    }

    // check alignment of pointers before casting
    if(BATCHED)
    {
        if(!isAligned(a, sizeof(Ti*)) || !isAligned(b, sizeof(Ti*)) || !isAligned(c, sizeof(To*))
           || !isAligned(d, sizeof(To*)))
            return rocblas_status_invalid_size;

        // Pass alpha and beta as simple array (stride of 1)
        return gemm_ex_chunking(handle,
                                trans_a,
                                trans_b,
                                unsigned(m),
                                unsigned(n),
                                unsigned(k),
                                h_alpha,
                                1, // using stride of 1 for alpha
                                (const Ti**)a,
                                unsigned(offsetAin),
                                unsigned(lda),
                                unsigned(stride_a),
                                (const Ti**)b,
                                unsigned(offsetBin),
                                unsigned(ldb),
                                unsigned(stride_b),
                                h_beta,
                                1, // using stride of 1 for beta
                                (const To**)c,
                                unsigned(offsetCin),
                                unsigned(ldc),
                                unsigned(stride_c),
                                (To**)d,
                                unsigned(offsetDin),
                                unsigned(ldd),
                                unsigned(stride_d),
                                unsigned(batch_count));
    }
    else
    {
        if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(To))
           || !isAligned(d, sizeof(To)))
            return rocblas_status_invalid_size;

        return gemm_ex_chunking(handle,
                                trans_a,
                                trans_b,
                                unsigned(m),
                                unsigned(n),
                                unsigned(k),
                                h_alpha,
                                1,
                                (const Ti*)a,
                                unsigned(offsetAin),
                                unsigned(lda),
                                unsigned(stride_a),
                                (const Ti*)b,
                                unsigned(offsetBin),
                                unsigned(ldb),
                                unsigned(stride_b),
                                h_beta,
                                1,
                                (const To*)c,
                                unsigned(offsetCin),
                                unsigned(ldc),
                                unsigned(stride_c),
                                (To*)d,
                                unsigned(offsetDin),
                                unsigned(ldd),
                                unsigned(stride_d),
                                unsigned(batch_count));
    }
}

#endif

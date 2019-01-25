/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef UTILITY_H
#define UTILITY_H
#include "rocblas.h"

#ifndef GOOGLE_TEST

// Load a scalar. If the argument is a pointer, dereference it; otherwise copy
// it. Allows the same kernels to be used for host and device scalars.

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(const T* xp)
{
    return *xp;
}

// For rocblas_half2, we broadcast a fp16 across two halves
template <>
__forceinline__ __device__ __host__ rocblas_half2 load_scalar(const rocblas_half2* xp)
{
    auto x = *reinterpret_cast<const _Float16*>(xp);
    return {x, x};
}

#endif // GOOGLE_TEST

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

// clang-format off
// return letter N,T,C in place of rocblas_operation enum
constexpr auto rocblas_transpose_letter(rocblas_operation trans)
{
    switch(trans)
    {
    case rocblas_operation_none:                return 'N';
    case rocblas_operation_transpose:           return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    default:                                    return ' ';
    }
}

// return letter L, R, B in place of rocblas_side enum
constexpr auto rocblas_side_letter(rocblas_side side)
{
    switch(side)
    {
    case rocblas_side_left:  return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both:  return 'B';
    default:                 return ' ';
    }
}

// return letter U, L, B in place of rocblas_fill enum
constexpr auto rocblas_fill_letter(rocblas_fill fill)
{
    switch(fill)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full:  return 'F';
    default:                 return ' ';
    }
}

// return letter N, U in place of rocblas_diagonal enum
constexpr auto rocblas_diag_letter(rocblas_diagonal diag)
{
    switch(diag)
    {
    case rocblas_diagonal_non_unit: return 'N';
    case rocblas_diagonal_unit:     return 'U';
    default:                        return ' ';
    }
}

// return precision string for rocblas_datatype
constexpr auto rocblas_datatype_string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r: return "f16_r";
    case rocblas_datatype_f32_r: return "f32_r";
    case rocblas_datatype_f64_r: return "f64_r";
    case rocblas_datatype_f16_c: return "f16_k";
    case rocblas_datatype_f32_c: return "f32_c";
    case rocblas_datatype_f64_c: return "f64_c";
    case rocblas_datatype_i8_r:  return "i8_r";
    case rocblas_datatype_u8_r:  return "u8_r";
    case rocblas_datatype_i32_r: return "i32_r";
    case rocblas_datatype_u32_r: return "u32_r";
    case rocblas_datatype_i8_c:  return "i8_c";
    case rocblas_datatype_u8_c:  return "u8_c";
    case rocblas_datatype_i32_c: return "i32_c";
    case rocblas_datatype_u32_c: return "u32_c";
    default:                     return "invalid";
    }
}

// return precision string for data type
template <typename> constexpr char rocblas_precision_string                [] = "invalid";
template <> constexpr char rocblas_precision_string<rocblas_half          >[] = "f16_r";
template <> constexpr char rocblas_precision_string<float                 >[] = "f32_r";
template <> constexpr char rocblas_precision_string<double                >[] = "f64_r";
template <> constexpr char rocblas_precision_string<int8_t                >[] = "i8_r";
template <> constexpr char rocblas_precision_string<uint8_t               >[] = "u8_r";
template <> constexpr char rocblas_precision_string<int32_t               >[] = "i32_r";
template <> constexpr char rocblas_precision_string<uint32_t              >[] = "u32_r";
template <> constexpr char rocblas_precision_string<rocblas_float_complex >[] = "f32_c";
template <> constexpr char rocblas_precision_string<rocblas_double_complex>[] = "f64_c";
#if 0 // Not implemented
template <> constexpr char rocblas_precision_string<rocblas_half_complex  >[] = "f16_c";
template <> constexpr char rocblas_precision_string<rocblas_i8_complex    >[] = "i8_c";
template <> constexpr char rocblas_precision_string<rocblas_u8_complex    >[] = "u8_c";
template <> constexpr char rocblas_precision_string<rocblas_i32_complex   >[] = "i32_c";
template <> constexpr char rocblas_precision_string<rocblas_u32_complex   >[] = "u32_c";
#endif

// clang-format on
#endif

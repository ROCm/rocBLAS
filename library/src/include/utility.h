/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef UTILITY_H
#define UTILITY_H
#include "definitions.h"
#include "rocblas.h"
#include <complex>
#include <hip/hip_runtime.h>
#include <type_traits>

#pragma STDC CX_LIMITED_RANGE ON

// half vectors
typedef _Float16 rocblas_half8 __attribute__((ext_vector_type(8)));
typedef _Float16 rocblas_half2 __attribute__((ext_vector_type(2)));

#ifndef GOOGLE_TEST
extern "C" __device__ rocblas_half2 llvm_fma_v2f16(rocblas_half2,
                                                   rocblas_half2,
                                                   rocblas_half2) __asm("llvm.fma.v2f16");

__device__ inline rocblas_half2
    rocblas_fmadd_half2(rocblas_half2 multiplier, rocblas_half2 multiplicand, rocblas_half2 addend)
{
    return llvm_fma_v2f16(multiplier, multiplicand, addend);
}

// Conjugate a value. For most types, simply return argument; for
// rocblas_float_complex and rocblas_double_complex, return std::conj(z)
template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
__device__ __host__ inline auto conj(const T& z)
{
    return z;
}

template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
__device__ __host__ inline auto conj(const T& z)
{
    return std::conj(z);
}

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
    case rocblas_datatype_f16_r:  return "f16_r";
    case rocblas_datatype_f32_r:  return "f32_r";
    case rocblas_datatype_f64_r:  return "f64_r";
    case rocblas_datatype_f16_c:  return "f16_c";
    case rocblas_datatype_f32_c:  return "f32_c";
    case rocblas_datatype_f64_c:  return "f64_c";
    case rocblas_datatype_i8_r:   return "i8_r";
    case rocblas_datatype_u8_r:   return "u8_r";
    case rocblas_datatype_i32_r:  return "i32_r";
    case rocblas_datatype_u32_r:  return "u32_r";
    case rocblas_datatype_i8_c:   return "i8_c";
    case rocblas_datatype_u8_c:   return "u8_c";
    case rocblas_datatype_i32_c:  return "i32_c";
    case rocblas_datatype_u32_c:  return "u32_c";
    case rocblas_datatype_bf16_r: return "bf16_r";
    case rocblas_datatype_bf16_c: return "bf16_c";
    default:                      return "invalid";
    }
}

// return sizeof rocblas_datatype
constexpr size_t rocblas_sizeof_datatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r: return 2;
    case rocblas_datatype_f32_r: return 4;
    case rocblas_datatype_f64_r: return 8;
    case rocblas_datatype_f16_c: return 4;
    case rocblas_datatype_f32_c: return 8;
    case rocblas_datatype_f64_c: return 16;
    case rocblas_datatype_i8_r:  return 1;
    case rocblas_datatype_u8_r:  return 1;
    case rocblas_datatype_i32_r: return 4;
    case rocblas_datatype_u32_r: return 4;
    case rocblas_datatype_i8_c:  return 2;
    case rocblas_datatype_u8_c:  return 2;
    case rocblas_datatype_i32_c: return 8;
    case rocblas_datatype_u32_c: return 8;
    default:                     return 0;
    }
}

// return rocblas_datatype from type
template <typename> static constexpr rocblas_datatype rocblas_datatype_from_type     = rocblas_datatype(-1);
template <> static constexpr auto rocblas_datatype_from_type<rocblas_half>           = rocblas_datatype_f16_r;
template <> static constexpr auto rocblas_datatype_from_type<float>                  = rocblas_datatype_f32_r;
template <> static constexpr auto rocblas_datatype_from_type<double>                 = rocblas_datatype_f64_r;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_float_complex>  = rocblas_datatype_f32_c;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_double_complex> = rocblas_datatype_f64_c;
template <> static constexpr auto rocblas_datatype_from_type<int8_t>                 = rocblas_datatype_i8_r;
template <> static constexpr auto rocblas_datatype_from_type<uint8_t>                = rocblas_datatype_u8_r;
template <> static constexpr auto rocblas_datatype_from_type<int32_t>                = rocblas_datatype_i32_r;
template <> static constexpr auto rocblas_datatype_from_type<uint32_t>               = rocblas_datatype_u32_r;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_bfloat16>       = rocblas_datatype_bf16_r;

// return precision string for data type
template <typename> static constexpr char rocblas_precision_string                [] = "invalid";
template <> static constexpr char rocblas_precision_string<rocblas_half          >[] = "f16_r";
template <> static constexpr char rocblas_precision_string<float                 >[] = "f32_r";
template <> static constexpr char rocblas_precision_string<double                >[] = "f64_r";
template <> static constexpr char rocblas_precision_string<int8_t                >[] = "i8_r";
template <> static constexpr char rocblas_precision_string<uint8_t               >[] = "u8_r";
template <> static constexpr char rocblas_precision_string<int32_t               >[] = "i32_r";
template <> static constexpr char rocblas_precision_string<uint32_t              >[] = "u32_r";
template <> static constexpr char rocblas_precision_string<rocblas_float_complex >[] = "f32_c";
template <> static constexpr char rocblas_precision_string<rocblas_double_complex>[] = "f64_c";
#if 0 // Not implemented
template <> static constexpr char rocblas_precision_string<rocblas_half_complex  >[] = "f16_c";
template <> static constexpr char rocblas_precision_string<rocblas_i8_complex    >[] = "i8_c";
template <> static constexpr char rocblas_precision_string<rocblas_u8_complex    >[] = "u8_c";
template <> static constexpr char rocblas_precision_string<rocblas_i32_complex   >[] = "i32_c";
template <> static constexpr char rocblas_precision_string<rocblas_u32_complex   >[] = "u32_c";
#endif

// clang-format on

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from those calls
 ******************************************************************************/
constexpr auto get_rocblas_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess:
        return rocblas_status_success;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return rocblas_status_memory_error;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return rocblas_status_invalid_pointer;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return rocblas_status_invalid_handle;

    // library using hip incorrectly
    case hipErrorInvalidValue:
        return rocblas_status_internal_error;

    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default:
        return rocblas_status_internal_error;
    }
}
#endif

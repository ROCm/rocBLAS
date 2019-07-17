/* ************************************************************************
 * Copyright 2016-2019 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 * \brief rocblas-types.h defines data types used by rocblas
 */

#pragma once
#ifndef _ROCBLAS_TYPES_H_
#define _ROCBLAS_TYPES_H_

#include "rocblas_bfloat16.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*! \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblas_destroy_handle().
 */
typedef struct _rocblas_handle* rocblas_handle;

// Forward declaration of hipStream_t
typedef struct ihipStream_t* hipStream_t;

// integer types
/*! \brief To specify whether int32 or int64 is used
 */
#if defined(rocblas_ILP64)
typedef int64_t rocblas_int;
typedef int64_t rocblas_long;
#else
typedef int32_t rocblas_int;
typedef int64_t rocblas_long;
#endif

// floating point types
typedef float    rocblas_float;
typedef double   rocblas_double;
typedef uint16_t rocblas_half; // TODO: should be replaced with a struct, to become a unique type

// complex types
#include "rocblas-complex-types.h"

/* ============================================================================================ */

/*! parameter constants.
 *  numbering is consistent with CBLAS, ACML and most standard C BLAS libraries
 */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Used to specify whether the matrix is to be transposed or not. */
typedef enum rocblas_operation_
{
    rocblas_operation_none      = 111, /**< Operate with the matrix. */
    rocblas_operation_transpose = 112, /**< Operate with the transpose of the matrix. */
    rocblas_operation_conjugate_transpose
    = 113 /**< Operate with the conjugate transpose of the matrix. */
} rocblas_operation;

/*! \brief Used by the Hermitian, symmetric and triangular matrix
 * routines to specify whether the upper or lower triangle is being referenced.
 */
typedef enum rocblas_fill_
{
    rocblas_fill_upper = 121, /**< Upper triangle. */
    rocblas_fill_lower = 122, /**< Lower triangle. */
    rocblas_fill_full  = 123
} rocblas_fill;

/*! \brief It is used by the triangular matrix routines to specify whether the
 * matrix is unit triangular.
 */
typedef enum rocblas_diagonal_
{
    rocblas_diagonal_non_unit = 131, /**< Non-unit triangular. */
    rocblas_diagonal_unit     = 132, /**< Unit triangular. */
} rocblas_diagonal;

/*! \brief Indicates the side matrix A is located relative to matrix B during multiplication. */
typedef enum rocblas_side_
{
    rocblas_side_left  = 141, /**< Multiply general matrix by symmetric,
                        Hermitian or triangular matrix on the left. */
    rocblas_side_right = 142, /**< Multiply general matrix by symmetric,
                        Hermitian or triangular matrix on the right. */
    rocblas_side_both  = 143
} rocblas_side;

/* ============================================================================================ */
/**
 *   @brief rocblas status codes definition
 */
typedef enum rocblas_status_
{
    rocblas_status_success         = 0, /**< success */
    rocblas_status_invalid_handle  = 1, /**< handle not initialized, invalid or null */
    rocblas_status_not_implemented = 2, /**< function is not implemented */
    rocblas_status_invalid_pointer = 3, /**< invalid pointer parameter */
    rocblas_status_invalid_size    = 4, /**< invalid size parameter */
    rocblas_status_memory_error    = 5, /**< failed internal memory allocation, copy or dealloc */
    rocblas_status_internal_error  = 6, /**< other internal library failure */
    rocblas_status_perf_degraded   = 7, /**< performance degraded due to low device memory */
    rocblas_status_size_query_mismatch = 8, /**< unmatched start/stop size query */
    rocblas_status_size_increased      = 9, /**< queried device memory size increased */
    rocblas_status_size_unchanged      = 10, /**< queried device memory size unchanged */
} rocblas_status;

/*! \brief Indicates the precision width of data stored in a blas type. */
typedef enum rocblas_datatype_
{
    rocblas_datatype_f16_r  = 150, /**< 16 bit floating point, real */
    rocblas_datatype_f32_r  = 151, /**< 32 bit floating point, real */
    rocblas_datatype_f64_r  = 152, /**< 64 bit floating point, real */
    rocblas_datatype_f16_c  = 153, /**< 16 bit floating point, complex */
    rocblas_datatype_f32_c  = 154, /**< 32 bit floating point, complex */
    rocblas_datatype_f64_c  = 155, /**< 64 bit floating point, complex */
    rocblas_datatype_i8_r   = 160, /**<  8 bit signed integer, real */
    rocblas_datatype_u8_r   = 161, /**<  8 bit unsigned integer, real */
    rocblas_datatype_i32_r  = 162, /**< 32 bit signed integer, real */
    rocblas_datatype_u32_r  = 163, /**< 32 bit unsigned integer, real */
    rocblas_datatype_i8_c   = 164, /**<  8 bit signed integer, complex */
    rocblas_datatype_u8_c   = 165, /**<  8 bit unsigned integer, complex */
    rocblas_datatype_i32_c  = 166, /**< 32 bit signed integer, complex */
    rocblas_datatype_u32_c  = 167, /**< 32 bit unsigned integer, complex */
    rocblas_datatype_bf16_r = 168, /**< 16 bit bfloat, real */
    rocblas_datatype_bf16_c = 169, /**< 16 bit bfloat, complex */
} rocblas_datatype;

/*! \brief Indicates the pointer is device pointer or host pointer */
typedef enum rocblas_pointer_mode_
{
    rocblas_pointer_mode_host   = 0,
    rocblas_pointer_mode_device = 1
} rocblas_pointer_mode;

/*! \brief Indicates if layer is active with bitmask*/
typedef enum rocblas_layer_mode_
{
    rocblas_layer_mode_none        = 0b0000000000,
    rocblas_layer_mode_log_trace   = 0b0000000001,
    rocblas_layer_mode_log_bench   = 0b0000000010,
    rocblas_layer_mode_log_profile = 0b0000000100,
} rocblas_layer_mode;

/*! \brief Indicates if layer is active with bitmask*/
typedef enum rocblas_gemm_algo_
{
    rocblas_gemm_algo_standard = 0b0000000000,
} rocblas_gemm_algo;

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 * \brief rocblas-types.h defines data types used by rocblas
 */

#ifndef _ROCBLAS_TYPES_H_
#define _ROCBLAS_TYPES_H_

// Request _Float16 type extension
#define __STDC_WANT_IEC_60559_TYPES_EXT__ 1

#include "rocblas_bfloat16.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ROCBLAS_EXPORT_NOINLINE __attribute__((visibility("default"))) __attribute__((noinline))

/*! \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocblas_destroy_handle().
 */
typedef struct _rocblas_handle* rocblas_handle;

// Forward declaration of hipStream_t
typedef struct ihipStream_t* hipStream_t;

// Forward declaration of hipEvent_t
typedef struct ihipEvent_t* hipEvent_t;

// integer types
// /*! \brief To specify whether int32 is used for LP64 or int64 is used for ILP64
//  */
#if defined(rocblas_ILP64)
typedef int64_t rocblas_int;
#else
typedef int32_t rocblas_int;
#endif

// /*! \brief Stride between matrices or vectors in strided_batched functions
//  */
#if defined(rocblas_ILP64)
typedef int64_t rocblas_stride;
#else
typedef int64_t rocblas_stride;
#endif

// floating point types
typedef float  rocblas_float;
typedef double rocblas_double;

#ifdef ROCM_USE_FLOAT16
typedef _Float16 rocblas_half;
#else
typedef struct rocblas_half
{
    uint16_t data;
} rocblas_half;
#endif

// complex types
#include "rocblas-complex-types.h"

/* ============================================================================================ */

/*! parameter constants.
 *  numbering is consistent with CBLAS, ACML and most standard C BLAS libraries
 */

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
    rocblas_status_invalid_pointer = 3, /**< invalid pointer argument */
    rocblas_status_invalid_size    = 4, /**< invalid size argument */
    rocblas_status_memory_error    = 5, /**< failed internal memory allocation, copy or dealloc */
    rocblas_status_internal_error  = 6, /**< other internal library failure */
    rocblas_status_perf_degraded   = 7, /**< performance degraded due to low device memory */
    rocblas_status_size_query_mismatch = 8, /**< unmatched start/stop size query */
    rocblas_status_size_increased      = 9, /**< queried device memory size increased */
    rocblas_status_size_unchanged      = 10, /**< queried device memory size unchanged */
    rocblas_status_invalid_value       = 11, /**< passed argument not valid */
    rocblas_status_continue            = 12, /**< nothing preventing function to proceed */
} rocblas_status;

/*! \brief Indicates the precision width of data stored in a blas type. */
typedef enum rocblas_datatype_
#if __cplusplus >= 201103L
    : int
#endif
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

/*! \brief Indicates if scalar pointers are on host or device. This is used for
*    scalars alpha and beta and for scalar function return values. */
typedef enum rocblas_pointer_mode_
{
    /*! \brief Scalar values affected by this variable will be located on the host. */
    rocblas_pointer_mode_host = 0,
    /*! \brief Scalar values affected by this variable will be located on the device. */
    rocblas_pointer_mode_device = 1
} rocblas_pointer_mode;

/*! \brief Indicates if layer is active with bitmask*/
typedef enum rocblas_layer_mode_
{
    /*! \brief No logging will take place. */
    rocblas_layer_mode_none = 0b0000000000,
    /*! \brief A line containing the function name and value of arguments passed will be printed with each rocBLAS function call. */
    rocblas_layer_mode_log_trace = 0b0000000001,
    /*! \brief Outputs a line each time a rocBLAS function is called, this line can be used with rocblas-bench to make the same call again. */
    rocblas_layer_mode_log_bench = 0b0000000010,
    /*! \brief Outputs a YAML description of each rocBLAS function called, along with its arguments and number of times it was called. */
    rocblas_layer_mode_log_profile = 0b0000000100,
} rocblas_layer_mode;

/*! \brief Indicates if layer is active with bitmask*/
typedef enum rocblas_gemm_algo_
{
    rocblas_gemm_algo_standard = 0b0000000000,
} rocblas_gemm_algo;

#endif

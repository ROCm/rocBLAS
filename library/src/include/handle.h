/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef HANDLE_H
#define HANDLE_H
#include <hip/hip_runtime_api.h>
#include <fstream>

#include "rocblas.h"

/*******************************************************************************
 * \brief rocblas_handle is a structure holding the rocblas library context.
 * It must be initialized using rocblas_create_handle() and the returned handle mus
 * It should be destroyed at the end using rocblas_destroy_handle().
 * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
******************************************************************************/
struct _rocblas_handle
{

    _rocblas_handle();
    ~_rocblas_handle();

    rocblas_status set_stream(hipStream_t stream);
    rocblas_status get_stream(hipStream_t* stream) const;

    void* get_trsm_Y();
    void* get_trsm_invA();
    void* get_trsm_invA_C();

    void* get_trsv_x();
    void* get_trsv_alpha();

    rocblas_int device;
    hipDeviceProp_t device_properties;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // default logging_mode is no logging
    rocblas_layer_mode layer_mode;

    // space allocated for trsm
    void* trsm_Y      = nullptr;
    void* trsm_invA   = nullptr;
    void* trsm_invA_C = nullptr;

    // space allocated for trsv
    void* trsv_x     = nullptr;
    void* trsv_alpha = nullptr;

    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ostream* log_trace_os;
    std::ostream* log_bench_os;
};

// work buffer size constants
#define WORKBUF_TRSM_A_BLKS 10
#define WORKBUF_TRSM_B_CHNK 32000
#define WORKBUF_TRSM_Y_SZ (32000 * 128 * sizeof(double))
#define WORKBUF_TRSM_INVA_SZ (128 * 128 * 10 * sizeof(double))
#define WORKBUF_TRSM_INVA_C_SZ (128 * 128 * 10 * sizeof(double) / 2)
#define WORKBUF_TRSV_X_SZ (131072 * sizeof(double))
#define WORKBUF_TRSV_ALPHA_SZ (1 * sizeof(double))

#endif

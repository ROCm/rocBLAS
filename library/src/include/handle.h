/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HANDLE_H
#define HANDLE_H

#include <fstream>
#include <iostream>
#include "rocblas.h"
#include "definitions.h"
#include <hip/hip_runtime_api.h>

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

    /*******************************************************************************
     * Exactly like CUBLAS, ROCBLAS only uses one stream for one API routine
     ******************************************************************************/

    /*******************************************************************************
     * set stream:
        This API assumes user has already created a valid stream
        Associate the following rocblas API call with this user provided stream
     ******************************************************************************/
    rocblas_status set_stream(hipStream_t user_stream)
    {
        // TODO: check the user_stream valid or not
        rocblas_stream = user_stream;
        return rocblas_status_success;
    }

    /*******************************************************************************
     * get stream
     ******************************************************************************/
    rocblas_status get_stream(hipStream_t* stream) const
    {
        *stream = rocblas_stream;
        return rocblas_status_success;
    }

    // trsm get pointers
    void* get_trsm_Y() const { return trsm_Y; }
    void* get_trsm_invA() const { return trsm_invA; }
    void* get_trsm_invA_C() const { return trsm_invA_C; }

    // trsv get pointers
    void* get_trsv_x() const { return trsv_x; }
    void* get_trsv_alpha() const { return trsv_alpha; }

    rocblas_int device;
    hipDeviceProp_t device_properties;

    // rocblas by default take the system default stream 0 users cannot create
    hipStream_t rocblas_stream = 0;

    // default pointer_mode is on host
    rocblas_pointer_mode pointer_mode = rocblas_pointer_mode_host;

    // space allocated for trsm
    void* trsm_Y      = nullptr;
    void* trsm_invA   = nullptr;
    void* trsm_invA_C = nullptr;

    // space allocated for trsv
    void* trsv_x     = nullptr;
    void* trsv_alpha = nullptr;

    // default logging_mode is no logging
    static rocblas_layer_mode layer_mode;

    // logging streams
    static std::ofstream log_trace_ofs;
    static std::ostream* log_trace_os;
    static std::ofstream log_bench_ofs;
    static std::ostream* log_bench_os;
    static std::ofstream log_profile_ofs;
    static std::ostream* log_profile_os;

    // static data for startup initialization
    static struct init
    {
        init();
    } handle_init;
};

namespace rocblas {
void reinit_logs(); // Reinitialize static data (for testing only)
}

// work buffer size constants
constexpr size_t WORKBUF_TRSM_A_BLKS    = 10;
constexpr size_t WORKBUF_TRSM_B_CHNK    = 32000;
constexpr size_t WORKBUF_TRSM_Y_SZ      = 32000 * 128 * sizeof(double);
constexpr size_t WORKBUF_TRSM_INVA_SZ   = 128 * 128 * 10 * sizeof(double);
constexpr size_t WORKBUF_TRSM_INVA_C_SZ = 128 * 128 * 10 * sizeof(double) / 2;
constexpr size_t WORKBUF_TRSV_X_SZ      = 131072 * sizeof(double);
constexpr size_t WORKBUF_TRSV_ALPHA_SZ  = sizeof(double);
#endif

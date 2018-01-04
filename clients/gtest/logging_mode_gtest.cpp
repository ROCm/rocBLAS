/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "rocblas.h"
#include "rocblas.hpp"
#include "utility.h"
#include "arg_check.h"
#include "rocblas_test_unique_ptr.hpp"
#include "handle.h"

using namespace std;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
     BLAS set-get_logging_mode:
=================================================================== */

TEST(checkin_auxilliary, set_logging_mode_get_logging_mode)
{
    rocblas_int N    = 1;
    rocblas_int incx = 1;
    float alpha_s    = 1.0;
    double alpha_d    = 1.0;
    rocblas_float_complex alpha_c    = 1.0;
    rocblas_double_complex alpha_z    = 1.0;

    rocblas_status status = rocblas_status_success;

    std::unique_ptr<rocblas_test::handle_struct> unique_ptr_handle(new rocblas_test::handle_struct);
    rocblas_handle handle = unique_ptr_handle->handle;

    rocblas_int size_x = N * incx;

    // allocate memory on device
    auto dx_float_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(float) * size_x),
                                         rocblas_test::device_free};

    auto dx_double_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(double) * size_x),
                                         rocblas_test::device_free};

    auto dx_float_complex_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(rocblas_float_complex) * size_x),
                                         rocblas_test::device_free};

    auto dx_double_complex_managed = rocblas_unique_ptr{rocblas_test::device_malloc(sizeof(rocblas_double_complex) * size_x),
                                         rocblas_test::device_free};

    float* dx_float = (float*)dx_float_managed.get();
    double* dx_double = (double*)dx_double_managed.get();
    rocblas_float_complex* dx_float_complex = (rocblas_float_complex*)dx_float_complex_managed.get();
    rocblas_double_complex* dx_double_complex = (rocblas_double_complex*)dx_double_complex_managed.get();
    if(!dx_float || !dx_double)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    status = rocblas_sscal(handle, N, &alpha_s, dx_float, incx);
    status = rocblas_dscal(handle, N, &alpha_d, dx_double, incx);
//  status = rocblas_cscal(handle, N, &alpha_c, dx_float_complex, incx);
//  status = rocblas_zscal(handle, N, &alpha_z, dx_double_complex, incx);

    FILE * fp;

    const size_t line_size = 300;
    char buffer[line_size];

    // open file in home directory
    const char *file_name = "/rocblas_logfile.yaml";
    char *home_dir = getenv("HOME");
    char *file_path = (char *) malloc(strlen(home_dir) + strlen(file_name) + 1);
    strncpy(file_path, home_dir, strlen(home_dir) + 1);
    strncat(file_path, file_name, strlen(file_name) + 1);
    printf("file_path=%s\n", file_path);
    fp = fopen(file_path, "r");
    free(file_path);

    if (fp == NULL)
    {
        printf("ERROR: rocblas_test: could not open logging file %s\n",file_path);
    }
    else
    {
        fflush(fp);
        rewind(fp);

        printf("-------------------------------------------------------------\n");
        while (fgets(buffer, line_size, fp))
        {
            printf("%s",buffer);
        }
        printf("-------------------------------------------------------------\n");

        fclose(fp);
    }
}

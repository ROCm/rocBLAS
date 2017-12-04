/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <stdexcept>
#include "rocblas.hpp"
#include "utility.h"

using namespace std;

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
     BLAS set-get_pointer_mode:
=================================================================== */

TEST(checkin_auxilliary, set_pointer_mode_get_pointer_mode)
{
    rocblas_status status     = rocblas_status_success;
    rocblas_pointer_mode mode = rocblas_pointer_mode_device;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);
    EXPECT_EQ(status, rocblas_status_success);

    status = rocblas_get_pointer_mode(handle, &mode);
    EXPECT_EQ(status, rocblas_status_success);

    EXPECT_EQ(rocblas_pointer_mode_device, mode);

    status = rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    EXPECT_EQ(status, rocblas_status_success);

    status = rocblas_get_pointer_mode(handle, &mode);
    EXPECT_EQ(status, rocblas_status_success);

    EXPECT_EQ(rocblas_pointer_mode_host, mode);

    rocblas_destroy_handle(handle);
}

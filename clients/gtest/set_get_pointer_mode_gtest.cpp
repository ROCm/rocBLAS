/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "rocblas.hpp"
#include "utility.hpp"

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
     BLAS set-get_pointer_mode:
=================================================================== */

TEST(quick_auxilliary, set_pointer_mode_get_pointer_mode)
{
    rocblas_pointer_mode mode = rocblas_pointer_mode_device;
    rocblas_local_handle handle;
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
    CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle, &mode));
    EXPECT_EQ(rocblas_pointer_mode_device, mode);
    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
    CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle, &mode));
    EXPECT_EQ(rocblas_pointer_mode_host, mode);
}

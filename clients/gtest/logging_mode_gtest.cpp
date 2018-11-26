/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>
#include "rocblas.h"
#include "rocblas.hpp"
#include "utility.h"
#include "testing_logging.hpp"

/* =====================================================================
README: This file contains testers to verify the correctness of
        BLAS routines with google test

        It is supposed to be played/used by advance / expert users
        Normal users only need to get the library routines without testers
     =================================================================== */

/* =====================================================================
     BLAS set-get_logging_mode:
=================================================================== */

TEST(quick_auxilliary, logging_float) { testing_logging<float>(); }
TEST(quick_auxilliary, logging_double) { testing_logging<double>(); }

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
#include "testing_logging.hpp"

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

TEST(checkin_auxilliary, logging_float) { testing_logging<float>(); }
TEST(checkin_auxilliary, logging_double) { testing_logging<double>(); }

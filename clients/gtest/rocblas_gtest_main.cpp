/* ************************************************************************
 * Copyright 2016 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <stdexcept>
#include "utility.h"

#define GTEST_DATA "rocblas_gtest.data"

// Return path of this executable
static string exepath()
{
    string pathstr;
    char* path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = rindex(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    // Open data file
    RocBLAS_TestData::init(exepath() + GTEST_DATA);

    // Device Query

    int device_id = 0;

    int device_count = query_device_property();

    if(device_count <= device_id)
    {
        printf("Error: invalid device ID. There may not be such device ID. Will exit \n");
        return -1;
    }
    else
    {
        set_device(device_id);
    }

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

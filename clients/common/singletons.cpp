/* ************************************************************************
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "singletons.hpp"

// global for device memory padding see d_vector.hpp
size_t g_DVEC_PAD = 4096;

void d_vector_set_pad_length(size_t pad)
{
    g_DVEC_PAD = pad;
}

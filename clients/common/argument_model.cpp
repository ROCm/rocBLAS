/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "argument_model.hpp"

// this should have been a member variable but due to the complex variadic template this singleton allows global control

static bool log_function_name = false;

void ArgumentModel_set_log_function_name(bool f)
{
    log_function_name = f;
}

bool ArgumentModel_get_log_function_name()
{
    return log_function_name;
}

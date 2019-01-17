/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_datatype2char.hpp"
#include <string>

rocblas_datatype string2rocblas_datatype(std::string value)
{
    if(value.compare("f16_r") == 0)
        return rocblas_datatype_f16_r;
    if(value.compare("f32_r") == 0)
        return rocblas_datatype_f32_r;
    if(value.compare("f64_r") == 0)
        return rocblas_datatype_f64_r;
    if(value.compare("f16_c") == 0)
        return rocblas_datatype_f32_c;
    if(value.compare("f32_c") == 0)
        return rocblas_datatype_f32_c;
    if(value.compare("f64_c") == 0)
        return rocblas_datatype_f64_c;
    if(value.compare("i8_r") == 0)
        return rocblas_datatype_i8_r;
    if(value.compare("i32_r") == 0)
        return rocblas_datatype_i32_r;
    if(value.compare("u8_r") == 0)
        return rocblas_datatype_u8_r;
    if(value.compare("u32_r") == 0)
        return rocblas_datatype_u32_r;
    if(value.compare("i8_c") == 0)
        return rocblas_datatype_i8_c;
    if(value.compare("i32_c") == 0)
        return rocblas_datatype_i32_c;
    if(value.compare("u8_c") == 0)
        return rocblas_datatype_u8_c;
    if(value.compare("u32_c") == 0)
        return rocblas_datatype_u32_c;
    return static_cast<rocblas_datatype>(-1);
}

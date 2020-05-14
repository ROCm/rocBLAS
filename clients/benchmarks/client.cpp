/* ************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <boost/program_options.hpp>

#include "rocblas.h"
#include "rocblas.hpp"
#include "rocblas_data.hpp"
#include "rocblas_datatype2string.hpp"
#include "rocblas_parse_data.hpp"
#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
// aux
#include "testing_set_get_matrix.hpp"
#include "testing_set_get_matrix_async.hpp"
#include "testing_set_get_vector.hpp"
#include "testing_set_get_vector_async.hpp"
// blas1
#include "testing_asum.hpp"
#include "testing_asum_batched.hpp"
#include "testing_asum_strided_batched.hpp"
#include "testing_axpy.hpp"
#include "testing_axpy_batched.hpp"
#include "testing_axpy_strided_batched.hpp"
#include "testing_copy.hpp"
#include "testing_copy_batched.hpp"
#include "testing_copy_strided_batched.hpp"
#include "testing_dot.hpp"
#include "testing_dot_batched.hpp"
#include "testing_dot_strided_batched.hpp"
#include "testing_iamax_iamin.hpp"
#include "testing_iamax_iamin_batched.hpp"
#include "testing_iamax_iamin_strided_batched.hpp"
#include "testing_nrm2.hpp"
#include "testing_nrm2_batched.hpp"
#include "testing_nrm2_strided_batched.hpp"
#include "testing_rot.hpp"
#include "testing_rot_batched.hpp"
#include "testing_rot_strided_batched.hpp"
#include "testing_rotg.hpp"
#include "testing_rotg_batched.hpp"
#include "testing_rotg_strided_batched.hpp"
#include "testing_rotm.hpp"
#include "testing_rotm_batched.hpp"
#include "testing_rotm_strided_batched.hpp"
#include "testing_rotmg.hpp"
#include "testing_rotmg_batched.hpp"
#include "testing_rotmg_strided_batched.hpp"
#include "testing_scal.hpp"
#include "testing_scal_batched.hpp"
#include "testing_scal_strided_batched.hpp"
#include "testing_swap.hpp"
#include "testing_swap_batched.hpp"
#include "testing_swap_strided_batched.hpp"
// blas2
#include "testing_gbmv.hpp"
#include "testing_gbmv_batched.hpp"
#include "testing_gbmv_strided_batched.hpp"
#include "testing_gemv.hpp"
#include "testing_gemv_batched.hpp"
#include "testing_gemv_strided_batched.hpp"
#include "testing_ger.hpp"
#include "testing_ger_batched.hpp"
#include "testing_ger_strided_batched.hpp"
#include "testing_hbmv.hpp"
#include "testing_hbmv_batched.hpp"
#include "testing_hbmv_strided_batched.hpp"
#include "testing_hemv.hpp"
#include "testing_hemv_batched.hpp"
#include "testing_hemv_strided_batched.hpp"
#include "testing_her.hpp"
#include "testing_her2.hpp"
#include "testing_her2_batched.hpp"
#include "testing_her2_strided_batched.hpp"
#include "testing_her_batched.hpp"
#include "testing_her_strided_batched.hpp"
#include "testing_hpmv.hpp"
#include "testing_hpmv_batched.hpp"
#include "testing_hpmv_strided_batched.hpp"
#include "testing_hpr.hpp"
#include "testing_hpr2.hpp"
#include "testing_hpr2_batched.hpp"
#include "testing_hpr2_strided_batched.hpp"
#include "testing_hpr_batched.hpp"
#include "testing_hpr_strided_batched.hpp"
#include "testing_sbmv.hpp"
#include "testing_sbmv_batched.hpp"
#include "testing_sbmv_strided_batched.hpp"
#include "testing_spmv.hpp"
#include "testing_spmv_batched.hpp"
#include "testing_spmv_strided_batched.hpp"
#include "testing_spr.hpp"
#include "testing_spr2.hpp"
#include "testing_spr2_batched.hpp"
#include "testing_spr2_strided_batched.hpp"
#include "testing_spr_batched.hpp"
#include "testing_spr_strided_batched.hpp"
#include "testing_symv.hpp"
#include "testing_symv_batched.hpp"
#include "testing_symv_strided_batched.hpp"
#include "testing_syr.hpp"
#include "testing_syr2.hpp"
#include "testing_syr2_batched.hpp"
#include "testing_syr2_strided_batched.hpp"
#include "testing_syr_batched.hpp"
#include "testing_syr_strided_batched.hpp"
#include "testing_tbmv.hpp"
#include "testing_tbmv_batched.hpp"
#include "testing_tbmv_strided_batched.hpp"
#include "testing_tbsv.hpp"
#include "testing_tbsv_batched.hpp"
#include "testing_tbsv_strided_batched.hpp"
#include "testing_tpmv.hpp"
#include "testing_tpmv_batched.hpp"
#include "testing_tpmv_strided_batched.hpp"
#include "testing_tpsv.hpp"
#include "testing_tpsv_batched.hpp"
#include "testing_tpsv_strided_batched.hpp"
#include "testing_trmv.hpp"
#include "testing_trmv_batched.hpp"
#include "testing_trmv_strided_batched.hpp"
// blas3 with no tensile
#include "testing_dgmm.hpp"
#include "testing_dgmm_batched.hpp"
#include "testing_dgmm_strided_batched.hpp"
#include "testing_geam.hpp"
#include "testing_geam_batched.hpp"
#include "testing_geam_strided_batched.hpp"
#include "testing_her2k.hpp"
#include "testing_her2k_batched.hpp"
#include "testing_her2k_strided_batched.hpp"
#include "testing_herk.hpp"
#include "testing_herk_batched.hpp"
#include "testing_herk_strided_batched.hpp"
#include "testing_symm_hemm.hpp"
#include "testing_symm_hemm_batched.hpp"
#include "testing_symm_hemm_strided_batched.hpp"
#include "testing_syr2k.hpp"
#include "testing_syr2k_batched.hpp"
#include "testing_syr2k_strided_batched.hpp"
#include "testing_syrk.hpp"
#include "testing_syrk_batched.hpp"
#include "testing_syrk_strided_batched.hpp"
//
#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#undef I

using namespace std::literals; // For std::string literals of form "str"s

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(const Arguments&), str_less>;

// Run a function by using map to map arg.function to function
void run_function(const func_map& map, const Arguments& arg, const std::string& msg = "")
{
    auto match = map.find(arg.function);
    if(match == map.end())
        throw std::invalid_argument("Invalid combination --function "s + arg.function
                                    + " --a_type "s + rocblas_datatype2string(arg.a_type) + msg);
    match->second(arg);
}

#if BUILD_WITH_TENSILE

#include "testing_gemm.hpp"
#include "testing_gemm_batched.hpp"
#include "testing_gemm_batched_ex.hpp"
#include "testing_gemm_ex.hpp"
#include "testing_gemm_strided_batched.hpp"
#include "testing_gemm_strided_batched_ex.hpp"
#include "testing_trmm.hpp"
#include "testing_trmm_batched.hpp"
#include "testing_trmm_strided_batched.hpp"
#include "testing_trsm.hpp"
#include "testing_trsm_batched.hpp"
#include "testing_trsm_batched_ex.hpp"
#include "testing_trsm_ex.hpp"
#include "testing_trsm_strided_batched.hpp"
#include "testing_trsm_strided_batched_ex.hpp"
#include "testing_trsv.hpp"
#include "testing_trsv_batched.hpp"
#include "testing_trsv_strided_batched.hpp"
#include "testing_trtri.hpp"
#include "testing_trtri_batched.hpp"
#include "testing_trtri_strided_batched.hpp"

// Template to dispatch testing_gemm_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_ex : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_ex<Ti,
                    To,
                    Tc,
                    std::enable_if_t<!std::is_same<Ti, void>{}
                                     && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                                          && std::is_same<Ti, rocblas_bfloat16>{})>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemm_ex", testing_gemm_ex<Ti, To, Tc>},
            {"gemm_batched_ex", testing_gemm_batched_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

// Template to dispatch testing_gemm_strided_batched_ex for performance tests
// When Ti == void or Ti == To == Tc == bfloat16, the test is marked invalid
template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_gemm_strided_batched_ex : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_gemm_strided_batched_ex<
    Ti,
    To,
    Tc,
    std::enable_if_t<!std::is_same<Ti, void>{}
                     && !(std::is_same<Ti, To>{} && std::is_same<Ti, Tc>{}
                          && std::is_same<Ti, rocblas_bfloat16>{})>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"gemm_strided_batched_ex", testing_gemm_strided_batched_ex<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

#endif // BUILD_WITH_TENSILE

template <typename T, typename U = T, typename = void>
struct perf_blas : rocblas_test_invalid
{
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same<T, float>{} || std::is_same<T, double>{}>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map
            = { {"set_get_vector", testing_set_get_vector<T>},
                {"set_get_matrix", testing_set_get_matrix<T>},
                // L1
                {"asum", testing_asum<T>},
                {"asum_batched", testing_asum_batched<T>},
                {"asum_strided_batched", testing_asum_strided_batched<T>},
                {"axpy", testing_axpy<T>},
                {"axpy_batched", testing_axpy_batched<T>},
                {"axpy_strided_batched", testing_axpy_strided_batched<T>},
                {"copy", testing_copy<T>},
                {"copy_batched", testing_copy_batched<T>},
                {"copy_strided_batched", testing_copy_strided_batched<T>},
                {"dot", testing_dot<T>},
                {"dot_batched", testing_dot_batched<T>},
                {"dot_strided_batched", testing_dot_strided_batched<T>},
                {"iamax", testing_iamax<T>},
                {"iamax_batched", testing_iamax_batched<T>},
                {"iamax_strided_batched", testing_iamax_strided_batched<T>},
                {"iamin", testing_iamin<T>},
                {"iamin_batched", testing_iamin_batched<T>},
                {"iamin_strided_batched", testing_iamin_strided_batched<T>},
                {"nrm2", testing_nrm2<T>},
                {"nrm2_batched", testing_nrm2_batched<T>},
                {"nrm2_strided_batched", testing_nrm2_strided_batched<T>},
                {"rotm", testing_rotm<T>},
                {"rotm_batched", testing_rotm_batched<T>},
                {"rotm_strided_batched", testing_rotm_strided_batched<T>},
                {"rotmg", testing_rotmg<T>},
                {"rotmg_batched", testing_rotmg_batched<T>},
                {"rotmg_strided_batched", testing_rotmg_strided_batched<T>},
                {"swap", testing_swap<T>},
                {"swap_batched", testing_swap_batched<T>},
                {"swap_strided_batched", testing_swap_strided_batched<T>},
                // L2
                {"gbmv", testing_gbmv<T>},
                {"gbmv_batched", testing_gbmv_batched<T>},
                {"gbmv_strided_batched", testing_gbmv_strided_batched<T>},
                {"geam", testing_geam<T>},
                {"geam_batched", testing_geam_batched<T>},
                {"geam_strided_batched", testing_geam_strided_batched<T>},
                {"gemv", testing_gemv<T>},
                {"gemv_batched", testing_gemv_batched<T>},
                {"gemv_strided_batched", testing_gemv_strided_batched<T>},
                {"ger", testing_ger<T, false>},
                {"ger_batched", testing_ger_batched<T, false>},
                {"ger_strided_batched", testing_ger_strided_batched<T, false>},
                {"spr", testing_spr<T>},
                {"spr_batched", testing_spr_batched<T>},
                {"spr_strided_batched", testing_spr_strided_batched<T>},
                {"spr2", testing_spr2<T>},
                {"spr2_batched", testing_spr2_batched<T>},
                {"spr2_strided_batched", testing_spr2_strided_batched<T>},
                {"syr", testing_syr<T>},
                {"syr_batched", testing_syr_batched<T>},
                {"syr_strided_batched", testing_syr_strided_batched<T>},
                {"syr2", testing_syr2<T>},
                {"syr2_batched", testing_syr2_batched<T>},
                {"syr2_strided_batched", testing_syr2_strided_batched<T>},
                {"sbmv", testing_sbmv<T>},
                {"sbmv_batched", testing_sbmv_batched<T>},
                {"sbmv_strided_batched", testing_sbmv_strided_batched<T>},
                {"spmv", testing_spmv<T>},
                {"spmv_batched", testing_spmv_batched<T>},
                {"spmv_strided_batched", testing_spmv_strided_batched<T>},
                {"symv", testing_symv<T>},
                {"symv_batched", testing_symv_batched<T>},
                {"symv_strided_batched", testing_symv_strided_batched<T>},
                {"tbmv", testing_tbmv<T>},
                {"tbmv_batched", testing_tbmv_batched<T>},
                {"tbmv_strided_batched", testing_tbmv_strided_batched<T>},
                {"tbsv", testing_tbsv<T>},
                {"tbsv_batched", testing_tbsv_batched<T>},
                {"tbsv_strided_batched", testing_tbsv_strided_batched<T>},
                {"tpmv", testing_tpmv<T>},
                {"tpmv_batched", testing_tpmv_batched<T>},
                {"tpmv_strided_batched", testing_tpmv_strided_batched<T>},
                {"tpsv", testing_tpsv<T>},
                {"tpsv_batched", testing_tpsv_batched<T>},
                {"tpsv_strided_batched", testing_tpsv_strided_batched<T>},
                {"trmv", testing_trmv<T>},
                {"trmv_batched", testing_trmv_batched<T>},
                {"trmv_strided_batched", testing_trmv_strided_batched<T>},
                // L3
                {"dgmm", testing_dgmm<T>},
                {"dgmm_batched", testing_dgmm_batched<T>},
                {"dgmm_strided_batched", testing_dgmm_strided_batched<T>},
                {"symm", testing_symm_hemm<T, false>},
                {"symm_batched", testing_symm_hemm_batched<T, false>},
                {"symm_strided_batched", testing_symm_hemm_strided_batched<T, false>},
                {"syrk", testing_syrk<T>},
                {"syrk_batched", testing_syrk_batched<T>},
                {"syrk_strided_batched", testing_syrk_strided_batched<T>},
                {"syr2k", testing_syr2k<T>},
                {"syr2k_batched", testing_syr2k_batched<T>},
                {"syr2k_strided_batched", testing_syr2k_strided_batched<T>},
                {"syrkx", testing_syr2k<T, false>},
                {"syrkx_batched", testing_syr2k_batched<T, false>},
                {"syrkx_strided_batched", testing_syr2k_strided_batched<T, false>},
#if BUILD_WITH_TENSILE
                {"trmm", testing_trmm<T>},
                {"trmm_batched", testing_trmm_batched<T>},
                {"trmm_strided_batched", testing_trmm_strided_batched<T>},
                {"trtri", testing_trtri<T>},
                {"trtri_batched", testing_trtri_batched<T>},
                {"trtri_strided_batched", testing_trtri_strided_batched<T>},
                {"gemm", testing_gemm<T>},
                {"gemm_batched", testing_gemm_batched<T>},
                {"gemm_strided_batched", testing_gemm_strided_batched<T>},
                {"trsm", testing_trsm<T>},
                {"trsm_ex", testing_trsm_ex<T>},
                {"trsm_batched", testing_trsm_batched<T>},
                {"trsm_batched_ex", testing_trsm_batched_ex<T>},
                {"trsm_strided_batched", testing_trsm_strided_batched<T>},
                {"trsm_strided_batched_ex", testing_trsm_strided_batched_ex<T>},
                {"trsv", testing_trsv<T>},
                {"trsv_batched", testing_trsv_batched<T>},
                {"trsv_strided_batched", testing_trsv_strided_batched<T>},
#endif
              };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same<T, rocblas_bfloat16>{}>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"dot", testing_dot<T>},
            {"dot_batched", testing_dot_batched<T>},
            {"dot_strided_batched", testing_dot_strided_batched<T>},
        };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T, U, std::enable_if_t<std::is_same<T, rocblas_half>{}>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map
            = { {"axpy", testing_axpy<T>},
                {"axpy_batched", testing_axpy_batched<T>},
                {"axpy_strided_batched", testing_axpy_strided_batched<T>},
                {"dot", testing_dot<T>},
                {"dot_batched", testing_dot_batched<T>},
                {"dot_strided_batched", testing_dot_strided_batched<T>},
#if BUILD_WITH_TENSILE
                {"gemm", testing_gemm<T>},
                {"gemm_batched", testing_gemm_batched<T>},
                {"gemm_strided_batched", testing_gemm_strided_batched<T>},
#endif
              };
        run_function(map, arg);
    }
};

template <typename T, typename U>
struct perf_blas<T,
                 U,
                 std::enable_if_t<std::is_same<T, rocblas_double_complex>{}
                                  || std::is_same<T, rocblas_float_complex>{}>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map
            = { {"asum", testing_asum<T>},
                {"asum_batched", testing_asum_batched<T>},
                {"asum_strided_batched", testing_asum_strided_batched<T>},
                {"axpy", testing_axpy<T>},
                {"axpy_batched", testing_axpy_batched<T>},
                {"axpy_strided_batched", testing_axpy_strided_batched<T>},
                {"copy", testing_copy<T>},
                {"copy_batched", testing_copy_batched<T>},
                {"copy_strided_batched", testing_copy_strided_batched<T>},
                {"dot", testing_dot<T>},
                {"dot_batched", testing_dot_batched<T>},
                {"dot_strided_batched", testing_dot_strided_batched<T>},
                {"dotc", testing_dotc<T>},
                {"dotc_batched", testing_dotc_batched<T>},
                {"dotc_strided_batched", testing_dotc_strided_batched<T>},
                {"iamax", testing_iamax<T>},
                {"iamax_batched", testing_iamax_batched<T>},
                {"iamax_strided_batched", testing_iamax_strided_batched<T>},
                {"iamin", testing_iamin<T>},
                {"iamin_batched", testing_iamin_batched<T>},
                {"iamin_strided_batched", testing_iamin_strided_batched<T>},
                {"nrm2", testing_nrm2<T>},
                {"nrm2_batched", testing_nrm2_batched<T>},
                {"nrm2_strided_batched", testing_nrm2_strided_batched<T>},
                {"swap", testing_swap<T>},
                {"swap_batched", testing_swap_batched<T>},
                {"swap_strided_batched", testing_swap_strided_batched<T>},
                // L2
                {"gbmv", testing_gbmv<T>},
                {"gbmv_batched", testing_gbmv_batched<T>},
                {"gbmv_strided_batched", testing_gbmv_strided_batched<T>},
                {"gemv", testing_gemv<T>},
                {"gemv_batched", testing_gemv_batched<T>},
                {"gemv_strided_batched", testing_gemv_strided_batched<T>},
                {"geru", testing_ger<T, false>},
                {"geru_batched", testing_ger_batched<T, false>},
                {"geru_strided_batched", testing_ger_strided_batched<T, false>},
                {"gerc", testing_ger<T, true>},
                {"gerc_batched", testing_ger_batched<T, true>},
                {"gerc_strided_batched", testing_ger_strided_batched<T, true>},
                {"hbmv", testing_hbmv<T>},
                {"hbmv_batched", testing_hbmv_batched<T>},
                {"hbmv_strided_batched", testing_hbmv_strided_batched<T>},
                {"hemv", testing_hemv<T>},
                {"hemv_batched", testing_hemv_batched<T>},
                {"hemv_strided_batched", testing_hemv_strided_batched<T>},
                {"her", testing_her<T>},
                {"her_batched", testing_her_batched<T>},
                {"her_strided_batched", testing_her_strided_batched<T>},
                {"her2", testing_her2<T>},
                {"her2_batched", testing_her2_batched<T>},
                {"her2_strided_batched", testing_her2_strided_batched<T>},
                {"hpmv", testing_hpmv<T>},
                {"hpmv_batched", testing_hpmv_batched<T>},
                {"hpmv_strided_batched", testing_hpmv_strided_batched<T>},
                {"hpr", testing_hpr<T>},
                {"hpr_batched", testing_hpr_batched<T>},
                {"hpr_strided_batched", testing_hpr_strided_batched<T>},
                {"hpr2", testing_hpr2<T>},
                {"hpr2_batched", testing_hpr2_batched<T>},
                {"hpr2_strided_batched", testing_hpr2_strided_batched<T>},
                {"spr", testing_spr<T>},
                {"spr_batched", testing_spr_batched<T>},
                {"spr_strided_batched", testing_spr_strided_batched<T>},
                {"syr", testing_syr<T>},
                {"syr_batched", testing_syr_batched<T>},
                {"syr_strided_batched", testing_syr_strided_batched<T>},
                {"syr2", testing_syr2<T>},
                {"syr2_batched", testing_syr2_batched<T>},
                {"syr2_strided_batched", testing_syr2_strided_batched<T>},
                {"tbmv", testing_tbmv<T>},
                {"tbmv_batched", testing_tbmv_batched<T>},
                {"tbmv_strided_batched", testing_tbmv_strided_batched<T>},
                {"tbsv", testing_tbsv<T>},
                {"tbsv_batched", testing_tbsv_batched<T>},
                {"tbsv_strided_batched", testing_tbsv_strided_batched<T>},
                {"tpmv", testing_tpmv<T>},
                {"tpmv_batched", testing_tpmv_batched<T>},
                {"tpmv_strided_batched", testing_tpmv_strided_batched<T>},
                {"tpsv", testing_tpsv<T>},
                {"tpsv_batched", testing_tpsv_batched<T>},
                {"tpsv_strided_batched", testing_tpsv_strided_batched<T>},
                {"symv", testing_symv<T>},
                {"symv_batched", testing_symv_batched<T>},
                {"symv_strided_batched", testing_symv_strided_batched<T>},
                {"trmv", testing_trmv<T>},
                {"trmv_batched", testing_trmv_batched<T>},
                {"trmv_strided_batched", testing_trmv_strided_batched<T>},
                // L3
                {"dgmm", testing_dgmm<T>},
                {"dgmm_batched", testing_dgmm_batched<T>},
                {"dgmm_strided_batched", testing_dgmm_strided_batched<T>},
                {"geam", testing_geam<T>},
                {"geam_batched", testing_geam_batched<T>},
                {"geam_strided_batched", testing_geam_strided_batched<T>},
                {"syrk", testing_syrk<T>},
                {"syrk_batched", testing_syrk_batched<T>},
                {"syrk_strided_batched", testing_syrk_strided_batched<T>},
                {"syr2k", testing_syr2k<T>},
                {"syr2k_batched", testing_syr2k_batched<T>},
                {"syr2k_strided_batched", testing_syr2k_strided_batched<T>},
                {"syrkx", testing_syr2k<T, false>},
                {"syrkx_batched", testing_syr2k_batched<T, false>},
                {"syrkx_strided_batched", testing_syr2k_strided_batched<T, false>},
                {"symm", testing_symm_hemm<T, false>},
                {"symm_batched", testing_symm_hemm_batched<T, false>},
                {"symm_strided_batched", testing_symm_hemm_strided_batched<T, false>},
                {"hemm", testing_symm_hemm<T, true>},
                {"hemm_batched", testing_symm_hemm_batched<T, true>},
                {"hemm_strided_batched", testing_symm_hemm_strided_batched<T, true>},
                {"herk", testing_herk<T>},
                {"herk_batched", testing_herk_batched<T>},
                {"herk_strided_batched", testing_herk_strided_batched<T>},
                {"her2k", testing_her2k<T>},
                {"her2k_batched", testing_her2k_batched<T>},
                {"her2k_strided_batched", testing_her2k_strided_batched<T>},
                {"herkx", testing_her2k<T, false>},
                {"herkx_batched", testing_her2k_batched<T, false>},
                {"herkx_strided_batched", testing_her2k_strided_batched<T, false>},
#if BUILD_WITH_TENSILE
                {"trtri", testing_trtri<T>},
                {"trtri_batched", testing_trtri_batched<T>},
                {"trtri_strided_batched", testing_trtri_strided_batched<T>},
                {"gemm", testing_gemm<T>},
                {"gemm_batched", testing_gemm_batched<T>},
                {"gemm_strided_batched", testing_gemm_strided_batched<T>},
                {"trsm", testing_trsm<T>},
                {"trsm_ex", testing_trsm_ex<T>},
                {"trsm_batched", testing_trsm_batched<T>},
                {"trsm_batched_ex", testing_trsm_batched_ex<T>},
                {"trsm_strided_batched", testing_trsm_strided_batched<T>},
                {"trsm_strided_batched_ex", testing_trsm_strided_batched_ex<T>},
                {"trsv", testing_trsv<T>},
                {"trsv_batched", testing_trsv_batched<T>},
                {"trsv_strided_batched", testing_trsv_strided_batched<T>},
                {"trmm", testing_trmm<T>},
                {"trmm_batched", testing_trmm_batched<T>},
                {"trmm_strided_batched", testing_trmm_strided_batched<T>},
#endif
              };
        run_function(map, arg);
    }
};

template <typename Ti, typename To = Ti, typename Tc = To, typename = void>
struct perf_blas_rot : rocblas_test_invalid
{
};

template <typename Ti, typename To, typename Tc>
struct perf_blas_rot<
    Ti,
    To,
    Tc,
    std::enable_if_t<(std::is_same<Ti, float>{} && std::is_same<Ti, To>{} && std::is_same<To, Tc>{})
                     || (std::is_same<Ti, double>{} && std::is_same<Ti, To>{}
                         && std::is_same<To, Tc>{})
                     || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{}
                         && std::is_same<Tc, rocblas_float_complex>{})
                     || (std::is_same<Ti, rocblas_float_complex>{} && std::is_same<To, float>{}
                         && std::is_same<Tc, float>{})
                     || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{}
                         && std::is_same<Tc, rocblas_double_complex>{})
                     || (std::is_same<Ti, rocblas_double_complex>{} && std::is_same<To, double>{}
                         && std::is_same<Tc, double>{})>> : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rot", testing_rot<Ti, To, Tc>},
            {"rot_batched", testing_rot_batched<Ti, To, Tc>},
            {"rot_strided_batched", testing_rot_strided_batched<Ti, To, Tc>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tb = Ta, typename = void>
struct perf_blas_scal : rocblas_test_invalid
{
};

template <typename Ta, typename Tb>
struct perf_blas_scal<
    Ta,
    Tb,
    std::enable_if_t<(std::is_same<Ta, double>{} && std::is_same<Tb, rocblas_double_complex>{})
                     || (std::is_same<Ta, float>{} && std::is_same<Tb, rocblas_float_complex>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, float>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, double>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, rocblas_float_complex>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, rocblas_double_complex>{})>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"scal", testing_scal<Ta, Tb>},
            {"scal_batched", testing_scal_batched<Ta, Tb>},
            {"scal_strided_batched", testing_scal_strided_batched<Ta, Tb>},
        };
        run_function(map, arg);
    }
};

template <typename Ta, typename Tb = Ta, typename = void>
struct perf_blas_rotg : rocblas_test_invalid
{
};

template <typename Ta, typename Tb>
struct perf_blas_rotg<
    Ta,
    Tb,
    std::enable_if_t<(std::is_same<Ta, rocblas_double_complex>{} && std::is_same<Tb, double>{})
                     || (std::is_same<Ta, rocblas_float_complex>{} && std::is_same<Tb, float>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, float>{})
                     || (std::is_same<Ta, Tb>{} && std::is_same<Ta, double>{})>>
    : rocblas_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"rotg", testing_rotg<Ta, Tb>},
            {"rotg_batched", testing_rotg_batched<Ta, Tb>},
            {"rotg_strided_batched", testing_rotg_strided_batched<Ta, Tb>},
        };
        run_function(map, arg, " --b_type "s + rocblas_datatype2string(arg.b_type));
    }
};

int run_bench_test(Arguments& arg)
{
    rocblas_initialize(); // Initialize rocBLAS

    rocblas_cout << std::ios::fixed << std::setprecision(7); // Set precision to 7 digits

    // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    arg.timing = 1;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

#if BUILD_WITH_TENSILE
    if(!strcmp(function, "gemm") || !strcmp(function, "gemm_batched"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;

        if(arg.lda < min_lda)
        {
            rocblas_cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            rocblas_cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            rocblas_cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
    }
    else if(!strcmp(function, "gemm_strided_batched"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;
        if(arg.lda < min_lda)
        {
            rocblas_cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            rocblas_cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            rocblas_cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }

        //      rocblas_int min_stride_a =
        //          arg.transA == 'N' ? arg.K * arg.lda : arg.M * arg.lda;
        //      rocblas_int min_stride_b =
        //          arg.transB == 'N' ? arg.N * arg.ldb : arg.K * arg.ldb;
        //      rocblas_int min_stride_a =
        //          arg.transA == 'N' ? arg.K * arg.lda : arg.M * arg.lda;
        //      rocblas_int min_stride_b =
        //          arg.transB == 'N' ? arg.N * arg.ldb : arg.K * arg.ldb;
        rocblas_int min_stride_c = arg.ldc * arg.N;
        //      if (arg.stride_a < min_stride_a)
        //      {
        //          rocblas_cout << "rocblas-bench INFO: stride_a < min_stride_a, set stride_a = " <<
        //          min_stride_a << std::endl;
        //          arg.stride_a = min_stride_a;
        //      }
        //      if (arg.stride_b < min_stride_b)
        //      {
        //          rocblas_cout << "rocblas-bench INFO: stride_b < min_stride_b, set stride_b = " <<
        //          min_stride_b << std::endl;
        //          arg.stride_b = min_stride_b;
        //      }
        if(arg.stride_c < min_stride_c)
        {
            rocblas_cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                         << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }
    }

    if(!strcmp(function, "gemm_ex") || !strcmp(function, "gemm_batched_ex"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;
        rocblas_int min_ldd = arg.M;

        if(arg.lda < min_lda)
        {
            rocblas_cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            rocblas_cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            rocblas_cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
        if(arg.ldd < min_ldd)
        {
            rocblas_cout << "rocblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd = min_ldd;
        }
        rocblas_gemm_dispatch<perf_gemm_ex>(arg);
    }
    else if(!strcmp(function, "gemm_strided_batched_ex"))
    {
        // adjust dimension for GEMM routines
        rocblas_int min_lda = arg.transA == 'N' ? arg.M : arg.K;
        rocblas_int min_ldb = arg.transB == 'N' ? arg.K : arg.N;
        rocblas_int min_ldc = arg.M;
        rocblas_int min_ldd = arg.M;
        if(arg.lda < min_lda)
        {
            rocblas_cout << "rocblas-bench INFO: lda < min_lda, set lda = " << min_lda << std::endl;
            arg.lda = min_lda;
        }
        if(arg.ldb < min_ldb)
        {
            rocblas_cout << "rocblas-bench INFO: ldb < min_ldb, set ldb = " << min_ldb << std::endl;
            arg.ldb = min_ldb;
        }
        if(arg.ldc < min_ldc)
        {
            rocblas_cout << "rocblas-bench INFO: ldc < min_ldc, set ldc = " << min_ldc << std::endl;
            arg.ldc = min_ldc;
        }
        if(arg.ldd < min_ldd)
        {
            rocblas_cout << "rocblas-bench INFO: ldd < min_ldd, set ldd = " << min_ldc << std::endl;
            arg.ldd = min_ldd;
        }
        rocblas_int min_stride_c = arg.ldc * arg.N;
        if(arg.stride_c < min_stride_c)
        {
            rocblas_cout << "rocblas-bench INFO: stride_c < min_stride_c, set stride_c = "
                         << min_stride_c << std::endl;
            arg.stride_c = min_stride_c;
        }

        rocblas_gemm_dispatch<perf_gemm_strided_batched_ex>(arg);
    }
    else
#endif
    {
        if(!strcmp(function, "scal") || !strcmp(function, "scal_batched")
           || !strcmp(function, "scal_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_scal>(arg);
        else if(!strcmp(function, "rotg") || !strcmp(function, "rotg_batched")
                || !strcmp(function, "rotg_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_rotg>(arg);
        else if(!strcmp(function, "rot") || !strcmp(function, "rot_batched")
                || !strcmp(function, "rot_strided_batched"))
            rocblas_blas1_dispatch<perf_blas_rot>(arg);
        else
            rocblas_simple_dispatch<perf_blas>(arg);
    }
    return 0;
}

int rocblas_bench_datafile()
{
    int ret = 0;
    for(Arguments arg : RocBLAS_TestData())
        ret |= run_bench_test(arg);
    test_cleanup::cleanup();
    return ret;
}

// Replace --batch with --batch_count for backward compatibility
void fix_batch(int argc, char* argv[])
{
    static char b_c[] = "--batch_count";
    for(int i = 1; i < argc; ++i)
        if(!strcmp(argv[i], "--batch"))
        {
            static int once = (rocblas_cerr << argv[0]
                                            << " warning: --batch is deprecated, and --batch_count "
                                               "should be used instead."
                                            << std::endl,
                               0);
            argv[i]         = b_c;
        }
}

using namespace boost::program_options;

int main(int argc, char* argv[])
try
{
    fix_batch(argc, argv);
    Arguments   arg;
    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string initialization;
    rocblas_int device_id;
    bool        datafile = rocblas_parse_data(argc, argv);

    options_description desc("rocblas-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<rocblas_int>(&arg.M)->default_value(128),
         "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
         "rows or columns in matrix.")

        ("sizen,n",
         value<rocblas_int>(&arg.N)->default_value(128),
         "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
         "BLAS-3: the number of rows or columns in matrix")

        ("sizek,k",
         value<rocblas_int>(&arg.K)->default_value(128),
         "Specific matrix size: BLAS-2: the number of sub or super-diagonals of A. BLAS-3: "
         "the number of columns in A and rows in B.")

        ("kl",
         value<rocblas_int>(&arg.KL)->default_value(128),
         "Specific matrix size: kl is only applicable to BLAS-2: The number of sub-diagonals "
         "of the banded matrix A.")

        ("ku",
         value<rocblas_int>(&arg.KU)->default_value(128),
         "Specific matrix size: ku is only applicable to BLAS-2: The number of super-diagonals "
         "of the banded matrix A.")

        ("lda",
         value<rocblas_int>(&arg.lda)->default_value(128),
         "Leading dimension of matrix A, is only applicable to BLAS-2 & BLAS-3.")

        ("ldb",
         value<rocblas_int>(&arg.ldb)->default_value(128),
         "Leading dimension of matrix B, is only applicable to BLAS-2 & BLAS-3.")

        ("ldc",
         value<rocblas_int>(&arg.ldc)->default_value(128),
         "Leading dimension of matrix C, is only applicable to BLAS-2 & BLAS-3.")

        ("ldd",
         value<rocblas_int>(&arg.ldd)->default_value(128),
         "Leading dimension of matrix D, is only applicable to BLAS-EX ")

        ("stride_a",
         value<rocblas_int>(&arg.stride_a)->default_value(128*128),
         "Specific stride of strided_batched matrix A, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_b",
         value<rocblas_int>(&arg.stride_b)->default_value(128*128),
         "Specific stride of strided_batched matrix B, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_c",
         value<rocblas_int>(&arg.stride_c)->default_value(128*128),
         "Specific stride of strided_batched matrix C, is only applicable to strided batched"
         "BLAS-2 and BLAS-3: second dimension * leading dimension.")

        ("stride_d",
         value<rocblas_int>(&arg.stride_d)->default_value(128*128),
         "Specific stride of strided_batched matrix D, is only applicable to strided batched"
         "BLAS_EX: second dimension * leading dimension.")

        ("stride_x",
         value<rocblas_int>(&arg.stride_x)->default_value(128*128),
         "Specific stride of strided_batched vector x, is only applicable to strided batched"
         "BLAS_2: second dimension.")

        ("stride_y",
         value<rocblas_int>(&arg.stride_y)->default_value(128*128),
         "Specific stride of strided_batched vector y, is only applicable to strided batched"
         "BLAS_2: leading dimension.")

        ("incx",
         value<rocblas_int>(&arg.incx)->default_value(1),
         "increment between values in x vector")

        ("incy",
         value<rocblas_int>(&arg.incy)->default_value(1),
         "increment between values in y vector")

        ("alpha",
          value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("alphai",
         value<double>(&arg.alphai)->default_value(0.0), "specifies the imaginary part of the scalar alpha")

        ("beta",
         value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("betai",
         value<double>(&arg.betai)->default_value(0.0), "specifies the imaginary part of the scalar beta")

        ("function,f",
         value<std::string>(&function),
         "BLAS function to test.")

        ("precision,r",
         value<std::string>(&precision)->default_value("f32_r"), "Precision. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("compute_type",
         value<std::string>(&compute_type), "Precision of computation. "
         "Options: h,s,d,c,z,f16_r,f32_r,f64_r,bf16_r,f32_c,f64_c,i8_r,i32_r")

        ("initialization",
         value<std::string>(&initialization)->default_value("rand_int"),
         "Intialize with random integers, trig functions sin and cos, or hpl-like input. "
         "Options: rand_int, trig_float, hpl")

        ("transposeA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("side",
         value<char>(&arg.side)->default_value('L'),
         "L = left, R = right. Only applicable to certain routines")

        ("uplo",
         value<char>(&arg.uplo)->default_value('U'),
         "U = upper, L = lower. Only applicable to certain routines") // xsymv xsyrk xsyr2k xtrsm xtrsm_ex
                                                                     // xtrmm xtrsv
        ("diag",
         value<char>(&arg.diag)->default_value('N'),
         "U = unit diagonal, N = non unit diagonal. Only applicable to certain routines") // xtrsm xtrsm_ex xtrsv xtrmm

        ("batch_count",
         value<rocblas_int>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines")

        ("verify,v",
         value<rocblas_int>(&arg.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<rocblas_int>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("cold_iters,j",
         value<rocblas_int>(&arg.cold_iters)->default_value(2),
         "Cold Iterations to run before entering the timing loop")

        ("algo",
         value<uint32_t>(&arg.algo)->default_value(0),
         "extended precision gemm algorithm")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(0),
         "extended precision gemm solution index")

        ("flags",
         value<uint32_t>(&arg.flags)->default_value(10),
         "extended precision gemm flags")

        ("device",
         value<rocblas_int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("c_noalias_d",
         bool_switch(&arg.c_noalias_d)->default_value(false),
         "C and D are stored in separate memory")

        ("fortran",
         bool_switch(&arg.fortran)->default_value(false),
         "Run using Fortran interface")

        ("help,h", "produces this help message")

        ("version", "Prints the version number");
    // clang-format on

    // Initialize rocBLAS; TODO: Remove this after it is determined why rocblas-bench
    // returns lower performance if this is executed after Boost parse_command_line().
    // Right now this causes 5-10 seconds of delay before processing the CLI arguments.
    rocblas_cerr << "Initializing rocBLAS..." << std::endl;
    rocblas_initialize();

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if(vm.count("help"))
    {
        rocblas_cout << desc << std::endl;
        return 0;
    }

    if(vm.find("version") != vm.end())
    {
        char blas_version[100];
        rocblas_get_version_string(blas_version, sizeof(blas_version));
        rocblas_cout << "rocBLAS version: " << blas_version << std::endl;
        return 0;
    }

    // Device Query
    rocblas_int device_count = query_device_property();

    rocblas_cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);

    if(datafile)
        return rocblas_bench_datafile();

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2rocblas_datatype(precision);
    if(prec == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string2rocblas_datatype(a_type);
    if(arg.a_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string2rocblas_datatype(b_type);
    if(arg.b_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string2rocblas_datatype(c_type);
    if(arg.c_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string2rocblas_datatype(d_type);
    if(arg.d_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    arg.compute_type = compute_type == "" ? prec : string2rocblas_datatype(compute_type);
    if(arg.compute_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    arg.initialization = string2rocblas_initialization(initialization);
    if(arg.initialization == static_cast<rocblas_initialization>(-1))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    return run_bench_test(arg);
}
catch(const std::invalid_argument& exp)
{
    rocblas_cerr << exp.what() << std::endl;
    return -1;
}

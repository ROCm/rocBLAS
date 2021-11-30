/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "rocblas_gemm_ext2.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas.h"
#include "utility.hpp"

namespace
{
    template <typename Ti, typename To, typename Tc>
    rocblas_status reference_gemm_ext2(rocblas_int    M,
                                       rocblas_int    N,
                                       rocblas_int    K,
                                       const Tc*      alpha,
                                       const Ti*      A,
                                       rocblas_stride row_stride_a,
                                       rocblas_stride col_stride_a,
                                       const Ti*      B,
                                       rocblas_stride row_stride_b,
                                       rocblas_stride col_stride_b,
                                       const Tc*      beta,
                                       const To*      C,
                                       rocblas_stride row_stride_c,
                                       rocblas_stride col_stride_c,
                                       To*            D,
                                       rocblas_stride row_stride_d,
                                       rocblas_stride col_stride_d)
    {
        for(rocblas_int row = 0; row < M; row++)
            for(rocblas_int col = 0; col < N; col++)
            {
                Tc t{};
                if(*alpha)
                    for(rocblas_int k = 0; k < K; k++)
                        t += Tc{A[row * row_stride_a + k * col_stride_a]}
                             * Tc{B[k * row_stride_b + col * col_stride_b]};
                D[row * row_stride_d + col * col_stride_d] = static_cast<To>(
                    *beta ? *beta * C[row * row_stride_c + col * col_stride_c] + *alpha * t
                          : *alpha * t);
            }
        return rocblas_status_success;
    }

    template <typename Ti, typename To = Ti, typename Tc = To>
    struct reference_gemm_ext2_call
    {
        rocblas_status operator()(rocblas_int    M,
                                  rocblas_int    N,
                                  rocblas_int    K,
                                  const void*    alpha,
                                  const void*    A,
                                  rocblas_stride row_stride_a,
                                  rocblas_stride col_stride_a,
                                  const void*    B,
                                  rocblas_stride row_stride_b,
                                  rocblas_stride col_stride_b,
                                  const void*    beta,
                                  const void*    C,
                                  rocblas_stride row_stride_c,
                                  rocblas_stride col_stride_c,
                                  void*          D,
                                  rocblas_stride row_stride_d,
                                  rocblas_stride col_stride_d) const
        {
            return reference_gemm_ext2(M,
                                       N,
                                       K,
                                       static_cast<const Tc*>(alpha),
                                       static_cast<const Ti*>(A),
                                       row_stride_a,
                                       col_stride_a,
                                       static_cast<const Ti*>(B),
                                       row_stride_b,
                                       col_stride_b,
                                       static_cast<const Tc*>(beta),
                                       static_cast<const To*>(C),
                                       row_stride_c,
                                       col_stride_c,
                                       static_cast<To*>(D),
                                       row_stride_d,
                                       col_stride_d);
        }
    };

    // gemm functions
    template <template <typename...> class GEMM, typename... Ts>
    auto gemm_dispatch(rocblas_datatype a_type,
                       rocblas_datatype b_type,
                       rocblas_datatype c_type,
                       rocblas_datatype d_type,
                       rocblas_datatype compute_type,
                       Ts&&... arg)
    {
        if(a_type == b_type && c_type == d_type)
        {
            if(a_type != c_type)
            {
                if(a_type == rocblas_datatype_i8_r && d_type == rocblas_datatype_i32_r
                   && compute_type == rocblas_datatype_i32_r)
                {
                    return GEMM<int8_t, int32_t, int32_t>{}(std::forward<Ts>(arg)...);
                }
                else if(a_type == rocblas_datatype_f16_r && d_type == rocblas_datatype_f32_r
                        && compute_type == rocblas_datatype_f32_r)
                {
                    return GEMM<rocblas_half, float, float>{}(std::forward<Ts>(arg)...);
                }
                else if(a_type == rocblas_datatype_bf16_r && d_type == rocblas_datatype_f32_r
                        && compute_type == rocblas_datatype_f32_r)
                {
                    return GEMM<rocblas_bfloat16, float, float>{}(std::forward<Ts>(arg)...);
                }
            }
            else if(d_type != compute_type)
            {
                if(d_type == rocblas_datatype_f16_r && compute_type == rocblas_datatype_f32_r)
                    return GEMM<rocblas_half, rocblas_half, float>{}(std::forward<Ts>(arg)...);
                else if(d_type == rocblas_datatype_bf16_r && compute_type == rocblas_datatype_f32_r)
                    return GEMM<rocblas_bfloat16, rocblas_bfloat16, float>{}(
                        std::forward<Ts>(arg)...);
            }
            else
            {
                switch(a_type)
                {
                case rocblas_datatype_f16_r:
                    return GEMM<rocblas_half>{}(std::forward<Ts>(arg)...);
                case rocblas_datatype_bf16_r:
                    return GEMM<rocblas_bfloat16>{}(std::forward<Ts>(arg)...);
                case rocblas_datatype_f32_r:
                    return GEMM<float>{}(std::forward<Ts>(arg)...);
                case rocblas_datatype_f64_r:
                    return GEMM<double>{}(std::forward<Ts>(arg)...);
                case rocblas_datatype_f32_c:
                    return GEMM<rocblas_float_complex>{}(std::forward<Ts>(arg)...);
                case rocblas_datatype_f64_c:
                    return GEMM<rocblas_double_complex>{}(std::forward<Ts>(arg)...);
                default:
                    break;
                }
            }
        }
        throw rocblas_status_not_implemented;
    }

    rocblas_status rocblas_gemm_ext2_impl(rocblas_handle    handle,
                                          rocblas_int       m,
                                          rocblas_int       n,
                                          rocblas_int       k,
                                          const void*       alpha,
                                          const void*       a,
                                          rocblas_datatype  a_type,
                                          rocblas_stride    row_stride_a,
                                          rocblas_stride    col_stride_a,
                                          const void*       b,
                                          rocblas_datatype  b_type,
                                          rocblas_stride    row_stride_b,
                                          rocblas_stride    col_stride_b,
                                          const void*       beta,
                                          const void*       c,
                                          rocblas_datatype  c_type,
                                          rocblas_stride    row_stride_c,
                                          rocblas_stride    col_stride_c,
                                          void*             d,
                                          rocblas_datatype  d_type,
                                          rocblas_stride    row_stride_d,
                                          rocblas_stride    col_stride_d,
                                          rocblas_datatype  compute_type,
                                          rocblas_gemm_algo algo,
                                          int32_t           solution_index,
                                          uint32_t          flags)
    {
        if(!handle)
            return rocblas_status_invalid_handle;

        const bool HPA = compute_type == rocblas_datatype_f32_r
                         && (a_type == rocblas_datatype_f16_r || a_type == rocblas_datatype_bf16_r);

        // Copy alpha and beta to host if on device
        rocblas_union_t alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h, beta_h, k, compute_type));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        if(!handle->is_device_memory_size_query())
        {
            // Perform logging
            auto layer_mode = handle->layer_mode;
            if(layer_mode
               & (rocblas_layer_mode_log_trace | rocblas_layer_mode_log_bench
                  | rocblas_layer_mode_log_profile))
            {
                auto a_type_string       = rocblas_datatype_string(a_type);
                auto b_type_string       = rocblas_datatype_string(b_type);
                auto c_type_string       = rocblas_datatype_string(c_type);
                auto d_type_string       = rocblas_datatype_string(d_type);
                auto compute_type_string = rocblas_datatype_string(compute_type);

                if(layer_mode & rocblas_layer_mode_log_trace)
                {
                    rocblas_internal_ostream alphass, betass;
                    if(log_trace_alpha_beta_ex(compute_type, alpha, beta, alphass, betass)
                       == rocblas_status_success)
                    {
                        log_trace(handle,
                                  "rocblas_gemm_ext2",
                                  m,
                                  n,
                                  k,
                                  alphass.str(),
                                  a,
                                  a_type_string,
                                  row_stride_a,
                                  col_stride_a,
                                  b,
                                  b_type_string,
                                  row_stride_b,
                                  col_stride_b,
                                  betass.str(),
                                  c,
                                  c_type_string,
                                  row_stride_c,
                                  col_stride_c,
                                  d,
                                  d_type_string,
                                  row_stride_d,
                                  col_stride_d,
                                  compute_type_string,
                                  algo,
                                  solution_index,
                                  rocblas_gemm_flags(flags));
                    }
                }

                if(layer_mode & rocblas_layer_mode_log_bench)
                {
                    std::string alphas, betas;
                    if(log_bench_alpha_beta_ex(compute_type, alpha, beta, alphas, betas)
                       == rocblas_status_success)
                    {
                        log_bench(handle,
                                  "./rocblas-bench -f gemm_ext2",
                                  "-m",
                                  m,
                                  "-n",
                                  n,
                                  "-k",
                                  k,
                                  alphas,
                                  "--a_type",
                                  a_type_string,
                                  "--row_stride_a",
                                  row_stride_a,
                                  "--col_stride_a",
                                  col_stride_a,
                                  "--b_type",
                                  b_type_string,
                                  "--row_stride_b",
                                  row_stride_b,
                                  "--col_stride_b",
                                  col_stride_b,
                                  betas,
                                  "--c_type",
                                  c_type_string,
                                  "--row_stride_c",
                                  row_stride_c,
                                  "--col_stride_c",
                                  col_stride_c,
                                  "--d_type",
                                  d_type_string,
                                  "--row_stride_d",
                                  row_stride_d,
                                  "--col_stride_d",
                                  col_stride_d,
                                  "--compute_type",
                                  compute_type_string,
                                  "--algo",
                                  algo,
                                  "--solution_index",
                                  solution_index,
                                  "--flags",
                                  flags);
                    }
                }

                if(layer_mode & rocblas_layer_mode_log_profile)
                {
                    log_profile(handle,
                                "rocblas_gemm_ext2",
                                "a_type",
                                a_type_string,
                                "b_type",
                                b_type_string,
                                "c_type",
                                c_type_string,
                                "d_type",
                                d_type_string,
                                "compute_type",
                                compute_type_string,
                                "M",
                                m,
                                "N",
                                n,
                                "K",
                                k,
                                "alpha",
                                value_category(alpha, compute_type),
                                "row_stride_a",
                                row_stride_a,
                                "col_stride_a",
                                col_stride_a,
                                "row_stride_b",
                                row_stride_b,
                                "col_stride_b",
                                col_stride_b,
                                "beta",
                                value_category(beta, compute_type),
                                "row_stride_c",
                                row_stride_c,
                                "col_stride_c",
                                col_stride_c,
                                "row_stride_d",
                                row_stride_d,
                                "col_stride_d",
                                col_stride_d,
                                "algo",
                                algo,
                                "solution_index",
                                solution_index,
                                "flags",
                                rocblas_gemm_flags(flags));
                }
            }
        }

        try
        {
            // sizes must not be negative
            if(m < 0 || n < 0 || k < 0)
                return rocblas_status_invalid_size;

            // We do not check strides for validity like BLAS checks leading dimensions
            // for validity, since we allow arbitrary strides including 0.

            // quick return
            // Note: k==0 is not a quick return, because C still has to be multiplied by beta
            if(!m || !n)
                return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                             : rocblas_status_success;

            // pointers must be valid
            if(!d || (!c && beta))
                return rocblas_status_invalid_pointer;
            if(k && (!a || !b) && alpha)
                return rocblas_status_invalid_pointer;

            rocblas_stride batch_stride = 1; // can be changed to 0 when Tensile bug is fixed
            rocblas_int    offset       = 0;
            rocblas_int    batch_count  = 1;

            auto status = rocblas_status_not_implemented;

            std::unique_ptr<void, void (*)(void*)> erase{
                nullptr, [](auto) { rocblas_suppress_tensile_error_messages() = false; }};
            rocblas_suppress_tensile_error_messages() = true;

            auto gemm_ext2 = [&] {
                return rocblas_gemm_ext2_template(handle,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  a,
                                                  a_type,
                                                  offset,
                                                  row_stride_a,
                                                  col_stride_a,
                                                  batch_stride,
                                                  b,
                                                  b_type,
                                                  offset,
                                                  row_stride_b,
                                                  col_stride_b,
                                                  batch_stride,
                                                  beta,
                                                  c,
                                                  c_type,
                                                  offset,
                                                  row_stride_c,
                                                  col_stride_c,
                                                  batch_stride,
                                                  d,
                                                  d_type,
                                                  offset,
                                                  row_stride_d,
                                                  col_stride_d,
                                                  batch_stride,
                                                  batch_count,
                                                  compute_type,
                                                  flags);
            };

            if(HPA && !handle->is_device_memory_size_query())
            {
                // Allocate GSU workspace in handle
                auto gsu_malloc = handle->gsu_malloc();
                status          = gemm_ext2();
            }
            else
            {
                status = gemm_ext2();
            }
            if(status == rocblas_status_success || handle->is_device_memory_size_query())
                return status;

            throw status;
        }
        catch(...)
        {
            // Fall back on slow, naive algorithm if not implemented in Tensile
            static auto& once = rocblas_cerr
                                << "\nWarning: Using slow on-host algorithm, because it "
                                   "is not implemented in Tensile yet."
                                << std::endl;

            size_t size_a = size_t(m - 1) * row_stride_a + size_t(k - 1) * col_stride_a + 1;
            size_t size_b = size_t(k - 1) * row_stride_b + size_t(n - 1) * col_stride_b + 1;
            size_t size_c = size_t(m - 1) * row_stride_c + size_t(n - 1) * col_stride_c + 1;
            size_t size_d = size_t(m - 1) * row_stride_d + size_t(n - 1) * col_stride_d + 1;

            std::unique_ptr<char[]> ha{a ? new char[rocblas_sizeof_datatype(a_type) * size_a]
                                         : nullptr};
            std::unique_ptr<char[]> hb{b ? new char[rocblas_sizeof_datatype(b_type) * size_b]
                                         : nullptr};
            std::unique_ptr<char[]> hc{c ? new char[rocblas_sizeof_datatype(c_type) * size_c]
                                         : nullptr};
            std::unique_ptr<char[]> hd{new char[rocblas_sizeof_datatype(d_type) * size_d]};

            if(a)
                RETURN_IF_HIP_ERROR(hipMemcpy(
                    ha.get(), a, rocblas_sizeof_datatype(a_type) * size_a, hipMemcpyDeviceToHost));
            if(b)
                RETURN_IF_HIP_ERROR(hipMemcpy(
                    hb.get(), b, rocblas_sizeof_datatype(b_type) * size_b, hipMemcpyDeviceToHost));
            if(c)
                RETURN_IF_HIP_ERROR(hipMemcpy(
                    hc.get(), c, rocblas_sizeof_datatype(c_type) * size_c, hipMemcpyDeviceToHost));

            rocblas_status status = gemm_dispatch<reference_gemm_ext2_call>(a_type,
                                                                            b_type,
                                                                            c_type,
                                                                            d_type,
                                                                            compute_type,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            alpha,
                                                                            ha.get(),
                                                                            row_stride_a,
                                                                            col_stride_a,
                                                                            hb.get(),
                                                                            row_stride_b,
                                                                            col_stride_b,
                                                                            beta,
                                                                            hc.get(),
                                                                            row_stride_c,
                                                                            col_stride_c,
                                                                            hd.get(),
                                                                            row_stride_d,
                                                                            col_stride_d);

            if(status == rocblas_status_success)
                RETURN_IF_HIP_ERROR(hipMemcpy(
                    d, hd.get(), rocblas_sizeof_datatype(d_type) * size_d, hipMemcpyHostToDevice));

            return status;
        }
    }
} // namespace

extern "C" rocblas_status rocblas_gemm_ext2(rocblas_handle    handle,
                                            rocblas_int       m,
                                            rocblas_int       n,
                                            rocblas_int       k,
                                            const void*       alpha,
                                            const void*       a,
                                            rocblas_datatype  a_type,
                                            rocblas_stride    row_stride_a,
                                            rocblas_stride    col_stride_a,
                                            const void*       b,
                                            rocblas_datatype  b_type,
                                            rocblas_stride    row_stride_b,
                                            rocblas_stride    col_stride_b,
                                            const void*       beta,
                                            const void*       c,
                                            rocblas_datatype  c_type,
                                            rocblas_stride    row_stride_c,
                                            rocblas_stride    col_stride_c,
                                            void*             d,
                                            rocblas_datatype  d_type,
                                            rocblas_stride    row_stride_d,
                                            rocblas_stride    col_stride_d,
                                            rocblas_datatype  compute_type,
                                            rocblas_gemm_algo algo,
                                            int32_t           solution_index,
                                            uint32_t          flags)
try
{
    return rocblas_gemm_ext2_impl(handle,
                                  m,
                                  n,
                                  k,
                                  alpha,
                                  a,
                                  a_type,
                                  row_stride_a,
                                  col_stride_a,
                                  b,
                                  b_type,
                                  row_stride_b,
                                  col_stride_b,
                                  beta,
                                  c,
                                  c_type,
                                  row_stride_c,
                                  col_stride_c,
                                  d,
                                  d_type,
                                  row_stride_d,
                                  col_stride_d,
                                  compute_type,
                                  algo,
                                  solution_index,
                                  flags);
}
catch(...)
{
    return exception_to_rocblas_status();
}

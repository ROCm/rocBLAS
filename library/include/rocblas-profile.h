#ifndef _ROCBLAS_PROFILE_H_
#define _ROCBLAS_PROFILE_H_

#include "rocblas-export.h"
#include "rocblas-types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief BLAS EX API

    \details
    GEMM_EX_PROFILE is an extension of GEMM_EX with the ability to send in hipEvents to Tensile for
    more accurate kernel profiling.

    @param[in]
    handle    [rocblas_handle]
              handle to the rocblas library context queue.
    @param[in]
    transA    [rocblas_operation]
              specifies the form of op( A ).
    @param[in]
    transB    [rocblas_operation]
              specifies the form of op( B ).
    @param[in]
    m         [rocblas_int]
              matrix dimension m.
    @param[in]
    n         [rocblas_int]
              matrix dimension n.
    @param[in]
    k         [rocblas_int]
              matrix dimension k.
    @param[in]
    alpha     [const void *]
              device pointer or host pointer specifying the scalar alpha. Same datatype as compute_type.
    @param[in]
    a         [void *]
              device pointer storing matrix A.
    @param[in]
    a_type    [rocblas_datatype]
              specifies the datatype of matrix A.
    @param[in]
    lda       [rocblas_int]
              specifies the leading dimension of A.
    @param[in]
    b         [void *]
              device pointer storing matrix B.
    @param[in]
    b_type    [rocblas_datatype]
              specifies the datatype of matrix B.
    @param[in]
    ldb       [rocblas_int]
              specifies the leading dimension of B.
    @param[in]
    beta      [const void *]
              device pointer or host pointer specifying the scalar beta. Same datatype as compute_type.
    @param[in]
    c         [void *]
              device pointer storing matrix C.
    @param[in]
    c_type    [rocblas_datatype]
              specifies the datatype of matrix C.
    @param[in]
    ldc       [rocblas_int]
              specifies the leading dimension of C.
    @param[out]
    d         [void *]
              device pointer storing matrix D.
    @param[in]
    d_type    [rocblas_datatype]
              specifies the datatype of matrix D.
    @param[in]
    ldd       [rocblas_int]
              specifies the leading dimension of D.
    @param[in]
    compute_type
              [rocblas_datatype]
              specifies the datatype of computation.
    @param[in]
    algo      [rocblas_gemm_algo]
              enumerant specifying the algorithm type.
    @param[in]
    solution_index
              [int32_t]
              reserved for future use.
    @param[in]
    flags     [uint32_t]
              reserved for future use.

    @param[in, out]
    startEvent         [hipEvent_t *]
              pointer to hipEvent used for marking start of profiling.

    @param[in, out]
    stopEvent          [hipEvent_t *]
              pointer to hipEvent used for marking end of profiling.

    ********************************************************************/
ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex_profile(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const void*       alpha,
                                                      const void*       a,
                                                      rocblas_datatype  a_type,
                                                      rocblas_int       lda,
                                                      const void*       b,
                                                      rocblas_datatype  b_type,
                                                      rocblas_int       ldb,
                                                      const void*       beta,
                                                      const void*       c,
                                                      rocblas_datatype  c_type,
                                                      rocblas_int       ldc,
                                                      void*             d,
                                                      rocblas_datatype  d_type,
                                                      rocblas_int       ldd,
                                                      rocblas_datatype  compute_type,
                                                      rocblas_gemm_algo algo,
                                                      int32_t           solution_index,
                                                      uint32_t          flags,
                                                      hipEvent_t*       startEvent,
                                                      hipEvent_t*       stopEvent);

#ifdef __cplusplus
}
#endif

#endif /* _ROCBLAS_PROFILE_H_ */

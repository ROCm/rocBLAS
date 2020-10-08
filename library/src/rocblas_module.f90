!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020 Advanced Micro Devices, Inc.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module rocblas_enums
    use iso_c_binding

    !!!!!!!!!!!!!!!!!!!!!!!
    !    rocBLAS types    !
    !!!!!!!!!!!!!!!!!!!!!!!
    enum, bind(c)
      enumerator :: rocblas_operation_none = 111
        enumerator :: rocblas_operation_transpose = 112
        enumerator :: rocblas_operation_conjugate_transpose = 113
    end enum

    enum, bind(c)
        enumerator :: rocblas_fill_upper = 121
        enumerator :: rocblas_fill_lower = 122
        enumerator :: rocblas_fill_full  = 123
    end enum

    enum, bind(c)
        enumerator :: rocblas_diagonal_non_unit = 131
        enumerator :: rocblas_diagonal_unit     = 132
    end enum

    enum, bind(c)
        enumerator :: rocblas_side_left  = 141
        enumerator :: rocblas_side_right = 142
        enumerator :: rocblas_side_both  = 143
    end enum

    enum, bind(c)
        enumerator :: rocblas_status_success         = 0
        enumerator :: rocblas_status_invalid_handle  = 1
        enumerator :: rocblas_status_not_implemented = 2
        enumerator :: rocblas_status_invalid_pointer = 3
        enumerator :: rocblas_status_invalid_size    = 4
        enumerator :: rocblas_status_memory_error    = 5
        enumerator :: rocblas_status_internal_error  = 6
        enumerator :: rocblas_status_perf_degraded   = 7
        enumerator :: rocblas_status_size_query_mismatch = 8
        enumerator :: rocblas_status_size_increased      = 9
        enumerator :: rocblas_status_size_unchanged      = 10
        enumerator :: rocblas_status_invalid_value       = 11
        enumerator :: rocblas_status_continue            = 12
    end enum

    enum, bind(c)
        enumerator :: rocblas_datatype_f16_r  = 150
        enumerator :: rocblas_datatype_f32_r  = 151
        enumerator :: rocblas_datatype_f64_r  = 152
        enumerator :: rocblas_datatype_f16_c  = 153
        enumerator :: rocblas_datatype_f32_c  = 154
        enumerator :: rocblas_datatype_f64_c  = 155
        enumerator :: rocblas_datatype_i8_r   = 160
        enumerator :: rocblas_datatype_u8_r   = 161
        enumerator :: rocblas_datatype_i32_r  = 162
        enumerator :: rocblas_datatype_u32_r  = 163
        enumerator :: rocblas_datatype_i8_c   = 164
        enumerator :: rocblas_datatype_u8_c   = 165
        enumerator :: rocblas_datatype_i32_c  = 166
        enumerator :: rocblas_datatype_u32_c  = 167
        enumerator :: rocblas_datatype_bf16_r = 168
        enumerator :: rocblas_datatype_bf16_c = 169
    end enum

    enum, bind(c)
        enumerator :: rocblas_pointer_mode_host = 0
        enumerator :: rocblas_pointer_mode_device = 1
    end enum

    enum, bind(c)
        enumerator :: rocblas_gemm_algo_standard = 0
    end enum

end module rocblas_enums

module rocblas
    use iso_c_binding

    !!!!!!!!!!!!!!!!!!!!!!!
    ! rocblas-auxiliary.h !
    !!!!!!!!!!!!!!!!!!!!!!!

    interface
        function rocblas_create_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocblas_create_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocblas_create_handle
    end interface

    interface
        function rocblas_destroy_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocblas_destroy_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocblas_destroy_handle
    end interface

    interface
        function rocblas_set_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_set_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocblas_set_stream
    end interface

    interface
        function rocblas_get_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_get_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocblas_get_stream
    end interface

    interface
        function rocblas_set_pointer_mode(handle, pointer_mode) &
                result(c_int) &
                bind(c, name = 'rocblas_set_pointer_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: pointer_mode
        end function rocblas_set_pointer_mode
    end interface

    interface
        function rocblas_get_pointer_mode(handle, pointer_mode) &
                result(c_int) &
                bind(c, name = 'rocblas_get_pointer_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int) :: pointer_mode
        end function rocblas_get_pointer_mode
    end interface

    interface
        function rocblas_pointer_to_mode(ptr) &
                result(c_int) &
                bind(c, name = 'rocblas_pointer_to_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
        end function rocblas_pointer_to_mode
    end interface

    interface
        function rocblas_set_vector(n, elem_size, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_set_vector')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_set_vector
    end interface

    interface
        function rocblas_get_vector(n, elem_size, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_get_vector')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_get_vector
    end interface

    interface
        function rocblas_set_matrix(rows, cols, elem_size, a, lda, b, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_set_matrix')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
        end function rocblas_set_matrix
    end interface

    interface
        function rocblas_get_matrix(rows, cols, elem_size, a, lda, b, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_get_matrix')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
        end function rocblas_get_matrix
    end interface

    interface
        function rocblas_set_vector_async(n, elem_size, x, incx, y, incy, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_set_vector_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: stream
        end function rocblas_set_vector_async
    end interface

    interface
        function rocblas_get_vector_async(n, elem_size, x, incx, y, incy, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_get_vector_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: n
            integer(c_int), value :: elem_size
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: stream
        end function rocblas_get_vector_async
    end interface

    interface
        function rocblas_set_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_set_matrix_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
            type(c_ptr), value :: stream
        end function rocblas_set_matrix_async
    end interface

    interface
        function rocblas_get_matrix_async(rows, cols, elem_size, a, lda, b, ldb, stream) &
                result(c_int) &
                bind(c, name = 'rocblas_get_matrix_async')
            use iso_c_binding
            implicit none
            integer(c_int), value :: rows
            integer(c_int), value :: cols
            integer(c_int), value :: elem_size
            type(c_ptr), value :: a
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: ldb
            type(c_ptr), value :: stream
        end function rocblas_get_matrix_async
    end interface

    interface
        function rocblas_set_start_stop_events(handle, start_event, stop_event) &
                result(c_int) &
                bind(c, name = 'rocblas_set_start_stop_events')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: start_event
            type(c_ptr), value :: stop_event
        end function rocblas_set_start_stop_events
    end interface

    !!!!!!!!!!!!!!!!!!!!!!!
    ! rocblas-functions.h !
    !!!!!!!!!!!!!!!!!!!!!!!

    !--------!
    ! blas 1 !
    !--------!

    ! scal
    interface
        function rocblas_sscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_sscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_sscal
    end interface

    interface
        function rocblas_dscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dscal
    end interface

    interface
        function rocblas_cscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_cscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_cscal
    end interface

    interface
        function rocblas_zscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_zscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_zscal
    end interface

    interface
        function rocblas_csscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_csscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_csscal
    end interface

    interface
        function rocblas_zdscal(handle, n, alpha, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_zdscal')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_zdscal
    end interface

    ! scal_batched
    interface
        function rocblas_sscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_sscal_batched
    end interface

    interface
        function rocblas_dscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dscal_batched
    end interface

    interface
        function rocblas_cscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_cscal_batched
    end interface

    interface
        function rocblas_zscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_zscal_batched
    end interface

    interface
        function rocblas_csscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_csscal_batched
    end interface

    interface
        function rocblas_zdscal_batched(handle, n, alpha, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdscal_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_zdscal_batched
    end interface

    ! scal_strided_batched
    interface
        function rocblas_sscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_sscal_strided_batched
    end interface

    interface
        function rocblas_dscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dscal_strided_batched
    end interface

    interface
        function rocblas_cscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_cscal_strided_batched
    end interface

    interface
        function rocblas_zscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_zscal_strided_batched
    end interface

    interface
        function rocblas_csscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_csscal_strided_batched
    end interface

    interface
        function rocblas_zdscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdscal_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_zdscal_strided_batched
    end interface

    ! copy
    interface
        function rocblas_scopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_scopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_scopy
    end interface

    interface
        function rocblas_dcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dcopy
    end interface

    interface
        function rocblas_ccopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_ccopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_ccopy
    end interface

    interface
        function rocblas_zcopy(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zcopy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zcopy
    end interface

    ! copy_batched
    interface
        function rocblas_scopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_scopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_scopy_batched
    end interface

    interface
        function rocblas_dcopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dcopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dcopy_batched
    end interface

    interface
        function rocblas_ccopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ccopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_ccopy_batched
    end interface

    interface
        function rocblas_zcopy_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zcopy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zcopy_batched
    end interface

    ! copy_strided_batched
    interface
        function rocblas_scopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_scopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_scopy_strided_batched
    end interface

    interface
        function rocblas_dcopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dcopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dcopy_strided_batched
    end interface

    interface
        function rocblas_ccopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ccopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_ccopy_strided_batched
    end interface

    interface
        function rocblas_zcopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zcopy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zcopy_strided_batched
    end interface

    ! dot
    interface
        function rocblas_sdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_sdot
    end interface

    interface
        function rocblas_ddot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_ddot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_ddot
    end interface

    interface
        function rocblas_hdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_hdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_hdot
    end interface

    interface
        function rocblas_bfdot(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_bfdot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_bfdot
    end interface

    interface
        function rocblas_cdotu(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotu')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_cdotu
    end interface

    interface
        function rocblas_cdotc(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_cdotc
    end interface

    interface
        function rocblas_zdotu(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotu')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_zdotu
    end interface

    interface
        function rocblas_zdotc(handle, n, x, incx, y, incy, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: result
        end function rocblas_zdotc
    end interface

    ! dot_batched
    interface
        function rocblas_sdot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sdot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sdot_batched
    end interface

    interface
        function rocblas_ddot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_ddot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_ddot_batched
    end interface

    interface
        function rocblas_hdot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_hdot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_hdot_batched
    end interface

    interface
        function rocblas_bfdot_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_bfdot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_bfdot_batched
    end interface

    interface
        function rocblas_cdotu_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotu_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotu_batched
    end interface

    interface
        function rocblas_cdotc_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotc_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotc_batched
    end interface

    interface
        function rocblas_zdotu_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotu_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotu_batched
    end interface

    interface
        function rocblas_zdotc_batched(handle, n, x, incx, y, incy, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotc_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotc_batched
    end interface

    ! dot_strided_batched
    interface
        function rocblas_sdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sdot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sdot_strided_batched
    end interface

    interface
        function rocblas_ddot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_ddot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_ddot_strided_batched
    end interface

    interface
        function rocblas_hdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_hdot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_hdot_strided_batched
    end interface

    interface
        function rocblas_bfdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_bfdot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_bfdot_strided_batched
    end interface

    interface
        function rocblas_cdotu_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotu_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotu_strided_batched
    end interface

    interface
        function rocblas_cdotc_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_cdotc_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_cdotc_strided_batched
    end interface

    interface
        function rocblas_zdotu_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotu_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotu_strided_batched
    end interface

    interface
        function rocblas_zdotc_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_zdotc_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_zdotc_strided_batched
    end interface

    ! swap
    interface
        function rocblas_sswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_sswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_sswap
    end interface

    interface
        function rocblas_dswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dswap
    end interface

    interface
        function rocblas_cswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_cswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_cswap
    end interface

    interface
        function rocblas_zswap(handle, n, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zswap')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zswap
    end interface

    ! swap_batched
    interface
        function rocblas_sswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_sswap_batched
    end interface

    interface
        function rocblas_dswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dswap_batched
    end interface

    interface
        function rocblas_cswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_cswap_batched
    end interface

    interface
        function rocblas_zswap_batched(handle, n, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zswap_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zswap_batched
    end interface

    ! swap_strided_batched
    interface
        function rocblas_sswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_sswap_strided_batched
    end interface

    interface
        function rocblas_dswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dswap_strided_batched
    end interface

    interface
        function rocblas_cswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_cswap_strided_batched
    end interface

    interface
        function rocblas_zswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zswap_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zswap_strided_batched
    end interface

    ! axpy
    interface
        function rocblas_haxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_haxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_haxpy
    end interface

    interface
        function rocblas_saxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_saxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_saxpy
    end interface

    interface
        function rocblas_daxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_daxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_daxpy
    end interface

    interface
        function rocblas_caxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_caxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_caxpy
    end interface

    interface
        function rocblas_zaxpy(handle, n, alpha, x, incx, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zaxpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zaxpy
    end interface

    ! axpy_batched
    interface
        function rocblas_haxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_haxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_haxpy_batched
    end interface

    interface
        function rocblas_saxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_saxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_saxpy_batched
    end interface

    interface
        function rocblas_daxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_daxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_daxpy_batched
    end interface

    interface
        function rocblas_caxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_caxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_caxpy_batched
    end interface

    interface
        function rocblas_zaxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zaxpy_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zaxpy_batched
    end interface

    ! axpy_strided_batched
    interface
        function rocblas_haxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_haxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_haxpy_strided_batched
    end interface

    interface
        function rocblas_saxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_saxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_saxpy_strided_batched
    end interface

    interface
        function rocblas_daxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_daxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_daxpy_strided_batched
    end interface

    interface
        function rocblas_caxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_caxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_caxpy_strided_batched
    end interface

    interface
        function rocblas_zaxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zaxpy_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zaxpy_strided_batched
    end interface

    ! asum
    interface
        function rocblas_sasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_sasum
    end interface

    interface
        function rocblas_dasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dasum
    end interface

    interface
        function rocblas_scasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_scasum
    end interface

    interface
        function rocblas_dzasum(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dzasum')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dzasum
    end interface

    ! asum_batched
    interface
        function rocblas_sasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sasum_batched
    end interface

    interface
        function rocblas_dasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dasum_batched
    end interface

    interface
        function rocblas_scasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scasum_batched
    end interface

    interface
        function rocblas_dzasum_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dzasum_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dzasum_batched
    end interface

    ! asum_strided_batched
    interface
        function rocblas_sasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_sasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_sasum_strided_batched
    end interface

    interface
        function rocblas_dasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dasum_strided_batched
    end interface

    interface
        function rocblas_scasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scasum_strided_batched
    end interface

    interface
        function rocblas_dzasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dzasum_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dzasum_strided_batched
    end interface

    ! nrm2
    interface
        function rocblas_snrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_snrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_snrm2
    end interface

    interface
        function rocblas_dnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dnrm2
    end interface

    interface
        function rocblas_scnrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scnrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_scnrm2
    end interface

    interface
        function rocblas_dznrm2(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dznrm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_dznrm2
    end interface

    ! nrm2_batched
    interface
        function rocblas_snrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_snrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_snrm2_batched
    end interface

    interface
        function rocblas_dnrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dnrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dnrm2_batched
    end interface

    interface
        function rocblas_scnrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scnrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scnrm2_batched
    end interface

    interface
        function rocblas_dznrm2_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dznrm2_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dznrm2_batched
    end interface

    ! nrm2_strided_batched
    interface
        function rocblas_snrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_snrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_snrm2_strided_batched
    end interface

    interface
        function rocblas_dnrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dnrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dnrm2_strided_batched
    end interface

    interface
        function rocblas_scnrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_scnrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_scnrm2_strided_batched
    end interface

    interface
        function rocblas_dznrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_dznrm2_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_dznrm2_strided_batched
    end interface

    ! amax
    interface
        function rocblas_isamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_isamax
    end interface

    interface
        function rocblas_idamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_idamax
    end interface

    interface
        function rocblas_icamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_icamax
    end interface

    interface
        function rocblas_izamax(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamax')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_izamax
    end interface

    ! amax_batched
    interface
        function rocblas_isamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamax_batched
    end interface

    interface
        function rocblas_idamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamax_batched
    end interface

    interface
        function rocblas_icamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamax_batched
    end interface

    interface
        function rocblas_izamax_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamax_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamax_batched
    end interface

    ! amax_strided_batched
    interface
        function rocblas_isamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamax_strided_batched
    end interface

    interface
        function rocblas_idamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamax_strided_batched
    end interface

    interface
        function rocblas_icamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamax_strided_batched
    end interface

    interface
        function rocblas_izamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamax_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamax_strided_batched
    end interface

    ! amin
    interface
        function rocblas_isamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_isamin
    end interface

    interface
        function rocblas_idamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_idamin
    end interface

    interface
        function rocblas_icamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_icamin
    end interface

    interface
        function rocblas_izamin(handle, n, x, incx, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamin')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: result
        end function rocblas_izamin
    end interface

    ! amin_batched
    interface
        function rocblas_isamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamin_batched
    end interface

    interface
        function rocblas_idamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamin_batched
    end interface

    interface
        function rocblas_icamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamin_batched
    end interface

    interface
        function rocblas_izamin_batched(handle, n, x, incx, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamin_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamin_batched
    end interface

    ! amin_strided_batched
    interface
        function rocblas_isamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_isamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_isamin_strided_batched
    end interface

    interface
        function rocblas_idamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_idamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_idamin_strided_batched
    end interface

    interface
        function rocblas_icamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_icamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_icamin_strided_batched
    end interface

    interface
        function rocblas_izamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result) &
                result(c_int) &
                bind(c, name = 'rocblas_izamin_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
            type(c_ptr), value :: result
        end function rocblas_izamin_strided_batched
    end interface

    ! rot
    interface
        function rocblas_srot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_srot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_srot
    end interface

    interface
        function rocblas_drot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_drot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_drot
    end interface

    interface
        function rocblas_crot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_crot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_crot
    end interface

    interface
        function rocblas_csrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_csrot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_csrot
    end interface

    interface
        function rocblas_zrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_zrot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_zrot
    end interface

    interface
        function rocblas_zdrot(handle, n, x, incx, y, incy, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_zdrot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_zdrot
    end interface

    ! rot_batched
    interface
        function rocblas_srot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_srot_batched
    end interface

    interface
        function rocblas_drot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_drot_batched
    end interface

    interface
        function rocblas_crot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_crot_batched
    end interface

    interface
        function rocblas_csrot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csrot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_csrot_batched
    end interface

    interface
        function rocblas_zrot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zrot_batched
    end interface

    interface
        function rocblas_zdrot_batched(handle, n, x, incx, y, incy, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdrot_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zdrot_batched
    end interface

    ! rot_strided_batched
    interface
        function rocblas_srot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_srot_strided_batched
    end interface

    interface
        function rocblas_drot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_drot_strided_batched
    end interface

    interface
        function rocblas_crot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_crot_strided_batched
    end interface

    interface
        function rocblas_csrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csrot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_csrot_strided_batched
    end interface

    interface
        function rocblas_zrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zrot_strided_batched
    end interface

    interface
        function rocblas_zdrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdrot_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zdrot_strided_batched
    end interface

    ! rotg
    interface
        function rocblas_srotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_srotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_srotg
    end interface

    interface
        function rocblas_drotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_drotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_drotg
    end interface

    interface
        function rocblas_crotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_crotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_crotg
    end interface

    interface
        function rocblas_zrotg(handle, a, b, c, s) &
                result(c_int) &
                bind(c, name = 'rocblas_zrotg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
        end function rocblas_zrotg
    end interface

    ! rotg_batched
    interface
        function rocblas_srotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_srotg_batched
    end interface

    interface
        function rocblas_drotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_drotg_batched
    end interface

    interface
        function rocblas_crotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_crotg_batched
    end interface

    interface
        function rocblas_zrotg_batched(handle, a, b, c, s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrotg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            type(c_ptr), value :: b
            type(c_ptr), value :: c
            type(c_ptr), value :: s
            integer(c_int), value :: batch_count
        end function rocblas_zrotg_batched
    end interface

    ! rotg_strided_batched
    interface
        function rocblas_srotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int64_t), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int64_t), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int64_t), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int64_t), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_srotg_strided_batched
    end interface

    interface
        function rocblas_drotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int64_t), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int64_t), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int64_t), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int64_t), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_drotg_strided_batched
    end interface

    interface
        function rocblas_crotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_crotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int64_t), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int64_t), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int64_t), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int64_t), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_crotg_strided_batched
    end interface

    interface
        function rocblas_zrotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zrotg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: a
            integer(c_int64_t), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int64_t), value :: stride_b
            type(c_ptr), value :: c
            integer(c_int64_t), value :: stride_c
            type(c_ptr), value :: s
            integer(c_int64_t), value :: stride_s
            integer(c_int), value :: batch_count
        end function rocblas_zrotg_strided_batched
    end interface

    ! rotm
    interface
        function rocblas_srotm(handle, n, x, incx, y, incy, param) &
                result(c_int) &
                bind(c, name = 'rocblas_srotm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
        end function rocblas_srotm
    end interface

    interface
        function rocblas_drotm(handle, n, x, incx, y, incy, param) &
                result(c_int) &
                bind(c, name = 'rocblas_drotm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
        end function rocblas_drotm
    end interface

    ! rotm_batched
    interface
        function rocblas_srotm_batched(handle, n, x, incx, y, incy, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotm_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_srotm_batched
    end interface

    interface
        function rocblas_drotm_batched(handle, n, x, incx, y, incy, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotm_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_drotm_batched
    end interface

    ! rotm_strided_batched
    interface
        function rocblas_srotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotm_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: param
            integer(c_int64_t), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_srotm_strided_batched
    end interface

    interface
        function rocblas_drotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotm_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: param
            integer(c_int64_t), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_drotm_strided_batched
    end interface

    ! rotmg
    interface
        function rocblas_srotmg(handle, d1, d2, x1, y1, param) &
                result(c_int) &
                bind(c, name = 'rocblas_srotmg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
        end function rocblas_srotmg
    end interface

    interface
        function rocblas_drotmg(handle, d1, d2, x1, y1, param) &
                result(c_int) &
                bind(c, name = 'rocblas_drotmg')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
        end function rocblas_drotmg
    end interface

    ! rotmg_batched
    interface
        function rocblas_srotmg_batched(handle, d1, d2, x1, y1, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotmg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_srotmg_batched
    end interface

    interface
        function rocblas_drotmg_batched(handle, d1, d2, x1, y1, param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotmg_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            type(c_ptr), value :: d2
            type(c_ptr), value :: x1
            type(c_ptr), value :: y1
            type(c_ptr), value :: param
            integer(c_int), value :: batch_count
        end function rocblas_drotmg_batched
    end interface

    ! rotmg_strided_batched
    interface
        function rocblas_srotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
             y1, stride_y1, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_srotmg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            integer(c_int64_t), value :: stride_d1
            type(c_ptr), value :: d2
            integer(c_int64_t), value :: stride_d2
            type(c_ptr), value :: x1
            integer(c_int64_t), value :: stride_x1
            type(c_ptr), value :: y1
            integer(c_int64_t), value :: stride_y1
            type(c_ptr), value :: param
            integer(c_int64_t), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_srotmg_strided_batched
    end interface

    interface
        function rocblas_drotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
             y1, stride_y1, param, stride_param, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_drotmg_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: d1
            integer(c_int64_t), value :: stride_d1
            type(c_ptr), value :: d2
            integer(c_int64_t), value :: stride_d2
            type(c_ptr), value :: x1
            integer(c_int64_t), value :: stride_x1
            type(c_ptr), value :: y1
            integer(c_int64_t), value :: stride_y1
            type(c_ptr), value :: param
            integer(c_int64_t), value :: stride_param
            integer(c_int), value :: batch_count
        end function rocblas_drotmg_strided_batched
    end interface

    !--------!
    ! blas 2 !
    !--------!

    ! gbmv
    interface
        function rocblas_sgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_sgbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_sgbmv
    end interface

    interface
        function rocblas_dgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dgbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dgbmv
    end interface

    interface
        function rocblas_cgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_cgbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_cgbmv
    end interface

    interface
        function rocblas_zgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zgbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zgbmv
    end interface

    ! gbmv_batched
    interface
        function rocblas_sgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_sgbmv_batched
    end interface

    interface
        function rocblas_dgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dgbmv_batched
    end interface

    interface
        function rocblas_cgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_cgbmv_batched
    end interface

    interface
        function rocblas_zgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zgbmv_batched
    end interface

    ! gbmv_strided_batched
    interface
        function rocblas_sgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_sgbmv_strided_batched
    end interface

    interface
        function rocblas_dgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dgbmv_strided_batched
    end interface

    interface
        function rocblas_cgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_cgbmv_strided_batched
    end interface

    interface
        function rocblas_zgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: kl
            integer(c_int), value :: ku
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zgbmv_strided_batched
    end interface

    ! gemv
    interface
        function rocblas_sgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_sgemv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_sgemv
    end interface

    interface
        function rocblas_dgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dgemv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dgemv
    end interface

    interface
        function rocblas_cgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_cgemv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_cgemv
    end interface

    interface
        function rocblas_zgemv(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zgemv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zgemv
    end interface

    ! gemv_batched
    interface
        function rocblas_sgemv_batched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgemv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_sgemv_batched
    end interface

    interface
        function rocblas_dgemv_batched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgemv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dgemv_batched
    end interface

    interface
        function rocblas_cgemv_batched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgemv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_cgemv_batched
    end interface

    interface
        function rocblas_zgemv_batched(handle, trans, m, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgemv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zgemv_batched
    end interface

    ! gemv_strided_batched
    interface
        function rocblas_sgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgemv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_sgemv_strided_batched
    end interface

    interface
        function rocblas_dgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgemv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dgemv_strided_batched
    end interface

    interface
        function rocblas_cgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgemv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_cgemv_strided_batched
    end interface

    interface
        function rocblas_zgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgemv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zgemv_strided_batched
    end interface

    ! hbmv
    interface
        function rocblas_chbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_chbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_chbmv
    end interface

    interface
        function rocblas_zhbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zhbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zhbmv
    end interface

    ! hbmv_batched
    interface
        function rocblas_chbmv_batched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_chbmv_batched
    end interface

    interface
        function rocblas_zhbmv_batched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zhbmv_batched
    end interface

    ! hbmv_strided_batched
    interface
        function rocblas_chbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_chbmv_strided_batched
    end interface

    interface
        function rocblas_zhbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zhbmv_strided_batched
    end interface

    ! hemv
    interface
        function rocblas_chemv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_chemv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_chemv
    end interface

    interface
        function rocblas_zhemv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zhemv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zhemv
    end interface

    ! hemv_batched
    interface
        function rocblas_chemv_batched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chemv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_chemv_batched
    end interface

    interface
        function rocblas_zhemv_batched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhemv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zhemv_batched
    end interface

    ! hemv_strided_batched
    interface
        function rocblas_chemv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chemv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_chemv_strided_batched
    end interface

    interface
        function rocblas_zhemv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhemv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zhemv_strided_batched
    end interface

    ! her
    interface
        function rocblas_cher(handle, uplo, n, alpha, &
                x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_cher')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_cher
    end interface

    interface
        function rocblas_zher(handle, uplo, n, alpha, &
                x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_zher')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_zher
    end interface

    ! her_batched
    interface
        function rocblas_cher_batched(handle, uplo, n, alpha, &
                x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cher_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_cher_batched
    end interface

    interface
        function rocblas_zher_batched(handle, uplo, n, alpha, &
                x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zher_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_zher_batched
    end interface

    ! her_strided_batched
    interface
        function rocblas_cher_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cher_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_cher_strided_batched
    end interface

    interface
        function rocblas_zher_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zher_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_zher_strided_batched
    end interface

    ! her2
    interface
        function rocblas_cher2(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_cher2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_cher2
    end interface

    interface
        function rocblas_zher2(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_zher2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_zher2
    end interface

    ! her2_batched
    interface
        function rocblas_cher2_batched(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cher2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_cher2_batched
    end interface

    interface
        function rocblas_zher2_batched(handle, uplo, n, alpha, &
                x, incx, y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zher2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_zher2_batched
    end interface

    ! her2_strided_batched
    interface
        function rocblas_cher2_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cher2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_cher2_strided_batched
    end interface

    interface
        function rocblas_zher2_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zher2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_zher2_strided_batched
    end interface

    ! hpmv
    interface
        function rocblas_chpmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_chpmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_chpmv
    end interface

    interface
        function rocblas_zhpmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zhpmv
    end interface

    ! hpmv_batched
    interface
        function rocblas_chpmv_batched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chpmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_chpmv_batched
    end interface

    interface
        function rocblas_zhpmv_batched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zhpmv_batched
    end interface

    ! hpmv_strided_batched
    interface
        function rocblas_chpmv_strided_batched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chpmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_chpmv_strided_batched
    end interface

    interface
        function rocblas_zhpmv_strided_batched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zhpmv_strided_batched
    end interface

    ! hpr
    interface
        function rocblas_chpr(handle, uplo, n, alpha, &
                x, incx, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_chpr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
        end function rocblas_chpr
    end interface

    interface
        function rocblas_zhpr(handle, uplo, n, alpha, &
                x, incx, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
        end function rocblas_zhpr
    end interface

    ! hpr_batched
    interface
        function rocblas_chpr_batched(handle, uplo, n, alpha, &
                x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chpr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_chpr_batched
    end interface

    interface
        function rocblas_zhpr_batched(handle, uplo, n, alpha, &
                x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_zhpr_batched
    end interface

    ! hpr_strided_batched
    interface
        function rocblas_chpr_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chpr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_chpr_strided_batched
    end interface

    interface
        function rocblas_zhpr_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_zhpr_strided_batched
    end interface

    ! hpr2
    interface
        function rocblas_chpr2(handle, uplo, n, alpha, &
                x, incx, y, incy, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_chpr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
        end function rocblas_chpr2
    end interface

    interface
        function rocblas_zhpr2(handle, uplo, n, alpha, &
                x, incx, y, incy, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
        end function rocblas_zhpr2
    end interface

    ! hpr2_batched
    interface
        function rocblas_chpr2_batched(handle, uplo, n, alpha, &
                x, incx, y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chpr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_chpr2_batched
    end interface

    interface
        function rocblas_zhpr2_batched(handle, uplo, n, alpha, &
                x, incx, y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_zhpr2_batched
    end interface

    ! hpr2_strided_batched
    interface
        function rocblas_chpr2_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chpr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_chpr2_strided_batched
    end interface

    interface
        function rocblas_zhpr2_strided_batched(handle, uplo, n, alpha, &
                x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhpr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_zhpr2_strided_batched
    end interface

    ! trmv
    interface
        function rocblas_strmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_strmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_strmv
    end interface

    interface
        function rocblas_dtrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dtrmv
    end interface

    interface
        function rocblas_ctrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ctrmv
    end interface

    interface
        function rocblas_ztrmv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ztrmv
    end interface

    ! trmv_batched
    interface
        function rocblas_strmv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_strmv_batched
    end interface

    interface
        function rocblas_dtrmv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dtrmv_batched
    end interface

    interface
        function rocblas_ctrmv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ctrmv_batched
    end interface

    interface
        function rocblas_ztrmv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ztrmv_batched
    end interface

    ! trmv_strided_batched
    interface
        function rocblas_strmv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_strmv_strided_batched
    end interface

    interface
        function rocblas_dtrmv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dtrmv_strided_batched
    end interface

    interface
        function rocblas_ctrmv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ctrmv_strided_batched
    end interface

    interface
        function rocblas_ztrmv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ztrmv_strided_batched
    end interface

    ! tpmv
    interface
        function rocblas_stpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_stpmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_stpmv
    end interface

    interface
        function rocblas_dtpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dtpmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dtpmv
    end interface

    interface
        function rocblas_ctpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ctpmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ctpmv
    end interface

    interface
        function rocblas_ztpmv(handle, uplo, transA, diag, m, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ztpmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ztpmv
    end interface

    ! tpmv_batched
    interface
        function rocblas_stpmv_batched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stpmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_stpmv_batched
    end interface

    interface
        function rocblas_dtpmv_batched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtpmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dtpmv_batched
    end interface

    interface
        function rocblas_ctpmv_batched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctpmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ctpmv_batched
    end interface

    interface
        function rocblas_ztpmv_batched(handle, uplo, transA, diag, m, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztpmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ztpmv_batched
    end interface

    ! tpmv_strided_batched
    interface
        function rocblas_stpmv_strided_batched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stpmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_stpmv_strided_batched
    end interface

    interface
        function rocblas_dtpmv_strided_batched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtpmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dtpmv_strided_batched
    end interface

    interface
        function rocblas_ctpmv_strided_batched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctpmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ctpmv_strided_batched
    end interface

    interface
        function rocblas_ztpmv_strided_batched(handle, uplo, transA, diag, m, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztpmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ztpmv_strided_batched
    end interface

    ! tbmv
    interface
        function rocblas_stbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_stbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_stbmv
    end interface

    interface
        function rocblas_dtbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dtbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dtbmv
    end interface

    interface
        function rocblas_ctbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ctbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ctbmv
    end interface

    interface
        function rocblas_ztbmv(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ztbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ztbmv
    end interface

    ! tbmv_batched
    interface
        function rocblas_stbmv_batched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_stbmv_batched
    end interface

    interface
        function rocblas_dtbmv_batched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dtbmv_batched
    end interface

    interface
        function rocblas_ctbmv_batched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ctbmv_batched
    end interface

    interface
        function rocblas_ztbmv_batched(handle, uplo, transA, diag, m, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ztbmv_batched
    end interface

    ! tbmv_strided_batched
    interface
        function rocblas_stbmv_strided_batched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_stbmv_strided_batched
    end interface

    interface
        function rocblas_dtbmv_strided_batched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dtbmv_strided_batched
    end interface

    interface
        function rocblas_ctbmv_strided_batched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ctbmv_strided_batched
    end interface

    interface
        function rocblas_ztbmv_strided_batched(handle, uplo, transA, diag, m, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ztbmv_strided_batched
    end interface

    ! tbsv
    interface
        function rocblas_stbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_stbsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_stbsv
    end interface

    interface
        function rocblas_dtbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dtbsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dtbsv
    end interface

    interface
        function rocblas_ctbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ctbsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ctbsv
    end interface

    interface
        function rocblas_ztbsv(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ztbsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ztbsv
    end interface

    ! tbsv_batched
    interface
        function rocblas_stbsv_batched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stbsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_stbsv_batched
    end interface

    interface
        function rocblas_dtbsv_batched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtbsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dtbsv_batched
    end interface

    interface
        function rocblas_ctbsv_batched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctbsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ctbsv_batched
    end interface

    interface
        function rocblas_ztbsv_batched(handle, uplo, transA, diag, n, k, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztbsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ztbsv_batched
    end interface

    ! tbsv_strided_batched
    interface
        function rocblas_stbsv_strided_batched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stbsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_stbsv_strided_batched
    end interface

    interface
        function rocblas_dtbsv_strided_batched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtbsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dtbsv_strided_batched
    end interface

    interface
        function rocblas_ctbsv_strided_batched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctbsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ctbsv_strided_batched
    end interface

    interface
        function rocblas_ztbsv_strided_batched(handle, uplo, transA, diag, n, k, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztbsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ztbsv_strided_batched
    end interface

    ! trsv
    interface
        function rocblas_strsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_strsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_strsv
    end interface

    interface
        function rocblas_dtrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dtrsv
    end interface

    interface
        function rocblas_ctrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ctrsv
    end interface

    interface
        function rocblas_ztrsv(handle, uplo, transA, diag, m, &
                A, lda, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ztrsv
    end interface

    ! trsv_batched
    interface
        function rocblas_strsv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_strsv_batched
    end interface

    interface
        function rocblas_dtrsv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dtrsv_batched
    end interface

    interface
        function rocblas_ctrsv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ctrsv_batched
    end interface

    interface
        function rocblas_ztrsv_batched(handle, uplo, transA, diag, m, &
                A, lda, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ztrsv_batched
    end interface

    ! trsv_strided_batched
    interface
        function rocblas_strsv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_strsv_strided_batched
    end interface

    interface
        function rocblas_dtrsv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dtrsv_strided_batched
    end interface

    interface
        function rocblas_ctrsv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ctrsv_strided_batched
    end interface

    interface
        function rocblas_ztrsv_strided_batched(handle, uplo, transA, diag, m, &
                A, lda, stride_A, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: m
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ztrsv_strided_batched
    end interface

    ! tpsv
    interface
        function rocblas_stpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_stpsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_stpsv
    end interface

    interface
        function rocblas_dtpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_dtpsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_dtpsv
    end interface

    interface
        function rocblas_ctpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ctpsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ctpsv
    end interface

    interface
        function rocblas_ztpsv(handle, uplo, transA, diag, n, &
                AP, x, incx) &
                result(c_int) &
                bind(c, name = 'rocblas_ztpsv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
        end function rocblas_ztpsv
    end interface

    ! tpsv_batched
    interface
        function rocblas_stpsv_batched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stpsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_stpsv_batched
    end interface

    interface
        function rocblas_dtpsv_batched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtpsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_dtpsv_batched
    end interface

    interface
        function rocblas_ctpsv_batched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctpsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ctpsv_batched
    end interface

    interface
        function rocblas_ztpsv_batched(handle, uplo, transA, diag, n, &
                AP, x, incx, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztpsv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
        end function rocblas_ztpsv_batched
    end interface

    ! tpsv_strided_batched
    interface
        function rocblas_stpsv_strided_batched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_stpsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_stpsv_strided_batched
    end interface

    interface
        function rocblas_dtpsv_strided_batched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtpsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_dtpsv_strided_batched
    end interface

    interface
        function rocblas_ctpsv_strided_batched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctpsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ctpsv_strided_batched
    end interface

    interface
        function rocblas_ztpsv_strided_batched(handle, uplo, transA, diag, n, &
                AP, stride_AP, x, incx, stride_x, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztpsv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_non_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            integer(c_int), value :: batch_count
        end function rocblas_ztpsv_strided_batched
    end interface

    ! symv
    interface
        function rocblas_ssymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_ssymv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_ssymv
    end interface

    interface
        function rocblas_dsymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dsymv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dsymv
    end interface

    interface
        function rocblas_csymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_csymv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_csymv
    end interface

    interface
        function rocblas_zsymv(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_zsymv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_zsymv
    end interface

    ! symv_batched
    interface
        function rocblas_ssymv_batched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssymv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_ssymv_batched
    end interface

    interface
        function rocblas_dsymv_batched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsymv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dsymv_batched
    end interface

    interface
        function rocblas_csymv_batched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csymv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_csymv_batched
    end interface

    interface
        function rocblas_zsymv_batched(handle, uplo, n, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsymv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_zsymv_batched
    end interface

    ! symv_strided_batched
    interface
        function rocblas_ssymv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssymv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_ssymv_strided_batched
    end interface

    interface
        function rocblas_dsymv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsymv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dsymv_strided_batched
    end interface

    interface
        function rocblas_csymv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csymv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_csymv_strided_batched
    end interface

    interface
        function rocblas_zsymv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsymv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_zsymv_strided_batched
    end interface

    ! spmv
    interface
        function rocblas_sspmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_sspmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_sspmv
    end interface

    interface
        function rocblas_dspmv(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dspmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dspmv
    end interface

    ! spmv_batched
    interface
        function rocblas_sspmv_batched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sspmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_sspmv_batched
    end interface

    interface
        function rocblas_dspmv_batched(handle, uplo, n, alpha, AP, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dspmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dspmv_batched
    end interface

    ! spmv_strided_batched
    interface
        function rocblas_sspmv_strided_batched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sspmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_sspmv_strided_batched
    end interface

    interface
        function rocblas_dspmv_strided_batched(handle, uplo, n, alpha, AP, stride_AP, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dspmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dspmv_strided_batched
    end interface

    ! sbmv
    interface
        function rocblas_ssbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_ssbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_ssbmv
    end interface

    interface
        function rocblas_dsbmv(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy) &
                result(c_int) &
                bind(c, name = 'rocblas_dsbmv')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
        end function rocblas_dsbmv
    end interface

    ! sbmv_batched
    interface
        function rocblas_ssbmv_batched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_ssbmv_batched
    end interface

    interface
        function rocblas_dsbmv_batched(handle, uplo, n, k, alpha, A, lda, &
                x, incx, beta, y, incy, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsbmv_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
        end function rocblas_dsbmv_batched
    end interface

    ! sbmv_strided_batched
    interface
        function rocblas_ssbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_ssbmv_strided_batched
    end interface

    interface
        function rocblas_dsbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsbmv_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: beta
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            integer(c_int), value :: batch_count
        end function rocblas_dsbmv_strided_batched
    end interface

    ! ger
    interface
        function rocblas_sger(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_sger')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_sger
    end interface

    interface
        function rocblas_dger(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_dger')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_dger
    end interface

    interface
        function rocblas_cgeru(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_cgeru')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_cgeru
    end interface

    interface
        function rocblas_cgerc(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_cgerc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_cgerc
    end interface

    interface
        function rocblas_zgeru(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_zgeru')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_zgeru
    end interface

    interface
        function rocblas_zgerc(handle, m, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_zgerc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_zgerc
    end interface

    ! ger_batched
    interface
        function rocblas_sger_batched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sger_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_sger_batched
    end interface

    interface
        function rocblas_dger_batched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dger_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_dger_batched
    end interface

    interface
        function rocblas_cgeru_batched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgeru_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_cgeru_batched
    end interface

    interface
        function rocblas_cgerc_batched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgerc_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_cgerc_batched
    end interface

    interface
        function rocblas_zgeru_batched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgeru_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_zgeru_batched
    end interface

    interface
        function rocblas_zgerc_batched(handle, m, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgerc_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_zgerc_batched
    end interface

    ! ger_strided_batched
    interface
        function rocblas_sger_strided_batched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sger_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_sger_strided_batched
    end interface

    interface
        function rocblas_dger_strided_batched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dger_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_dger_strided_batched
    end interface

    interface
        function rocblas_cgeru_strided_batched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgeru_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_cgeru_strided_batched
    end interface

    interface
        function rocblas_cgerc_strided_batched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgerc_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_cgerc_strided_batched
    end interface

    interface
        function rocblas_zgeru_strided_batched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgeru_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_zgeru_strided_batched
    end interface

    interface
        function rocblas_zgerc_strided_batched(handle, m, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgerc_strided_batched')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_zgerc_strided_batched
    end interface

    ! spr
    interface
        function rocblas_sspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_sspr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
        end function rocblas_sspr
    end interface

    interface
        function rocblas_dspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_dspr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
        end function rocblas_dspr
    end interface

    interface
        function rocblas_cspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_cspr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
        end function rocblas_cspr
    end interface

    interface
        function rocblas_zspr(handle, uplo, n, alpha, x, incx, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_zspr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
        end function rocblas_zspr
    end interface

    ! spr_batched
    interface
        function rocblas_sspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sspr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_sspr_batched
    end interface

    interface
        function rocblas_dspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dspr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_dspr_batched
    end interface

    interface
        function rocblas_cspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cspr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_cspr_batched
    end interface

    interface
        function rocblas_zspr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zspr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_zspr_batched
    end interface

    ! spr_strided_batched
    interface
        function rocblas_sspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sspr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_sspr_strided_batched
    end interface

    interface
        function rocblas_dspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dspr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_dspr_strided_batched
    end interface

    interface
        function rocblas_cspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cspr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_cspr_strided_batched
    end interface

    interface
        function rocblas_zspr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zspr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_zspr_strided_batched
    end interface

    ! spr2
    interface
        function rocblas_sspr2(handle, uplo, n, alpha, x, incx, &
                y, incy, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_sspr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
        end function rocblas_sspr2
    end interface

    interface
        function rocblas_dspr2(handle, uplo, n, alpha, x, incx, &
                y, incy, AP) &
                result(c_int) &
                bind(c, name = 'rocblas_dspr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
        end function rocblas_dspr2
    end interface

    ! spr2_batched
    interface
        function rocblas_sspr2_batched(handle, uplo, n, alpha, x, incx, &
                y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sspr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_sspr2_batched
    end interface

    interface
        function rocblas_dspr2_batched(handle, uplo, n, alpha, x, incx, &
                y, incy, AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dspr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: AP
            integer(c_int), value :: batch_count
        end function rocblas_dspr2_batched
    end interface

    ! spr2_strided_batched
    interface
        function rocblas_sspr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sspr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_sspr2_strided_batched
    end interface

    interface
        function rocblas_dspr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, AP, stride_AP, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dspr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: AP
            integer(c_int64_t), value :: stride_AP
            integer(c_int), value :: batch_count
        end function rocblas_dspr2_strided_batched
    end interface

    ! syr
    interface
        function rocblas_ssyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_ssyr
    end interface

    interface
        function rocblas_dsyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_dsyr
    end interface

    interface
        function rocblas_csyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_csyr
    end interface

    interface
        function rocblas_zsyr(handle, uplo, n, alpha, x, incx, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_zsyr
    end interface

    ! syr_batched
    interface
        function rocblas_ssyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_ssyr_batched
    end interface

    interface
        function rocblas_dsyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_dsyr_batched
    end interface

    interface
        function rocblas_csyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_csyr_batched
    end interface

    interface
        function rocblas_zsyr_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_zsyr_batched
    end interface

    ! syr_strided_batched
    interface
        function rocblas_ssyr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_ssyr_strided_batched
    end interface

    interface
        function rocblas_dsyr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_dsyr_strided_batched
    end interface

    interface
        function rocblas_csyr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_csyr_strided_batched
    end interface

    interface
        function rocblas_zsyr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_zsyr_strided_batched
    end interface

    ! syr2
    interface
        function rocblas_ssyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_ssyr2
    end interface

    interface
        function rocblas_dsyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_dsyr2
    end interface

    interface
        function rocblas_csyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_csyr2
    end interface

    interface
        function rocblas_zsyr2(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function rocblas_zsyr2
    end interface

    ! syr2_batched
    interface
        function rocblas_ssyr2_batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_ssyr2_batched
    end interface

    interface
        function rocblas_dsyr2_batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_dsyr2_batched
    end interface

    interface
        function rocblas_csyr2_batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_csyr2_batched
    end interface

    interface
        function rocblas_zsyr2_batched(handle, uplo, n, alpha, x, incx, &
                y, incy, A, lda, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr2_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int), value :: batch_count
        end function rocblas_zsyr2_batched
    end interface

    ! syr2_strided_batched
    interface
        function rocblas_ssyr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_ssyr2_strided_batched
    end interface

    interface
        function rocblas_dsyr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_dsyr2_strided_batched
    end interface

    interface
        function rocblas_csyr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_csyr2_strided_batched
    end interface

    interface
        function rocblas_zsyr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x, &
                y, incy, stride_y, A, lda, stride_A, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr2_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: y
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stride_y
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            integer(c_int), value :: batch_count
        end function rocblas_zsyr2_strided_batched
    end interface

    !--------!
    ! blas 3 !
    !--------!

    ! hemm
    interface
        function rocblas_chemm(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_chemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_chemm
    end interface

    interface
        function rocblas_zhemm(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zhemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zhemm
    end interface

    ! hemm_batched
    interface
        function rocblas_chemm_batched(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_chemm_batched
    end interface

    interface
        function rocblas_zhemm_batched(handle, side, uplo, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zhemm_batched
    end interface

    ! hemm_strided_batched
    interface
        function rocblas_chemm_strided_batched(handle, side, uplo, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_chemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_chemm_strided_batched
    end interface

    interface
        function rocblas_zhemm_strided_batched(handle, side, uplo, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zhemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zhemm_strided_batched
    end interface

    ! herk
    interface
        function rocblas_cherk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_cherk')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_cherk
    end interface

    interface
        function rocblas_zherk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zherk')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zherk
    end interface

    ! herk_batched
    interface
        function rocblas_cherk_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cherk_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_cherk_batched
    end interface

    interface
        function rocblas_zherk_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zherk_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zherk_batched
    end interface

    ! herk_strided_batched
    interface
        function rocblas_cherk_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cherk_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_cherk_strided_batched
    end interface

    interface
        function rocblas_zherk_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zherk_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zherk_strided_batched
    end interface

    ! her2k
    interface
        function rocblas_cher2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_cher2k')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_cher2k
    end interface

    interface
        function rocblas_zher2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zher2k')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zher2k
    end interface

    ! her2k_batched
    interface
        function rocblas_cher2k_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cher2k_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_cher2k_batched
    end interface

    interface
        function rocblas_zher2k_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zher2k_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zher2k_batched
    end interface

    ! her2k_strided_batched
    interface
        function rocblas_cher2k_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cher2k_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_cher2k_strided_batched
    end interface

    interface
        function rocblas_zher2k_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zher2k_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zher2k_strided_batched
    end interface

    ! herkx
    interface
        function rocblas_cherkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_cherkx')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_cherkx
    end interface

    interface
        function rocblas_zherkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zherkx')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zherkx
    end interface

    ! herkx_batched
    interface
        function rocblas_cherkx_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cherkx_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_cherkx_batched
    end interface

    interface
        function rocblas_zherkx_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zherkx_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zherkx_batched
    end interface

    ! herkx_strided_batched
    interface
        function rocblas_cherkx_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cherkx_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_cherkx_strided_batched
    end interface

    interface
        function rocblas_zherkx_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zherkx_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zherkx_strided_batched
    end interface

    ! symm
    interface
        function rocblas_ssymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_ssymm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_ssymm
    end interface

    interface
        function rocblas_dsymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_dsymm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_dsymm
    end interface

    interface
        function rocblas_csymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_csymm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_csymm
    end interface

    interface
        function rocblas_zsymm(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zsymm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zsymm
    end interface

    ! symm_batched
    interface
        function rocblas_ssymm_batched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssymm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_ssymm_batched
    end interface

    interface
        function rocblas_dsymm_batched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsymm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_dsymm_batched
    end interface

    interface
        function rocblas_csymm_batched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csymm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_csymm_batched
    end interface

    interface
        function rocblas_zsymm_batched(handle, side, uplo, m, n, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsymm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zsymm_batched
    end interface

    ! symm_strided_batched
    interface
        function rocblas_ssymm_strided_batched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssymm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_ssymm_strided_batched
    end interface

    interface
        function rocblas_dsymm_strided_batched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsymm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_dsymm_strided_batched
    end interface

    interface
        function rocblas_csymm_strided_batched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csymm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_csymm_strided_batched
    end interface

    interface
        function rocblas_zsymm_strided_batched(handle, side, uplo, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsymm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zsymm_strided_batched
    end interface

    ! syrk
    interface
        function rocblas_ssyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyrk')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_ssyrk
    end interface

    interface
        function rocblas_dsyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyrk')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_dsyrk
    end interface

    interface
        function rocblas_csyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_csyrk')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_csyrk
    end interface

    interface
        function rocblas_zsyrk(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyrk')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zsyrk
    end interface

    ! syrk_batched
    interface
        function rocblas_ssyrk_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyrk_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_ssyrk_batched
    end interface

    interface
        function rocblas_dsyrk_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyrk_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_dsyrk_batched
    end interface

    interface
        function rocblas_csyrk_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyrk_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_csyrk_batched
    end interface

    interface
        function rocblas_zsyrk_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyrk_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zsyrk_batched
    end interface

    ! syrk_strided_batched
    interface
        function rocblas_ssyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyrk_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_ssyrk_strided_batched
    end interface

    interface
        function rocblas_dsyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyrk_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_dsyrk_strided_batched
    end interface

    interface
        function rocblas_csyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyrk_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_csyrk_strided_batched
    end interface

    interface
        function rocblas_zsyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyrk_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zsyrk_strided_batched
    end interface

    ! syr2k
    interface
        function rocblas_ssyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr2k')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_ssyr2k
    end interface

    interface
        function rocblas_dsyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr2k')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_dsyr2k
    end interface

    interface
        function rocblas_csyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr2k')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_csyr2k
    end interface

    interface
        function rocblas_zsyr2k(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr2k')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zsyr2k
    end interface

    ! syr2k_batched
    interface
        function rocblas_ssyr2k_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr2k_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_ssyr2k_batched
    end interface

    interface
        function rocblas_dsyr2k_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr2k_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_dsyr2k_batched
    end interface

    interface
        function rocblas_csyr2k_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr2k_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_csyr2k_batched
    end interface

    interface
        function rocblas_zsyr2k_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr2k_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zsyr2k_batched
    end interface

    ! syr2k_strided_batched
    interface
        function rocblas_ssyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyr2k_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_ssyr2k_strided_batched
    end interface

    interface
        function rocblas_dsyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyr2k_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_dsyr2k_strided_batched
    end interface

    interface
        function rocblas_csyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyr2k_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_csyr2k_strided_batched
    end interface

    interface
        function rocblas_zsyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyr2k_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zsyr2k_strided_batched
    end interface

    ! syrkx
    interface
        function rocblas_ssyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyrkx')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_ssyrkx
    end interface

    interface
        function rocblas_dsyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyrkx')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_dsyrkx
    end interface

    interface
        function rocblas_csyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_csyrkx')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_csyrkx
    end interface

    interface
        function rocblas_zsyrkx(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyrkx')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zsyrkx
    end interface

    ! syrkx_batched
    interface
        function rocblas_ssyrkx_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyrkx_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_ssyrkx_batched
    end interface

    interface
        function rocblas_dsyrkx_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyrkx_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_dsyrkx_batched
    end interface

    interface
        function rocblas_csyrkx_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyrkx_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_csyrkx_batched
    end interface

    interface
        function rocblas_zsyrkx_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyrkx_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zsyrkx_batched
    end interface

    ! syrkx_strided_batched
    interface
        function rocblas_ssyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ssyrkx_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_ssyrkx_strided_batched
    end interface

    interface
        function rocblas_dsyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dsyrkx_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_dsyrkx_strided_batched
    end interface

    interface
        function rocblas_csyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_csyrkx_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_csyrkx_strided_batched
    end interface

    interface
        function rocblas_zsyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zsyrkx_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zsyrkx_strided_batched
    end interface

    ! trmm
    interface
        function rocblas_strmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_strmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_strmm
    end interface

    interface
        function rocblas_dtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_dtrmm
    end interface

    interface
        function rocblas_ctrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_ctrmm
    end interface

    interface
        function rocblas_ztrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_ztrmm
    end interface

    ! trmm_batched
    interface
        function rocblas_strmm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_strmm_batched
    end interface

    interface
        function rocblas_dtrmm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_dtrmm_batched
    end interface

    interface
        function rocblas_ctrmm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_ctrmm_batched
    end interface

    interface
        function rocblas_ztrmm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_ztrmm_batched
    end interface

    ! trmm_strided_batched
    interface
        function rocblas_strmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_strmm_strided_batched
    end interface

    interface
        function rocblas_dtrmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_dtrmm_strided_batched
    end interface

    interface
        function rocblas_ctrmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_ctrmm_strided_batched
    end interface

    interface
        function rocblas_ztrmm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_ztrmm_strided_batched
    end interface

    ! trtri
    interface
        function rocblas_strtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'rocblas_strtri')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
        end function rocblas_strtri
    end interface

    interface
        function rocblas_dtrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrtri')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
        end function rocblas_dtrtri
    end interface

    interface
        function rocblas_ctrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrtri')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
        end function rocblas_ctrtri
    end interface

    interface
        function rocblas_ztrtri(handle, uplo, diag, n, &
                A, lda, invA, ldinvA) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrtri')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
        end function rocblas_ztrtri
    end interface

    ! trtri_batched
    interface
        function rocblas_strtri_batched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strtri_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int), value :: batch_count
        end function rocblas_strtri_batched
    end interface

    interface
        function rocblas_dtrtri_batched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrtri_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int), value :: batch_count
        end function rocblas_dtrtri_batched
    end interface

    interface
        function rocblas_ctrtri_batched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrtri_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int), value :: batch_count
        end function rocblas_ctrtri_batched
    end interface

    interface
        function rocblas_ztrtri_batched(handle, uplo, diag, n, &
                A, lda, invA, ldinvA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrtri_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int), value :: batch_count
        end function rocblas_ztrtri_batched
    end interface

    ! trtri_strided_batched
    interface
        function rocblas_strtri_strided_batched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strtri_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int64_t), value :: stride_invA
            integer(c_int), value :: batch_count
        end function rocblas_strtri_strided_batched
    end interface

    interface
        function rocblas_dtrtri_strided_batched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrtri_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int64_t), value :: stride_invA
            integer(c_int), value :: batch_count
        end function rocblas_dtrtri_strided_batched
    end interface

    interface
        function rocblas_ctrtri_strided_batched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrtri_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int64_t), value :: stride_invA
            integer(c_int), value :: batch_count
        end function rocblas_ctrtri_strided_batched
    end interface

    interface
        function rocblas_ztrtri_strided_batched(handle, uplo, diag, n, &
                A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrtri_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: invA
            integer(c_int), value :: ldinvA
            integer(c_int64_t), value :: stride_invA
            integer(c_int), value :: batch_count
        end function rocblas_ztrtri_strided_batched
    end interface

    ! trsm
    interface
        function rocblas_strsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_strsm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_strsm
    end interface

    interface
        function rocblas_dtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrsm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_dtrsm
    end interface

    interface
        function rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrsm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_ctrsm
    end interface

    interface
        function rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrsm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
        end function rocblas_ztrsm
    end interface

    ! trsm_batched
    interface
        function rocblas_strsm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strsm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_strsm_batched
    end interface

    interface
        function rocblas_dtrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrsm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_dtrsm_batched
    end interface

    interface
        function rocblas_ctrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrsm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_ctrsm_batched
    end interface

    interface
        function rocblas_ztrsm_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, B, ldb, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrsm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
        end function rocblas_ztrsm_batched
    end interface

    ! trsm_strided_batched
    interface
        function rocblas_strsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_strsm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_strsm_strided_batched
    end interface

    interface
        function rocblas_dtrsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dtrsm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_dtrsm_strided_batched
    end interface

    interface
        function rocblas_ctrsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ctrsm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_ctrsm_strided_batched
    end interface

    interface
        function rocblas_ztrsm_strided_batched(handle, side, uplo, transA, diag, m, n, alpha, &
                A, lda, stride_A, B, ldb, stride_B, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ztrsm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_full)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
        end function rocblas_ztrsm_strided_batched
    end interface

    ! gemm
    interface
        function rocblas_hgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_hgemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_hgemm
    end interface

    interface
        function rocblas_sgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_sgemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_sgemm
    end interface

    interface
        function rocblas_dgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_dgemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_dgemm
    end interface

    interface
        function rocblas_cgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_cgemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_cgemm
    end interface

    interface
        function rocblas_zgemm(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zgemm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zgemm
    end interface

    ! gemm_batched
    interface
        function rocblas_hgemm_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_hgemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_hgemm_batched
    end interface

    interface
        function rocblas_sgemm_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_sgemm_batched
    end interface

    interface
        function rocblas_dgemm_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_dgemm_batched
    end interface

    interface
        function rocblas_cgemm_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_cgemm_batched
    end interface

    interface
        function rocblas_zgemm_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, B, ldb, beta, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgemm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zgemm_batched
    end interface

    ! gemm_strided_batched
    interface
        function rocblas_hgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_hgemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_hgemm_strided_batched
    end interface

    interface
        function rocblas_sgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_sgemm_strided_batched
    end interface

    interface
        function rocblas_dgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_dgemm_strided_batched
    end interface

    interface
        function rocblas_cgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_cgemm_strided_batched
    end interface

    interface
        function rocblas_zgemm_strided_batched(handle, transA, transB, m, n, k, alpha, &
                A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgemm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zgemm_strided_batched
    end interface

    ! dgmm
    interface
        function rocblas_sdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_sdgmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_sdgmm
    end interface

    interface
        function rocblas_ddgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_ddgmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_ddgmm
    end interface

    interface
        function rocblas_cdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_cdgmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_cdgmm
    end interface

    interface
        function rocblas_zdgmm(handle, side, m, n, &
                A, lda, x, incx, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zdgmm')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zdgmm
    end interface

    ! dgmm_batched
    interface
        function rocblas_sdgmm_batched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sdgmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_sdgmm_batched
    end interface

    interface
        function rocblas_ddgmm_batched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ddgmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_ddgmm_batched
    end interface

    interface
        function rocblas_cdgmm_batched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cdgmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_cdgmm_batched
    end interface

    interface
        function rocblas_zdgmm_batched(handle, side, m, n, &
                A, lda, x, incx, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdgmm_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zdgmm_batched
    end interface

    ! dgmm_strided_batched
    interface
        function rocblas_sdgmm_strided_batched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sdgmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_sdgmm_strided_batched
    end interface

    interface
        function rocblas_ddgmm_strided_batched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_ddgmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_ddgmm_strided_batched
    end interface

    interface
        function rocblas_cdgmm_strided_batched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cdgmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_cdgmm_strided_batched
    end interface

    interface
        function rocblas_zdgmm_strided_batched(handle, side, m, n, &
                A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zdgmm_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: x
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stride_x
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zdgmm_strided_batched
    end interface

    ! geam
    interface
        function rocblas_sgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_sgeam')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_sgeam
    end interface

    interface
        function rocblas_dgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_dgeam')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_dgeam
    end interface

    interface
        function rocblas_cgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_cgeam')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_cgeam
    end interface

    interface
        function rocblas_zgeam(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocblas_zgeam')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocblas_zgeam
    end interface

    ! geam_batched
    interface
        function rocblas_sgeam_batched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgeam_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_sgeam_batched
    end interface

    interface
        function rocblas_dgeam_batched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgeam_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_dgeam_batched
    end interface

    interface
        function rocblas_cgeam_batched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgeam_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_cgeam_batched
    end interface

    interface
        function rocblas_zgeam_batched(handle, transA, transB, m, n, alpha, &
                A, lda, beta, B, ldb, C, ldc, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgeam_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int), value :: batch_count
        end function rocblas_zgeam_batched
    end interface

    ! geam_strided_batched
    interface
        function rocblas_sgeam_strided_batched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_sgeam_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_sgeam_strided_batched
    end interface

    interface
        function rocblas_dgeam_strided_batched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_dgeam_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_dgeam_strided_batched
    end interface

    interface
        function rocblas_cgeam_strided_batched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_cgeam_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_cgeam_strided_batched
    end interface

    interface
        function rocblas_zgeam_strided_batched(handle, transA, transB, m, n, alpha, &
                A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
                result(c_int) &
                bind(c, name = 'rocblas_zgeam_strided_batched')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: beta
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_C
            integer(c_int), value :: batch_count
        end function rocblas_zgeam_strided_batched
    end interface

    !-----------------!
    ! blas Extensions !
    !-----------------!

    ! axpy_ex
    interface
        function rocblas_axpy_ex(handle, n, alpha, alpha_type, x, x_type, incx, &
                y, y_type, incy, execution_type) &
                result(c_int) &
                bind(c, name = 'rocblas_axpy_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(rocblas_datatype_f16_r)), value :: alpha_type
            type(c_ptr), value :: x
            integer(kind(rocblas_datatype_f16_r)), value :: x_type
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(rocblas_datatype_f16_r)), value :: y_type
            integer(c_int), value :: incy
            integer(kind(rocblas_datatype_f16_r)), value :: execution_type
        end function rocblas_axpy_ex
    end interface

    interface
        function rocblas_axpy_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx, &
                y, y_type, incy, batch_count, execution_type) &
                result(c_int) &
                bind(c, name = 'rocblas_axpy_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(rocblas_datatype_f16_r)), value :: alpha_type
            type(c_ptr), value :: x
            integer(kind(rocblas_datatype_f16_r)), value :: x_type
            integer(c_int), value :: incx
            type(c_ptr), value :: y
            integer(kind(rocblas_datatype_f16_r)), value :: y_type
            integer(c_int), value :: incy
            integer(c_int), value :: batch_count
            integer(kind(rocblas_datatype_f16_r)), value :: execution_type
        end function rocblas_axpy_batched_ex
    end interface

    interface
        function rocblas_axpy_strided_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx, stridex, &
                y, y_type, incy, stridey, batch_count, execution_type) &
                result(c_int) &
                bind(c, name = 'rocblas_axpy_strided_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(rocblas_datatype_f16_r)), value :: alpha_type
            type(c_ptr), value :: x
            integer(kind(rocblas_datatype_f16_r)), value :: x_type
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            type(c_ptr), value :: y
            integer(kind(rocblas_datatype_f16_r)), value :: y_type
            integer(c_int), value :: incy
            integer(c_int64_t), value :: stridey
            integer(c_int), value :: batch_count
            integer(kind(rocblas_datatype_f16_r)), value :: execution_type
        end function rocblas_axpy_strided_batched_ex
    end interface

    ! scal_ex
    interface
        function rocblas_scal_ex(handle, n, alpha, alpha_type, x, x_type, incx, execution_type) &
                result(c_int) &
                bind(c, name = 'rocblas_scal_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(rocblas_datatype_f16_r)), value :: alpha_type
            type(c_ptr), value :: x
            integer(kind(rocblas_datatype_f16_r)), value :: x_type
            integer(c_int), value :: incx
            integer(kind(rocblas_datatype_f16_r)), value :: execution_type
        end function rocblas_scal_ex
    end interface

    interface
        function rocblas_scal_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx, &
                batch_count, execution_type) &
                result(c_int) &
                bind(c, name = 'rocblas_scal_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(rocblas_datatype_f16_r)), value :: alpha_type
            type(c_ptr), value :: x
            integer(kind(rocblas_datatype_f16_r)), value :: x_type
            integer(c_int), value :: incx
            integer(c_int), value :: batch_count
            integer(kind(rocblas_datatype_f16_r)), value :: execution_type
        end function rocblas_scal_batched_ex
    end interface

    interface
        function rocblas_scal_strided_batched_ex(handle, n, alpha, alpha_type, x, x_type, incx, stridex, &
                batch_count, execution_type) &
                result(c_int) &
                bind(c, name = 'rocblas_scal_strided_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            integer(kind(rocblas_datatype_f16_r)), value :: alpha_type
            type(c_ptr), value :: x
            integer(kind(rocblas_datatype_f16_r)), value :: x_type
            integer(c_int), value :: incx
            integer(c_int64_t), value :: stridex
            integer(c_int), value :: batch_count
            integer(kind(rocblas_datatype_f16_r)), value :: execution_type
        end function rocblas_scal_strided_batched_ex
    end interface

    ! gemm_ex
    interface
        function rocblas_gemm_ex(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, &
                compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'rocblas_gemm_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: a
            integer(kind(rocblas_datatype_f16_r)), value :: a_type
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(kind(rocblas_datatype_f16_r)), value :: b_type
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: c
            integer(kind(rocblas_datatype_f16_r)), value :: c_type
            integer(c_int), value :: ldc
            type(c_ptr), value :: d
            integer(kind(rocblas_datatype_f16_r)), value :: d_type
            integer(c_int), value :: ldd
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
            integer(kind(rocblas_gemm_algo_standard)), value :: algo
            integer(c_int32_t), value :: solution_index
            ! No unsigned types in fortran. If larger values are needed
            ! we will need a workaround.
            integer(c_int32_t), value :: flags
        end function rocblas_gemm_ex
    end interface

    interface
        function rocblas_gemm_batched_ex(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                b, b_type, ldb, beta, c, c_type, ldc, d, d_type, ldd, &
                batch_count, compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'rocblas_gemm_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: a
            integer(kind(rocblas_datatype_f16_r)), value :: a_type
            integer(c_int), value :: lda
            type(c_ptr), value :: b
            integer(kind(rocblas_datatype_f16_r)), value :: b_type
            integer(c_int), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: c
            integer(kind(rocblas_datatype_f16_r)), value :: c_type
            integer(c_int), value :: ldc
            type(c_ptr), value :: d
            integer(kind(rocblas_datatype_f16_r)), value :: d_type
            integer(c_int), value :: ldd
            integer(c_int), value :: batch_count
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
            integer(kind(rocblas_gemm_algo_standard)), value :: algo
            integer(c_int32_t), value :: solution_index
            ! No unsigned types in fortran. If larger values are needed
            ! we will need a workaround.
            integer(c_int32_t), value :: flags
        end function rocblas_gemm_batched_ex
    end interface

    interface
        function rocblas_gemm_strided_batched_ex(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
                b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, d, d_type, ldd, stride_d, &
                batch_count, compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'rocblas_gemm_strided_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_operation_none)), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: a
            integer(kind(rocblas_datatype_f16_r)), value :: a_type
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_a
            type(c_ptr), value :: b
            integer(kind(rocblas_datatype_f16_r)), value :: b_type
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_b
            type(c_ptr), value :: beta
            type(c_ptr), value :: c
            integer(kind(rocblas_datatype_f16_r)), value :: c_type
            integer(c_int), value :: ldc
            integer(c_int64_t), value :: stride_c
            type(c_ptr), value :: d
            integer(kind(rocblas_datatype_f16_r)), value :: d_type
            integer(c_int), value :: ldd
            integer(c_int64_t), value :: stride_d
            integer(c_int), value :: batch_count
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
            integer(kind(rocblas_gemm_algo_standard)), value :: algo
            integer(c_int32_t), value :: solution_index
            ! No unsigned types in fortran. If larger values are needed
            ! we will need a workaround.
            integer(c_int32_t), value :: flags
        end function rocblas_gemm_strided_batched_ex
    end interface

    interface
        function rocblas_gemm_ext2(handle, m, n, k, alpha, a, a_type, row_stride_a, col_stride_a, &
             b, b_type, row_stride_b, col_stride_b, beta, c, c_type, row_stride_c, col_stride_c, &
             d, d_type, row_stride_d, col_stride_d, compute_type, algo, solution_index, flags) &
                result(c_int) &
                bind(c, name = 'rocblas_gemm_ext2')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: a
            integer(kind(rocblas_datatype_f16_r)), value :: a_type
            integer(c_int64_t), value :: row_stride_a, col_stride_a
            type(c_ptr), value :: b
            integer(kind(rocblas_datatype_f16_r)), value :: b_type
            integer(c_int64_t), value :: row_stride_b, col_stride_b
            type(c_ptr), value :: beta
            type(c_ptr), value :: c
            integer(kind(rocblas_datatype_f16_r)), value :: c_type
            integer(c_int64_t), value :: row_stride_c, col_stride_c
            type(c_ptr), value :: d
            integer(kind(rocblas_datatype_f16_r)), value :: d_type
            integer(c_int64_t), value :: row_stride_d, col_stride_d
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
            integer(kind(rocblas_gemm_algo_standard)), value :: algo
            integer(c_int32_t), value :: solution_index
            ! No unsigned types in fortran. If larger values are needed
            ! we will need a workaround.
            integer(c_int32_t), value :: flags
        end function rocblas_gemm_ext2
    end interface

    ! trsm_ex
    interface
        function rocblas_trsm_ex(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
                B, ldb, invA, invA_size, compute_type) &
                result(c_int) &
                bind(c, name = 'rocblas_trsm_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_upper)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: invA
            integer(c_int), value :: invA_size
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
        end function rocblas_trsm_ex
    end interface

    interface
        function rocblas_trsm_batched_ex(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
                B, ldb, batch_count, invA, invA_size, compute_type) &
                result(c_int) &
                bind(c, name = 'rocblas_trsm_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_upper)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int), value :: batch_count
            type(c_ptr), value :: invA
            integer(c_int), value :: invA_size
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
        end function rocblas_trsm_batched_ex
    end interface

    interface
        function rocblas_trsm_strided_batched_ex(handle, side, uplo, transA, diag, m, n, alpha, A, lda, stride_A, &
                B, ldb, stride_B, batch_count, invA, invA_size, stride_invA, compute_type) &
                result(c_int) &
                bind(c, name = 'rocblas_trsm_strided_batched_ex')
            use iso_c_binding
            use rocblas_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(rocblas_side_left)), value :: side
            integer(kind(rocblas_fill_upper)), value :: uplo
            integer(kind(rocblas_operation_none)), value :: transA
            integer(kind(rocblas_diagonal_unit)), value :: diag
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: alpha
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            integer(c_int64_t), value :: stride_A
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            integer(c_int64_t), value :: stride_B
            integer(c_int), value :: batch_count
            type(c_ptr), value :: invA
            integer(c_int), value :: invA_size
            integer(c_int64_t), value :: stride_invA
            integer(kind(rocblas_datatype_f16_r)), value :: compute_type
        end function rocblas_trsm_strided_batched_ex
    end interface


end module rocblas

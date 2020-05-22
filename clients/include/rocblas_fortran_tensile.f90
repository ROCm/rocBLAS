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

! This file interfaces between the unit test infrastructure and the Fortran
! rocblas module.

module rocblas_interface
    use iso_c_binding
    use rocblas

    contains

    ! ==========
    !     L2
    ! ==========
    ! trsv
    function rocblas_strsv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_strsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_strsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_strsv_fortran

    function rocblas_dtrsv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dtrsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_dtrsv_fortran

    function rocblas_ctrsv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ctrsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_ctrsv_fortran

    function rocblas_ztrsv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ztrsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztrsv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_ztrsv_fortran

    ! trsv_batched
    function rocblas_strsv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_strsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_strsv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_strsv_batched_fortran

    function rocblas_dtrsv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtrsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtrsv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_dtrsv_batched_fortran

    function rocblas_ctrsv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctrsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctrsv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_ctrsv_batched_fortran

    function rocblas_ztrsv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztrsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztrsv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_ztrsv_batched_fortran

    ! trsv_strided_batched
    function rocblas_strsv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_strsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_strsv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_strsv_strided_batched_fortran

    function rocblas_dtrsv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtrsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtrsv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_dtrsv_strided_batched_fortran

    function rocblas_ctrsv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctrsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctrsv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ctrsv_strided_batched_fortran

    function rocblas_ztrsv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztrsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztrsv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ztrsv_strided_batched_fortran

end module rocblas_interface

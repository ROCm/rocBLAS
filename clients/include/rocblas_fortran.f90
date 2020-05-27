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

    !--------!
    ! blas 1 !
    !--------!

    ! scal
    function rocblas_sscal_fortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_sscal_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = rocblas_sscal(handle, n, alpha, x, incx)
        return
    end function rocblas_sscal_fortran

    function rocblas_dscal_fortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dscal_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = rocblas_dscal(handle, n, alpha, x, incx)
        return
    end function rocblas_dscal_fortran

    function rocblas_cscal_fortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_cscal_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = rocblas_cscal(handle, n, alpha, x, incx)
        return
    end function rocblas_cscal_fortran

    function rocblas_zscal_fortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_zscal_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = rocblas_zscal(handle, n, alpha, x, incx)
        return
    end function rocblas_zscal_fortran

    function rocblas_csscal_fortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_csscal_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = rocblas_csscal(handle, n, alpha, x, incx)
        return
    end function rocblas_csscal_fortran

    function rocblas_zdscal_fortran(handle, n, alpha, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_zdscal_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int) :: res
        res = rocblas_zdscal(handle, n, alpha, x, incx)
        return
    end function rocblas_zdscal_fortran

    ! scal_batched
    function rocblas_sscal_batched_fortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sscal_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_sscal_batched(handle, n, alpha, x, incx, batch_count)
        return
    end function rocblas_sscal_batched_fortran

    function rocblas_dscal_batched_fortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dscal_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_dscal_batched(handle, n, alpha, x, incx, batch_count)
        return
    end function rocblas_dscal_batched_fortran

    function rocblas_cscal_batched_fortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cscal_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_cscal_batched(handle, n, alpha, x, incx, batch_count)
        return
    end function rocblas_cscal_batched_fortran

    function rocblas_zscal_batched_fortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zscal_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zscal_batched(handle, n, alpha, x, incx, batch_count)
        return
    end function rocblas_zscal_batched_fortran

    function rocblas_csscal_batched_fortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csscal_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_csscal_batched(handle, n, alpha, x, incx, batch_count)
        return
    end function rocblas_csscal_batched_fortran

    function rocblas_zdscal_batched_fortran(handle, n, alpha, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zdscal_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zdscal_batched(handle, n, alpha, x, incx, batch_count)
        return
    end function rocblas_zdscal_batched_fortran

    ! scal_strided_batched
    function rocblas_sscal_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sscal_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_sscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function rocblas_sscal_strided_batched_fortran

    function rocblas_dscal_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dscal_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_dscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function rocblas_dscal_strided_batched_fortran

    function rocblas_cscal_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cscal_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_cscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function rocblas_cscal_strided_batched_fortran

    function rocblas_zscal_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zscal_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function rocblas_zscal_strided_batched_fortran

    function rocblas_csscal_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csscal_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_csscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function rocblas_csscal_strided_batched_fortran

    function rocblas_zdscal_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zdscal_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zdscal_strided_batched(handle, n, alpha, x, incx, stride_x, batch_count)
        return
    end function rocblas_zdscal_strided_batched_fortran

    ! copy
    function rocblas_scopy_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_scopy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_scopy(handle, n, x, incx, y, incy)
        return
    end function rocblas_scopy_fortran

    function rocblas_dcopy_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dcopy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_dcopy(handle, n, x, incx, y, incy)
        return
    end function rocblas_dcopy_fortran

    function rocblas_ccopy_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_ccopy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_ccopy(handle, n, x, incx, y, incy)
        return
    end function rocblas_ccopy_fortran

    function rocblas_zcopy_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zcopy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_zcopy(handle, n, x, incx, y, incy)
        return
    end function rocblas_zcopy_fortran

    ! copy_batched
    function rocblas_scopy_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_scopy_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_scopy_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_scopy_batched_fortran

    function rocblas_dcopy_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dcopy_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_dcopy_batched(handle, n,  x, incx, y, incy, batch_count)
        return
    end function rocblas_dcopy_batched_fortran

    function rocblas_ccopy_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ccopy_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_ccopy_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_ccopy_batched_fortran

    function rocblas_zcopy_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zcopy_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zcopy_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_zcopy_batched_fortran

    ! copy_strided_batched
    function rocblas_scopy_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_scopy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_scopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_scopy_strided_batched_fortran

    function rocblas_dcopy_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dcopy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dcopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_dcopy_strided_batched_fortran

    function rocblas_ccopy_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ccopy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ccopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_ccopy_strided_batched_fortran

    function rocblas_zcopy_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zcopy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zcopy_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_zcopy_strided_batched_fortran

    ! dot
    function rocblas_sdot_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_sdot_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_sdot(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_sdot_fortran

    function rocblas_ddot_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_ddot_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_ddot(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_ddot_fortran

    function rocblas_hdot_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_hdot_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_hdot(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_hdot_fortran

    function rocblas_bfdot_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_bfdot_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_bfdot(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_bfdot_fortran

    function rocblas_cdotu_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_cdotu_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_cdotu(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_cdotu_fortran

    function rocblas_cdotc_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_cdotc_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_cdotc(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_cdotc_fortran

    function rocblas_zdotu_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_zdotu_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_zdotu(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_zdotu_fortran

    function rocblas_zdotc_fortran(handle, n, x, incx, y, incy, result) &
            result(res) &
            bind(c, name = 'rocblas_zdotc_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_zdotc(handle, n, x, incx, y, incy, result)
        return
    end function rocblas_zdotc_fortran

    ! dot_batched
    function rocblas_sdot_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_sdot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sdot_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_sdot_batched_fortran

    function rocblas_ddot_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_ddot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ddot_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_ddot_batched_fortran

    function rocblas_hdot_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_hdot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_hdot_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_hdot_batched_fortran

    function rocblas_bfdot_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_bfdot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_bfdot_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_bfdot_batched_fortran

    function rocblas_cdotu_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_cdotu_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdotu_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_cdotu_batched_fortran

    function rocblas_cdotc_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_cdotc_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdotc_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_cdotc_batched_fortran

    function rocblas_zdotu_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_zdotu_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdotu_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_zdotu_batched_fortran

    function rocblas_zdotc_batched_fortran(handle, n, x, incx, y, incy, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_zdotc_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdotc_batched(handle, n, x, incx, y, incy, batch_count, result)
        return
    end function rocblas_zdotc_batched_fortran

    ! dot_strided_batched
    function rocblas_sdot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_sdot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_sdot_strided_batched_fortran

    function rocblas_ddot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_ddot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ddot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_ddot_strided_batched_fortran

    function rocblas_hdot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_hdot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_hdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_hdot_strided_batched_fortran

    function rocblas_bfdot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_bfdot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_bfdot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_bfdot_strided_batched_fortran

    function rocblas_cdotu_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_cdotu_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdotu_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_cdotu_strided_batched_fortran

    function rocblas_cdotc_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_cdotc_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdotc_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_cdotc_strided_batched_fortran

    function rocblas_zdotu_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_zdotu_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdotu_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_zdotu_strided_batched_fortran

    function rocblas_zdotc_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_zdotc_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdotc_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
        return
    end function rocblas_zdotc_strided_batched_fortran

    ! swap
    function rocblas_sswap_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_sswap_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_sswap(handle, n, x, incx, y, incy)
        return
    end function rocblas_sswap_fortran

    function rocblas_dswap_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dswap_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_dswap(handle, n, x, incx, y, incy)
        return
    end function rocblas_dswap_fortran

    function rocblas_cswap_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_cswap_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_cswap(handle, n, x, incx, y, incy)
        return
    end function rocblas_cswap_fortran

    function rocblas_zswap_fortran(handle, n, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zswap_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_zswap(handle, n, x, incx, y, incy)
        return
    end function rocblas_zswap_fortran

    ! swap_batched
    function rocblas_sswap_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sswap_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_sswap_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_sswap_batched_fortran

    function rocblas_dswap_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dswap_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_dswap_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_dswap_batched_fortran

    function rocblas_cswap_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cswap_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_cswap_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_cswap_batched_fortran

    function rocblas_zswap_batched_fortran(handle, n, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zswap_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zswap_batched(handle, n, x, incx, y, incy, batch_count)
        return
    end function rocblas_zswap_batched_fortran

    ! swap_strided_batched
    function rocblas_sswap_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sswap_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_sswap_strided_batched_fortran

    function rocblas_dswap_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dswap_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_dswap_strided_batched_fortran

    function rocblas_cswap_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cswap_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_cswap_strided_batched_fortran

    function rocblas_zswap_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zswap_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zswap_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_zswap_strided_batched_fortran

    ! axpy
    function rocblas_haxpy_fortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_haxpy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_haxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function rocblas_haxpy_fortran

    function rocblas_saxpy_fortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_saxpy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_saxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function rocblas_saxpy_fortran

    function rocblas_daxpy_fortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_daxpy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_daxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function rocblas_daxpy_fortran

    function rocblas_caxpy_fortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_caxpy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_caxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function rocblas_caxpy_fortran

    function rocblas_zaxpy_fortran(handle, n, alpha, x, incx, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zaxpy_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: alpha
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int) :: res
        res = rocblas_zaxpy(handle, n, alpha, x, incx, y, incy)
        return
    end function rocblas_zaxpy_fortran

    ! axpy_batched
    function rocblas_haxpy_batched_fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_haxpy_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_haxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function rocblas_haxpy_batched_fortran

    function rocblas_saxpy_batched_fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_saxpy_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_saxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function rocblas_saxpy_batched_fortran

    function rocblas_daxpy_batched_fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_daxpy_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_daxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function rocblas_daxpy_batched_fortran

    function rocblas_caxpy_batched_fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_caxpy_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_caxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function rocblas_caxpy_batched_fortran

    function rocblas_zaxpy_batched_fortran(handle, n, alpha, x, incx, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zaxpy_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zaxpy_batched(handle, n, alpha, x, incx, y, incy, batch_count)
        return
    end function rocblas_zaxpy_batched_fortran

    ! axpy_strided_batched
    function rocblas_haxpy_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_haxpy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_haxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_haxpy_strided_batched_fortran

    function rocblas_saxpy_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_saxpy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_saxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_saxpy_strided_batched_fortran

    function rocblas_daxpy_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_daxpy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_daxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_daxpy_strided_batched_fortran

    function rocblas_caxpy_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_caxpy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_caxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_caxpy_strided_batched_fortran

    function rocblas_zaxpy_strided_batched_fortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zaxpy_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zaxpy_strided_batched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
        return
    end function rocblas_zaxpy_strided_batched_fortran

    ! asum
    function rocblas_sasum_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_sasum_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_sasum(handle, n, x, incx, result)
        return
    end function rocblas_sasum_fortran

    function rocblas_dasum_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_dasum_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dasum(handle, n, x, incx, result)
        return
    end function rocblas_dasum_fortran

    function rocblas_scasum_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_scasum_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_scasum(handle, n, x, incx, result)
        return
    end function rocblas_scasum_fortran

    function rocblas_dzasum_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_dzasum_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dzasum(handle, n, x, incx, result)
        return
    end function rocblas_dzasum_fortran

    ! asum_batched
    function rocblas_sasum_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_sasum_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_sasum_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_sasum_batched_fortran

    function rocblas_dasum_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dasum_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dasum_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_dasum_batched_fortran

    function rocblas_scasum_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_scasum_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_scasum_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_scasum_batched_fortran

    function rocblas_dzasum_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dzasum_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dzasum_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_dzasum_batched_fortran

    ! asum_strided_batched
    function rocblas_sasum_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_sasum_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_sasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_sasum_strided_batched_fortran

    function rocblas_dasum_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dasum_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_dasum_strided_batched_fortran

    function rocblas_scasum_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_scasum_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_scasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_scasum_strided_batched_fortran

    function rocblas_dzasum_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dzasum_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dzasum_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_dzasum_strided_batched_fortran

    ! nrm2
    function rocblas_snrm2_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_snrm2_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_snrm2(handle, n, x, incx, result)
        return
    end function rocblas_snrm2_fortran

    function rocblas_dnrm2_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_dnrm2_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dnrm2(handle, n, x, incx, result)
        return
    end function rocblas_dnrm2_fortran

    function rocblas_scnrm2_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_scnrm2_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_scnrm2(handle, n, x, incx, result)
        return
    end function rocblas_scnrm2_fortran

    function rocblas_dznrm2_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_dznrm2_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dznrm2(handle, n, x, incx, result)
        return
    end function rocblas_dznrm2_fortran

    ! nrm2_batched
    function rocblas_snrm2_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_snrm2_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_snrm2_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_snrm2_batched_fortran

    function rocblas_dnrm2_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dnrm2_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dnrm2_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_dnrm2_batched_fortran

    function rocblas_scnrm2_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_scnrm2_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_scnrm2_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_scnrm2_batched_fortran

    function rocblas_dznrm2_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dznrm2_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dznrm2_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_dznrm2_batched_fortran

    ! nrm2_strided_batched
    function rocblas_snrm2_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_snrm2_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_snrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_snrm2_strided_batched_fortran

    function rocblas_dnrm2_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dnrm2_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dnrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_dnrm2_strided_batched_fortran

    function rocblas_scnrm2_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_scnrm2_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_scnrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_scnrm2_strided_batched_fortran

    function rocblas_dznrm2_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_dznrm2_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_dznrm2_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_dznrm2_strided_batched_fortran

    ! amax
    function rocblas_isamax_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_isamax_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_isamax(handle, n, x, incx, result)
        return
    end function rocblas_isamax_fortran

    function rocblas_idamax_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_idamax_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_idamax(handle, n, x, incx, result)
        return
    end function rocblas_idamax_fortran

    function rocblas_icamax_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_icamax_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_icamax(handle, n, x, incx, result)
        return
    end function rocblas_icamax_fortran

    function rocblas_izamax_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_izamax_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_izamax(handle, n, x, incx, result)
        return
    end function rocblas_izamax_fortran

    ! amax_batched
    function rocblas_isamax_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_isamax_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_isamax_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_isamax_batched_fortran

    function rocblas_idamax_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_idamax_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_idamax_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_idamax_batched_fortran

    function rocblas_icamax_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_icamax_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_icamax_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_icamax_batched_fortran

    function rocblas_izamax_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_izamax_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_izamax_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_izamax_batched_fortran

    ! amax_strided_batched
    function rocblas_isamax_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_isamax_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_isamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_isamax_strided_batched_fortran

    function rocblas_idamax_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_idamax_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_idamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_idamax_strided_batched_fortran

    function rocblas_icamax_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_icamax_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_icamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_icamax_strided_batched_fortran

    function rocblas_izamax_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_izamax_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_izamax_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_izamax_strided_batched_fortran

    ! amin
    function rocblas_isamin_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_isamin_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_isamin(handle, n, x, incx, result)
        return
    end function rocblas_isamin_fortran

    function rocblas_idamin_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_idamin_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_idamin(handle, n, x, incx, result)
        return
    end function rocblas_idamin_fortran

    function rocblas_icamin_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_icamin_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_icamin(handle, n, x, incx, result)
        return
    end function rocblas_icamin_fortran

    function rocblas_izamin_fortran(handle, n, x, incx, result) &
            result(res) &
            bind(c, name = 'rocblas_izamin_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_izamin(handle, n, x, incx, result)
        return
    end function rocblas_izamin_fortran

    ! amin_batched
    function rocblas_isamin_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_isamin_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_isamin_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_isamin_batched_fortran

    function rocblas_idamin_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_idamin_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_idamin_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_idamin_batched_fortran

    function rocblas_icamin_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_icamin_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_icamin_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_icamin_batched_fortran

    function rocblas_izamin_batched_fortran(handle, n, x, incx, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_izamin_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_izamin_batched(handle, n, x, incx, batch_count, result)
        return
    end function rocblas_izamin_batched_fortran

    ! amin_strided_batched
    function rocblas_isamin_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_isamin_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_isamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_isamin_strided_batched_fortran

    function rocblas_idamin_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_idamin_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_idamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_idamin_strided_batched_fortran

    function rocblas_icamin_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_icamin_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_icamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_icamin_strided_batched_fortran

    function rocblas_izamin_strided_batched_fortran(handle, n, x, incx, stride_x, batch_count, result) &
            result(res) &
            bind(c, name = 'rocblas_izamin_strided_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        integer(c_int64_t), value :: stride_x
        integer(c_int), value :: batch_count
        type(c_ptr), value :: result
        integer(c_int) :: res
        res = rocblas_izamin_strided_batched(handle, n, x, incx, stride_x, batch_count, result)
        return
    end function rocblas_izamin_strided_batched_fortran

    ! rot
    function rocblas_srot_fortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'rocblas_srot_fortran')
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
        integer(c_int) :: res
        res = rocblas_srot(handle, n, x, incx, y, incy, c, s)
        return
    end function rocblas_srot_fortran

    function rocblas_drot_fortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'rocblas_drot_fortran')
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
        integer(c_int) :: res
        res = rocblas_drot(handle, n, x, incx, y, incy, c, s)
        return
    end function rocblas_drot_fortran

    function rocblas_crot_fortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'rocblas_crot_fortran')
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
        integer(c_int) :: res
        res = rocblas_crot(handle, n, x, incx, y, incy, c, s)
        return
    end function rocblas_crot_fortran

    function rocblas_csrot_fortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'rocblas_csrot_fortran')
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
        integer(c_int) :: res
        res = rocblas_csrot(handle, n, x, incx, y, incy, c, s)
        return
    end function rocblas_csrot_fortran

    function rocblas_zrot_fortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'rocblas_zrot_fortran')
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
        integer(c_int) :: res
        res = rocblas_zrot(handle, n, x, incx, y, incy, c, s)
        return
    end function rocblas_zrot_fortran

    function rocblas_zdrot_fortran(handle, n, x, incx, y, incy, c, s) &
            result(res) &
            bind(c, name = 'rocblas_zdrot_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdrot(handle, n, x, incx, y, incy, c, s)
        return
    end function rocblas_zdrot_fortran

    ! rot_batched
    function rocblas_srot_batched_fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_srot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function rocblas_srot_batched_fortran

    function rocblas_drot_batched_fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_drot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function rocblas_drot_batched_fortran

    function rocblas_crot_batched_fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_crot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_crot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function rocblas_crot_batched_fortran

    function rocblas_csrot_batched_fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csrot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csrot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function rocblas_csrot_batched_fortran

    function rocblas_zrot_batched_fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zrot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zrot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function rocblas_zrot_batched_fortran

    function rocblas_zdrot_batched_fortran(handle, n, x, incx, y, incy, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zdrot_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdrot_batched(handle, n, x, incx, y, incy, c, s, batch_count)
        return
    end function rocblas_zdrot_batched_fortran

    ! rot_strided_batched
    function rocblas_srot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_srot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function rocblas_srot_strided_batched_fortran

    function rocblas_drot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_drot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function rocblas_drot_strided_batched_fortran

    function rocblas_crot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_crot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_crot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function rocblas_crot_strided_batched_fortran

    function rocblas_csrot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csrot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function rocblas_csrot_strided_batched_fortran

    function rocblas_zrot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zrot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function rocblas_zrot_strided_batched_fortran

    function rocblas_zdrot_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zdrot_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdrot_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
        return
    end function rocblas_zdrot_strided_batched_fortran

    ! rotg
    function rocblas_srotg_fortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'rocblas_srotg_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = rocblas_srotg(handle, a, b, c, s)
        return
    end function rocblas_srotg_fortran

    function rocblas_drotg_fortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'rocblas_drotg_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = rocblas_drotg(handle, a, b, c, s)
        return
    end function rocblas_drotg_fortran

    function rocblas_crotg_fortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'rocblas_crotg_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = rocblas_crotg(handle, a, b, c, s)
        return
    end function rocblas_crotg_fortran

    function rocblas_zrotg_fortran(handle, a, b, c, s) &
            result(res) &
            bind(c, name = 'rocblas_zrotg_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int) :: res
        res = rocblas_zrotg(handle, a, b, c, s)
        return
    end function rocblas_zrotg_fortran

    ! rotg_batched
    function rocblas_srotg_batched_fortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srotg_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_srotg_batched(handle, a, b, c, s, batch_count)
        return
    end function rocblas_srotg_batched_fortran

    function rocblas_drotg_batched_fortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drotg_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_drotg_batched(handle, a, b, c, s, batch_count)
        return
    end function rocblas_drotg_batched_fortran

    function rocblas_crotg_batched_fortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_crotg_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_crotg_batched(handle, a, b, c, s, batch_count)
        return
    end function rocblas_crotg_batched_fortran

    function rocblas_zrotg_batched_fortran(handle, a, b, c, s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zrotg_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: a
        type(c_ptr), value :: b
        type(c_ptr), value :: c
        type(c_ptr), value :: s
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_zrotg_batched(handle, a, b, c, s, batch_count)
        return
    end function rocblas_zrotg_batched_fortran

    ! rotg_strided_batched
    function rocblas_srotg_strided_batched_fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srotg_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_srotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function rocblas_srotg_strided_batched_fortran

    function rocblas_drotg_strided_batched_fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drotg_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_drotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function rocblas_drotg_strided_batched_fortran

    function rocblas_crotg_strided_batched_fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_crotg_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_crotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function rocblas_crotg_strided_batched_fortran

    function rocblas_zrotg_strided_batched_fortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zrotg_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zrotg_strided_batched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
        return
    end function rocblas_zrotg_strided_batched_fortran

    ! rotm
    function rocblas_srotm_fortran(handle, n, x, incx, y, incy, param) &
            result(res) &
            bind(c, name = 'rocblas_srotm_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = rocblas_srotm(handle, n, x, incx, y, incy, param)
        return
    end function rocblas_srotm_fortran

    function rocblas_drotm_fortran(handle, n, x, incx, y, incy, param) &
            result(res) &
            bind(c, name = 'rocblas_drotm_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: n
        type(c_ptr), value :: x
        integer(c_int), value :: incx
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = rocblas_drotm(handle, n, x, incx, y, incy, param)
        return
    end function rocblas_drotm_fortran

    ! rotm_batched
    function rocblas_srotm_batched_fortran(handle, n, x, incx, y, incy, param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srotm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_srotm_batched(handle, n, x, incx, y, incy, param, batch_count)
        return
    end function rocblas_srotm_batched_fortran

    function rocblas_drotm_batched_fortran(handle, n, x, incx, y, incy, param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drotm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_drotm_batched(handle, n, x, incx, y, incy, param, batch_count)
        return
    end function rocblas_drotm_batched_fortran

    ! rotm_strided_batched
    function rocblas_srotm_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
        stride_param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srotm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_srotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
            stride_param, batch_count)
        return
    end function rocblas_srotm_strided_batched_fortran

    function rocblas_drotm_strided_batched_fortran(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
        stride_param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drotm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_drotm_strided_batched(handle, n, x, incx, stride_x, y, incy, stride_y, param,&
            stride_param, batch_count)
        return
    end function rocblas_drotm_strided_batched_fortran

    ! rotmg
    function rocblas_srotmg_fortran(handle, d1, d2, x1, y1, param) &
            result(res) &
            bind(c, name = 'rocblas_srotmg_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = rocblas_srotmg(handle, d1, d2, x1, y1, param)
        return
    end function rocblas_srotmg_fortran

    function rocblas_drotmg_fortran(handle, d1, d2, x1, y1, param) &
            result(res) &
            bind(c, name = 'rocblas_drotmg_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int) :: res
        res = rocblas_drotmg(handle, d1, d2, x1, y1, param)
        return
    end function rocblas_drotmg_fortran

    ! rotmg_batched
    function rocblas_srotmg_batched_fortran(handle, d1, d2, x1, y1, param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srotmg_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_srotmg_batched(handle, d1, d2, x1, y1, param, batch_count)
        return
    end function rocblas_srotmg_batched_fortran

    function rocblas_drotmg_batched_fortran(handle, d1, d2, x1, y1, param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drotmg_batched_fortran')
        use iso_c_binding
        implicit none
        type(c_ptr), value :: handle
        type(c_ptr), value :: d1
        type(c_ptr), value :: d2
        type(c_ptr), value :: x1
        type(c_ptr), value :: y1
        type(c_ptr), value :: param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_drotmg_batched(handle, d1, d2, x1, y1, param, batch_count)
        return
    end function rocblas_drotmg_batched_fortran

    ! rotmg_strided_batched
    function rocblas_srotmg_strided_batched_fortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
            y1, stride_y1, param, stride_param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_srotmg_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_srotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1,&
            param, stride_param, batch_count)
        return
    end function rocblas_srotmg_strided_batched_fortran

    function rocblas_drotmg_strided_batched_fortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
            y1, stride_y1, param, stride_param, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_drotmg_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_drotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1,&
            param, stride_param, batch_count)
        return
    end function rocblas_drotmg_strided_batched_fortran

    !--------!
    ! blas 2 !
    !--------!

    ! gbmv
    function rocblas_sgbmv_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_sgbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_sgbmv_fortran

    function rocblas_dgbmv_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dgbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_dgbmv_fortran

    function rocblas_cgbmv_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_cgbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_cgbmv_fortran

    function rocblas_zgbmv_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zgbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_zgbmv_fortran

    ! gbmv_batched
    function rocblas_sgbmv_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sgbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function rocblas_sgbmv_batched_fortran

    function rocblas_dgbmv_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dgbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function rocblas_dgbmv_batched_fortran

    function rocblas_cgbmv_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function rocblas_cgbmv_batched_fortran

    function rocblas_zgbmv_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgbmv_batched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx,&
              beta, y, incy, batch_count)
    end function rocblas_zgbmv_batched_fortran

    ! gbmv_strided_batched
    function rocblas_sgbmv_strided_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sgbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function rocblas_sgbmv_strided_batched_fortran

    function rocblas_dgbmv_strided_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dgbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function rocblas_dgbmv_strided_batched_fortran

    function rocblas_cgbmv_strided_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function rocblas_cgbmv_strided_batched_fortran

    function rocblas_zgbmv_strided_batched_fortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgbmv_strided_batched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x,&
              beta, y, incy, stride_y, batch_count)
    end function rocblas_zgbmv_strided_batched_fortran

    ! gemv
    function rocblas_sgemv_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_sgemv_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_sgemv_fortran

    function rocblas_dgemv_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dgemv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_dgemv_fortran

    function rocblas_cgemv_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_cgemv_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_cgemv_fortran

    function rocblas_zgemv_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zgemv_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_zgemv_fortran

    ! gemv_batched
    function rocblas_sgemv_batched_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sgemv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_sgemv_batched_fortran

    function rocblas_dgemv_batched_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dgemv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_dgemv_batched_fortran

    function rocblas_cgemv_batched_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgemv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_cgemv_batched_fortran

    function rocblas_zgemv_batched_fortran(handle, trans, m, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgemv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgemv_batched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_zgemv_batched_fortran

    ! gemv_strided_batched
    function rocblas_sgemv_strided_batched_fortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sgemv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_sgemv_strided_batched_fortran

    function rocblas_dgemv_strided_batched_fortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dgemv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_dgemv_strided_batched_fortran

    function rocblas_cgemv_strided_batched_fortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgemv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_cgemv_strided_batched_fortran

    function rocblas_zgemv_strided_batched_fortran(handle, trans, m, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgemv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgemv_strided_batched(handle, trans, m, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_zgemv_strided_batched_fortran

    ! hbmv
    function rocblas_chbmv_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_chbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_chbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_chbmv_fortran

    function rocblas_zhbmv_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zhbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_zhbmv_fortran

    ! hbmv_batched
    function rocblas_chbmv_batched_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chbmv_batched(handle, uplo, n, k, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function rocblas_chbmv_batched_fortran

    function rocblas_zhbmv_batched_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhbmv_batched(handle, uplo, n, k, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function rocblas_zhbmv_batched_fortran

    ! hbmv_strided_batched
    function rocblas_chbmv_strided_batched_fortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_chbmv_strided_batched_fortran

    function rocblas_zhbmv_strided_batched_fortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhbmv_strided_batched(handle, uplo, n, k, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_zhbmv_strided_batched_fortran

    ! hemv
    function rocblas_chemv_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_chemv_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_chemv_fortran

    function rocblas_zhemv_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zhemv_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    end function rocblas_zhemv_fortran

    ! hemv_batched
    function rocblas_chemv_batched_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chemv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemv_batched(handle, uplo, n, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function rocblas_chemv_batched_fortran

    function rocblas_zhemv_batched_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhemv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhemv_batched(handle, uplo, n, alpha, A, lda,&
              x, incx, beta, y, incy, batch_count)
    end function rocblas_zhemv_batched_fortran

    ! hemv_strided_batched
    function rocblas_chemv_strided_batched_fortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chemv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_chemv_strided_batched_fortran

    function rocblas_zhemv_strided_batched_fortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhemv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhemv_strided_batched(handle, uplo, n, alpha, A, lda, stride_A,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_zhemv_strided_batched_fortran

    ! her
    function rocblas_cher_fortran(handle, uplo, n, alpha, &
            x, incx, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_cher_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher(handle, uplo, n, alpha, x, incx, A, lda)
    end function rocblas_cher_fortran

    function rocblas_zher_fortran(handle, uplo, n, alpha, &
            x, incx, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_zher_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher(handle, uplo, n, alpha, x, incx, A, lda)
    end function rocblas_zher_fortran
    
    ! her_batched
    function rocblas_cher_batched_fortran(handle, uplo, n, alpha, &
            x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cher_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    end function rocblas_cher_batched_fortran

    function rocblas_zher_batched_fortran(handle, uplo, n, alpha, &
            x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zher_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher_batched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
    end function rocblas_zher_batched_fortran

    ! her_strided_batched
    function rocblas_cher_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cher_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              A, lda, stride_A, batch_count)
    end function rocblas_cher_strided_batched_fortran

    function rocblas_zher_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zher_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              A, lda, stride_A, batch_count)
    end function rocblas_zher_strided_batched_fortran

    ! her2
    function rocblas_cher2_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_cher2_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher2(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda)
    end function rocblas_cher2_fortran

    function rocblas_zher2_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_zher2_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher2(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda)
    end function rocblas_zher2_fortran

    ! her2_batched
    function rocblas_cher2_batched_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cher2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher2_batched(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda, batch_count)
    end function rocblas_cher2_batched_fortran

    function rocblas_zher2_batched_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zher2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher2_batched(handle, uplo, n, alpha, x, incx,&
              y, incy, A, lda, batch_count)
    end function rocblas_zher2_batched_fortran

    ! her2_strided_batched
    function rocblas_cher2_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cher2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_cher2_strided_batched_fortran

    function rocblas_zher2_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zher2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_zher2_strided_batched_fortran

    ! hpmv
    function rocblas_chpmv_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_chpmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpmv(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy)
    end function rocblas_chpmv_fortran

    function rocblas_zhpmv_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zhpmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpmv(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy)
    end function rocblas_zhpmv_fortran

    ! hpmv_batched
    function rocblas_chpmv_batched_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chpmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpmv_batched(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy, batch_count)
    end function rocblas_chpmv_batched_fortran

    function rocblas_zhpmv_batched_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhpmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpmv_batched(handle, uplo, n, alpha, AP,&
              x, incx, beta, y, incy, batch_count)
    end function rocblas_zhpmv_batched_fortran

    ! hpmv_strided_batched
    function rocblas_chpmv_strided_batched_fortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chpmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpmv_strided_batched(handle, uplo, n, alpha, AP, stride_AP,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_chpmv_strided_batched_fortran

    function rocblas_zhpmv_strided_batched_fortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhpmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpmv_strided_batched(handle, uplo, n, alpha, AP, stride_AP,&
              x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_zhpmv_strided_batched_fortran

    ! hpr
    function rocblas_chpr_fortran(handle, uplo, n, alpha, &
            x, incx, AP) &
            result(res) &
            bind(c, name = 'rocblas_chpr_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpr(handle, uplo, n, alpha, x, incx, AP)
    end function rocblas_chpr_fortran

    function rocblas_zhpr_fortran(handle, uplo, n, alpha, &
            x, incx, AP) &
            result(res) &
            bind(c, name = 'rocblas_zhpr_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpr(handle, uplo, n, alpha, x, incx, AP)
    end function rocblas_zhpr_fortran

    ! hpr_batched
    function rocblas_chpr_batched_fortran(handle, uplo, n, alpha, &
            x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chpr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    end function rocblas_chpr_batched_fortran

    function rocblas_zhpr_batched_fortran(handle, uplo, n, alpha, &
            x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhpr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpr_batched(handle, uplo, n, alpha, x, incx, AP, batch_count)
    end function rocblas_zhpr_batched_fortran

    ! hpr_strided_batched
    function rocblas_chpr_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chpr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              AP, stride_AP, batch_count)
    end function rocblas_chpr_strided_batched_fortran

    function rocblas_zhpr_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhpr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpr_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              AP, stride_AP, batch_count)
    end function rocblas_zhpr_strided_batched_fortran

    ! hpr2
    function rocblas_chpr2_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP) &
            result(res) &
            bind(c, name = 'rocblas_chpr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpr2(handle, uplo, n, alpha, x, incx,&
              y, incy, AP)
    end function rocblas_chpr2_fortran

    function rocblas_zhpr2_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP) &
            result(res) &
            bind(c, name = 'rocblas_zhpr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpr2(handle, uplo, n, alpha, x, incx,&
              y, incy, AP)
    end function rocblas_zhpr2_fortran

    ! hpr2_batched
    function rocblas_chpr2_batched_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chpr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpr2_batched(handle, uplo, n, alpha, x, incx,&
              y, incy, AP, batch_count)
    end function rocblas_chpr2_batched_fortran

    function rocblas_zhpr2_batched_fortran(handle, uplo, n, alpha, &
            x, incx, y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhpr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpr2_batched(handle, uplo, n, alpha, x, incx,&
              y, incy, AP, batch_count)
    end function rocblas_zhpr2_batched_fortran

    ! hpr2_strided_batched
    function rocblas_chpr2_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chpr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chpr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, AP, stride_AP, batch_count)
    end function rocblas_chpr2_strided_batched_fortran

    function rocblas_zhpr2_strided_batched_fortran(handle, uplo, n, alpha, &
            x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhpr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhpr2_strided_batched(handle, uplo, n, alpha, x, incx, stride_x,&
              y, incy, stride_y, AP, stride_AP, batch_count)
    end function rocblas_zhpr2_strided_batched_fortran

    ! trmv
    function rocblas_strmv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_strmv_fortran')
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
        res = rocblas_strmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_strmv_fortran

    function rocblas_dtrmv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dtrmv_fortran')
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
        res = rocblas_dtrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_dtrmv_fortran

    function rocblas_ctrmv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ctrmv_fortran')
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
        res = rocblas_ctrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_ctrmv_fortran

    function rocblas_ztrmv_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ztrmv_fortran')
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
        res = rocblas_ztrmv(handle, uplo, transA, diag, m,&
              A, lda, x, incx)
    end function rocblas_ztrmv_fortran

    ! trmv_batched
    function rocblas_strmv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_strmv_batched_fortran')
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
        res = rocblas_strmv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_strmv_batched_fortran

    function rocblas_dtrmv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtrmv_batched_fortran')
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
        res = rocblas_dtrmv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_dtrmv_batched_fortran

    function rocblas_ctrmv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctrmv_batched_fortran')
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
        res = rocblas_ctrmv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_ctrmv_batched_fortran

    function rocblas_ztrmv_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztrmv_batched_fortran')
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
        res = rocblas_ztrmv_batched(handle, uplo, transA, diag, m,&
              A, lda, x, incx, batch_count)
    end function rocblas_ztrmv_batched_fortran

    ! trmv_strided_batched
    function rocblas_strmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_strmv_strided_batched_fortran')
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
        res = rocblas_strmv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_strmv_strided_batched_fortran

    function rocblas_dtrmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtrmv_strided_batched_fortran')
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
        res = rocblas_dtrmv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_dtrmv_strided_batched_fortran

    function rocblas_ctrmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctrmv_strided_batched_fortran')
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
        res = rocblas_ctrmv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ctrmv_strided_batched_fortran

    function rocblas_ztrmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztrmv_strided_batched_fortran')
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
        res = rocblas_ztrmv_strided_batched(handle, uplo, transA, diag, m,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ztrmv_strided_batched_fortran

    ! tpmv
    function rocblas_stpmv_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_stpmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_stpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function rocblas_stpmv_fortran

    function rocblas_dtpmv_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dtpmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function rocblas_dtpmv_fortran

    function rocblas_ctpmv_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ctpmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function rocblas_ctpmv_fortran

    function rocblas_ztpmv_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ztpmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztpmv(handle, uplo, transA, diag, m,&
              AP, x, incx)
    end function rocblas_ztpmv_fortran

    ! tpmv_batched
    function rocblas_stpmv_batched_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stpmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stpmv_batched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function rocblas_stpmv_batched_fortran

    function rocblas_dtpmv_batched_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtpmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtpmv_batched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function rocblas_dtpmv_batched_fortran

    function rocblas_ctpmv_batched_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctpmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctpmv_batched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function rocblas_ctpmv_batched_fortran

    function rocblas_ztpmv_batched_fortran(handle, uplo, transA, diag, m, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztpmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztpmv_batched(handle, uplo, transA, diag, m,&
              AP, x, incx, batch_count)
    end function rocblas_ztpmv_batched_fortran

    ! tpmv_strided_batched
    function rocblas_stpmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stpmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stpmv_strided_batched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_stpmv_strided_batched_fortran

    function rocblas_dtpmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtpmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtpmv_strided_batched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_dtpmv_strided_batched_fortran

    function rocblas_ctpmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctpmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctpmv_strided_batched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_ctpmv_strided_batched_fortran

    function rocblas_ztpmv_strided_batched_fortran(handle, uplo, transA, diag, m, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztpmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztpmv_strided_batched(handle, uplo, transA, diag, m,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_ztpmv_strided_batched_fortran

    ! tbmv
    function rocblas_stbmv_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_stbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_stbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function rocblas_stbmv_fortran

    function rocblas_dtbmv_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dtbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function rocblas_dtbmv_fortran

    function rocblas_ctbmv_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ctbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function rocblas_ctbmv_fortran

    function rocblas_ztbmv_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ztbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztbmv(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx)
    end function rocblas_ztbmv_fortran

    ! tbmv_batched
    function rocblas_stbmv_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stbmv_batched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_stbmv_batched_fortran

    function rocblas_dtbmv_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtbmv_batched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_dtbmv_batched_fortran

    function rocblas_ctbmv_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctbmv_batched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_ctbmv_batched_fortran

    function rocblas_ztbmv_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztbmv_batched(handle, uplo, transA, diag, m, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_ztbmv_batched_fortran

    ! tbmv_strided_batched
    function rocblas_stbmv_strided_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stbmv_strided_batched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_stbmv_strided_batched_fortran

    function rocblas_dtbmv_strided_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtbmv_strided_batched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_dtbmv_strided_batched_fortran

    function rocblas_ctbmv_strided_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctbmv_strided_batched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ctbmv_strided_batched_fortran

    function rocblas_ztbmv_strided_batched_fortran(handle, uplo, transA, diag, m, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztbmv_strided_batched(handle, uplo, transA, diag, m, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ztbmv_strided_batched_fortran

    ! tbsv
    function rocblas_stbsv_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_stbsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_stbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function rocblas_stbsv_fortran

    function rocblas_dtbsv_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dtbsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function rocblas_dtbsv_fortran

    function rocblas_ctbsv_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ctbsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function rocblas_ctbsv_fortran

    function rocblas_ztbsv_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ztbsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztbsv(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx)
    end function rocblas_ztbsv_fortran

    ! tbsv_batched
    function rocblas_stbsv_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stbsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stbsv_batched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_stbsv_batched_fortran

    function rocblas_dtbsv_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtbsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtbsv_batched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_dtbsv_batched_fortran

    function rocblas_ctbsv_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctbsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctbsv_batched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_ctbsv_batched_fortran

    function rocblas_ztbsv_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztbsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztbsv_batched(handle, uplo, transA, diag, n, k,&
              A, lda, x, incx, batch_count)
    end function rocblas_ztbsv_batched_fortran

    ! tbsv_strided_batched
    function rocblas_stbsv_strided_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stbsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stbsv_strided_batched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_stbsv_strided_batched_fortran

    function rocblas_dtbsv_strided_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtbsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtbsv_strided_batched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_dtbsv_strided_batched_fortran

    function rocblas_ctbsv_strided_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctbsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctbsv_strided_batched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ctbsv_strided_batched_fortran

    function rocblas_ztbsv_strided_batched_fortran(handle, uplo, transA, diag, n, k, &
            A, lda, stride_A, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztbsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztbsv_strided_batched(handle, uplo, transA, diag, n, k,&
              A, lda, stride_A, x, incx, stride_x, batch_count)
    end function rocblas_ztbsv_strided_batched_fortran

    ! tpsv
    function rocblas_stpsv_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_stpsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_stpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function rocblas_stpsv_fortran

    function rocblas_dtpsv_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_dtpsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function rocblas_dtpsv_fortran

    function rocblas_ctpsv_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ctpsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function rocblas_ctpsv_fortran

    function rocblas_ztpsv_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx) &
            result(res) &
            bind(c, name = 'rocblas_ztpsv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztpsv(handle, uplo, transA, diag, n,&
              AP, x, incx)
    end function rocblas_ztpsv_fortran

    ! tpsv_batched
    function rocblas_stpsv_batched_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stpsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stpsv_batched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function rocblas_stpsv_batched_fortran

    function rocblas_dtpsv_batched_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtpsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtpsv_batched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function rocblas_dtpsv_batched_fortran

    function rocblas_ctpsv_batched_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctpsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctpsv_batched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function rocblas_ctpsv_batched_fortran

    function rocblas_ztpsv_batched_fortran(handle, uplo, transA, diag, n, &
            AP, x, incx, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztpsv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztpsv_batched(handle, uplo, transA, diag, n,&
              AP, x, incx, batch_count)
    end function rocblas_ztpsv_batched_fortran

    ! tpsv_strided_batched
    function rocblas_stpsv_strided_batched_fortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_stpsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_stpsv_strided_batched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_stpsv_strided_batched_fortran

    function rocblas_dtpsv_strided_batched_fortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dtpsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dtpsv_strided_batched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_dtpsv_strided_batched_fortran

    function rocblas_ctpsv_strided_batched_fortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ctpsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ctpsv_strided_batched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_ctpsv_strided_batched_fortran

    function rocblas_ztpsv_strided_batched_fortran(handle, uplo, transA, diag, n, &
            AP, stride_AP, x, incx, stride_x, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ztpsv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ztpsv_strided_batched(handle, uplo, transA, diag, n,&
              AP, stride_AP, x, incx, stride_x, batch_count)
    end function rocblas_ztpsv_strided_batched_fortran

    ! symv
    function rocblas_ssymv_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_ssymv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function rocblas_ssymv_fortran

    function rocblas_dsymv_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dsymv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function rocblas_dsymv_fortran

    function rocblas_csymv_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_csymv_fortran')
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
        integer(c_int) :: res
        res = rocblas_csymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function rocblas_csymv_fortran

    function rocblas_zsymv_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_zsymv_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsymv(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function rocblas_zsymv_fortran

    ! symv_batched
    function rocblas_ssymv_batched_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssymv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssymv_batched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_ssymv_batched_fortran

    function rocblas_dsymv_batched_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsymv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsymv_batched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_dsymv_batched_fortran

    function rocblas_csymv_batched_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csymv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csymv_batched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_csymv_batched_fortran

    function rocblas_zsymv_batched_fortran(handle, uplo, n, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsymv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsymv_batched(handle, uplo, n, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_zsymv_batched_fortran

    ! symv_strided_batched
    function rocblas_ssymv_strided_batched_fortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssymv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssymv_strided_batched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_ssymv_strided_batched_fortran

    function rocblas_dsymv_strided_batched_fortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsymv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsymv_strided_batched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_dsymv_strided_batched_fortran

    function rocblas_csymv_strided_batched_fortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csymv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csymv_strided_batched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_csymv_strided_batched_fortran

    function rocblas_zsymv_strided_batched_fortran(handle, uplo, n, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsymv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsymv_strided_batched(handle, uplo, n, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_zsymv_strided_batched_fortran

    ! spmv
    function rocblas_sspmv_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_sspmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspmv(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy)
    end function rocblas_sspmv_fortran

    function rocblas_dspmv_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dspmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspmv(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy)
    end function rocblas_dspmv_fortran

    ! spmv_batched
    function rocblas_sspmv_batched_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sspmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspmv_batched(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy, batch_count)
    end function rocblas_sspmv_batched_fortran

    function rocblas_dspmv_batched_fortran(handle, uplo, n, alpha, AP, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dspmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspmv_batched(handle, uplo, n, alpha,&
              AP, x, incx, beta, y, incy, batch_count)
    end function rocblas_dspmv_batched_fortran

    ! spmv_strided_batched
    function rocblas_sspmv_strided_batched_fortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sspmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspmv_strided_batched(handle, uplo, n, alpha,&
              AP, stride_AP, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_sspmv_strided_batched_fortran

    function rocblas_dspmv_strided_batched_fortran(handle, uplo, n, alpha, AP, stride_AP, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dspmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspmv_strided_batched(handle, uplo, n, alpha,&
              AP, stride_AP, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_dspmv_strided_batched_fortran

    ! sbmv
    function rocblas_ssbmv_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_ssbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssbmv(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function rocblas_ssbmv_fortran

    function rocblas_dsbmv_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy) &
            result(res) &
            bind(c, name = 'rocblas_dsbmv_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsbmv(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy)
    end function rocblas_dsbmv_fortran

    ! sbmv_batched
    function rocblas_ssbmv_batched_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssbmv_batched(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_ssbmv_batched_fortran

    function rocblas_dsbmv_batched_fortran(handle, uplo, n, k, alpha, A, lda, &
            x, incx, beta, y, incy, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsbmv_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsbmv_batched(handle, uplo, n, k, alpha,&
              A, lda, x, incx, beta, y, incy, batch_count)
    end function rocblas_dsbmv_batched_fortran

    ! sbmv_strided_batched
    function rocblas_ssbmv_strided_batched_fortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssbmv_strided_batched(handle, uplo, n, k, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_ssbmv_strided_batched_fortran

    function rocblas_dsbmv_strided_batched_fortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsbmv_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsbmv_strided_batched(handle, uplo, n, k, alpha,&
              A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
    end function rocblas_dsbmv_strided_batched_fortran

    ! ger
    function rocblas_sger_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_sger_fortran')
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
        integer(c_int) :: res
        res = rocblas_sger(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_sger_fortran

    function rocblas_dger_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_dger_fortran')
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
        integer(c_int) :: res
        res = rocblas_dger(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_dger_fortran

    function rocblas_cgeru_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_cgeru_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgeru(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_cgeru_fortran

    function rocblas_cgerc_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_cgerc_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgerc(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_cgerc_fortran

    function rocblas_zgeru_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_zgeru_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgeru(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_zgeru_fortran

    function rocblas_zgerc_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_zgerc_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgerc(handle, m, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_zgerc_fortran

    ! ger_batched
    function rocblas_sger_batched_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sger_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sger_batched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_sger_batched_fortran

    function rocblas_dger_batched_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dger_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dger_batched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_dger_batched_fortran

    function rocblas_cgeru_batched_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgeru_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgeru_batched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_cgeru_batched_fortran

    function rocblas_cgerc_batched_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgerc_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgerc_batched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_cgerc_batched_fortran

    function rocblas_zgeru_batched_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgeru_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgeru_batched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_zgeru_batched_fortran

    function rocblas_zgerc_batched_fortran(handle, m, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgerc_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgerc_batched(handle, m, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_zgerc_batched_fortran

    ! ger_strided_batched
    function rocblas_sger_strided_batched_fortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sger_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sger_strided_batched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_sger_strided_batched_fortran

    function rocblas_dger_strided_batched_fortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dger_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dger_strided_batched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_dger_strided_batched_fortran

    function rocblas_cgeru_strided_batched_fortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgeru_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgeru_strided_batched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_cgeru_strided_batched_fortran

    function rocblas_cgerc_strided_batched_fortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgerc_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgerc_strided_batched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_cgerc_strided_batched_fortran

    function rocblas_zgeru_strided_batched_fortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgeru_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgeru_strided_batched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_zgeru_strided_batched_fortran

    function rocblas_zgerc_strided_batched_fortran(handle, m, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgerc_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgerc_strided_batched(handle, m, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_zgerc_strided_batched_fortran

    ! spr
    function rocblas_sspr_fortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'rocblas_sspr_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function rocblas_sspr_fortran

    function rocblas_dspr_fortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'rocblas_dspr_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function rocblas_dspr_fortran

    function rocblas_cspr_fortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'rocblas_cspr_fortran')
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
        integer(c_int) :: res
        res = rocblas_cspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function rocblas_cspr_fortran

    function rocblas_zspr_fortran(handle, uplo, n, alpha, x, incx, AP) &
            result(res) &
            bind(c, name = 'rocblas_zspr_fortran')
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
        integer(c_int) :: res
        res = rocblas_zspr(handle, uplo, n, alpha,&
              x, incx, AP)
    end function rocblas_zspr_fortran

    ! spr_batched
    function rocblas_sspr_batched_fortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sspr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspr_batched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function rocblas_sspr_batched_fortran

    function rocblas_dspr_batched_fortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dspr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspr_batched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function rocblas_dspr_batched_fortran

    function rocblas_cspr_batched_fortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cspr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cspr_batched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function rocblas_cspr_batched_fortran

    function rocblas_zspr_batched_fortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zspr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zspr_batched(handle, uplo, n, alpha,&
              x, incx, AP, batch_count)
    end function rocblas_zspr_batched_fortran

    ! spr_strided_batched
    function rocblas_sspr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sspr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function rocblas_sspr_strided_batched_fortran

    function rocblas_dspr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dspr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function rocblas_dspr_strided_batched_fortran

    function rocblas_cspr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cspr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cspr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function rocblas_cspr_strided_batched_fortran

    function rocblas_zspr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zspr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zspr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, AP, stride_AP, batch_count)
    end function rocblas_zspr_strided_batched_fortran

    ! spr2
    function rocblas_sspr2_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP) &
            result(res) &
            bind(c, name = 'rocblas_sspr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspr2(handle, uplo, n, alpha,&
              x, incx, y, incy, AP)
    end function rocblas_sspr2_fortran

    function rocblas_dspr2_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP) &
            result(res) &
            bind(c, name = 'rocblas_dspr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspr2(handle, uplo, n, alpha,&
              x, incx, y, incy, AP)
    end function rocblas_dspr2_fortran

    ! spr2_batched
    function rocblas_sspr2_batched_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sspr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspr2_batched(handle, uplo, n, alpha,&
              x, incx, y, incy, AP, batch_count)
    end function rocblas_sspr2_batched_fortran

    function rocblas_dspr2_batched_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dspr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspr2_batched(handle, uplo, n, alpha,&
              x, incx, y, incy, AP, batch_count)
    end function rocblas_dspr2_batched_fortran

    ! spr2_strided_batched
    function rocblas_sspr2_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sspr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sspr2_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count)
    end function rocblas_sspr2_strided_batched_fortran

    function rocblas_dspr2_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, AP, stride_AP, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dspr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dspr2_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count)
    end function rocblas_dspr2_strided_batched_fortran

    ! syr
    function rocblas_ssyr_fortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_ssyr_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function rocblas_ssyr_fortran

    function rocblas_dsyr_fortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_dsyr_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function rocblas_dsyr_fortran

    function rocblas_csyr_fortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_csyr_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function rocblas_csyr_fortran

    function rocblas_zsyr_fortran(handle, uplo, n, alpha, x, incx, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_zsyr_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr(handle, uplo, n, alpha,&
              x, incx, A, lda)
    end function rocblas_zsyr_fortran

    ! syr_batched
    function rocblas_ssyr_batched_fortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr_batched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function rocblas_ssyr_batched_fortran

    function rocblas_dsyr_batched_fortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr_batched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function rocblas_dsyr_batched_fortran

    function rocblas_csyr_batched_fortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr_batched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function rocblas_csyr_batched_fortran

    function rocblas_zsyr_batched_fortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyr_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr_batched(handle, uplo, n, alpha,&
              x, incx, A, lda, batch_count)
    end function rocblas_zsyr_batched_fortran

    ! syr_strided_batched
    function rocblas_ssyr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function rocblas_ssyr_strided_batched_fortran

    function rocblas_dsyr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function rocblas_dsyr_strided_batched_fortran

    function rocblas_csyr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function rocblas_csyr_strided_batched_fortran

    function rocblas_zsyr_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyr_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, A, lda, stride_A, batch_count)
    end function rocblas_zsyr_strided_batched_fortran

    ! syr2
    function rocblas_ssyr2_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_ssyr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_ssyr2_fortran

    function rocblas_dsyr2_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_dsyr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_dsyr2_fortran

    function rocblas_csyr2_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_csyr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_csyr2_fortran

    function rocblas_zsyr2_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda) &
            result(res) &
            bind(c, name = 'rocblas_zsyr2_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr2(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda)
    end function rocblas_zsyr2_fortran

    ! syr2_batched
    function rocblas_ssyr2_batched_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr2_batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_ssyr2_batched_fortran

    function rocblas_dsyr2_batched_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr2_batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_dsyr2_batched_fortran

    function rocblas_csyr2_batched_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr2_batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_csyr2_batched_fortran

    function rocblas_zsyr2_batched_fortran(handle, uplo, n, alpha, x, incx, &
            y, incy, A, lda, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyr2_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr2_batched(handle, uplo, n, alpha,&
              x, incx, y, incy, A, lda, batch_count)
    end function rocblas_zsyr2_batched_fortran

    ! syr2_strided_batched
    function rocblas_ssyr2_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr2_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_ssyr2_strided_batched_fortran

    function rocblas_dsyr2_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr2_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_dsyr2_strided_batched_fortran

    function rocblas_csyr2_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr2_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_csyr2_strided_batched_fortran

    function rocblas_zsyr2_strided_batched_fortran(handle, uplo, n, alpha, x, incx, stride_x, &
            y, incy, stride_y, A, lda, stride_A, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyr2_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr2_strided_batched(handle, uplo, n, alpha,&
              x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
    end function rocblas_zsyr2_strided_batched_fortran

    !--------!
    ! blas 3 !
    !--------!

    ! hemm
    function rocblas_chemm_fortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_chemm_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemm(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_chemm_fortran

    function rocblas_zhemm_fortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zhemm_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhemm(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_zhemm_fortran

    ! hemm_batched
    function rocblas_chemm_batched_fortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chemm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemm_batched(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_chemm_batched_fortran

    function rocblas_zhemm_batched_fortran(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhemm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zhemm_batched(handle, side, uplo, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_zhemm_batched_fortran

    ! hemm_strided_batched
    function rocblas_chemm_strided_batched_fortran(handle, side, uplo, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_chemm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemm_strided_batched(handle, side, uplo, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_chemm_strided_batched_fortran

    function rocblas_zhemm_strided_batched_fortran(handle, side, uplo, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zhemm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_chemm_strided_batched(handle, side, uplo, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zhemm_strided_batched_fortran

    ! herk
    function rocblas_cherk_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_cherk_fortran')
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
        integer(c_int) :: res
        res = rocblas_cherk(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc)
    end function rocblas_cherk_fortran

    function rocblas_zherk_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zherk_fortran')
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
        integer(c_int) :: res
        res = rocblas_zherk(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc)
    end function rocblas_zherk_fortran

    ! herk_batched
    function rocblas_cherk_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cherk_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cherk_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count)
    end function rocblas_cherk_batched_fortran

    function rocblas_zherk_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zherk_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zherk_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count)
    end function rocblas_zherk_batched_fortran

    ! herk_strided_batched
    function rocblas_cherk_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cherk_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cherk_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function rocblas_cherk_strided_batched_fortran

    function rocblas_zherk_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zherk_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zherk_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zherk_strided_batched_fortran

    ! her2k
    function rocblas_cher2k_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_cher2k_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher2k(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_cher2k_fortran

    function rocblas_zher2k_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zher2k_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher2k(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_zher2k_fortran

    ! her2k_batched
    function rocblas_cher2k_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cher2k_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher2k_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_cher2k_batched_fortran

    function rocblas_zher2k_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zher2k_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher2k_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_zher2k_batched_fortran

    ! her2k_strided_batched
    function rocblas_cher2k_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cher2k_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cher2k_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_cher2k_strided_batched_fortran

    function rocblas_zher2k_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zher2k_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zher2k_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zher2k_strided_batched_fortran

    ! herkx
    function rocblas_cherkx_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_cherkx_fortran')
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
        integer(c_int) :: res
        res = rocblas_cherkx(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_cherkx_fortran

    function rocblas_zherkx_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zherkx_fortran')
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
        integer(c_int) :: res
        res = rocblas_zherkx(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_zherkx_fortran

    ! herkx_batched
    function rocblas_cherkx_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cherkx_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cherkx_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_cherkx_batched_fortran

    function rocblas_zherkx_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zherkx_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zherkx_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_zherkx_batched_fortran

    ! herkx_strided_batched
    function rocblas_cherkx_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cherkx_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cherkx_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_cherkx_strided_batched_fortran

    function rocblas_zherkx_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zherkx_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zherkx_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zherkx_strided_batched_fortran

    ! symm
    function rocblas_ssymm_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_ssymm_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssymm(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_ssymm_fortran

    function rocblas_dsymm_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_dsymm_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsymm(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_dsymm_fortran

    function rocblas_csymm_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_csymm_fortran')
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
        integer(c_int) :: res
        res = rocblas_csymm(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_csymm_fortran

    function rocblas_zsymm_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zsymm_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsymm(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_zsymm_fortran

    ! symm_batched
    function rocblas_ssymm_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssymm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssymm_batched(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_ssymm_batched_fortran

    function rocblas_dsymm_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsymm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsymm_batched(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_dsymm_batched_fortran

    function rocblas_csymm_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csymm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csymm_batched(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_csymm_batched_fortran

    function rocblas_zsymm_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsymm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsymm_batched(handle, side, uplo, m, n, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_zsymm_batched_fortran

    ! symm_strided_batched
    function rocblas_ssymm_strided_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssymm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssymm_strided_batched(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_ssymm_strided_batched_fortran

    function rocblas_dsymm_strided_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsymm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsymm_strided_batched(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_dsymm_strided_batched_fortran

    function rocblas_csymm_strided_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csymm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csymm_strided_batched(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_csymm_strided_batched_fortran

    function rocblas_zsymm_strided_batched_fortran(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsymm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsymm_strided_batched(handle, side, uplo, m, n, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zsymm_strided_batched_fortran

    ! syrk
    function rocblas_ssyrk_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_ssyrk_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyrk(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc)
    end function rocblas_ssyrk_fortran

    function rocblas_dsyrk_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_dsyrk_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyrk(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc)
    end function rocblas_dsyrk_fortran

    function rocblas_csyrk_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_csyrk_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyrk(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc)
    end function rocblas_csyrk_fortran

    function rocblas_zsyrk_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zsyrk_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyrk(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc)
    end function rocblas_zsyrk_fortran

    ! syrk_batched
    function rocblas_ssyrk_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyrk_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyrk_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count)
    end function rocblas_ssyrk_batched_fortran

    function rocblas_dsyrk_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyrk_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyrk_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count)
    end function rocblas_dsyrk_batched_fortran

    function rocblas_csyrk_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyrk_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyrk_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count)
    end function rocblas_csyrk_batched_fortran

    function rocblas_zsyrk_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyrk_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyrk_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, beta, C, ldc, batch_count)
    end function rocblas_zsyrk_batched_fortran

    ! syrk_strided_batched
    function rocblas_ssyrk_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyrk_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function rocblas_ssyrk_strided_batched_fortran

    function rocblas_dsyrk_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyrk_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function rocblas_dsyrk_strided_batched_fortran

    function rocblas_csyrk_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyrk_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function rocblas_csyrk_strided_batched_fortran

    function rocblas_zsyrk_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyrk_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyrk_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zsyrk_strided_batched_fortran

    ! syr2k
    function rocblas_ssyr2k_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_ssyr2k_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr2k(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_ssyr2k_fortran

    function rocblas_dsyr2k_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_dsyr2k_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr2k(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_dsyr2k_fortran

    function rocblas_csyr2k_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_csyr2k_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr2k(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_csyr2k_fortran

    function rocblas_zsyr2k_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zsyr2k_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr2k(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_zsyr2k_fortran

    ! syr2k_batched
    function rocblas_ssyr2k_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyr2k_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr2k_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_ssyr2k_batched_fortran

    function rocblas_dsyr2k_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyr2k_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr2k_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_dsyr2k_batched_fortran

    function rocblas_csyr2k_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyr2k_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr2k_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_csyr2k_batched_fortran

    function rocblas_zsyr2k_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyr2k_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr2k_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_zsyr2k_batched_fortran

    ! syr2k_strided_batched
    function rocblas_ssyr2k_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyr2k_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_ssyr2k_strided_batched_fortran

    function rocblas_dsyr2k_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyr2k_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_dsyr2k_strided_batched_fortran

    function rocblas_csyr2k_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyr2k_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_csyr2k_strided_batched_fortran

    function rocblas_zsyr2k_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyr2k_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyr2k_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zsyr2k_strided_batched_fortran

    ! syrkx
    function rocblas_ssyrkx_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_ssyrkx_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyrkx(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_ssyrkx_fortran

    function rocblas_dsyrkx_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_dsyrkx_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyrkx(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_dsyrkx_fortran

    function rocblas_csyrkx_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_csyrkx_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyrkx(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_csyrkx_fortran

    function rocblas_zsyrkx_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zsyrkx_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyrkx(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc)
    end function rocblas_zsyrkx_fortran

    ! syrkx_batched
    function rocblas_ssyrkx_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyrkx_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyrkx_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_ssyrkx_batched_fortran

    function rocblas_dsyrkx_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyrkx_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyrkx_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_dsyrkx_batched_fortran

    function rocblas_csyrkx_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyrkx_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyrkx_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_csyrkx_batched_fortran

    function rocblas_zsyrkx_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyrkx_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyrkx_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, B, ldb, beta, C, ldc, batch_count)
    end function rocblas_zsyrkx_batched_fortran

    ! syrkx_strided_batched
    function rocblas_ssyrkx_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ssyrkx_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ssyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_ssyrkx_strided_batched_fortran

    function rocblas_dsyrkx_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dsyrkx_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dsyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_dsyrkx_strided_batched_fortran

    function rocblas_csyrkx_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_csyrkx_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_csyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_csyrkx_strided_batched_fortran

    function rocblas_zsyrkx_strided_batched_fortran(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zsyrkx_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zsyrkx_strided_batched(handle, uplo, transA, n, k, alpha, &
            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
    end function rocblas_zsyrkx_strided_batched_fortran

    ! dgmm
    function rocblas_sdgmm_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_sdgmm_fortran')
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
        integer(c_int) :: res
        res = rocblas_sdgmm(handle, side, m, n, &
            A, lda, x, incx, C, ldc)
    end function rocblas_sdgmm_fortran

    function rocblas_ddgmm_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_ddgmm_fortran')
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
        integer(c_int) :: res
        res = rocblas_ddgmm(handle, side, m, n, &
            A, lda, x, incx, C, ldc)
    end function rocblas_ddgmm_fortran

    function rocblas_cdgmm_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_cdgmm_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdgmm(handle, side, m, n, &
            A, lda, x, incx, C, ldc)
    end function rocblas_cdgmm_fortran

    function rocblas_zdgmm_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zdgmm_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdgmm(handle, side, m, n, &
            A, lda, x, incx, C, ldc)
    end function rocblas_zdgmm_fortran

    ! dgmm_batched
    function rocblas_sdgmm_batched_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sdgmm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sdgmm_batched(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count)
    end function rocblas_sdgmm_batched_fortran

    function rocblas_ddgmm_batched_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ddgmm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ddgmm_batched(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count)
    end function rocblas_ddgmm_batched_fortran

    function rocblas_cdgmm_batched_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cdgmm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdgmm_batched(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count)
    end function rocblas_cdgmm_batched_fortran

    function rocblas_zdgmm_batched_fortran(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zdgmm_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdgmm_batched(handle, side, m, n, &
            A, lda, x, incx, C, ldc, batch_count)
    end function rocblas_zdgmm_batched_fortran

    ! dgmm_strided_batched
    function rocblas_sdgmm_strided_batched_fortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sdgmm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sdgmm_strided_batched(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function rocblas_sdgmm_strided_batched_fortran

    function rocblas_ddgmm_strided_batched_fortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_ddgmm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_ddgmm_strided_batched(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function rocblas_ddgmm_strided_batched_fortran

    function rocblas_cdgmm_strided_batched_fortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cdgmm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cdgmm_strided_batched(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function rocblas_cdgmm_strided_batched_fortran

    function rocblas_zdgmm_strided_batched_fortran(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zdgmm_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zdgmm_strided_batched(handle, side, m, n, &
            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
    end function rocblas_zdgmm_strided_batched_fortran

    ! geam
    function rocblas_sgeam_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_sgeam_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgeam(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc)
    end function rocblas_sgeam_fortran

    function rocblas_dgeam_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_dgeam_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgeam(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc)
    end function rocblas_dgeam_fortran

    function rocblas_cgeam_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_cgeam_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgeam(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc)
    end function rocblas_cgeam_fortran

    function rocblas_zgeam_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc) &
            result(res) &
            bind(c, name = 'rocblas_zgeam_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgeam(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc)
    end function rocblas_zgeam_fortran

    ! geam_batched
    function rocblas_sgeam_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sgeam_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgeam_batched(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count)
    end function rocblas_sgeam_batched_fortran

    function rocblas_dgeam_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dgeam_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgeam_batched(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count)
    end function rocblas_dgeam_batched_fortran

    function rocblas_cgeam_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgeam_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgeam_batched(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count)
    end function rocblas_cgeam_batched_fortran

    function rocblas_zgeam_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgeam_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgeam_batched(handle, transA, transB, m, n, alpha, &
            A, lda, beta, B, ldb, C, ldc, batch_count)
    end function rocblas_zgeam_batched_fortran

    ! geam_strided_batched
    function rocblas_sgeam_strided_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_sgeam_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_sgeam_strided_batched(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function rocblas_sgeam_strided_batched_fortran

    function rocblas_dgeam_strided_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_dgeam_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_dgeam_strided_batched(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function rocblas_dgeam_strided_batched_fortran

    function rocblas_cgeam_strided_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_cgeam_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_cgeam_strided_batched(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function rocblas_cgeam_strided_batched_fortran

    function rocblas_zgeam_strided_batched_fortran(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
            result(res) &
            bind(c, name = 'rocblas_zgeam_strided_batched_fortran')
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
        integer(c_int) :: res
        res = rocblas_zgeam_strided_batched(handle, transA, transB, m, n, alpha, &
            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
    end function rocblas_zgeam_strided_batched_fortran

end module rocblas_interface

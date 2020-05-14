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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
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
        integer(c_int), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int), value :: stride_s
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
        integer(c_int), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int), value :: stride_s
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
        integer(c_int), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int), value :: stride_s
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
        integer(c_int), value :: stride_a
        type(c_ptr), value :: b
        integer(c_int), value :: stride_b
        type(c_ptr), value :: c
        integer(c_int), value :: stride_c
        type(c_ptr), value :: s
        integer(c_int), value :: stride_s
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
        type(c_ptr), value :: param
        integer(c_int), value :: stride_param
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
        integer(c_int), value :: stride_x
        type(c_ptr), value :: y
        integer(c_int), value :: incy
        integer(c_int), value :: stride_y
        type(c_ptr), value :: param
        integer(c_int), value :: stride_param
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
        integer(c_int), value :: stride_d1
        type(c_ptr), value :: d2
        integer(c_int), value :: stride_d2
        type(c_ptr), value :: x1
        integer(c_int), value :: stride_x1
        type(c_ptr), value :: y1
        integer(c_int), value :: stride_y1
        type(c_ptr), value :: param
        integer(c_int), value :: stride_param
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
        integer(c_int), value :: stride_d1
        type(c_ptr), value :: d2
        integer(c_int), value :: stride_d2
        type(c_ptr), value :: x1
        integer(c_int), value :: stride_x1
        type(c_ptr), value :: y1
        integer(c_int), value :: stride_y1
        type(c_ptr), value :: param
        integer(c_int), value :: stride_param
        integer(c_int), value :: batch_count
        integer(c_int) :: res
        res = rocblas_drotmg_strided_batched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1,&
            param, stride_param, batch_count)
        return
    end function rocblas_drotmg_strided_batched_fortran

end module rocblas_interface
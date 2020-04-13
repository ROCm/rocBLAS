/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

inline bool dcld(rocblas_int ld)
{
    // Purpose
    // =======
    //
    // The size of the leading dimension of a two-dimensional array may
    // cause severe problems. Often when an array with a 'critical' leading
    // dimension is referenced, the execution time becomes significantly
    // longer than expected. This is caused by shortcomings of the memory
    // system.
    //
    // The function DCLD returns .TRUE. if the leading dimension LD is
    // critical and .FALSE. if it is not critical. In this implementation
    // DCLD is designed to detect critical leading dimensions in an
    // environment with a multi-way associative cache. Parameters defining
    // cache characteristics are adjustable to match different machines.
    // It may be rewarding to rewrite DCLD for a machine with a different
    // cache policy.
    //
    // The cache lines in a multi-way associative cache are divided among a
    // number of partitions, each containing the same number of lines. Each
    // address of main memory is mapped into a particular partition. The
    // number of lines in a partition equals the associativity. For example,
    // in a four way associative cache, each partition contain four cache
    // lines.
    //
    // Data are transferred between the cache and main memory according to
    // an associative mapping scheme. A transfer of a data word from main
    // memory to cache is accomplished as follows. A unit of data
    // (data line) in main memory, with the size of a cache line, and
    // containing several contiguous data words including the referenced
    // one, is mapped (copied) to a certain partition in the cache memory.
    // The partition is determined by the location of the element in the
    // main memory and the associative mapping scheme. A replacement
    // algorithm makes room for the data line in one of the cache lines in
    // the selected partition. For example, an LRU-based (Least Recently
    // Used) replacement algorithm places the data line in the least
    // recently 'touched' cache line in the selected partition.
    //
    // Input
    // =====
    //
    // LD     - On entry, LD specifies the leading dimension of a
    //          2-dimensional array. Unchanged on exit.
    //
    // User specified parameters for DCLD
    // ================================
    //
    // LNSZ   - Size of a cache line in number of bytes.
    //
    // NPRT   - Number of partitions in the cache memory.
    //
    // PRTSZ  - The number of cache lines in a partition that can be used
    //          exclusively to hold a local array containing a matrix block
    //          during the execution of a GEMM-Based Level 3 BLAS routine.
    //          The remaining cache lines may be occupied by scalars,
    //          vectors and possibly program code depending on the system.
    //
    // LOLIM  - Leading dimensions smaller than or equal to LOLIM are not
    //          considered critical.
    //
    // DP     - Number of bytes in a double-precision word.
    //
    //
    // Local Variables and Parameters
    // ==============================
    //
    // ONEWAY - The maximum number of double precision words that can be
    //          stored in the cache memory if only a single cache line in
    //          each partition may be used.
    //
    // UPDIF  - The difference between the multiple of LD that is nearest
    //          ONEWAY, or nearest a multiple of ONEWAY, and the nearest
    //          multiple of ONEWAY that is larger than LD. In number of
    //          double precision words.
    //
    // MXDIF  - If both UPDIF and LD - UPDIF are less than MXDIF, and LD
    //          is greater than LOLIM, then the leading dimension is
    //          considered critical. Otherwise, the leading dimension is
    //          considered not critical.
    //
    //
    // -- Written in December-1993.
    //    GEMM-Based Level 3 BLAS.
    //    Per Ling, Institute of Information Processing,
    //    University of Umea, Sweden.
    //
    //    .. User specified parameters for DCLD ..

    rocblas_int lnsz  = 64;
    rocblas_int nprt  = 128;
    rocblas_int prtsz = 3;
    rocblas_int lolim = 64;
    rocblas_int dp    = 8;

    //    .. Parameters ..
    rocblas_int oneway = (lnsz * nprt) / dp;
    rocblas_int mxdif  = lnsz / (dp * prtsz);
    //    ..
    //    .. Executable Statements ..
    //
    if(ld <= lolim)
    {
        return false;
    }
    else
    {
        rocblas_int updif     = ((ld / oneway) * oneway + oneway) % ld;
        rocblas_int min_updif = updif < ld - updif ? updif : ld - updif;
        if(min_updif <= mxdif)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

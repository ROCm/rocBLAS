#!/usr/bin/python3

# a timing script for FFTs and convolutions using OpenMP

import sys, getopt
import numpy as np
from math import *
import subprocess
import os
import re # regexp package
import shutil
import tempfile
import re

usage = '''A timing script for rocblas

Usage:
\ttiming.py
\t\t-I          make transform in-place
\t\t-a <int>    number of samples used for median per problem size
\t\t-i <int>    number of iterations averaged per sample
\t\t-j <int>    number of cold/warmup iterations before timing iterations
\t\t-o <string> name of output file
\t\t-R          set transform to be real/complex or complex/real
\t\t-w <string> set working directory for rocblas-bench
\t\t-n <int>    minimum problem size in all directions for now
\t\t-N <int>    maximum problem size in all directions for now
\t\t-f <string> precision: float(default) or double
\t\t-b <int>    batch size
\t\t-g <int>    device number
\t\t-t <string> data type: time or gflops (default: time)'''

# \t\t-y <int>    minimum problem size in y direction
# \t\t-Y <int>    maximum problem size in Y direction
# \t\t-z <int>    minimum problem size in z direction
# \t\t-Z <int>    maximum problem size in Z direction

def runcase(workingdir, mval, nval, kval, ntrial, precision, nbatch,
            devicenum, logfilename, function, side, uplo, diag, transA, transB, alpha, beta, incx, incy, lda, ldb, ldc, algo, iters, cold_iters):
    progname = "rocblas-bench"
    prog = os.path.join(workingdir, progname)

    cmd = []
    cmd.append(prog)

    cmd.append("-f")
    cmd.append(function)

    cmd.append("--transposeA")
    cmd.append(str(transA))

    cmd.append("--transposeB")
    cmd.append(str(transB))

    cmd.append("-m")
    cmd.append(str(mval))

    cmd.append("-n")
    cmd.append(str(nval))

    cmd.append("-k")
    cmd.append(str(kval))

    cmd.append("--side")
    cmd.append(str(side))

    cmd.append("--uplo")
    cmd.append(str(uplo))

    cmd.append("--diag")
    cmd.append(str(diag))

    cmd.append("--lda")
    cmd.append(str(lda))

    cmd.append("--ldb")
    cmd.append(str(ldb))

    cmd.append("--ldc")
    cmd.append(str(ldc))

    cmd.append("--alpha")
    cmd.append(str(alpha))

    cmd.append("--beta")
    cmd.append(str(beta))

    cmd.append("--incx")
    cmd.append(str(incx))

    cmd.append("--incy")
    cmd.append(str(incy))

    cmd.append("-i")
    cmd.append(str(iters))

    if (cold_iters >= 0):
        cmd.append("-j")
        cmd.append(str(cold_iters))

    cmd.append("-r")
    cmd.append(precision)

    cmd.append("--batch_count")
    cmd.append(str(nbatch))

    cmd.append("--algo")
    cmd.append(str(algo))

    cmd.append("--device")
    cmd.append(str(devicenum))

    print(" ".join(cmd))

    fout = tempfile.TemporaryFile(mode="w+")

    for x in range(ntrial):
        proc = subprocess.Popen(cmd, cwd=os.path.join(workingdir,"..",".."), stdout=fout, stderr=fout,
                                env=os.environ.copy())
        proc.wait()
        rc = proc.returncode

        if rc != 0:
            lines = fout.readlines()
            logfile = open(logfilename, "a")
            logfile.write('\n'.join(lines))
            logfile.close()
            print("\twell, that didn't work")
            print(rc)
            print(" ".join(cmd))
            return [], [], []

    us_vals = []
    gf_vals = []
    bw_vals = []

    fout.seek(0)

    lines = fout.readlines()
    logfile = open(logfilename, "a")
    logfile.write('\n'.join(lines))
    logfile.close()

    gf_string = "rocblas-Gflops"
    bw_string = "rocblas-GB/s"
    us_string = "us"
    for i in range(0, len(lines)):
        if re.search(r"\b" + re.escape(us_string) + r"\b", lines[i]) is not None:
            us_line = lines[i].strip().split(",")
            index = [idx for idx, s in enumerate(us_line) if us_string in s][0] #us_line.index()
            us_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
        if gf_string in lines[i]:
            gf_line = lines[i].split(",")
            index = gf_line.index(gf_string)
            gf_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))
        if bw_string in lines[i]:
            bw_line = lines[i].split(",")
            index = bw_line.index(bw_string)
            bw_vals.append(float(re.split(r',\s*(?![^()]*\))', lines[i+1])[index]))


    fout.close()

    return us_vals, gf_vals, bw_vals

def incrementParam(cur, max, mul, step_size, done):
    if cur < max:
        if mul == 1 and cur * step_size <= max:
            return cur*step_size, False
        elif mul != 1 and cur + step_size <= max:
            return cur+step_size, False
    return cur, done


def main(argv):
    workingdir = "."
    nmin = 1
    nmax = 1024
    kmin = 1
    kmax = 1
    mmin = 1
    mmax = 1
    ntrial = 1
    iters = 1
    cold_iters = -1  # default to not set
    outfilename = "timing.dat"
    precision = "f32_r"
    nbatch = 1
    algo = 0
    # datatype = "time"
    radix = 2
    step_size = 10
    devicenum = 0
    function = ""
    alpha = 1
    beta = 1
    incx = 1
    incy = 1
    precision = ""
    transA = "N"
    transB = "N"
    lda = 1
    ldb = 1
    ldc = 1
    LDA = 1
    LDB = 1
    LDC = 1
    initialization = ""
    step_mult = 0
    side = "L"
    uplo = "L"
    diag = "N"

    try:
        print(argv)
        opts, args = getopt.getopt(argv,"hb:d:I:i:j:o:Rt:w:m:n:k:M:N:K:y:Y:z:Z:f:r:g:p:s:a:x", ["side=", "uplo=", "diag=", "incx=", "incy=",
         "alpha=", "beta=", "transA=", "transB=", "lda=", "ldb=", "ldc=", "LDA=", "LDB=", "LDC=", "algo=", "initialization="])
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-o"):
            outfilename = arg
        # elif opt in ("-t"):
        #     if arg not in ["time", "gflops"]:
        #         print("data type must be time or gflops")
        #         print(usage)
        #         sys.exit(1)
        #     datatype = arg
        elif opt in ("-R"):
            rcfft = True
        elif opt in ("-w"):
            workingdir = arg
        elif opt in ("-i"):
            iters = int(arg)
        elif opt in ("-j"):
            cold_iters = int(arg)
        elif opt in ("-a"):
            ntrial = int(arg)
        elif opt in ("-m"):
            mmin = int(arg)
        elif opt in ("-M"):
            mmax = int(arg)
        elif opt in ("-n"):
            nmin = int(arg)
        elif opt in ("-N"):
            nmax = int(arg)
        elif opt in ("-k"):
            kmin = int(arg)
        elif opt in ("-K"):
            kmax = int(arg)
        elif opt in ("-b"):
            nbatch = int(arg)
        elif opt in ("-r"):
            radix = int(arg)
        elif opt in ("-s"):
            step_size = int(arg)
        elif opt in ("-x"):
            step_mult = 1
        elif opt in ("-f"):
            function = arg
        elif opt in ("--incx"):
            incx = int(float(arg))
        elif opt in ("--side"):
            side = arg
        elif opt in ("--uplo"):
            uplo = arg
        elif opt in ("--diag"):
            diag = arg
        elif opt in ("--incy"):
            incy = int(float(arg))
        elif opt in ("--alpha"):
            alpha = float(arg)
        elif opt in ("--beta"):
            beta = float(arg)
        elif opt in ("--transA"):
            transA = arg
        elif opt in ("--transB"):
            transB = arg
        elif opt in ("--lda"):
            lda = int(arg)
        elif opt in ("--ldb"):
            ldb = int(arg)
        elif opt in ("--ldc"):
            ldc = int(arg)
        elif opt in ("--LDA"):
            LDA = int(arg)
        elif opt in ("--LDB"):
            LDB = int(arg)
        elif opt in ("--LDC"):
            LDC = int(arg)
        elif opt in ("--algo"):
            algo = int(arg)
        elif opt in ("--initialization"):
            initialization = arg
        elif opt in ("-p"):
            precision = arg
        elif opt in ("-g"):
            devicenum = int(arg)

    print("workingdir: "+ workingdir)
    print("outfilename: "+ outfilename)
    print("ntrial: " + str(ntrial))
    print("nmin: "+ str(nmin) + " nmax: " + str(nmax))
    print("batch-size: " + str(nbatch))
    # print("data type: " + datatype)
    print("device number: " + str(devicenum))

    progname = "rocblas-bench"
    prog = os.path.join(workingdir, progname)
    if not os.path.isfile(prog):
        print("**** Error: unable to find " + prog)
        sys.exit(1)

    logfilename = outfilename + ".log"
    print("log filename: "  + logfilename)
    logfile = open(logfilename, "w+")
    metadatastring = "# " + " ".join(sys.argv)  + "\n"
    logfile.write(metadatastring)
    logfile.close()

    outfile = open(outfilename, "w+")
    outfile.write(metadatastring)
    outfile.close()

    mval = mmin
    nval = nmin
    kval = kmin
    done = False
    while(not done):
        us, gf, bw = runcase(workingdir, mval, nval, kval, ntrial,
                          precision, nbatch, devicenum, logfilename, function, side, uplo, diag, transA, transB, alpha, beta, incx, incy, lda, ldb, ldc, algo, iters, cold_iters)
        #print(seconds)
        with open(outfilename, 'a') as outfile:
            if function == 'trsm':
                if side == 'R':
                    outfile.write(str(mval))
                else:
                    outfile.write(str(nval))
            else:
                outfile.write(str(nval))
            outfile.write("\t")
            outfile.write('{:4.0f}'.format(len(us)))
            for time in us:
                outfile.write('{:12.2f}'.format(time))
            outfile.write('{:4.0f}'.format(len(gf)))
            for flops in gf:
                outfile.write('{:10.2f}'.format(flops))
            outfile.write('{:4.0f}'.format(len(bw)))
            for bandwidth in bw:
                outfile.write('{:10.2f}'.format(bandwidth))
            outfile.write("\n")

        done = True
        mval, done = incrementParam(mval, mmax, step_mult, step_size, done)
        nval, done = incrementParam(nval, nmax, step_mult, step_size, done)
        kval, done = incrementParam(kval, kmax, step_mult, step_size, done)
        lda, done = incrementParam(lda, LDA, step_mult, step_size, done)
        ldb, done = incrementParam(ldb, LDB, step_mult, step_size, done)
        ldc, done = incrementParam(ldc, LDC, step_mult, step_size, done)



if __name__ == "__main__":
    main(sys.argv[1:])


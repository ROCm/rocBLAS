#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCFFT_EXPORT //g' ../library/include/rocblas.h > rocblas.h
doxygen Doxyfile

cd source
make clean
make html
cd ..

rm rocblas.h

#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sed -e 's/ROCBLAS_EXPORT //g' ../library/include/rocfft.h > rocfft.h
doxygen Doxyfile
rm rocfft.h

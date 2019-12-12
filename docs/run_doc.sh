#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

sh run_doxygen.sh

cd source
make clean
make html
make latexpdf
cd ..

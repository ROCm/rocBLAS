#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="TENSILE_VERSION 4.39.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.40.0"

OLD_TENSILE_HASH="3a4153ac72b11c5258e7a5687bbb11d9bc739680"
NEW_TENSILE_HASH="2a3d8fea619ccefe003ddc1de8c114f758774d81"

OLD_ROCBLAS_VERSION="4.1.0"
NEW_ROCBLAS_VERSION="4.2.0"

OLD_SO_VERSION="rocblas_SOVERSION 4.1"
NEW_SO_VERSION="rocblas_SOVERSION 4.2"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt

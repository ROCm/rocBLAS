#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="TENSILE_VERSION 4.40.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.41.0"

OLD_TENSILE_HASH="d2813593f97e892303c8870a3efb301d48ba52f3"
NEW_TENSILE_HASH="8fd47da00b782b53706c4c08ac28a1a7026946d4"

OLD_ROCBLAS_VERSION="4.2.0"
NEW_ROCBLAS_VERSION="4.3.0"

OLD_SO_VERSION="rocblas_SOVERSION 4.2"
NEW_SO_VERSION="rocblas_SOVERSION 4.3"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt

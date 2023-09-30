#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.38.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.39.0"

OLD_TENSILE_HASH="614e5d9eae7da8abec5ed4629d14d624c3b5aa3c"
NEW_TENSILE_HASH="44f174e5f3d9e8dec9783fd1f114f1adb2b661e9"

OLD_SO_VERSION="rocblas_SOVERSION 4.0.0"
NEW_SO_VERSION="rocblas_SOVERSION 4.0"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt

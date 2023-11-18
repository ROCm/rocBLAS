#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.39.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.40.0"

OLD_TENSILE_HASH="3a4153ac72b11c5258e7a5687bbb11d9bc739680"
NEW_TENSILE_HASH="d23b84ae6499b922831c11e75b1daa58d520fe01"

OLD_SO_VERSION="rocblas_SOVERSION 4.1"
NEW_SO_VERSION="rocblas_SOVERSION 4.1"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt

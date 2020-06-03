#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.24.0"
NEW_ROCBLAS_VERSION="2.25.0"

OLD_TENSILE_VERSION="Tensile 4.20.0"
NEW_TENSILE_VERSION="Tensile 4.20.0"

OLD_TENSILE_HASH="b38a7e0ce0cb898594d25c4a77ca6a55b10730b8"
NEW_TENSILE_HASH="ea87bcaa91c99672aa7d0b0a7981dbdd8b5eb2f5"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


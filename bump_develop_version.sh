#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.26.0"
NEW_ROCBLAS_VERSION="2.27.0"

OLD_TENSILE_VERSION="Tensile 4.21.0"
NEW_TENSILE_VERSION="Tensile 4.21.0"

OLD_TENSILE_HASH="148626ac3fbae797455522260ae56c6fb1e7e6cf"
NEW_TENSILE_HASH="79482dece074bea9b1a87792cbbe26b1ebd6e06e"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


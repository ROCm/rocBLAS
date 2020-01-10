#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.14.0"
NEW_ROCBLAS_VERSION="2.15.0"

OLD_TENSILE_VERSION="Tensile 4.14.0"
NEW_TENSILE_VERSION="Tensile 4.14.0"

OLD_TENSILE_HASH="d52cdbf1f3ef0e860cb8a7dc2acbd9fdb46260e6"
NEW_TENSILE_HASH="d52cdbf1f3ef0e860cb8a7dc2acbd9fdb46260e6"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


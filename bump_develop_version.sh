#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.14.1"
NEW_ROCBLAS_VERSION="2.15.1"

OLD_TENSILE_VERSION="Tensile 4.15.0"
NEW_TENSILE_VERSION="Tensile 4.15.0"

OLD_TENSILE_HASH="31ac86aaa5e4d5b892e50a0e61c63e37b54b22bd"
NEW_TENSILE_HASH="394e6ae2f08004b0f24f46aa9f2b9da42a8f185f"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


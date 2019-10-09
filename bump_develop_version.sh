#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch 
# - after running this script merge master into develop 

OLD_ROCBLAS_VERSION="2.10.0"
NEW_ROCBLAS_VERSION="2.11.0"

OLD_TENSILE_VERSION="Tensile 4.12.0"
NEW_TENSILE_VERSION="Tensile 4.13.0"

OLD_TENSILE_HASH="fe4f721886d07eef6251cea4225e027181022aa5"
NEW_TENSILE_HASH="a9379f4e42efb754c9a618047bfbf292d74dfd0f"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


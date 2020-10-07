#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.32.0"
NEW_ROCBLAS_VERSION="2.33.0"

OLD_TENSILE_VERSION="Tensile 4.24.0"
NEW_TENSILE_VERSION="Tensile 4.24.0"

OLD_TENSILE_HASH="3ddca8e109c4633613f5c8289fb399a19d058826"
NEW_TENSILE_HASH="31e10427e1612ea7d1b015980aa7e6ef7f1bc8ad"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


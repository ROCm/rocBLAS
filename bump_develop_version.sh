#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.28.0"
NEW_ROCBLAS_VERSION="2.29.0"

OLD_TENSILE_VERSION="Tensile 4.22.0"
NEW_TENSILE_VERSION="Tensile 4.22.0"

OLD_TENSILE_HASH="efb7d485af43175b83bdfda094b61b5653ba2dae"
NEW_TENSILE_HASH="bb27142d6445ab3077ece9f5db2e662850a6ae46"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch 
# - after running this script merge master into develop 

OLD_ROCBLAS_VERSION="2.12.0"
NEW_ROCBLAS_VERSION="2.13.0"

OLD_TENSILE_VERSION="Tensile 4.14.0"
NEW_TENSILE_VERSION="Tensile 4.14.0"

OLD_TENSILE_HASH="78e72fbb35c3fba688deab3895f70f2f39f537af"
NEW_TENSILE_HASH="78e72fbb35c3fba688deab3895f70f2f39f537af"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


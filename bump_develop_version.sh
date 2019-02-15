#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch 
# - after running this script merge master into develop 

OLD_ROCBLAS_VERSION="2.1.1"
NEW_ROCBLAS_VERSION="2.2.1"

OLD_TENSILE_VERSION="tensile_tag d81a997e7cace7da2edfb8c5d40a9b6b2387bdf9"
NEW_TENSILE_VERSION="tensile_tag \"develop\""

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt


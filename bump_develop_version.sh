#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch 
# - after running this script merge master into develop 

OLD_ROCBLAS_VERSION="2.0.0"
NEW_ROCBLAS_VERSION="2.1.0"

OLD_TENSILE_VERSION="tensile_tag 7e50085d9e310349152f686275d0ad98cc58b130"
NEW_TENSILE_VERSION="tensile_tag \"develop\""

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt


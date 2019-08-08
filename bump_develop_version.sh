#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch 
# - after running this script merge master into develop 

OLD_ROCBLAS_VERSION="2.6.0"
NEW_ROCBLAS_VERSION="2.7.0"

OLD_TENSILE_VERSION="tensile_tag f5b33e22367807ca5bff1002b6e7e8939409d961"
NEW_TENSILE_VERSION="tensile_tag develop"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt


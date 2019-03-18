#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch 
# - after running this script merge master into develop 

OLD_ROCBLAS_VERSION="2.2.2"
NEW_ROCBLAS_VERSION="2.3.2"

OLD_TENSILE_VERSION="tensile_tag 8f4183bd4a94df0d1faad30645be80957631a77f"
NEW_TENSILE_VERSION="tensile_tag \"develop\""

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt


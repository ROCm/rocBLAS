#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.34.0"
NEW_ROCBLAS_VERSION="2.35.0"

OLD_TENSILE_VERSION="Tensile 4.25.0"
NEW_TENSILE_VERSION="Tensile 4.25.0"

OLD_TENSILE_HASH="c7f2ddca8586e67be75d0aaeec371d07a2f96ae7"
NEW_TENSILE_HASH="becb32f9af16c8003712acbdae4ce54a6376113a"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


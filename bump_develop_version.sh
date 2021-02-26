#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_ROCBLAS_VERSION="2.38.0"
NEW_ROCBLAS_VERSION="2.39.0"

OLD_TENSILE_VERSION="Tensile 4.27.0"
NEW_TENSILE_VERSION="Tensile 4.27.0"

OLD_TENSILE_HASH="1a8369a4baf528926a179cea86c630176d17298a"
NEW_TENSILE_HASH="ec7a04df20b3b0248de61add042525f49210209d"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt


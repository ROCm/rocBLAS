#!/bin/bash

# This script needs to be edited to bump new master hash to new staging hash for new release.
# - run this script after running bump_master_version.sh and merging staging into master
# - run this script in master branch
# - after running this script merge master into staging

OLD_TENSILE_HASH="ed3af9d8bc94258464881c4684daf651bc254a4c"
NEW_TENSILE_HASH="cdc4ab982d64f9b6ff98cd65dcb4ded610397786"

OLD_ROCBLAS_VERSION="2.46.0"
NEW_ROCBLAS_VERSION="2.47.0"

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt


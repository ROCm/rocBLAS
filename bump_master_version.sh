#!/bin/bash

# This script needs to be edited to bump old develop version to new master version for new release.
# - run this script in develop branch
# - after running this script merge develop into master
# - after running this script and merging develop into master, run bump_develop_version.sh in master and
#   merge master into develop

OLD_ROCBLAS_VERSION="2.25.0"
NEW_ROCBLAS_VERSION="2.26.0"

OLD_TENSILE_VERSION="Tensile 4.20.0"
NEW_TENSILE_VERSION="Tensile 4.21.0"

OLD_TENSILE_HASH="e9c3142c3ae4402715809970fc5bced6bf2ec384"
NEW_TENSILE_HASH="148626ac3fbae797455522260ae56c6fb1e7e6cf"

OLD_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.6.0"
NEW_MINIMUM_REQUIRED_VERSION="MinimumRequiredVersion: 4.7.1"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

#only update yaml files for a Tensile major version change
#for FILE in library/src/blas3/Tensile/Logic/*/*yaml
#do
#  sed -i "s/${OLD_MINIMUM_REQUIRED_VERSION}/${NEW_MINIMUM_REQUIRED_VERSION}/" $FILE
#a9379f4e42efb754c9a618047bfbf292d74dfd0fdone

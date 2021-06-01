#!/bin/bash

# This script needs to be edited to bump develop version after feature complete
# - run this script in develop branch

OLD_ROCBLAS_VERSION="2.39.0"
NEW_ROCBLAS_VERSION="2.40.0"

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

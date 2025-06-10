#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.44.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.44.0"

OLD_TENSILE_HASH="7126604cd7ff65bd9eb7d458830e67fb15d78cac"
NEW_TENSILE_HASH="3f758d57616084520788537d59fe90ffacd5b4ea"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_HIPBLASLT_VERSION}/${NEW_HIPBLASLT_VERSION}/g" CMakeLists.txt

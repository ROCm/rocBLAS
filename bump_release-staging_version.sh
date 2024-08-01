#!/bin/bash

# For the release-staging branch his script bumps the Tensile version and hash

OLD_TENSILE_VERSION="TENSILE_VERSION 4.41.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.42.0"

OLD_TENSILE_HASH="34a42538e425c330b52694e3adf9fcda537db1dd"
NEW_TENSILE_HASH="df4be50f2cf7abb86b2fc7af171802a8b16e043a"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

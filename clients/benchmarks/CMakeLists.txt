# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
# ########################################################################

# Depending on whether we are building for 64 or 32 bit, construct common paths and names that subdirectories can reference for their use
if( rocBLAS_BUILD64 )
	set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${rocBLAS-clients_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x64" )
	set( INCLUDE_DIR include )
else( )
	set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${rocBLAS-clients_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x32" )
	set( INCLUDE_DIR include )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
	set( CPACK_SOURCE_GENERATOR "ZIP" )
	set( CPACK_GENERATOR "ZIP" )
else( )
	set( CPACK_SOURCE_GENERATOR "TGZ" )
	set( CPACK_GENERATOR "TGZ" )
endif( )

set( CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${rocBLAS-clients_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-Source")
# set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set( CPACK_PACKAGE_VERSION_MAJOR ${rocBLAS-clients_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${rocBLAS-clients_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${rocBLAS-clients_VERSION_PATCH} )
set( CPACK_PACKAGE_VERSION_TWEAK ${rocBLAS-clients_VERSION_TWEAK} )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute BLAS library package")
set( CPACK_PACKAGE_VENDOR "AMD")
set( CPACK_SOURCE_IGNORE_FILES "/\\\\.git/;/\\\\.hg/;/\\\\.svn/;" )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )

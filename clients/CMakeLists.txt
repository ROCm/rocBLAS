# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

add_definitions(-D_ROCBLAS_INTERNAL_BFLOAT16_)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# This project may compile dependencies for clients
project( rocblas-clients LANGUAGES CXX Fortran )

set(rocblas_f90_source_clients
    include/rocblas_fortran.f90
)
set(rocblas_f90_source_clients_tensile
    include/rocblas_fortran_tensile.f90
)

if( BUILD_WITH_TENSILE )
  add_library(rocblas_fortran_client ${rocblas_f90_source_clients} ${rocblas_f90_source_clients_tensile})
else()
  add_library(rocblas_fortran_client ${rocblas_f90_source_clients})
endif()
add_dependencies(rocblas_fortran_client rocblas_fortran)
include_directories(${CMAKE_BINARY_DIR}/include)

if(OS_ID_rhel OR OS_ID_centos OR OS_ID_sles)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libgomp -pthread -stdlib=libstdc++ -std=c++14")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( NOT TARGET rocblas )
  find_package( rocblas REQUIRED CONFIG PATHS ${ROCM_PATH}/rocblas /opt/rocm/rocblas )
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} /opt/rocm )
find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} )

# Quietly look for CUDA, but if not found it's not an error
# The presense of hip is not sufficient to determine if we want a rocm or cuda backend
if( NOT IGNORE_CUDA )
  find_package( CUDA QUIET )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_BENCHMARKS )
  add_subdirectory( benchmarks )
endif( )

if( BUILD_CLIENTS_TESTS )
  add_subdirectory( gtest )
endif( )

set( ROCBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/rocblas_common.yaml")
add_custom_command( OUTPUT "${ROCBLAS_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_common.yaml "${ROCBLAS_COMMON}"
                    DEPENDS include/rocblas_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocblas_template.yaml")
add_custom_command( OUTPUT "${ROCBLAS_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_template.yaml "${ROCBLAS_TEMPLATE}"
                    DEPENDS include/rocblas_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/rocblas_gentest.py")
add_custom_command( OUTPUT "${ROCBLAS_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/rocblas_gentest.py "${ROCBLAS_GENTEST}"
                    DEPENDS common/rocblas_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

add_custom_target( rocblas-common DEPENDS "${ROCBLAS_COMMON}" "${ROCBLAS_TEMPLATE}" "${ROCBLAS_GENTEST}" )


# TODO: move to rocm-cmake
include(CMakeParseArguments)

function(rocm_create_package_clients)
    set(options)
    set(oneValueArgs LIB_NAME DESCRIPTION SECTION MAINTAINER VERSION)
    set(multiValueArgs DEPENDS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set( PACKAGE_REQUIREMENTS "${PARSE_LIB_NAME} (>=${PARSE_VERSION})")
    if(CPACK_SYSTEM_NAME MATCHES "ubuntu-18.04")
      set(PACKAGE_REQUIREMENTS "${PACKAGE_REQUIREMENTS}, libboost-program-options-dev (>=1.65.1)")
      set(PACKAGE_REQUIREMENTS "${PACKAGE_REQUIREMENTS}, libomp-dev (>=5.0.1)")
      set(PACKAGE_REQUIREMENTS "${PACKAGE_REQUIREMENTS}, gfortran (>=7.4.0)")
    elseif(CPACK_SYSTEM_NAME MATCHES "ubuntu-16.04")
      set(PACKAGE_REQUIREMENTS "${PACKAGE_REQUIREMENTS}, libboost-program-options-dev (>=1.58.0)")
      set(PACKAGE_REQUIREMENTS "${PACKAGE_REQUIREMENTS}, libomp-dev (>=3.7.0)")
      set(PACKAGE_REQUIREMENTS "${PACKAGE_REQUIREMENTS}, gfortran (>=4.8.5)")
    endif()
    message(${PACKAGE_REQUIREMENTS})

    string(CONCAT PACKAGE_NAME ${PARSE_LIB_NAME} "-clients-" ${PARSE_VERSION} "-" ${CPACK_SYSTEM_NAME} ".deb")
    string(CONCAT DEB_CONTROL_FILE_CONTENT "Package: " ${PARSE_LIB_NAME} "-clients"
                                           "\nVersion: " ${PARSE_VERSION}
                                           "\nSection: " ${PARSE_SECTION}
                                           "\nPriority: optional"
                                           "\nArchitecture: amd64"
                                           "\nMaintainer: " ${PARSE_MAINTAINER}
                                           "\nDescription: " ${PARSE_DESCRIPTION}
                                           "\nDepends: " ${PACKAGE_REQUIREMENTS} "\n\n")
    set(PACKAGE_DIR "${PROJECT_BINARY_DIR}/package")
    if(EXISTS "${PACKAGE_DIR}")
        file(REMOVE_RECURSE "${PACKAGE_DIR}")
    endif()
    set(PACKAGE_INSTALL_DIR "${PACKAGE_DIR}/${CPACK_PACKAGING_INSTALL_PREFIX}/${PARSE_LIB_NAME}")
    file(MAKE_DIRECTORY "${PACKAGE_INSTALL_DIR}/bin")
    ### AMD BLIS is not packaged right now, so we have to put the so files into /opt/rocm/lib temporarily
    ### BLIS will be packaged soon and this will not be needed
    set(BLIS_LIB_DIR "${PACKAGE_DIR}/${CPACK_PACKAGING_INSTALL_PREFIX}/lib")
    set(ROCBLAS_FORT_LIB_DIR "${PACKAGE_INSTALL_DIR}/lib")
    file(MAKE_DIRECTORY "${BLIS_LIB_DIR}")
    file(MAKE_DIRECTORY "${ROCBLAS_FORT_LIB_DIR}")
    file(WRITE "${PROJECT_BINARY_DIR}/package/DEBIAN/control" ${DEB_CONTROL_FILE_CONTENT})
    add_custom_target(package_clients
        COMMAND ${CMAKE_COMMAND} -E remove -f "${PACKAGE_INSTALL_DIR}/*"
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/../../deps/blis/lib/*" "${BLIS_LIB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/../clients/*rocblas_fortran_client*" "${ROCBLAS_FORT_LIB_DIR}"
        COMMAND ln -r -s "${ROCBLAS_FORT_LIB_DIR}/*rocblas_fortran_client*" "${PACKAGE_DIR}/${CPACK_PACKAGING_INSTALL_PREFIX}/lib/"
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/*" "${PACKAGE_INSTALL_DIR}/bin"
        COMMAND dpkg -b "${PROJECT_BINARY_DIR}/package/"  ${PACKAGE_NAME})
endfunction(rocm_create_package_clients)


if (BUILD_CLIENTS_SAMPLES OR  BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_SELFTEST OR BUILD_CLIENTS_RIDER)

rocm_create_package_clients(LIB_NAME rocblas
                            DESCRIPTION "Radeon Open Compute BLAS library"
                            MAINTAINER "rocblas-maintainer@amd.com>"
                            SECTION "dev"
                            VERSION ${rocblas_VERSION})
endif()

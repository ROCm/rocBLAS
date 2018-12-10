###############################################################################
# Copyright (C) 2018 Advanced Micro Devices, Inc.
################################################################################

# TODO: move this function to https://github.com/RadeonOpenCompute/rocm-cmake/blob/master/share/rocm/cmake/ROCMSetupVersion.cmake

macro(rocm_set_parent VAR)
  set(${VAR} ${ARGN} PARENT_SCOPE)
  set(${VAR} ${ARGN})
endmacro()

function(rocm_get_git_commit_id OUTPUT_VERSION)
  set(options)
  set(oneValueArgs VERSION DIRECTORY)
  set(multiValueArgs)

  cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(_version ${PARSE_VERSION})

  set(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  if(PARSE_DIRECTORY)
    set(DIRECTORY ${PARSE_DIRECTORY})
  endif()

  find_program(GIT NAMES git)

  if(GIT)
    set(GIT_COMMAND ${GIT} describe --dirty --long --match [0-9]*)
    execute_process(COMMAND ${GIT_COMMAND}
      WORKING_DIRECTORY ${DIRECTORY}
      OUTPUT_VARIABLE GIT_TAG_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE RESULT
      ERROR_QUIET)
    if(${RESULT} EQUAL 0)
      set(_version ${GIT_TAG_VERSION})
    else()
      execute_process(COMMAND ${GIT_COMMAND} --always
	WORKING_DIRECTORY ${DIRECTORY}
	OUTPUT_VARIABLE GIT_TAG_VERSION
	OUTPUT_STRIP_TRAILING_WHITESPACE
	RESULT_VARIABLE RESULT
	ERROR_QUIET)
      if(${RESULT} EQUAL 0)
	set(_version ${GIT_TAG_VERSION})
      endif()
    endif()
  endif()
  rocm_set_parent(${OUTPUT_VERSION} ${_version})
endfunction()

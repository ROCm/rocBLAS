# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
#
# ########################################################################

set(GRAPHING_SCRIPTS measurePerformance.py
    plotPerformance.py
    blasPerformanceTesting.py
    errorHandler.py
    performanceUtility.py
    README.txt
  )

foreach( SCRIPT ${GRAPHING_SCRIPTS} )
  #Copy a file to another location.
  configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${CMAKE_CURRENT_BINARY_DIR}/${SCRIPT} @ONLY )
endforeach( )

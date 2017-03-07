# ########################################################################
# Copyright 2016 Advanced Micro Devices, Inc.
#
# ########################################################################

set(GRAPHING_SCRIPTS 	measurePerformance.py 
						plotPerformance.py 
						blasPerformanceTesting.py 
						errorHandler.py 
						performanceUtility.py
						)

if( WIN32 )
		install( FILES ${GRAPHING_SCRIPTS} DESTINATION bin${SUFFIX_BIN} )
else ( )
		install( FILES ${GRAPHING_SCRIPTS} DESTINATION share/rocBLAS )
endif( )

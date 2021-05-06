# find_package(PythonInterp)
# # TODO: Check PYTHON_VERSION_MAJOR

find_program(VIRTUALENV_PYTHON_EXE python3)
if(NOT VIRTUALENV_PYTHON_EXE)
    find_program(VIRTUALENV_PYTHON_EXE python)
endif()
get_filename_component(VIRTUALENV_PYTHON_EXENAME ${VIRTUALENV_PYTHON_EXE} NAME CACHE)

set(VIRTUALENV_HOME_DIR ${CMAKE_BINARY_DIR}/virtualenv CACHE PATH "Path to virtual environment")

function(virtualenv_create)
    message("${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages --clear")
    execute_process(
      COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages --clear
    )

    if(WIN32)
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/Scripts CACHE PATH "Path to virtualenv bin directory")
    else()
        set(VIRTUALENV_BIN_DIR ${VIRTUALENV_HOME_DIR}/bin CACHE PATH "Path to virtualenv bin directory")
    endif()
endfunction()

function(virtualenv_install)
    virtualenv_create()

    message("${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install ${ARGN}")
    execute_process(
      RESULT_VARIABLE rc
      COMMAND ${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -m pip install ${ARGN}
    )
    if(rc)
        message(FATAL_ERROR ${rc})
    endif()
endfunction()

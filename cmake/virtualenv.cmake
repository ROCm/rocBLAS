# find_package(PythonInterp)
# # TODO: Check PYTHON_VERSION_MAJOR

find_program(VIRTUALENV_PYTHON_EXE python3)
if(NOT VIRTUALENV_PYTHON_EXE)
    find_program(VIRTUALENV_PYTHON_EXE python)
endif()

set(VIRTUALENV_HOME_DIR ${CMAKE_BINARY_DIR}/virtualenv CACHE PATH "Path to virtual environment")

function(virtualenv_create)
    execute_process(
        COMMAND ${VIRTUALENV_PYTHON_EXE} -m venv ${VIRTUALENV_HOME_DIR} --system-site-packages
    )
endfunction()

function(virtualenv_install)
    virtualenv_create()
    # TODO: Check result
    message("${VIRTUALENV_HOME_DIR}/bin/pip install ${ARGN}")
    execute_process( COMMAND ${VIRTUALENV_HOME_DIR}/bin/pip install ${ARGN} )
endfunction()

# find_package(PythonInterp)
# # TODO: Check PYTHON_VERSION_MAJOR

find_program(VIRTUALENV_PYTHON_EXE python)


set(VIRTUALENV_SOURCE_DIR ${CMAKE_BINARY_DIR}/virtualenv-source CACHE PATH "Path to virtualenv source")
set(VIRTUALENV_HOME_DIR ${CMAKE_BINARY_DIR}/virtualenv CACHE PATH "Path to virtual environment")

set(VIRTUALENV_VERSION 15.1.0)
function(virtualenv_create)
    if(NOT EXISTS ${VIRTUALENV_HOME_DIR}/bin/python)
        file(DOWNLOAD https://pypi.python.org/packages/d4/0c/9840c08189e030873387a73b90ada981885010dd9aea134d6de30cd24cb8/virtualenv-${VIRTUALENV_VERSION}.tar.gz
            ${VIRTUALENV_SOURCE_DIR}/virtualenv-${VIRTUALENV_VERSION}.tar.gz
            STATUS status LOG log
            EXPECTED_HASH MD5=44e19f4134906fe2d75124427dc9b716
        )

        list(GET status 0 status_code)
        list(GET status 1 status_string)

        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error: downloading
                'https://pypi.python.org/packages/source/v/virtualenv/virtualenv-${VIRTUALENV_VERSION}.tar.gz' failed
                status_code: ${status_code}
                status_string: ${status_string}
                log: ${log}
            ")
        endif()

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${VIRTUALENV_SOURCE_DIR}/virtualenv-${VIRTUALENV_VERSION}.tar.gz
            WORKING_DIRECTORY ${VIRTUALENV_SOURCE_DIR}
        )

        execute_process(
            COMMAND ${VIRTUALENV_PYTHON_EXE} ${VIRTUALENV_SOURCE_DIR}/virtualenv-${VIRTUALENV_VERSION}/virtualenv.py ${VIRTUALENV_HOME_DIR}
        )
    endif()
endfunction()

function(virtualenv_install)
    virtualenv_create()
    # TODO: Check result
    message("${VIRTUALENV_HOME_DIR}/pip install ${ARGN}")
    execute_process(
        COMMAND ${VIRTUALENV_HOME_DIR}/bin/python ${VIRTUALENV_HOME_DIR}/bin/pip install ${ARGN}
    )
endfunction()

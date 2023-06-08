// This file is for AMD Continuous Integration use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    String centos7 = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'
    String hipccCompileFlags = ""
    if (jobName.contains('hipclang'))
    {
        //default in the hipclang docker containers. May change later on
        hipccCompileFlags = "export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=2'"
    }
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        pullRequest.labels.each
        {
            if (it == "noTensile")
            {
                project.paths.build_command = project.paths.build_command.replaceAll(' -c', ' -cn')
            }
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${centos7}
                echo Original HIPCC_COMPILE_FLAGS_APPEND: \$HIPCC_COMPILE_FLAGS_APPEND
                ${hipccCompileFlags}
                CXX=/opt/rocm/bin/hipcc ${project.paths.build_command}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String installPackage = ""
    if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
    {
        installPackage = 'sudo rpm -i rocblas*.rpm'
    } else
    {
        installPackage = 'sudo dpkg -i rocblas*.deb'
    }

    String runTests = ""
    String testXMLPath = "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"

    String gtestArgs = ""
    String xnackVar = ""

    def hmmTestCommand= ''
    if (platform.jenkinsLabel.contains('gfx90a') && gfilter.contains('nightly'))
    {
        hmmTestCommand = """
                            HSA_XNACK=1 GTEST_LISTENER=NO_PASS_LINE_IN_LOG \$ROCBLAS_TEST --gtest_output=xml:test_detail_hmm.xml --gtest_color=yes --gtest_filter=*HMM*-*known_bug*
                         """
    }

//  For the trmm_outofplace tests the deprecated functions rocblas_Xtrmm_outofplace
//  are called by rocblas-test, and the new functions rocblas_Xtrmm are called by
//  rocblas_v3-test. Both call it with the arguments A, B, C.

//  This is done in testing_trmm_outofplace.hpp where rocblas.hpp maps the
//  string rocblas_trmm_outofplace_fn to the deprecated rocblas_Xtrmm_outofplace
//  for rocblas-test and to the new function rocblas_Xtrmm for rocblas_v3-test. Thus
//  rocblas-test tests the deprecated function and rocblas_v3-test tests the new function.

//  The below v3TestCommand calls rocblas_v3-test to verify the new functions
//  rocblas_Xtrmm with arguments A, B, C (the new in-place/outofplace API).
    def v3TestCommand= ''
    if (platform.jenkinsLabel.contains('gfx90a') && gfilter.contains('nightly'))
    {
        v3TestCommand = """
                            GTEST_LISTENER=NO_PASS_LINE_IN_LOG \$ROCBLAS_v3_TEST --gtest_output=xml:test_detail_V3.xml --gtest_color=yes --gtest_filter=*trmm_outofplace*quick*-*known_bug*
                         """
    }


    if (platform.jenkinsLabel.contains('ubuntu'))
    {
        runTests = """
                    pushd ${project.paths.project_build_prefix}
                    mv build build_BAK
                    ROCBLAS_TEST=/opt/rocm/bin/rocblas-test
                    ROCBLAS_v3_TEST=/opt/rocm/bin/rocblas_v3-test
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${hmmTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${v3TestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    mv build_BAK build
                    popd
                   """
        testXMLPath = "${project.paths.project_build_prefix}/test_detail*.xml"
    } else
    {
        runTests = """
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ROCBLAS_TEST=./rocblas-test
                    ROCBLAS_v3_TEST=./rocblas_v3-test
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${hmmTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${v3TestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                   """
    }


    def command = """#!/usr/bin/env bash
                    set -x
                    pushd ${project.paths.project_build_prefix}/build/release/package
                    ${installPackage}
                    popd
                    ${runTests}
                  """

    platform.runCommand(this, command)
    junit testXMLPath
}

def runPackageCommand(platform, project)
{
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")
        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
        def cleanCommand = """#!/usr/bin/env bash
                                set -x
                                cd ${project.paths.project_build_prefix}/build/
                                find -name '*.o' -delete
                                find -type d -name '*build_tmp*' -exec rm -rf {} +
                                find -type d -name '*_CPack_Packages*' -exec rm -rf {} +
                           """
        platform.runCommand(this, cleanCommand)
}

return this

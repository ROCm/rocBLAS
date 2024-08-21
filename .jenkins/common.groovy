// This file is for AMD Continuous Integration use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.


def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    String centos7 = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'
    String dynamicBuildCommand = project.paths.build_command
    String dynamicOptions = ""

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("noTensile"))
        {
            dynamicBuildCommand = dynamicBuildCommand + ' -n'
        }

        // in PR if we are targeting develop branch build ONLY what CI pipeline will test, unless bug label
        if (env.CHANGE_TARGET == "develop" && !pullRequest.labels.contains("bug"))
        {
            // requires at command execution time ${auxiliary.gfxTargetParser()} to set gfx_var variable
            dynamicOptions = dynamicOptions + ' -a \$gfx_arch'
        }

        if (env.CHANGE_TARGET == "develop" && pullRequest.labels.contains("ci:static-libraries"))
        {
            // test PR as static pipeline may be infrequent
            dynamicOptions = dynamicOptions + ' --static'
        }
    }
    // these 908 nodes have too few CPU cores to build full fat library (temporary workaround)
    // contains question to remove after testing PR before merging!!!
    else if (env.BRANCH_NAME ==~ /develop/ && platform.jenkinsLabel.contains('gfx908'))
    {
        // mimimal fat binary
        dynamicOptions = dynamicOptions + ' -a "gfx908;gfx1101"'
    }

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${centos7}
                ${auxiliary.gfxTargetParser()}
                ${dynamicBuildCommand} ${dynamicOptions}
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
    }
    else
    {
        installPackage = 'sudo dpkg -i rocblas*.deb'
    }

    String runTests = ""
    String testXMLPath = "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"

    String gtestArgs = ""
    String xnackVar = ""

    String gtestCommonEnv = "ROCBLAS_CLIENT_RAM_GB_LIMIT=95"
    String checkNumericsEnv = "ROCBLAS_CHECK_NUMERICS=6" // report status 4 & log 2 on fail
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("help wanted"))
        {
            gtestCommonEnv += " GTEST_LISTENER=PASS_LINE_IN_LOG"
        }
    }

    def hmmTestCommand= ''
    if (platform.jenkinsLabel.contains('gfx90a') && gfilter.contains('nightly'))
    {
        hmmTestCommand = """
                            ${gtestCommonEnv} HSA_XNACK=1 \$ROCBLAS_TEST --gtest_output=xml:test_detail_hmm.xml --gtest_color=yes --gtest_filter=*HMM*-*known_bug*
                         """
    }

    def rocBLASTestCommand = ''
    def checkNumericsTestCommand= ''
    if (project.buildName.contains('weekly'))
    {
            rocBLASTestCommand = """
                                    ${gtestCommonEnv} \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                                 """

            // Enable check numerics only for checkNumericsTestCommand
            checkNumericsTestCommand = """
                                    ${gtestCommonEnv} ${checkNumericsEnv} \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=*blas1/pre_checkin*:*blas2/pre_checkin*:*blas3/pre_checkin*:*blas3_tensile/pre_checkin*:*blas_ex/pre_checkin*:-*known_bug*:*repeatability_check*:*get_solutions*
                                 """
    }
    else
    {
            rocBLASTestCommand = """
                                    ${gtestCommonEnv} \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                                 """
    }

    if (platform.jenkinsLabel.contains('ubuntu'))
    {
        runTests = """
                    pushd ${project.paths.project_build_prefix}
                    mv build build_BAK
                    ROCBLAS_TEST=/opt/rocm/bin/rocblas-test
                    ${rocBLASTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${hmmTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${checkNumericsTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    mv build_BAK build
                    popd
                   """
        testXMLPath = "${project.paths.project_build_prefix}/test_detail*.xml"
    }
    else
    {
        runTests = """
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ROCBLAS_TEST=./rocblas-test
                    ${rocBLASTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${hmmTestCommand}
                    if (( \$? != 0 )); then
                        exit 1
                    fi
                    ${checkNumericsTestCommand}
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
}

def runPackageCommand(platform, project, boolean debug=false)
{
    String buildTypeDir = debug ? 'debug' : 'release'
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/${buildTypeDir}")
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
    def cleanCommand = """#!/usr/bin/env bash
                            set -x
                            cd ${project.paths.project_build_prefix}/build/
                            find -name '*.o.d' -delete
                            find -name '*.o' -delete
                            find -type d -name '*build_tmp*' -exec rm -rf {} +
                            find -type d -name '*_CPack_Packages*' -exec rm -rf {} +
                        """
    platform.runCommand(this, cleanCommand)
}

return this

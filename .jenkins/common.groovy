// This file is for AMD Continuous Integration use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.


def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    String centos7 = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'
    String hipccCompileFlags = ""
    String dynamicBuildCommand = project.paths.build_command
    String dynamicOptions = ""

    if (jobName.contains('hipclang'))
    {
        //default in the hipclang docker containers. May change later on
        hipccCompileFlags = "export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=2'"
    }
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
            dynamicOptions = ' -a \$gfx_arch'
        }
    }

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${centos7}
                echo Original HIPCC_COMPILE_FLAGS_APPEND: \$HIPCC_COMPILE_FLAGS_APPEND
                ${hipccCompileFlags}
                ${auxiliary.gfxTargetParser()}
                CXX=/opt/rocm/bin/hipcc ${dynamicBuildCommand} ${dynamicOptions}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, settings)
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

    String gtestCommonEnv = "ROCBLAS_CLIENT_RAM_GB_LIMIT=95"
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("help wanted"))
        {
            gtestCommonEnv += " GTEST_LISTENER=PASS_LINE_IN_LOG"
        }
    }

    def hmmTestCommand= ''
    if (platform.jenkinsLabel.contains('gfx90a') && settings.gfilter.contains('nightly'))
    {
        hmmTestCommand = """
                            ${gtestCommonEnv} HSA_XNACK=1 \$ROCBLAS_TEST --gtest_output=xml:test_detail_hmm.xml --gtest_color=yes --gtest_filter=*HMM*-*known_bug*
                         """
    }

    def rocBLASTestCommand = ''

    // Enable check numerics only for weekly tests
    if (project.buildName.contains('weekly'))
    {
            rocBLASTestCommand = """
                                    ${gtestCommonEnv} ROCBLAS_CHECK_NUMERICS=4 \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=${settings.gfilter}-*known_bug*
                                 """
    }
    else
    {
            rocBLASTestCommand = """
                                    ${gtestCommonEnv} \$ROCBLAS_TEST --gtest_output=xml --gtest_color=yes --gtest_filter=${settings.gfilter}-*known_bug*
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
                   """
    }

    def LD_PATH = ''
    if (settings.addressSanitizer)
    {
        LD_PATH = """
                    export ASAN_LIB_PATH=\$(/opt/rocm/llvm/bin/clang -print-file-name=libclang_rt.asan-x86_64.so)
                    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(dirname "\${ASAN_LIB_PATH}")
                  """
    }

    def command = """#!/usr/bin/env bash
                    set -x
                    pushd ${project.paths.project_build_prefix}/build/release/package
                    ${installPackage}
                    popd
                    ${LD_PATH}
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

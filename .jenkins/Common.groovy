// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project)
{
    project.paths.construct_build_prefix()

    String compiler = platform.jenkinsLabel.contains('hip-clang') ? 'hipcc' : 'hcc'
    String hipClang = platform.jenkinsLabel.contains('hip-clang') ? '--hip-clang' : ''
    String sles = platform.jenkinsLabel.contains('sles') ? '/usr/bin/sudo --preserve-env' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${sles} LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/${compiler} ${project.paths.build_command} ${hipClang}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release",true)
        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this


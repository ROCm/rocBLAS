#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

import java.nio.file.Path;

rocBLASCI:
{

    def rocblas = new rocProject('rocBLAS')
    // customize for project
    rocblas.paths.build_command = './install.sh -lasm_ci -c'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['ubuntu && gfx900', 'centos7 && gfx900', 'centos7 && gfx906', 'sles && gfx906'], rocblas)

    boolean formatCheck = true

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()

        def command

        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                    """
        }
        else if(platform.jenkinsLabel.contains('sles'))
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc sudo ${project.paths.build_command}
                    """
        }

        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command}
                    """
        }
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        String sudo = auxiliary.sudo(platform.jenkinsLabel)
        def gfilter = auxiliary.isJobStartedByTimer() ? "*nightly*" : "*quick*:*pre_checkin*"

        def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release/clients/staging
                        ${sudo} LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                    """

        platform.runCommand(this, command)
        junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
    }

    def packageCommand =
    {
        platform, project->
        String sudo = auxiliary.sudo(platform.jenkinsLabel)

        def command

        if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    ${sudo} make package
                    ${sudo} mkdir -p package
                    ${sudo} mv *.rpm package/
                    ${sudo} rpm -qlp package/*.rpm
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
        }
        else if(platform.jenkinsLabel.contains('hip-clang') || platform.jenkinsLabel.contains('sles'))
        {
            packageCommand = null
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make package
                    make package_clients
                    mkdir -p package
                    mv *.deb package/
                    mv clients/*.deb package/
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
        }
    }

    buildProject(rocblas, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}

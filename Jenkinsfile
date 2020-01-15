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
    rocblas.paths.build_command = './install.sh -lasm_ci -c -oV3'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900 && centos7 && hip-clang', 'gfx906 && centos7 && hip-clang', 'gfx900 && ubuntu && hip-clang',
				'gfx906 && ubuntu && hip-clang', 'gfx908 && ubuntu && hip-clang'], rocblas)

    boolean formatCheck = true

    rocblas.timeout.compiler = 300
    rocblas.timeout.test = 600

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
	def prepend = platform.jenkinsLabel.contains('centos') ? "/opt/rh/devtoolset-7/root/usr/bin:/opt/rocm/bin" : "/opt/rocm/bin"
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    export PATH=${prepend}:$PATH
                    LD_LIBRARY_PATH=/opt/rocm/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                    """
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
                        ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                    """

        platform.runCommand(this, command)
        junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
    }

    def packageCommand =
    {
        platform, project->

        def command = """
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

    buildProject(rocblas, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

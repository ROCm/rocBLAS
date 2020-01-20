#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

properties(auxiliary.setProperties())

rocBLASCI:
{

    def rocblas = new rocProject('rocBLAS', "HipClang")

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx900 && centos7 && hip-clang', 'gfx906 && ubuntu && hip-clang'], rocblas)

    rocblas.paths.build_command = './install.sh -lasm_ci -c'
    rocblas.timeout.compile = 300
    rocblas.timeout.test = 600

    boolean formatCheck = true

    def commonGroovy

    def compileCommand =
    {
        platform, project->
        
        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/Common.groovy"
        commonGroovy.runCompileCommand(platform, project)
    }

    def testCommand =
    {
        platform, project->
        
        def gfilter = "*quick*:*pre_checkin*"
        commonGroovy.runTestCommand(platform, project, gfilter)
    }

    def packageCommand =
    {
        platform, project->
        
        commonGroovy.runPackageCommand(platform, project)
    }

    buildProject(rocblas, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

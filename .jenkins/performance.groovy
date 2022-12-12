#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean buildStatic=false)
{
    def reference = (env.BRANCH_NAME ==~ /PR-\d+/) ? 'develop' : 'master'

    project.paths.construct_build_prefix()
    dir("${project.paths.project_build_prefix}/ref-repo") {
        git branch: "${reference}", url: 'https://github.com/ROCmSoftwarePlatform/rocBLAS-internal.git', credentialsId: 'ab8d4444-4620-4189-a9ce-c16035c854dd'
    }

    String centos7 = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'
    String hipccCompileFlags = ""
    if (jobName.contains('hipclang'))
    {
        //default in the hipclang docker containers. May change later on
        hipccCompileFlags = "export HIPCC_COMPILE_FLAGS_APPEND='-O3 -Wno-format-nonliteral -parallel-jobs=2'"
    }
    String get_arch = ""
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        pullRequest.labels.each
        {
            if (it == "noTensile")
            {
                project.paths.build_command = project.paths.build_command.replaceAll(' -c', ' -cn')
            }
        }
        get_arch = auxiliary.gfxTargetParser()
        project.paths.build_command += "a \$gfx_arch"
    }

    def command = """#!/usr/bin/env bash
                set -x
                set -e
                cd ${project.paths.project_build_prefix}
                ${centos7}
                ${get_arch}
                echo Original HIPCC_COMPILE_FLAGS_APPEND: \$HIPCC_COMPILE_FLAGS_APPEND
                ${hipccCompileFlags}
                CXX=/opt/rocm/bin/hipcc ${project.paths.build_command}
                cd ref-repo
                CXX=/opt/rocm/bin/hipcc ${project.paths.build_command}
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, boolean debug=false)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String testBinaryName = debug ? 'rocblas-test-d' : 'rocblas-test'
    String directory = debug ? 'debug' : 'release'

    withCredentials([gitUsernamePassword(credentialsId: 'ab8d4444-4620-4189-a9ce-c16035c854dd', gitToolName: 'git-tool')])
    {
        platform.runCommand(
            this,
            """
            cd ${project.paths.build_prefix}
            git clone https://github.com/ROCmSoftwarePlatform/rocPTS.git -b release/rocpts-rel-1.0
            cd rocPTS
            python3 -m pip install build
            python3 -m build
            python3 -m pip install .
            """
        )
    }

    writeFile(
        file: project.paths.project_build_prefix + "/record_pts.py",
        text: libraryResource("com/amd/scripts/record_pts.py")
    )

    def setupBranch = env.CHANGE_ID ? 'git checkout -b $BRANCH_NAME' : 'git checkout $BRANCH_NAME'
    def command = """#!/usr/bin/env bash
                set -x
                pwd
                cd ${project.paths.project_build_prefix}
                ${setupBranch}
                python3 scripts/performance/pts/write_pts_report.py build/release/clients/staging/rocblas-bench rocBLAS_PTS_Benchmarks/ build-new scripts/performance/pts/benchmarks/gemv_problems.yaml scripts/performance/pts/benchmarks/axpy_problems.yaml scripts/performance/pts/benchmarks/gemm_problems.yaml scripts/performance/pts/benchmarks/trsm_problems.yaml scripts/performance/pts/benchmarks/symv_problems.yaml
                python3 scripts/performance/pts/write_pts_report.py ref-repo/build/release/clients/staging/rocblas-bench rocBLAS_PTS_Benchmarks/ build-reference scripts/performance/pts/benchmarks/gemv_problems.yaml scripts/performance/pts/benchmarks/axpy_problems.yaml scripts/performance/pts/benchmarks/gemm_problems.yaml scripts/performance/pts/benchmarks/trsm_problems.yaml scripts/performance/pts/benchmarks/symv_problems.yaml
                for dataset in ./rocBLAS_PTS_Benchmarks/*/;
                    do python3 ./record_pts.py --dataset-path \$dataset --reference-dataset build-reference --new-dataset build-new -l pts_rocblas_benchmark_data
                done
            """
    withCredentials([usernamePassword(credentialsId: 'PTS_API_ID_KEY_PROD', usernameVariable: 'PTS_API_ID', passwordVariable: 'PTS_API_KEY')])
    {
        platform.runCommand(this, command)
    }
}

def runCI =
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocBLAS-internal', 'Performance')

    prj.paths.build_command = './install.sh -c'
    prj.defaults.ccache = true
    prj.timeout.compile = 600
    prj.timeout.test = 600
    prj.libraryDependencies = ['rocBLAS-internal']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy
    def gpus = []

    def compileCommand =
    {
        platform, project->

        gpus.add(platform.gpu)
        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        runCompileCommand(platform, project, jobName)
    }

    def testCommand =
    {
        platform, project->

        runTestCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
    // def commentString = "Performance reports: \n" + "Commit hashes: \n"
    // for(parentHash in prj.gitParentHashes) {
    //      commentString += "${parentHash} \n"
    // }
    // for (gpu in gpus) {
    //     commentString += "[${gpu} report](${JOB_URL}/${dataType}-precision-${gpu})\n"
    // }
    // boolean commentExists = false
    // for (prComment in pullRequest.comments) {
    //     if (prComment.body.contains("Performance reports:"))
    //     {
    //         commentExists = true
    //         prComment.body = commentString
    //     }
    // }
    // if (!commentExists) {
    //     def comment = pullRequest.comment(commentString)
    // }
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900','gfx906']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([ubuntu18:['gfx906']], urlJobName)
        }
    }
}

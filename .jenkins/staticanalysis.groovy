#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for AMD Continuous Integration use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCI =
{
    nodeDetails, jobName->

    def settings = [formatCheck: true,
                    addressSanitizer: false,
                    gfilter: "*quick*:*pre_checkin*"]

    def prj  = new rocProject('rocBLAS', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean staticAnalysis = true

    buildProject(prj, settings.formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
    stage(urlJobName) {
        runCI([ubuntu20:['any']], urlJobName)
    }
}

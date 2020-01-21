#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@perftest') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

properties(auxiliary.setProperties())

rocBLASCI:
{

    def rocblas = new rocProject('rocBLAS', 'PerformanceTest')
    // customize for project
    rocblas.paths.build_command = './install.sh -lasm_ci -c'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['ubuntu && gfx803 && perf', 'ubuntu && gfx900 && perf'], rocblas)

    boolean formatCheck = true

    def commonGroovy

    rocblas.timeout.test = 600

    def compileCommand =
    {
        platform, project->

        // Print out available environment variables
        echo sh(script: 'env|sort', returnStdout: true)

        String gpuLabel = project.email.gpuLabel(platform.jenkinsLabel)
        def command = """#!/usr/bin/env bash
                        set -x
                        pwd
                        cd ${project.paths.project_build_prefix}
                        workingdir=`pwd`

                        shopt expand_aliases
                        shopt -s expand_aliases
                        shopt expand_aliases

                        python -V
                        alias python=python3
                        python -V

                        # wget http://10.216.151.18:8080/job/Performance/job/${project.name}/job/develop/lastSuccessfulBuild/artifact/*zip*/archive.zip
                        wget http://10.216.151.18:8080/job/Performance/job/rocBLAS/job/PR-895/lastSuccessfulBuild/artifact/*zip*/archive.zip
                        wgetreturn=\$?
                        if [[ \$wgetreturn -eq 8 ]]; then
                            echo "Download error"
                        else
                            unzip -o archive.zip
                            tar -xvf archive/*/*/perfoutput_${gpuLabel}.tar
                            mv perfoutput perfoutput_${gpuLabel} 
                        fi

                        tar -cvf perfoutput_${gpuLabel}.tar perfoutput_${gpuLabel}

                        if [[ -z "${env.CHANGE_ID}" ]]
                        then
                            echo "This is not a pull request"
                        else
                            echo "This is a pull request"
                        fi
                    """
        platform.runCommand(this, command)

        // commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/Common.groovy"
        // commonGroovy.runCompileCommand(platform, project)
    }

    def testCommand =
    {
        platform, project->
        
        // String gpuLabel = project.email.gpuLabel(platform.jenkinsLabel)
        // def command = """#!/usr/bin/env bash
        //                 set -x
        //                 pwd
        //                 cd ${project.paths.project_build_prefix}
        //                 workingdir=`pwd`

        //                 shopt expand_aliases
        //                 shopt -s expand_aliases
        //                 shopt expand_aliases

        //                 python -V
        //                 alias python=python3
        //                 python -V

        //                 # get device name from /dev/dri
        //                 devicename=\$(echo \$(ls /dev/dri) | sed 's/.*\\(card[0-9]\\).*/\\1/')
        //                 echo \$devicename
        //                 # get device num from device name
        //                 devicenum=\$(echo \$devicename | sed 's/.*\\([0-9]\\).*/\\1/')
        //                 echo \$devicenum
        //                 echo ${gpuLabel}
        //                 export PATH=/opt/asy/bin:${PATH}
        //                 pushd scripts/performance/blas/
        //                 python alltime.py -A \$workingdir/build/release/clients/staging -o \$workingdir/perfoutput -i perf.yaml -S 0 -g 0 -d \$devicenum
        //                 echo "Uploading Data..."
        //                 python uploadData.py -a ${gpuLabel} -f \$workingdir/perfoutput
        //                 popd

        //                 ls \$workingdir/perfoutput
        //                 cat \$workingdir/perfoutput/specs.txt
                        
        //                 # wget http://10.216.151.18:8080/job/Performance/job/${project.name}/job/develop/lastSuccessfulBuild/artifact/*zip*/archive.zip
        //                 wget http://10.216.151.18:8080/job/Performance/job/rocBLAS/job/PR-895/lastSuccessfulBuild/artifact/*zip*/archive.zip
        //                 wgetreturn=\$?
        //                 pushd scripts/performance/blas/
        //                 if [[ \$wgetreturn -eq 8 ]]; then
        //                     echo "Download error"
        //                     python alltime.py -T -o \$workingdir/perfoutput -S 0 -g 1 -i perf.yaml -d \$devicenum
        //                 else
        //                     unzip -o archive.zip
        //                     tar -xvf archive/*/*/perfoutput_${gpuLabel}.tar
        //                     mv perfoutput perfoutput_${gpuLabel}
        //                     python alltime.py -T -o \$workingdir/perfoutput -b \$workingdir/perfoutput_${gpuLabel} -g 1 -d \$devicenum -i perf.yaml
        //                 fi
                        
        //                 popd

        //                 tar -cvf perfoutput_${gpuLabel}.tar perfoutput

        //                 if [[ -z "${env.CHANGE_ID}" ]]
        //                 then
        //                     echo "This is not a pull request"
        //                 else
        //                     echo "This is a pull request"
        //                 fi
        //             """
        // platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->
        
        String gpuLabel = project.email.gpuLabel(platform.jenkinsLabel)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/perfoutput_${gpuLabel}.tar""")
    }

    buildProject(rocblas, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}

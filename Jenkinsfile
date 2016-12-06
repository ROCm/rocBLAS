#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
  disableConcurrentBuilds(),
  pipelineTriggers([cron('H/30 * * * *')])])

currentBuild.result = "SUCCESS"

// Currently, YADP (yet-another-docker-plugin v0.1.0-rc30) does not load balance between clouds with the same label
// They recommend to use docker swarm, but not yet work with docker 1.12 'swarm mode'
// Manually load balance by picking a particular machine
node('rocm-1.3 && hawaii')
{
  def node_list = env.NODE_LABELS.tokenize()
  // sh "echo node_list: ${node_list}"

  def scm_dir = pwd()
  def build_dir_debug = "${scm_dir}/../build/debug"
  def build_dir_release = "${scm_dir}/../build/release"

  // Run the containers preperation script
  // Note, exported environment variables do not live outside of sh step
  sh ". /home/jenkins/prep-env.sh"

  // The following try block performs build steps
  try
  {
    dir("${scm_dir}") {
      stage("Clone") {
        checkout scm
      }
    }

    withEnv(["PATH=${PATH}:/opt/rocm/bin"]) {

      // Record important versions of software layers we use
      sh '''clang++ --version
            cmake --version
            hcc --version
            hipconfig --version
      '''

      //Jenkins plugin that adds color terminal support to output 'Console Output'; requires bash shell
      wrap([$class: 'AnsiColorBuildWrapper', 'colorMapName': 'XTerm'])
      {

        dir("${build_dir_release}")
        {
          stage("configure clang release") {
              sh """#!/usr/bin/env bash
                cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_BUILD_TYPE=Release \
                -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_WITH_TENSILE=ON \
                -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang ${scm_dir}
                """
          }

          stage("Build")
          {
              if (env.NODE_LABELS ==~ /.*fiji.*/)
              {
              sh 'echo Target Fiji ISA'
                withEnv(['HCC_AMDGPU_TARGET=AMD:AMDGPU:8:0:3'])
                {
                  sh '''#!/usr/bin/env bash
                        make -j 8
                    '''
                }
              }
              else if (env.NODE_LABELS ==~ /.*hawaii.*/)
              {
                sh 'echo Target Hawaii ISA'
                withEnv(['HCC_AMDGPU_TARGET=AMD:AMDGPU:7:0:1'])
                {
                    sh '''#!/usr/bin/env bash
                          make -j 8
                      '''
                }
              }
          }

          stage("Package Debian") {
            sh 'cd library-build; make package'
            archive includes: 'library-build/*.deb'
          }

          // Cap the maximum amount of testing to be a few hours; assume failure if the time limit is hit
          timeout(time: 2, unit: 'HOURS')
          {
            stage("unit tests") {
              // To trim test time, only execute single digit tests
              sh '''#!/usr/bin/env bash
                    cd clients-build/tests-build/staging
                    ./rocblas-test --gtest_output=xml --gtest_filter=*/?
                '''
              junit 'clients-build/tests-build/staging/*.xml'
            }

            stage("samples")
            {
              sh "cd clients-build/samples-build; ./example-sscal"
            }
          }

        }

      }
    }
  }
  catch( err )
  {
      currentBuild.result = "FAILURE"

      def email_list = emailextrecipients([
              [$class: 'CulpritsRecipientProvider']
      ])

      // CulpritsRecipientProvider below doesn't help, because nobody uses their real email address
      // emailext  to: "kent.knox@amd.com", recipientProviders: [[$class: 'CulpritsRecipientProvider']],
      //       subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
      //       body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

      // Disable email for now
      mail  to: "kent.knox@amd.com, david.tanner@amd.com, tingxing.dong@amd.com",
            subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
            body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

      throw err
  }
}

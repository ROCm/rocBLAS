#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

// def build_type="Debug"
// def build_type_postfix="-d"
def build_type="Release"
def build_type_postfix=""

// Currently, YADP (yet-another-docker-plugin v0.1.0-rc30) does not load balance between clouds with the same label
// They recommend to use docker swarm, but not yet work with docker 1.12 'swarm mode'
// Manually load balance by picking a particular machine
node('rocm-1.6 && fiji')
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
      stage("Clone")
      {
        checkout scm

        if( fileExists( 'cmake/build-version.cmake' ) )
        {
          def cmake_version_file = readFile( 'cmake/build-version.cmake' ).trim()
          //echo "cmake_version_file:\n${cmake_version_file}"

          cmake_version_file = cmake_version_file.replaceAll(/(\d+\.)(\d+\.)(\d+\.)\d+/, "\$1\$2\$3${env.BUILD_ID}")
          cmake_version_file = cmake_version_file.replaceAll(/VERSION_TWEAK\s+\d+/, "VERSION_TWEAK ${env.BUILD_ID}")
          //echo "cmake_version_file:\n${cmake_version_file}"
          writeFile( file: 'cmake/build-version.cmake', text: cmake_version_file )
        }
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
                cmake -DCMAKE_CXX_COMPILER=clang++-3.8 -DCMAKE_C_COMPILER=clang-3.8 -DCMAKE_BUILD_TYPE=${build_type} -DCMAKE_PREFIX_PATH=/opt/boost/clang -DBUILD_SHARED_LIBS=ON -DBUILD_LIBRARY=ON -DBUILD_WITH_TENSILE=ON \
                -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DCPACK_PACKAGING_INSTALL_PREFIX=/opt/rocm/rocblas ${scm_dir}
                """
          }

          stage("Build")
          {
              sh '''#!/usr/bin/env bash
                    make -j $(nproc)
                '''
          }

          stage("Package Debian") {
            sh 'cd library-build; make package'
            archiveArtifacts artifacts: 'library-build/*.deb', fingerprint: true
            archiveArtifacts artifacts: 'library-build/*.rpm', fingerprint: true
            sh "sudo dpkg -c library-build/*.deb"
         }

          // Cap the maximum amount of testing to be a few hours; assume failure if the time limit is hit
          timeout(time: 1, unit: 'HOURS')
          {
            stage("unit tests") {
              sh """#!/usr/bin/env bash
                    cd clients-build/tests-build/staging
                    ./rocblas-test${build_type_postfix} --gtest_output=xml
                """
              junit 'clients-build/tests-build/staging/*.xml'
            }

            stage("samples")
            {
              sh "cd clients-build/samples-build; ./example-sscal${build_type_postfix}"
            }
          }

        }

      }
    }
  }
  catch( err )
  {
      def email_list = emailextrecipients([
              [$class: 'CulpritsRecipientProvider']
      ])

      // CulpritsRecipientProvider below doesn't help, because nobody uses their real email address
      // emailext  to: "kent.knox@amd.com", recipientProviders: [[$class: 'CulpritsRecipientProvider']],
      //       subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
      //       body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

      // Disable email for now
      mail  to: "kent.knox@amd.com, david.tanner@amd.com, tingxing.dong@amd.com, andrew.chapman@amd.com",
            subject: "${env.JOB_NAME} finished with FAILUREs",
            body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

      throw err
  }
}

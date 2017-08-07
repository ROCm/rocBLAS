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
node('rocm-1.6')
{
  def node_list = env.NODE_LABELS.tokenize()
  // sh "echo node_list: ${node_list}"

  def scm_dir = pwd()
  def build_dir_debug = "${scm_dir}/../build/debug"
  def build_dir_release = "${scm_dir}/../build/release"

  // The following try block performs build steps
  try
  {
    dir("${scm_dir}") {
      stage("Clone")
      {
        checkout scm

        if( fileExists( 'CMakeLists.txt' ) )
        {
          def cmake_version_file = readFile( 'CMakeLists.txt' ).trim()
          //echo "cmake_version_file:\n${cmake_version_file}"

          cmake_version_file = cmake_version_file.replaceAll(/(\d+\.)(\d+\.)(\d+\.)\d+/, "\$1\$2\$3${env.BUILD_ID}")
          //echo "cmake_version_file:\n${cmake_version_file}"
          writeFile( file: 'CMakeLists.txt', text: cmake_version_file )
        }
      }
    }


    withEnv(["PATH=${PATH}:/opt/rocm/bin"]) {

      // Record important versions of software layers we use
      sh '''cmake --version
            hcc --version
            hipconfig --version
         '''

      dir("${build_dir_release}")
      {
        stage("configure clang release") {
            sh """#!/usr/bin/env bash
                  mkdir -p deps; cd deps
                  cmake -DBUILD_BOOST=OFF ${scm_dir}/deps
                  sudo make -j \$(nproc) install

                  cd ${build_dir_release}
                  CXX=/opt/rocm/bin/hcc cmake -DCMAKE_INSTALL_PREFIX=package -DCMAKE_PREFIX_PATH=/opt/boost/gcc \
                  -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON ${scm_dir}
                """
        }

        stage("Build")
        {
            sh '''#!/usr/bin/env bash
                  make -j $(nproc)
              '''
        }

        stage("Package Debian") {
          sh 'make package'
          archiveArtifacts artifacts: '*.deb', fingerprint: true
          archiveArtifacts artifacts: '*.rpm', fingerprint: true
          sh "sudo dpkg -c *.deb"
        }

        // Cap the maximum amount of testing to be a few hours; assume failure if the time limit is hit
        timeout(time: 1, unit: 'HOURS')
        {
          stage("unit tests") {
            sh """#!/usr/bin/env bash
                  cd clients/staging
                  ./rocblas-test${build_type_postfix} --gtest_output=xml
              """
            junit 'clients/staging/*.xml'
          }

          stage("samples")
          {
            sh "cd clients/staging; ./example-sscal${build_type_postfix}"
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
      // mail  to: "kent.knox@amd.com, david.tanner@amd.com, tingxing.dong@amd.com, andrew.chapman@amd.com",
      //       subject: "${env.JOB_NAME} finished with FAILUREs",
      //       body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

      throw err
  }
}

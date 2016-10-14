#!/usr/bin/env groovy

node('rocm')
{
  //     sh 'env | sort'
    def scm_dir = pwd()
    def build_dir_debug = "${scm_dir}/../build/debug"
    def build_dir_release = "${scm_dir}/../build/release"

    // // print versions of hcc and hip
    // sh "/opt/rocm/hcc-lc/bin/hcc --version; /opt/rocm/hip/bin/hipconfig"

    // Prepare build environment and set up tools
    sh ". /home/jenkins/prep-env.sh"

    // Record important versions of software layers we use
    sh '''clang++ --version
          cmake --version
          /opt/rocm/bin/hcc --version
          /opt/rocm/bin/hipconfig --version
    '''

    // The following try block performs build steps
    currentBuild.result = "SUCCESS"
    try
    {
      dir("${scm_dir}") {
        stage("Clone") {
          checkout scm
        }
      }

      dir("${build_dir_release}") {
        stage("configure clang release") {
          withEnv(['CXXFLAGS=-I /usr/include/c++/4.8 -I /usr/include/x86_64-linux-gnu/c++/4.8  -I /usr/include/x86_64-linux-gnu', 'HIP_PATH=/opt/rocm/hip']) {
            sh "cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang ${scm_dir}"
          }
        }

        stage("Build") {
          withEnv(['HCC_AMDGPU_TARGET=AMD:AMDGPU:7:0:1,AMD:AMDGPU:8:0:3']) {
            sh 'make -j 8'
          }
        }

        stage("Package Debian") {
          sh 'cd rocblas-build; make package'
          archive includes: 'library-build/*.deb'
        }

        stage("unit tests") {
          sh '''
              cd clients-build/tests-build/staging
              ./rocblas-test-d --gtest_output=xml
          '''
          junit 'clients-build/tests-build/staging/*.xml'
        }

        stage("samples") {
          sh "cd clients-build/samples-build; make; ./example_sscal"
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
        // mail  to: "kent.knox@amd.com, david.tanner@amd.com, tingxing.dong@amd.com",
        //       subject: "${env.JOB_NAME} finished with ${currentBuild.result}",
        //       body: "Node: ${env.NODE_NAME}\nSee ${env.BUILD_URL}\n\n" + err.toString()

        throw err
    }
}

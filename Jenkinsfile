#!/usr/bin/env groovy

currentBuild.result = "SUCCESS"
node('rocm-1.3 && fiji')
{
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

      dir("${build_dir_release}") {
        stage("configure clang release") {
          // withEnv(['CXXFLAGS=-I /usr/include/c++/4.8 -I /usr/include/x86_64-linux-gnu/c++/4.8  -I /usr/include/x86_64-linux-gnu', 'HIP_PATH=/opt/rocm/hip']) {
          // --amdgpu-target=AMD:AMDGPU:8:0:3
            sh "cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_LIBRARY=ON -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_WITH_TENSILE=ON -DHIP_ROOT=/opt/rocm/hip -DBOOST_ROOT=/opt/boost/clang ${scm_dir}"
          // }
        }

        stage("Build") {
          // withEnv(['HCC_AMDGPU_TARGET=AMD:AMDGPU:7:0:1,AMD:AMDGPU:8:0:3']) {
            sh 'make -j 8'
          // }
        }

        stage("Package Debian") {
          sh 'cd library-build; make package'
          archive includes: 'library-build/*.deb'
        }

        stage("unit tests") {
          // To trim test time, only execute single digit tests
          sh '''
              cd clients-build/tests-build/staging
              ./rocblas-test --gtest_output=xml --gtest_filter=*/?
          '''
          junit 'clients-build/tests-build/staging/*.xml'
        }

        stage("samples") {
          sh "cd clients-build/samples-build; ./example-sscal"
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

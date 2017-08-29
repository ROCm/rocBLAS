#!/usr/bin/env groovy

// Generated from snippet generator 'properties; set job properties'
properties([buildDiscarder(logRotator(
    artifactDaysToKeepStr: '',
    artifactNumToKeepStr: '',
    daysToKeepStr: '',
    numToKeepStr: '10')),
    disableConcurrentBuilds(),
    // parameters([booleanParam( name: 'push_image_to_docker_hub', defaultValue: false, description: 'Push rocblas image to rocm docker-hub' )]),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])

////////////////////////////////////////////////////////////////////////
// -- AUXILLARY HELPER FUNCTIONS
import hudson.FilePath;
import java.nio.file.Path;

////////////////////////////////////////////////////////////////////////
// Return build number of upstream job
@NonCPS
int get_upstream_build_num( )
{
    def upstream_cause = currentBuild.rawBuild.getCause( hudson.model.Cause$UpstreamCause )
    if( upstream_cause == null)
      return 0

    return upstream_cause.getUpstreamBuild()
}

////////////////////////////////////////////////////////////////////////
// Return project name of upstream job
@NonCPS
String get_upstream_build_project( )
{
    def upstream_cause = currentBuild.rawBuild.getCause( hudson.model.Cause$UpstreamCause )
    if( upstream_cause == null)
      return null

    return upstream_cause.getUpstreamProject()
}

////////////////////////////////////////////////////////////////////////
// Calculate the relative path between two sub-directories from a common root
@NonCPS
String g_relativize( String root_string, String rel_source, String rel_build )
{
  Path root_path = new File( root_string ).toPath( )
  Path path_src = root_path.resolve( rel_source )
  Path path_build = root_path.resolve( rel_build )

  return path_build.relativize( path_src ).toString( )
}

////////////////////////////////////////////////////////////////////////
// Construct the relative path of the build directory
String build_directory_rel( project_paths paths, compiler_args hcc_args )
{
  if( hcc_args.build_config.equalsIgnoreCase( 'release' ) )
  {
    return paths.build_prefix + "/${paths.project_name}" + '/release';
  }
  else
  {
    return paths.build_prefix + "/${paths.project_name}" + '/debug';
  }
}

////////////////////////////////////////////////////////////////////////
// Lots of images are created above; no apparent way to delete images:tags with docker global variable
def docker_clean_images( String org, String image_name )
{
  // Check if any images exist first grepping for image names
  int docker_images = sh( script: "docker images | grep \"${org}/${image_name}\"", returnStatus: true )

  // The script returns a 0 for success (images were found )
  if( docker_images == 0 )
  {
    // run bash script to clean images:tags after successful pushing
    sh "docker images | grep \"${org}/${image_name}\" | awk '{print \$1 \":\" \$2}' | xargs docker rmi"
  }
}

////////////////////////////////////////////////////////////////////////
// -- BUILD RELATED FUNCTIONS

////////////////////////////////////////////////////////////////////////
// Checkout source code, source dependencies and update version number numbers
// Returns a relative path to the directory where the source exists in the workspace
String rocblas_checkout_and_version( project_paths paths )
{
  String src_rel = paths.src_prefix + "/${paths.project_name}"

  dir( "${src_rel}" )
  {
    // checkout rocblas
    checkout([
      $class: 'GitSCM',
      branches: scm.branches,
      doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
      extensions: scm.extensions + [[$class: 'CleanCheckout']],
      userRemoteConfigs: scm.userRemoteConfigs
    ])

    if( fileExists( 'CMakeLists.txt' ) )
    {
      def cmake_version_file = readFile( 'CMakeLists.txt' ).trim()
      //echo "cmake_version_file:\n${cmake_version_file}"

      cmake_version_file = cmake_version_file.replaceAll(/(\d+\.)(\d+\.)(\d+\.)\d+/, "\$1\$2\$3${env.BUILD_ID}")
      //echo "cmake_version_file:\n${cmake_version_file}"
      writeFile( file: 'CMakeLists.txt', text: cmake_version_file )
    }
  }

  return src_rel
}


////////////////////////////////////////////////////////////////////////
// This creates the docker image that we use to build the project in
// The docker images contains all dependencies, including OS platform, to build
def docker_build_image( docker_args d_args, project_paths paths )
{
  String build_image_name = "build-rocblas-hip-artifactory"
  def build_image = null

  dir( paths.src_relpath )
  {
    def user_uid = sh( script: 'id -u', returnStdout: true ).trim()

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // build_image = docker.build( "${paths.project_name}/${build_image_name}:latest", "--pull -f docker/${docker_file} --build-arg user_uid=${user_uid} --build-arg base_image=${from_image} ." )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh "docker build -t ${paths.project_name}/${build_image_name}:latest -f docker/${d_args.docker_file} ${d_args.docker_build_args} --build-arg user_uid=${user_uid} --build-arg base_image=${d_args.from_image} ."
    build_image = docker.image( "${paths.project_name}/${build_image_name}:latest" )
  }

  return build_image
}

////////////////////////////////////////////////////////////////////////
// This encapsulates the cmake configure, build and package commands
// Leverages docker containers to encapsulate the build in a fixed environment
def docker_build_inside_image( def build_image, compiler_args c_args, docker_args d_args, project_paths paths )
{
  // Construct a relative path from build directory to src directory; used to invoke cmake
  String rel_path_to_src = g_relativize( pwd( ), paths.src_relpath, paths.build_relpath )

  String build_type_postfix = null
  if( c_args.build_config.equalsIgnoreCase( 'release' ) )
  {
    build_type_postfix = ""
  }
  else
  {
    build_type_postfix = "-d"
  }

  build_image.inside( d_args.docker_run_args )
  {
      withEnv(["CXX=${c_args.compiler_path}", 'CLICOLOR_FORCE=1'])
      {
        // Build library & clients
        sh  """#!/usr/bin/env bash
            set -x
            rm -rf ${paths.build_relpath} && mkdir -p ${paths.build_relpath} && cd ${paths.build_relpath}
            cmake -DCMAKE_INSTALL_PREFIX=package -DBUILD_CLIENTS=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON ${rel_path_to_src}
            make -j \$(nproc) install
          """
      }

    stage( "Test ${c_args.compiler_name} ${c_args.build_config}" )
    {
      // Cap the maximum amount of testing to be a few hours; assume failure if the time limit is hit
      timeout(time: 1, unit: 'HOURS')
      {
        sh """#!/usr/bin/env bash
              set -x
              cd ${paths.build_relpath}/clients/staging
              ./rocblas-test${build_type_postfix} --gtest_output=xml --gtest_color=yes
          """
        junit "${paths.build_relpath}/clients/staging/*.xml"

        sh """#!/usr/bin/env bash
              set -x
              cd ${paths.build_relpath}/clients/staging; ./example-sscal${build_type_postfix}
        """
      }

      // Only upload 1 set of packages, so we don't have a race condition uploading packages
      if( c_args.compiler_name.toLowerCase( ).startsWith( 'hcc-' ) )
      {
        sh  """#!/usr/bin/env bash
            set -x
            cd ${paths.build_relpath}
            make package
          """

        sh  """#!/usr/bin/env bash
            mkdir -p ${c_args.build_config}/${c_args.compiler_name}
            mv ${paths.build_relpath}/*.deb ${c_args.build_config}/${c_args.compiler_name}
            dpkg -c ${c_args.build_config}/${c_args.compiler_name}/*.deb
        """

        archiveArtifacts artifacts: "${c_args.build_config}/${c_args.compiler_name}/*.deb", fingerprint: true
      }
    }
  }

  return void
}

////////////////////////////////////////////////////////////////////////
// This builds a fresh docker image FROM a clean base image, with no build dependencies included
// Uploads the new docker image to internal artifactory
String docker_upload_artifactory( String hcc_ver, String artifactory_org, String from_image, String rocblas_src_rel, String build_dir_rel )
{
  def rocblas_install_image = null
  String image_name = "rocblas-${hcc_ver}-ubuntu-16.04"

  stage( 'artifactory' )
  {
    println "artifactory_org: ${artifactory_org}"

    //  We copy the docker files into the bin directory where the .deb lives so that it's a clean build everytime
    sh "cp -r ${rocblas_src_rel}/docker/* ${build_dir_rel}"

    // Docker 17.05 introduced the ability to use ARG values in FROM statements
    // Docker inspect failing on FROM statements with ARG https://issues.jenkins-ci.org/browse/JENKINS-44836
    // rocblas_install_image = docker.build( "${artifactory_org}/${image_name}:${env.BUILD_NUMBER}", "--pull -f ${build_dir_rel}/dockerfile-rocblas-ubuntu-16.04 --build-arg base_image=${from_image} ${build_dir_rel}" )

    // JENKINS-44836 workaround by using a bash script instead of docker.build()
    sh "docker build -t ${artifactory_org}/${image_name} --pull -f ${build_dir_rel}/dockerfile-rocblas-ubuntu-16.04 --build-arg base_image=${from_image} ${build_dir_rel}"
    rocblas_install_image = docker.image( "${artifactory_org}/${image_name}" )

    // The connection to artifactory can fail sometimes, but this should not be treated as a build fail
    try
    {
      // Don't push pull requests to artifactory, these tend to accumulate over time
      if( env.BRANCH_NAME.toLowerCase( ).startsWith( 'pr-' ) )
      {
        println 'Pull Request (PR-xxx) detected; NOT pushing to artifactory'
      }
      else
      {
        docker.withRegistry('http://compute-artifactory:5001', 'artifactory-cred' )
        {
          rocblas_install_image.push( "${env.BUILD_NUMBER}" )
          rocblas_install_image.push( 'latest' )
        }
      }
    }
    catch( err )
    {
      currentBuild.result = 'SUCCESS'
    }
  }

  return image_name
}

////////////////////////////////////////////////////////////////////////
// Uploads the new docker image to the public docker-hub
def docker_upload_dockerhub( String local_org, String image_name, String remote_org )
{
  stage( 'docker-hub' )
  {
    // Do not treat failures to push to docker-hub as a build fail
    try
    {
      sh  """#!/usr/bin/env bash
          set -x
          echo inside sh
          docker tag ${local_org}/${image_name} ${remote_org}/${image_name}
        """

      docker_hub_image = docker.image( "${remote_org}/${image_name}" )

      docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-cred' )
      {
        docker_hub_image.push( "${env.BUILD_NUMBER}" )
        docker_hub_image.push( 'latest' )
      }
    }
    catch( err )
    {
      currentBuild.result = 'SUCCESS'
    }
  }
}

////////////////////////////////////////////////////////////////////////
// hip_integration_testing
// This function sets up compilation and testing of HiP on a compiler downloaded from an upstream build
// Integration testing is centered around docker and constructing clean test environments every time

// NOTES: I have implemeneted integration testing 3 different ways, and I've come to the conclusion nothing is perfect
// 1.  I've tried having HCC push the test compiler to artifactory, and having HiP download the test docker image from artifactory
//     a.  The act of uploading and downloading images from artifactory takes minutes
//     b.  There is no good way of deleting images from a repository.  You have to use an arcane CURL command and I don't know how
//        to keep the password secret.  These test integration images are meant to be ephemeral.
// 2.  I tried 'docker save' to export a docker image into a tarball, and transfering the image through 'copy artifacts plugin'
//     a.  The HCC docker image uncompressed is over 1GB
//     b.  Compressing the docker image takes even longer than uploading the image to artifactory
// 3.  Download the HCC .deb and dockerfile through 'copy artifacts plugin'.  Create a new HCC image on the fly
//     a.  There is inefficency in building a new ubuntu image and installing HCC twice (once in HCC build, once here)
//     b.  This solution doesn't scale when we start testing downstream libraries

// I've implemented solution #3 above, probably transitioning to #2 down the line (probably without compression)
String hip_integration_testing( String inside_args, String job, String build_config )
{
  // Attempt to make unique docker image names for each build, to support concurrent builds
  // Mangle docker org name with upstream build info
  String testing_org_name = 'hip-test-' + get_upstream_build_project( ).replaceAll('/','-') + '-' + get_upstream_build_num( )

  // Tag image name with this build number
  String hip_test_image_name = "hip:${env.BUILD_NUMBER}"

  def rocblas_integration_image = null

  dir( 'integration-testing' )
  {
    deleteDir( )

    // This invokes 'copy artifact plugin' to copy archived files from upstream build
    step([$class: 'CopyArtifact', filter: 'archive/**/*.deb, docker/dockerfile-*',
      fingerprintArtifacts: true, projectName: get_upstream_build_project( ), flatten: true,
      selector: [$class: 'TriggeredBuildSelector', allowUpstreamDependencies: false, fallbackToLastSuccessful: false, upstreamFilterStrategy: 'UseGlobalSetting'],
      target: '.' ])

    docker.build( "${testing_org_name}/${hip_test_image_name}", "-f dockerfile-hip-ubuntu-16.04 ." )
  }

  // Checkout source code, dependencies and version files
  String rocblas_src_rel = checkout_and_version( job )

  // Conctruct a binary directory path based on build config
  String rocblas_bin_rel = build_directory_rel( build_config );

  // Build rocblas inside of the build environment
  rocblas_integration_image = docker_build_image( job, testing_org_name, '', rocblas_src_rel, "${testing_org_name}/${hip_test_image_name}" )

  docker_build_inside_image( rocblas_integration_image, inside_args, job, '', build_config, rocblas_src_rel, rocblas_bin_rel )

  docker_clean_images( testing_org_name, '*' )
}

////////////////////////////////////////////////////////////////////////
// -- MAIN
// Following this line is the start of MAIN of this Jenkinsfile
//String job_name = env.JOB_NAME.toLowerCase( )

// sh  """
//     set -x
// # printf '\033[31mHello World\033[0m'
// # echo "TERM=${env.TERM}"
// # echo "LANG=${env.LANG}"
// # echo "SHELL=${env.SHELL}"
// """

// Docker related variables gathered together to reduce parameter bloat on function calls
class docker_args implements Serializable
{
  String from_image
  String docker_file
  String docker_run_args
  String docker_build_args
}

// Docker related variables gathered together to reduce parameter bloat on function calls
class compiler_args implements Serializable
{
  String compiler_name
  String build_config
  String compiler_path
}

// Paths variables bundled together to reduce parameter bloat on function calls
class project_paths implements Serializable
{
  String project_name
  String src_prefix
  String src_relpath
  String build_prefix
  String build_relpath
}

// Integration testing is a special path which implies testing of an upsteam build of hcc,
// but does not need testing across older builds of hcc or cuda.
// params.hip_integration_test is set in HIP build
// NOTE: hip does not currently set this bit; this is non-functioning at this time
if( params.hip_integration_test )
{
  println "Enabling rocblas integration testing pass"

  node('docker && rocm')
  {
    hip_integration_testing( '--device=/dev/kfd', 'hip-ctu', 'Release' )
  }

  return
}

// The following launches 3 builds in parallel: hcc-ctu, hcc-1.6 and cuda
parallel hcc_ctu:
{
  node( 'docker && rocm' )
  {
    def hcc_docker_args = new docker_args(
        from_image:'compute-artifactory:5001/rocm-developer-tools/hip/master/hip-hcc-ctu-ubuntu-16.04:latest',
        docker_file:'dockerfile-build-hip-hcc-ctu-ubuntu-16.04',
        docker_run_args:'--device=/dev/kfd',
        docker_build_args:' --pull' )

    def hcc_compiler_args = new compiler_args(
        compiler_name:'hcc-ctu',
        build_config:'Release',
        compiler_path:'/opt/rocm/bin/hcc' )

    def paths = new project_paths(
        project_name:'rocblas',
        src_prefix:'src',
        build_prefix:'build' )

    ansiColor( 'vga' )
    {
      stage( "Build ${hcc_compiler_args.compiler_name} ${hcc_compiler_args.build_config}" )
      {
        // Checkout source code, dependencies and version files
        paths.src_relpath = rocblas_checkout_and_version( paths )

        // Conctruct a binary directory path based on build config
        paths.build_relpath = build_directory_rel( paths, hcc_compiler_args );

        // Create/reuse a docker image that represents the rocblas build environment
        def rocblas_build_image = docker_build_image( hcc_docker_args, paths )

        // Print system information for the log
        rocblas_build_image.inside( hcc_docker_args.docker_run_args )
        {
          sh  """
              set -x
              /opt/rocm/bin/rocm_agent_enumerator -t ALL
              /opt/rocm/bin/hcc --version
            """
        }

        // Build rocblas inside of the build environment
        docker_build_inside_image( rocblas_build_image, hcc_compiler_args, hcc_docker_args, paths )
    }

      // // After a successful build, upload a docker image of the results
      // String rocblas_image_name = docker_upload_artifactory( hcc_ver, job_name, from_image, rocblas_src_rel, rocblas_bin_rel )

      // if( params.push_image_to_docker_hub )
      // {
      //   docker_upload_dockerhub( job_name, rocblas_image_name, 'rocm' )
      //   docker_clean_images( 'rocm', rocblas_image_name )
      // }
      // docker_clean_images( job_name, rocblas_image_name )
    }
  }
},
hcc_1_6:
{
  node( 'docker && rocm' )
  {
    def hcc_docker_args = new docker_args(
        from_image:'rocm/rocm-terminal:latest',
        docker_file:'dockerfile-build-rocm-terminal',
        docker_run_args:'--device=/dev/kfd',
        docker_build_args:' --pull' )

    def hcc_compiler_args = new compiler_args(
        compiler_name:'hcc-1.6',
        build_config:'Release',
        compiler_path:'/opt/rocm/bin/hcc' )

    def paths = new project_paths(
        project_name:'rocblas',
        src_prefix:'src',
        build_prefix:'build' )

    ansiColor( 'vga' )
    {
      stage( "Build ${hcc_compiler_args.compiler_name} ${hcc_compiler_args.build_config}" )
      {
        // Checkout source code, dependencies and version files
        paths.src_relpath = rocblas_checkout_and_version( paths )

        // Conctruct a binary directory path based on build config
        paths.build_relpath = build_directory_rel( paths, hcc_compiler_args );

        // Create/reuse a docker image that represents the rocblas build environment
        def rocblas_build_image = docker_build_image( hcc_docker_args, paths )

        // Print system information for the log
        rocblas_build_image.inside( hcc_docker_args.docker_run_args )
        {
          sh  """#!/usr/bin/env bash
              set -x

              /opt/rocm/bin/rocm_agent_enumerator -t ALL
              /opt/rocm/bin/hcc --version
            """
        }

        // Build rocblas inside of the build environment
        docker_build_inside_image( rocblas_build_image, hcc_compiler_args, hcc_docker_args, paths )
      }

      // Not pushing rocblas-hcc-1.6 builds at this time; saves a minute and nobody needs?
      // String rocblas_image_name = docker_upload_artifactory( hcc_ver, job_name, from_image, rocblas_src_rel, rocblas_bin_rel )
      // docker_clean_images( job_name, rocblas_image_name )
    }
  }
}
//,
// Disabling: nvcc fails cmake configure because nvcc identifies itself as gnu 5.4 compiler, but fails feature checks
// nvcc:
// {
//   node( 'docker && cuda' )
//   {
//     ////////////////////////////////////////////////////////////////////////
//     // Block of string constants customizing behavior for cuda
//     String nvcc_ver = 'nvcc-8.0'
//     String from_image = 'nvidia/cuda:8.0-devel'
//     String docker_file = 'dockerfile-build-nvidia-cuda-8'

//     ansiColor( 'vga' )
//     {
//       // This unfortunately hardcodes the driver version nvidia_driver_375.74 in the volume mount.  Research if a way
//       // exists to get volume driver to customize the volume names to leave out driver version
//       String inside_args = '''--device=/dev/nvidiactl --device=/dev/nvidia0 --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools
//           --volume-driver=nvidia-docker --volume=nvidia_driver_375.74:/usr/local/nvidia:ro''';

//       // Checkout source code, dependencies and version files
//       String rocblas_src_rel = rocblas_checkout_and_version( 'src', nvcc_ver )

//       // We pull public nvidia images
//       def rocblas_build_image = docker_build_image( docker_file, nvcc_ver, 'rocblas', ' --pull', rocblas_src_rel, from_image )

//       // Print system information for the log
//       rocblas_build_image.inside( inside_args )
//       {
//         sh  """#!/usr/bin/env bash
//             set -x
//             nvidia-smi
//             nvcc --version
//           """
//       }

//       // Conctruct a binary directory path based on build config
//       String rocblas_bin_rel = build_directory_rel( 'build', build_config );

//       // Build rocblas inside of the build environment
//       docker_build_inside_image( rocblas_build_image, inside_args, nvcc_ver, '', '/opt/rocm/bin/hipcc', build_config, rocblas_src_rel, rocblas_bin_rel )
//     }
//   }
// }

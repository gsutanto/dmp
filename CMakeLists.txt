cmake_minimum_required(VERSION 2.8.3)
project(amd_clmc_dmp)

add_definitions(-DEIGEN_RUNTIME_NO_MALLOC -DEIGEN_STACK_ALLOCATION_LIMIT=0)

# required to update the rpath during installation. has to be *before* add_executable and add_library
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) # tried also with TRUE
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)                                                
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)


# set global compile type
set(CMAKE_BUILD_TYPE RelWithDebInfo) # Optimization with debugging info
#set(CMAKE_BUILD_TYPE Release)       # Optimization 
#set(CMAKE_BUILD_TYPE Debug)         # Debug

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Weffc++ -pedantic -std=c++0x")

find_package(catkin REQUIRED COMPONENTS Eigen) # 14.04
#find_package(Eigen REQUIRED)                   # 12.04
find_package(Boost COMPONENTS thread system filesystem REQUIRED)
find_package(Threads)

#add_definitions(${EIGEN_DEFINITIONS})
#link_directories(${EIGEN_LIBRARY_DIRS})

include_directories(
    include
    ${Eigen_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

get_filename_component(DATA_DIR data ABSOLUTE)
get_filename_component(MATLAB_DIR matlab ABSOLUTE)
get_filename_component(PLOT_DIR plot ABSOLUTE)
get_filename_component(RT_ERRORS_DIR rt_errors ABSOLUTE)

set(COMPILE_DEFS DMP_DATA_DIR=\"${DATA_DIR}\"
  DMP_MATLAB_DIR=\"${MATLAB_DIR}\"
  DMP_PLOT_DIR=\"${PLOT_DIR}\"
  DMP_RT_ERRORS_DIR=\"${RT_ERRORS_DIR}\"
)

add_library(${PROJECT_NAME} SHARED 
    src/paths.cpp
    src/utility/utility.cpp
    src/utility/utility_cartesian.cpp
    src/utility/utility_quaternion.cpp
    src/utility/DataIO.cpp
    src/utility/NormalDistributionRandomVectorGenerator.cpp
    src/utility/RealTimeAssertor.cpp
    src/dmp_state/DMPState.cpp
    src/dmp_state/QuaternionDMPState.cpp
    src/dmp_param/TauSystem.cpp
    src/dmp_param/DMPUnrollInitParams.cpp
    src/dmp_goal_system/GoalSystem.cpp
    src/dmp_goal_system/QuaternionGoalSystem.cpp
    src/dmp_base/CanonicalSystem.cpp
    src/dmp_base/FunctionApproximator.cpp
    src/dmp_base/TransformationSystem.cpp
    src/dmp_base/LearningSystem.cpp
    src/dmp_base/DMPDataIO.cpp
    src/dmp_base/DMP.cpp
    src/dmp_base/LoggedDMPVariables.cpp
    src/dmp_discrete/CanonicalSystemDiscrete.cpp
    src/dmp_discrete/FuncApproximatorDiscrete.cpp
    src/dmp_discrete/TransformSystemDiscrete.cpp
    src/dmp_discrete/LearningSystemDiscrete.cpp
    src/dmp_discrete/DMPDataIODiscrete.cpp
    src/dmp_discrete/DMPDiscrete.cpp
    src/dmp_discrete/LoggedDMPDiscreteVariables.cpp
    src/dmp_1D/DMPDiscrete1D.cpp
    src/dmp_multi_dim/DMPDiscreteMultiDim.cpp
    src/cart_dmp/cart_coord_dmp/CartesianCoordTransformer.cpp
    src/cart_dmp/cart_coord_dmp/CartesianCoordDMP.cpp
    src/cart_dmp/quat_dmp/TransformSystemQuaternion.cpp
    src/cart_dmp/quat_dmp/QuaternionDMP.cpp
    src/dmp_coupling/learn_obs_avoid/LearnObsAvoidDataIO.cpp
    src/dmp_coupling/learn_obs_avoid/TCLearnObsAvoidFeatureParameter.cpp
    src/dmp_coupling/learn_obs_avoid/TransformCouplingLearnObsAvoid.cpp
    src/dmp_coupling/learn_tactile_feedback/TransformCouplingLearnTactileFeedback.cpp
    src/neural_nets/PMNN.cpp
)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(${PROJECT_NAME} PROPERTIES
  COMPILE_DEFINITIONS "${COMPILE_DEFS}"
)


### catkin exports
### this will allow other catkin packages in the same workspace to link against the libraries generated here
### and get access to the include folder
catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS Eigen    # 14.04
  DEPENDS Boost Threading # 14.04
#  DEPENDS Eigen Boost Threading # 12.04
  INCLUDE_DIRS include ${Eigen_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}
)

## Test executables
#set_source_files_properties(src/dmp_test/dmp_1D/dmp_1D_test.cpp PROPERTIES COMPILE_FLAGS -Weffc++)
add_executable(amd_clmc_dmp_1D_demo src/dmp_test/dmp_1D/dmp_1D_test.cpp)
target_link_libraries(amd_clmc_dmp_1D_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo src/dmp_test/cart_dmp/cart_coord_dmp/cart_coord_dmp_single_traj_training_test.cpp)
target_link_libraries(amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo src/dmp_test/cart_dmp/cart_coord_dmp/cart_coord_dmp_multi_traj_training_test.cpp)
target_link_libraries(amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_quat_dmp_unroll_demo src/dmp_test/cart_dmp/quat_dmp/quat_dmp_unroll_test.cpp)
target_link_libraries(amd_clmc_dmp_quat_dmp_unroll_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_dc_loa_so_sb_single_demo_demo src/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/dc_loa_so_sb_single_demo_test.cpp)
target_link_libraries(amd_clmc_dmp_dc_loa_so_sb_single_demo_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_dc_loa_so_sb_multi_demo_demo src/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/dc_loa_so_sb_multi_demo_test.cpp)
target_link_libraries(amd_clmc_dmp_dc_loa_so_sb_multi_demo_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_ct_loa_learn_algo_verification_demo src/dmp_test/dmp_coupling/learn_obs_avoid/learning_algo_verification/ct_loa_learn_algo_verification.cpp)
target_link_libraries(amd_clmc_dmp_ct_loa_learn_algo_verification_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_ct_loa_so_sb_multi_demo_vicon_learning_demo src/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/ct_loa_so_sb_multi_demo_vicon_learning_test.cpp)
target_link_libraries(amd_clmc_dmp_ct_loa_so_sb_multi_demo_vicon_learning_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_ct_loa_so_sb_multi_demo_vicon_unrolling_demo src/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/ct_loa_so_sb_multi_demo_vicon_unrolling_test.cpp)
target_link_libraries(amd_clmc_dmp_ct_loa_so_sb_multi_demo_vicon_unrolling_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_ct_loa_so_sb_multi_demo_vicon_NN_unrolling_demo src/dmp_test/dmp_coupling/learn_obs_avoid/static_obs/single_baseline/ct_loa_so_sb_multi_demo_vicon_NN_unrolling_test.cpp)
target_link_libraries(amd_clmc_dmp_ct_loa_so_sb_multi_demo_vicon_NN_unrolling_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_pmnn_demo src/dmp_test/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/pmnn_test.cpp)
target_link_libraries(amd_clmc_dmp_pmnn_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_executable(amd_clmc_dmp_cart_dmp_pmnn_fitted_ct_unroll_demo src/dmp_test/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/cart_dmp_pmnn_fitted_ct_unroll_test.cpp)
target_link_libraries(amd_clmc_dmp_cart_dmp_pmnn_fitted_ct_unroll_demo ${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

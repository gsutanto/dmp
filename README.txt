To compile:
CATKIN
------
source /opt/ros/groovy/setup.bash
Go to inside amd_clmc/workspace/ folder (directory that contains src/ folder (and also possibly build/, devel/, and install/ folders))
catkin_make install

CMAKE
-----
Go to inside amd_clmc_dmp/ folder
mkdir build
cd build/
cmake ..
make


To execute:
CATKIN
------
Go to inside amd_clmc/workspace/install/demos/ folder
(1) 1-Dimensional DMP Test:
./amd_clmc_dmp_1D_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-g new_goal] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
(it will generate files and/or folders of files inside ../plot/dmp_1D/ directory, for plotting on MATLAB)

(2) Cartesian DMP Test:
(2.a) Learning from Single Trajectory
./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
(it will generate files and/or folders of files inside ../plot/cart_dmp/cart_coord_dmp/single_traj_training/ directory, for plotting on MATLAB)
(2.b) Learning from Multiple Trajectory
./amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
(it will generate files and/or folders of files inside ../plot/cart_dmp/cart_coord_dmp/multi_traj_training/ directory, for plotting on MATLAB)

(3) Learning Obstacle Avoidance Test:
(3.a) Single Demonstration Setting
./amd_clmc_dmp_dc_loa_so_sb_single_demo_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-o loa_output_file_path] [-e rt_err_file_path]
(3.b) Multiple Demonstration Settings
./amd_clmc_dmp_dc_loa_so_sb_multi_demo_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r regularization_const] [-o loa_output_file_path] [-e rt_err_file_path]


To plot:
Open MATLAB
(1) 1-Dimensional DMP Test:
Go to inside amd_clmc/workspace/install/plot/dmp_1D/ folder
Execute 'plot_dmp_1D([is_plot_goal_trajetory(0/1)], [is_plot_canonical_sys(0/1)], [is_plot_forcing_term(0/1)], [is_plot_forcing_term_vs_canonical_state(0/1)])' from MATLAB command line

(2) Cartesian DMP Test:
(2.a) Learning from Single Trajectory
Go to inside amd_clmc/workspace/install/plot/cart_dmp/cart_coord_dmp/single_traj_training/ folder
Execute 'plot_cart_coord_dmp_single_traj_training' from MATLAB command line
(2.b) Learning from Multiple Trajectory
Go to inside amd_clmc/workspace/install/plot/cart_dmp/cart_coord_dmp/multi_traj_training/ folder
Execute 'plot_cart_coord_dmp_multi_traj_training([is_plot_training_trajectories(0/1)])' from MATLAB command line

(3) Learning Obstacle Avoidance Test:
Go to inside amd_clmc/workspace/install/plot/dmp_coupling/learn_obs_avoid/tau_invariance_evaluation/ folder
Execute 'plot_dmp_coupling_learn_obs_avoid([obstacle_position_selection(0/1)])' from MATLAB command line


To run software tests:
$ cd <workspace>/src/catkin/planning/amd_clmc_dmp/software_test/
$ ./get_original_software_test_dir_abs_path.sh
$ cd <workspace>/devel/lib/amd_clmc_dmp/
$ cp -r <workspace>/src/catkin/planning/amd_clmc_dmp/software_test/ ../
$ ../software_test/dmp_software_test.sh
If there are some messages produced (by Unix's diff program), then it is possible that the software has been changed, and it is not producing consistent results anymore. Please check the software and try to return it to a consistent state, or reconcile the differences.

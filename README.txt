To execute on Python:
------
Open Spyder IDE ( https://docs.spyder-ide.org/installation.html )
(1) 1-Dimensional DMP Test:
Open the file <workspace>/src/catkin/planning/amd_clmc_dmp/python/dmp_test/dmp_1D/dmp_1D_test.py
Execute ("Run") the file inside the Spyder IDE.
(it will generate some plots comparing the training/demonstration data (in dotted blue line) versus the unrolled trajectory from the learned DMP primitive (in solid green line), 
 as well as some plots about the DMP basis function activations versus time and some plots about canonical state variables' trajectories)

(2) Cartesian DMP Test:
(2.a) Learning from Single Trajectory
Open the file <workspace>/src/catkin/planning/amd_clmc_dmp/python/dmp_test/cart_dmp/cart_coord_dmp/cart_coord_dmp_single_traj_training_test.py
Execute ("Run") the file inside the Spyder IDE.
(it will generate some plots comparing the training/demonstration data (in dotted blue line) versus the unrolled trajectory from the learned DMP primitive (in solid green line))
(2.b) Learning from Multiple Trajectories
Open the file <workspace>/src/catkin/planning/amd_clmc_dmp/python/dmp_test/cart_dmp/cart_coord_dmp/cart_coord_dmp_multi_traj_training_test.py
Execute ("Run") the file inside the Spyder IDE.
(it will generate some plots comparing the training/demonstration data (in dotted blue lines) versus the unrolled trajectory from the learned DMP primitive (in solid green line))

(3) Quaternion DMP Test:
(3.a) Learning from Single Trajectory
Open the file <workspace>/src/catkin/planning/amd_clmc_dmp/python/dmp_test/cart_dmp/quat_dmp/quat_dmp_single_traj_training_test.py
Execute ("Run") the file inside the Spyder IDE.
(it will generate some plots comparing the training/demonstration data (in dotted blue line) versus the unrolled trajectory from the learned DMP primitive (in solid green line))
(3.b) Learning from Multiple Trajectories
Open the file <workspace>/src/catkin/planning/amd_clmc_dmp/python/dmp_test/cart_dmp/quat_dmp/quat_dmp_multi_traj_training_test.py
Execute ("Run") the file inside the Spyder IDE.
(it will generate some plots comparing the training/demonstration data (in dotted blue lines) versus the unrolled trajectory from the learned DMP primitive (in solid green line))



To compile:
CATKIN
------
source /opt/ros/indigo/setup.bash
Go to inside <workspace>/ folder (directory that contains src/ folder (and also possibly build/, devel/, and install/ folders))
catkin_make install

CMAKE
-----
Go to inside amd_clmc_dmp/ folder
mkdir build
cd build/
cmake ..
make


To execute on C++:
(C++ implementation is HARD real-time safe for execution in a robot control loop; depends on Eigen C++ Linear Algebra library)
------
Go to inside <workspace>/devel/lib/amd_clmc_dmp/ folder (if using CATKIN installation)
(1) 1-Dimensional DMP Test:
./amd_clmc_dmp_1D_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-g new_goal] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
(it will generate files and/or folders of files inside ../plot/dmp_1D/ directory, for plotting on MATLAB)

(2) Cartesian DMP Test:
(2.a) Learning from Single Trajectory
./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-h time_goal_change(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
(it will generate files and/or folders of files inside ../plot/cart_dmp/cart_coord_dmp/single_traj_training/ directory, for plotting on MATLAB)
(2.b) Learning from Multiple Trajectories
./amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r time_reproduce_max(>=0)] [-t tau_reproduce(>=0)] [-e rt_err_file_path]
(it will generate files and/or folders of files inside ../plot/cart_dmp/cart_coord_dmp/multi_traj_training/ directory, for plotting on MATLAB)

(3) Learning Obstacle Avoidance Test:
(3.a) Single Demonstration Setting
./amd_clmc_dmp_dc_loa_so_sb_single_demo_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-o loa_output_file_path] [-e rt_err_file_path]
(3.b) Multiple Demonstration Settings
./amd_clmc_dmp_dc_loa_so_sb_multi_demo_demo [-f formula_type(0=_SCHAAL_ OR 1=_HOFFMANN_)] [-c canonical_order(1 OR 2)] [-m learning_method(0=_SCHAAL_LWR_METHOD_ OR 1=_JPETERS_GLOBAL_LEAST_SQUARES_METHOD_)] [-r regularization_const] [-o loa_output_file_path] [-e rt_err_file_path]


To plot (C++ results to be plotted on MATLAB):
Open MATLAB
(1) 1-Dimensional DMP Test:
Go to inside <workspace>/install/plot/dmp_1D/ folder
Execute 'plot_dmp_1D([is_plot_goal_trajetory(0/1)], [is_plot_canonical_sys(0/1)], [is_plot_forcing_term(0/1)], [is_plot_forcing_term_vs_canonical_state(0/1)])' from MATLAB command line

(2) Cartesian DMP Test:
(2.a) Learning from Single Trajectory
Go to inside <workspace>/install/plot/cart_dmp/cart_coord_dmp/single_traj_training/ folder
Execute 'plot_cart_coord_dmp_single_traj_training' from MATLAB command line
(2.b) Learning from Multiple Trajectories
Go to inside <workspace>/install/plot/cart_dmp/cart_coord_dmp/multi_traj_training/ folder
Execute 'plot_cart_coord_dmp_multi_traj_training([is_plot_training_trajectories(0/1)])' from MATLAB command line

(3) Learning Obstacle Avoidance Test:
Go to inside <workspace>/install/plot/dmp_coupling/learn_obs_avoid/tau_invariance_evaluation/ folder
Execute 'plot_dmp_coupling_learn_obs_avoid([obstacle_position_selection(0/1)])' from MATLAB command line



To run software tests (comparing results between C++ vs MATLAB vs Python implementations, should be pretty similar from each other):
$ cd <workspace>/src/catkin/planning/amd_clmc_dmp/software_test/
$ ./get_original_software_test_dir_abs_path.sh
$ cd <workspace>/devel/lib/amd_clmc_dmp/
$ cp -r <workspace>/src/catkin/planning/amd_clmc_dmp/software_test/ ../
$ ../software_test/dmp_software_test.sh
If there are some messages produced (by Unix's diff program), then it is possible that the software has been changed, and it is not producing consistent results anymore. Please check the software and try to return it to a consistent state, or reconcile the differences.
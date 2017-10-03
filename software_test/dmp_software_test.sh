#!/usr/bin/tcsh -f


#IF USING VALGRIND, HERE IS THE EXAMPLE (BUT THE PID WILL BE DIFFERENT, SO IT IS NOT COMPARABLE):
#valgrind --leak-check=full -v ./amd_clmc_dmp_1D_demo -f 0 -c 1 -e ../software_test/test_rt_err_dmp_1D_test_0_1.txt >& ../software_test/test_dmp_1D_test_0_1.txt


# Execution Test and Comparison with Well-Known Result:
#../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 0 -c 1 -m 0" result_dmp_1D_test_0_1_0.txt	# became part of ../software_test/execute_matlab_tests_and_compare_w_cpp_tests.sh script execution
../software_test/execute_cpp_test.sh "./amd_clmc_dmp_1D_demo -f 0 -c 1 -m 0" test_cpp_dmp_1D_test_0_1_0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 0 -c 1 -m 1" result_dmp_1D_test_0_1_1.txt
#../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 0 -c 2 -m 0" result_dmp_1D_test_0_2_0.txt	# became part of ../software_test/execute_matlab_tests_and_compare_w_cpp_tests.sh script execution
../software_test/execute_cpp_test.sh "./amd_clmc_dmp_1D_demo -f 0 -c 2 -m 0" test_cpp_dmp_1D_test_0_2_0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 0 -c 2 -m 1" result_dmp_1D_test_0_2_1.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 1 -c 1 -m 0" result_dmp_1D_test_1_1_0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 1 -c 1 -m 1" result_dmp_1D_test_1_1_1.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 1 -c 2 -m 0" result_dmp_1D_test_1_2_0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 1 -c 2 -m 1" result_dmp_1D_test_1_2_1.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 0 -c 2 -m 0 -r 6.0 -h 1.0 -g 10.0 -t 4.0" result_dmp_1D_test_0_2_0_6.0_1.0_10.0_4.0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 0 -c 2 -m 1 -r 6.0 -h 1.0 -g 10.0 -t 4.0" result_dmp_1D_test_0_2_1_6.0_1.0_10.0_4.0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 1 -c 1 -m 0 -r 6.0 -h 1.0 -g 10.0 -t 4.0" result_dmp_1D_test_1_1_0_6.0_1.0_10.0_4.0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_1D_demo -f 1 -c 1 -m 1 -r 6.0 -h 1.0 -g 10.0 -t 4.0" result_dmp_1D_test_1_1_1_6.0_1.0_10.0_4.0.txt

#../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 0 -c 1" result_cart_coord_dmp_single_traj_training_test_0_1.txt	# became part of ../software_test/execute_matlab_tests_and_compare_w_cpp_tests.sh script execution
../software_test/execute_cpp_test.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 0 -c 1" test_cpp_cart_coord_dmp_single_traj_training_test_0_1.txt
#../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 0 -c 2" result_cart_coord_dmp_single_traj_training_test_0_2.txt	# became part of ../software_test/execute_matlab_tests_and_compare_w_cpp_tests.sh script execution
../software_test/execute_cpp_test.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 0 -c 2" test_cpp_cart_coord_dmp_single_traj_training_test_0_2.txt
#../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 1 -c 1" result_cart_coord_dmp_single_traj_training_test_1_1.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 1 -c 2" result_cart_coord_dmp_single_traj_training_test_1_2.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 0 -c 2 -r 6.0 -h 1.0 -t 6.0" result_cart_coord_dmp_single_traj_training_test_0_2_6.0_1.0_6.0.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_single_traj_training_demo -f 0 -c 2 -r 6.0 -h 2.0 -t 6.0" result_cart_coord_dmp_single_traj_training_test_0_2_6.0_2.0_6.0.txt

#../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo -f 0 -c 2" result_cart_coord_dmp_multi_traj_training_test_0_2.txt	# became part of ../software_test/execute_matlab_tests_and_compare_w_cpp_tests.sh script execution
../software_test/execute_cpp_test.sh "./amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo -f 0 -c 2" test_cpp_cart_coord_dmp_multi_traj_training_test_0_2.txt
../software_test/compare_execution_result.sh "./amd_clmc_dmp_cart_coord_dmp_multi_traj_training_demo -f 1 -c 2" result_cart_coord_dmp_multi_traj_training_test_1_2.txt

../software_test/execute_cpp_test.sh "./amd_clmc_dmp_quat_dmp_unroll_demo" test_cpp_quat_dmp_unroll_test.txt

../software_test/execute_cpp_test.sh "./amd_clmc_dmp_ffnn_final_phase_lwr_per_dims_demo" test_cpp_ffnn_final_phase_lwr_per_dims_test.txt

../software_test/execute_cpp_test.sh "./amd_clmc_dmp_cart_dmp_nnlwr_fitted_ct_unroll_demo -o ../software_test/" test_cpp_cart_dmp_nnlwr_fitted_ct_unroll_test.txt

../software_test/execute_matlab_tests_and_compare_w_cpp_tests.sh

#../software_test/compare_execution_result.sh "./amd_clmc_dmp_dc_loa_so_sb_single_demo_demo -f 0 -c 1 -o ../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/single_demo/param_settings/param_set_1/Schaal_1st_order_CanonicalSys/" result_dc_loa_so_sb_single_demo_test_0_1.txt
#temp#../software_test/compare_execution_result.sh "./amd_clmc_dmp_dc_loa_so_sb_single_demo_demo -f 0 -c 2 -o ../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/single_demo/param_settings/param_set_1/Schaal_2nd_order_CanonicalSys/" result_dc_loa_so_sb_single_demo_test_0_2.txt
#temp#../software_test/compare_execution_result.sh "./amd_clmc_dmp_dc_loa_so_sb_single_demo_demo -f 1 -c 1 -o ../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/single_demo/param_settings/param_set_1/Hoffmann_1st_order_CanonicalSys/" result_dc_loa_so_sb_single_demo_test_1_1.txt
#../software_test/compare_execution_result.sh "./amd_clmc_dmp_dc_loa_so_sb_single_demo_demo -f 1 -c 2 -o ../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/single_demo/param_settings/param_set_1/Hoffmann_2nd_order_CanonicalSys/" result_dc_loa_so_sb_single_demo_test_1_2.txt

#../software_test/compare_execution_result.sh "./amd_clmc_dmp_dc_loa_so_sb_multi_demo_demo -f 0 -c 2" result_dc_loa_so_sb_multi_demo_test_0_2.txt

#temp#../software_test/compare_execution_result.sh "./amd_clmc_dmp_ct_loa_learn_algo_verification_demo -f 0 -c 2" result_ct_loa_learn_algo_verification_test_0_2.txt


# Delete Test Files:
rm -f ../software_test/test_*

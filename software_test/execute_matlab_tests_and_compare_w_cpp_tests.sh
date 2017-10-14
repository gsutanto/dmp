#!/bin/bash

amd_clmc_dmp_software_test_dir_abs_path=$(<../software_test/amd_clmc_dmp_software_test_dir_abs_path.txt)
devel_software_test_dir_abs_path=$(pwd)/../software_test/

matlab -nodisplay -nodesktop -nosplash -nojvm -r \
"cd('$amd_clmc_dmp_software_test_dir_abs_path/../matlab/dmp_1D/'); \
testDMP1DFitAndUnroll('$devel_software_test_dir_abs_path/'); \
cd('$amd_clmc_dmp_software_test_dir_abs_path/../matlab/cart_dmp/cart_coord_dmp/'); \
testCartCoordDMPFitAndUnroll('$devel_software_test_dir_abs_path/'); \
cd('$amd_clmc_dmp_software_test_dir_abs_path/../matlab/cart_dmp/quat_dmp/'); \
testQuaternionDMPUnroll('$devel_software_test_dir_abs_path/'); \
cd('$amd_clmc_dmp_software_test_dir_abs_path/../data/dmp_coupling/learn_tactile_feedback/scraping/neural_nets/pmnn/python_models/'); \
load('prim_1_Ctt_test_prediction.mat'); \
dlmwrite('$devel_software_test_dir_abs_path/test_matlab_pmnn_test.txt', Ctt_test_prediction, 'delimiter', ' ', 'precision', '%.5f'); \
cd('$amd_clmc_dmp_software_test_dir_abs_path/../matlab/dmp_coupling/learn_tactile_feedback/neural_nets/unroll/for_verification/'); \
testCoupledCartDMPUnroll('$devel_software_test_dir_abs_path/'); \
cd('$amd_clmc_dmp_software_test_dir_abs_path/../matlab/utilities/'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_dmp_1D_test_0_1_0.txt', '$devel_software_test_dir_abs_path/test_matlab_dmp_1D_test_0_1_0.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_dmp_1D_test_0_2_0.txt', '$devel_software_test_dir_abs_path/test_matlab_dmp_1D_test_0_2_0.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_cart_coord_dmp_single_traj_training_test_0_1.txt', '$devel_software_test_dir_abs_path/test_matlab_cart_coord_dmp_single_traj_training_test_0_1.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_cart_coord_dmp_single_traj_training_test_0_2.txt', '$devel_software_test_dir_abs_path/test_matlab_cart_coord_dmp_single_traj_training_test_0_2.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_cart_coord_dmp_multi_traj_training_test_0_2.txt', '$devel_software_test_dir_abs_path/test_matlab_cart_coord_dmp_multi_traj_training_test_0_2.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_quat_dmp_unroll_test.txt', '$devel_software_test_dir_abs_path/test_matlab_quat_dmp_unroll_test.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_pmnn_test.txt', '$devel_software_test_dir_abs_path/test_matlab_pmnn_test.txt', 6.001e-5); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_coupled_cart_dmp_unroll_test_prim1', '$devel_software_test_dir_abs_path/test_matlab_coupled_cart_dmp_unroll_test_prim1.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_coupled_cart_dmp_unroll_test_prim2', '$devel_software_test_dir_abs_path/test_matlab_coupled_cart_dmp_unroll_test_prim2.txt'); \
compareTwoNumericFiles('$devel_software_test_dir_abs_path/test_cpp_coupled_cart_dmp_unroll_test_prim3', '$devel_software_test_dir_abs_path/test_matlab_coupled_cart_dmp_unroll_test_prim3.txt'); \
quit";

echo "execute_matlab_tests_and_compare_w_cpp_tests.sh script execution done!"

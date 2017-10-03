% Before executing this script, please make sure that the variables are
% available in the workspace! This can be done, for example by commands:
% load('trained_NN_2.mat');                 % load the *.mat file containing the trained neural network (NN) model
% genFunction(net,'trained_NN_2_netFcn');   % convert the trained NN model into a MATLAB *.m file

vicon_data_dir_path     = '../../../../data/dmp_coupling/learn_obs_avoid/static_obs/data_multi_demo_vicon_static/trained_NN_params/';

dlmwrite([vicon_data_dir_path, 'b1.txt'], b1, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'b2.txt'], b2, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'b3.txt'], b3, 'delimiter', ' ', 'precision', 100);

dlmwrite([vicon_data_dir_path, 'IW1_1.txt'], IW1_1, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'LW2_1.txt'], LW2_1, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'LW3_2.txt'], LW3_2, 'delimiter', ' ', 'precision', 100);

dlmwrite([vicon_data_dir_path, 'x1_step1_gain.txt'], x1_step1_gain, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'x1_step1_xoffset.txt'], x1_step1_xoffset, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'x1_step1_ymin.txt'], x1_step1_ymin, 'delimiter', ' ', 'precision', 100);

dlmwrite([vicon_data_dir_path, 'y1_step1_gain.txt'], y1_step1_gain, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'y1_step1_xoffset.txt'], y1_step1_xoffset, 'delimiter', ' ', 'precision', 100);
dlmwrite([vicon_data_dir_path, 'y1_step1_ymin.txt'], y1_step1_ymin, 'delimiter', ' ', 'precision', 100);
% Author: Giovanni Sutanto
% Date  : August 01, 2016

close   all;
clc;

addpath('../utilities/');
addpath('../vicon/');

data_global_coord   = prepareDemoDatasetLOAVicon;

save(['../data/data_multi_demo_vicon_static_global_coord.mat'], 'data_global_coord');
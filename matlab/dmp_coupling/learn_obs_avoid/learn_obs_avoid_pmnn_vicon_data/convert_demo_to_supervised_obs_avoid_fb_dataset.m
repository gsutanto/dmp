% Author: Giovanni Sutanto
% Date  : October 2017
% Description:
%   Convert (segmented) demonstrations into 
%   supervised obstacle avoidance feedback dataset.

clear  	all;
close   all;
clc;

addpath('../../../utilities/');
addpath('../../../cart_dmp/cart_coord_dmp/');
addpath('../vicon/');

%% Demo Dataset Preparation

data_global_coord   = prepareDemoDatasetLOAVicon;

save('data_multi_demo_vicon_static_global_coord.mat', 'data_global_coord');

% end of Demo Dataset Preparation
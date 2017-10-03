clear all;
close all;
clc;

addpath('../../../../matlab/dmp_coupling/learn_obs_avoid/utilities/');

X       = dlmread('loa/X.txt');
Ct      = dlmread('loa/Ct_target.txt');

% Performing ARD
tic
disp(['Performing ARD:']);
[ w_ard, nmse_ard, Ct_fit_ard ] = learnUsingARD( X, Ct );
toc

w_ard   = w_ard.';
w_gT    = dlmread('../../../../data/dmp_coupling/learn_obs_avoid/learning_algo_verification/loa_synthetic_weights.txt');

diff_w  = w_ard - w_gT;

mse_w_ard   = mean(mean((diff_w).^2));
disp(['mse(w_gT-w_ard) = ', num2str(mse_w_ard)]);

dlmwrite('loa/learn_obs_avoid_weights_matrix_ARD.txt', w_ard, 'delimiter', ' ');
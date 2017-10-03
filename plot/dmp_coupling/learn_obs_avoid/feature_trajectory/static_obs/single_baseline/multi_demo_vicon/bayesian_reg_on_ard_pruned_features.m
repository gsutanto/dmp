clear all;
close all;
clc;

addpath('../../../../../../../matlab/dmp_coupling/learn_obs_avoid/utilities/');

X                           = dlmread('X.txt');
Ct                          = dlmread('Ct_target.txt');

D                           = size(Ct,2);

w                           = zeros(size(X,2), D);

num_iter                    = 1;
debug_interval              = 1;
debug_mode                  = 1;
alpha_min_threshold         = 0;
max_abs_weight_threshold    = 5e3;

w_ard                       = dlmread('learn_obs_avoid_weights_matrix_ARD.txt');

for d=1:D
    relevant_feature_idx    = find(w_ard(d,:) ~= 0);
    X_relevant              = X(:,relevant_feature_idx);
    w_proxy_br_d            = zeros(size(X_relevant,2), 1);
    [w_br_d, r_br_idx, cfit_hist, w_hist, log10_a_hist] = BayesianRegression( X_relevant, Ct(:,d), num_iter, debug_interval, debug_mode, alpha_min_threshold, max_abs_weight_threshold );
    w_proxy_br_d(r_br_idx,1)                            = w_br_d;
    w(relevant_feature_idx,d)                           = w_proxy_br_d;
end

w                           = w.';

dlmwrite('learn_obs_avoid_weights_matrix_bayesian_reg_on_ard_pruned_features.txt', w, 'delimiter', ' ');
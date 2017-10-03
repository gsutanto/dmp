clear           all;
close           all;
clc;

data_dir        = '../../../../../plot/dmp_coupling/learn_obs_avoid/feature_trajectory/static_obs/single_baseline/';

% exp_type        = 'single_demo/';
exp_type        = 'multi_demo/';

% feat_type       = 'feat0/';
% feat_type       = 'feat1/';

% param_table     = dlmread(strcat(exp_type, feat_type, 'parameter_table.txt'));
% X               = dlmread(strcat(exp_type, feat_type, 'X.txt'));
% Ct_target       = dlmread(strcat(exp_type, feat_type, 'Ct_target.txt'));

param_table     = dlmread(strcat(data_dir, exp_type, 'parameter_table.txt'));
X               = dlmread(strcat(data_dir, exp_type, 'X.txt'));
Ct_target       = dlmread(strcat(data_dir, exp_type, 'Ct_target.txt'));

nonzero_features    = any(X);
rind                = find(nonzero_features==1)';
param_table_new     = param_table;
param_table         = param_table_new(rind,:);
X_new               = X;
X                   = X_new(:,rind);

% mean_Ctt        = mean(Ct_target);
% Ctt_temp        = bsxfun(@minus,Ct_target,mean_Ctt);
% Ct_target       = Ctt_temp;

options.noise           = 1;        % initial output noise variance
options.threshold       = 1e-5;     % threshold for convergence
options.numIterations   = 10000;    % max number of EM iterations

result                  = cell(3,1);
w_vbls                  = zeros(size(X, 2), 3);
for d = 1:3
    result{d,1}         = vbls(X, Ct_target(:,d), options);
    w_vbls(:,d)         = result{d,1}.b_mean;
end

Ct_fit          = X * w_vbls;

figure;
hold            on;
plot3(Ct_target(:,1), Ct_target(:,2), Ct_target(:,3));
plot3(Ct_fit(:,1), Ct_fit(:,2), Ct_fit(:,3));
legend('Ct_t_a_r_g_e_t', 'Ct_f_i_t');
hold            off;

var_ct          = var(Ct_target, 1);
mse_vbls        = mean( (Ct_fit-Ct_target).^2 );
nmse_vbls       = mse_vbls./var_ct;
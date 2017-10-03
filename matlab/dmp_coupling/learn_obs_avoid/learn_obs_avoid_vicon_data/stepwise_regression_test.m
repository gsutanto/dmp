clear all;
close all;
clc;

load('4_X.mat');
load('4_Ct_target.mat');
load('4_loa_feat_param.mat');

[b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(X, Ct_target);

save('4_stepwisefit_result.mat', 'b', 'se', 'pval', 'inmodel', 'stats', 'nextstep', 'history');

var_Ct_target   = var(Ct_target,1);
history_rmse    = history.rmse;
history_mse     = history_rmse.^2;
history_nmse    = history_mse/var_Ct_target;

figure;
axis equal;
plot(history_mse);
xlabel('stepwisefit iter #');
ylabel('MSE');
title('MSE vs Iteration Number of Stepwise-Regression');

figure;
axis equal;
plot(history_nmse);
xlabel('stepwisefit iter #');
ylabel('NMSE');
title('NMSE vs Iteration Number of Stepwise-Regression');

figure;
axis equal;
plot(history.df0);
xlabel('stepwisefit iter #');
ylabel('Degree of Freedom');
title('Degree of Freedom vs Iteration Number of Stepwise-Regression');

figure;
axis equal;
plot(log10(max(abs(history.B))));
xlabel('stepwisefit iter #');
ylabel('log10(max(abs(w)))');
title('Logarithmic Max Absolute Weights (Magnitude) vs Iteration Number of Stepwise-Regression');
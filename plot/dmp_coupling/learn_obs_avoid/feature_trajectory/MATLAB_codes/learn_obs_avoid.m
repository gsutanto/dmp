clear all;
close all;
clc;

addpath('../../../../../../../matlab/dmp_coupling/learn_obs_avoid/utilities/');

is_performing_ard           = 1;
is_performing_bayesian_reg  = 0;
is_performing_lasso         = 0;

max_abs_weight_threshold    = 1e5;
num_iter_ard                = 1;
num_iter_br                 = 1;

X                           = dlmread('X.txt');
Ct                          = dlmread('Ct_target.txt');

param_table                 = dlmread('parameter_table.txt');
param_table                 = param_table.';
param_table(2,:)            = sqrt(1 ./ param_table(2,:));

D                           = size(Ct,2);

rank_XTX                    = rank(X.'*X);
dim_XTX                     = size(X,2);
percentage_rank_XTX         = (100.0 * rank_XTX)/dim_XTX;
fprintf('rank(X^T*X)  = %d of matrix dimension %d (%f %%)\n', ...
        rank_XTX, dim_XTX, percentage_rank_XTX);

if (is_performing_ard)
    disp(['----------------']);
    % Performing ARD
    tic
    disp(['Performing ARD:']);
    [ w_ard, nmse_learning_ard, Ct_fit_ard ] = learnUsingARD( X, Ct, max_abs_weight_threshold, num_iter_ard );
    toc

    w_ard                       = w_ard.';
    dlmwrite('learn_obs_avoid_weights_matrix_ARD.txt', w_ard, 'delimiter', ' ');
    
    param_table_w_stacked_ard   = [param_table; w_ard];
    
    figure;
    for d=1:size(param_table_w_stacked_ard,1)
        subplot(size(param_table_w_stacked_ard,1),1,d);
        hold on;
            if (d > size(param_table,1))
                title(['w (with ARD) d=',num2str(d-size(param_table,1))]);
            elseif (d==1)
                title(['param #',num2str(d),': beta']);
            elseif (d==2)
                title(['1/sqrt(param #',num2str(d),'): 1/sqrt(k)']);
            elseif ((d==3) && (size(param_table,1)==3))
                title(['param #',num2str(d),': s']);
            end
            plot(param_table_w_stacked_ard(d,:));
        hold off;
    end
    drawnow;
    
    % some curiosity on picked parameters:
    % all axes (x,y,z)
    w_non_zero_xyz_idx      = find(any(w_ard) == 1);
    w_zero_xyz_idx          = setdiff(1:size(w_ard,2), w_non_zero_xyz_idx);
    param_non_zero_xyz      = param_table(:,w_non_zero_xyz_idx);
    param_zero_xyz          = param_table(:,w_zero_xyz_idx);
    
    % only y and z axes
    w_non_zero_yz_idx       = find(any(w_ard(2:3,:)) == 1);
    w_zero_yz_idx           = setdiff(1:size(w_ard,2), w_non_zero_xyz_idx);
    param_non_zero_yz       = param_table(:,w_non_zero_yz_idx);
    param_zero_yz           = param_table(:,w_zero_yz_idx);

    disp(['ARD Result:']);
    disp(['nmse learning    = ', num2str(nmse_learning_ard)]);
    disp(['----------------']);
end

if (is_performing_bayesian_reg)
    disp(['----------------']);
    % Performing Bayesian Regression
    tic
    disp(['Performing Bayesian Regression:']);
    [ w_br, nmse_learning_br, Ct_fit_br ] = learnUsingBayesianRegression( X, Ct, max_abs_weight_threshold, num_iter_br );
    toc

    w_br                    = w_br.';
    dlmwrite('learn_obs_avoid_weights_matrix_BayesianReg.txt', w_br, 'delimiter', ' ');

    disp(['Bayesian Regression Result:']);
    disp(['nmse learning	= ', num2str(nmse_learning_br)]);
    disp(['----------------']);
end

if (is_performing_lasso)
    disp(['----------------']);
    % Performing LASSO
    tic
    disp(['Performing LASSO:']);
    [ w_lasso, nmse_learning_lasso, Ct_fit_lasso ] = learnUsingLASSO( X, Ct );
    toc

    w_lasso                     = w_lasso.';
    dlmwrite('learn_obs_avoid_weights_matrix_LASSO.txt', w_lasso, 'delimiter', ' ');

    disp(['LASSO Result:']);
    disp(['nmse learning    = ', num2str(nmse_learning_lasso)]);
    disp(['----------------']);
end
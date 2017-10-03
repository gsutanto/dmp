function [ nmse_train_cell, nmse_test_cell, w_cell, w_per_dim_cell, max_w_cell, min_w_cell, sparsity_w_cell ] = learnAndTestGeneralizationLeave_P_Out( X, Ct, learning_methods_cell, is_using_all_settings, feature_row_idx, num_lo_set )
    if (is_using_all_settings == 0) % just use one setting/obstacle position
        num_lo_trials               = length(feature_row_idx{1,1});
    else % use all settings/obstacle positions
        num_lo_trials               = 2 * length(feature_row_idx);
    end
    D                               = size(Ct,2);
    N_learning_method               = size(learning_methods_cell,2);

    nmse_train_cell                 = cell(size(num_lo_set, 2),N_learning_method);
    nmse_test_cell                  = cell(size(num_lo_set, 2),N_learning_method);
    w_cell                          = cell(size(num_lo_set, 2),N_learning_method);
    w_per_dim_cell                  = cell(size(num_lo_set, 2),N_learning_method);
    max_w_cell                      = cell(size(num_lo_set, 2),N_learning_method);
    min_w_cell                      = cell(size(num_lo_set, 2),N_learning_method);
    sparsity_w_cell                 = cell(size(num_lo_set, 2),N_learning_method);

    for n=1:size(num_lo_set, 2)
        for l_idx=1:N_learning_method
            nmse_train_cell{n,l_idx}        = zeros(num_lo_trials,D);
            nmse_test_cell{n,l_idx}         = zeros(num_lo_trials,D);
            w_cell{n,l_idx}                 = cell(num_lo_trials,1);
            w_per_dim_cell{n,l_idx}         = cell(1,D);
            for d=1:D
                w_per_dim_cell{n,l_idx}{1,d}= zeros(size(X,2), num_lo_trials);
            end
            max_w_cell{n,l_idx}             = zeros(num_lo_trials,D);
            min_w_cell{n,l_idx}             = zeros(num_lo_trials,D);
            sparsity_w_cell{n,l_idx}        = zeros(num_lo_trials,D);
        end
        
        for j=1:num_lo_trials
            disp(['performing generalization test Leave-',num2str(num_lo_set(1, n)),'-Out out of ',num2str(size(num_lo_set, 2)),' trials',...
                  ', j=',num2str(j),'/',num2str(num_lo_trials)]);

            all_idx                 = [1:size(X,1)];
            test_idx                = [];
            if (is_using_all_settings == 0) % just use one setting/obstacle position
                lo_set              = randsample(1:length(feature_row_idx{1,1}), num_lo_set(1, n));
                for k=1:length(lo_set)
                    test_idx        = union(test_idx, feature_row_idx{1,1}{1,lo_set(k)});
                end
            else % use all settings/obstacle positions
                lo_set              = randsample(1:length(feature_row_idx), num_lo_set(1, n));
                for k=1:length(lo_set)
                    for t=1:size(feature_row_idx{1,lo_set(k)},2) % all demonstrations/trajectories belonging to the randomly picked setting are taken out, and become part of the test set
                        test_idx    = union(test_idx, feature_row_idx{1,lo_set(k)}{1,t});
                    end
                end
            end
            disp(['leave-out set is: ', num2str(lo_set)]);
            train_idx               = setdiff(all_idx,test_idx);
            
            % Bug Checking:
            all_idx                 = [1:size(X,1)];
            supposed_all_idx        = union(train_idx, test_idx);
            supposed_null_set       = setdiff(all_idx, supposed_all_idx);
            if (isempty(supposed_null_set) ~= 1)
                disp('BUG: indexing bug; NOT all data samples are covered');
                return;
            end

            disp('--------------------------------------------------------');
            for l_idx=1:N_learning_method
                disp(learning_methods_cell{1,l_idx});
                w_cell{n,l_idx}{j,1}    = zeros(size(X,2), D);

                for d=1:D
                    disp([' => dim: ', num2str(d)]);
                    
                    tic
                    if (strcmp(learning_methods_cell{1,l_idx},'ARD') == 1)
                        precision_cap           = 0;
                        abs_weights_threshold   = 1e5;
                        
                        [w_ard_d,r_ard_idx] = ARD( X(train_idx,:), Ct(train_idx,d), 0, precision_cap, abs_weights_threshold );
                        w_cell{n,l_idx}{j,1}(r_ard_idx,d)   = w_ard_d;
                    elseif (strcmp(learning_methods_cell{1,l_idx},'BAYESIAN_REG') == 1)
                        num_iter                = 200;
                        debug_interval          = 4;
                        debug_mode              = 0;
                        alpha_min_threshold     = 0;
                        max_abs_weight_threshold= 1e5;
                        
                        [w_br_d, r_br_idx]      = BayesianRegression( X(train_idx,:), Ct(train_idx,d), num_iter, debug_interval, debug_mode, alpha_min_threshold, max_abs_weight_threshold );
                        w_cell{n,l_idx}{j,1}(r_br_idx,d)    = w_br_d;
                    elseif (strcmp(learning_methods_cell{1,l_idx},'RIDGE') == 1)
                        XX          = X(train_idx,:).' * X(train_idx,:);
                        xc          = X(train_idx,:).' * Ct(train_idx,d);

                        reg         = 1e-5;
                        A           = reg*eye(size(XX,2));
                        w_cell{n,l_idx}{j,1}(:,d)           = (A + XX)\xc;
                    elseif (strcmp(learning_methods_cell{1,l_idx},'LASSO') == 1)
                        alpha       = 1;
                        num_lambda  = 3;
                        [w_lasso_d]	= lasso( X(train_idx,:), Ct(train_idx,d), 'Alpha', alpha, 'NumLambda', num_lambda );
                        w_cell{n,l_idx}{j,1}(:,d)           = w_lasso_d(:,1);   % pick the column of w_lasso_d corresponding to the smallest regularization constant
                    end
                    toc
                    
                    w_per_dim_cell{n,l_idx}{1,d}(:,j)   = w_cell{n,l_idx}{j,1}(:,d);
                    max_w_cell{n,l_idx}(j,d)            = max(w_ard_d);
                    min_w_cell{n,l_idx}(j,d)            = min(w_ard_d);
                    sparsity_w_cell{n,l_idx}(j,d)       = length(r_ard_idx);
                end

                [ mse_train, nmse_train_cell{n,l_idx}(j,:), Ct_fit_train ] = computeNMSE( X(train_idx,:), w_cell{n,l_idx}{j,1}, Ct(train_idx,:) );
                [ mse_test, nmse_test_cell{n,l_idx}(j,:), Ct_fit_test ] = computeNMSE( X(test_idx,:), w_cell{n,l_idx}{j,1}, Ct(test_idx,:) );
                disp(['   nmse_train = ',num2str(nmse_train_cell{n,l_idx}(j,:))]);
                disp(['   nmse_test  = ',num2str(nmse_test_cell{n,l_idx}(j,:))]);
                disp(['   max_w      = ',num2str(max_w_cell{n,l_idx}(j,:))]);
                disp(['   min_w      = ',num2str(min_w_cell{n,l_idx}(j,:))]);
                disp(['   sparsity_w = ',num2str(sparsity_w_cell{n,l_idx}(j,:))]);
                disp('--------------------------------------------------------');
            end
            disp('--------------------------------------------------------');
        end
    end
end

